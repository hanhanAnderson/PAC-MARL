import copy

from torch.autograd import Variable

from components.episode_buffer import EpisodeBatch
from modules.mixers.vdn import VDNMixer
from modules.mixers.qmix import QMixer
from modules.mixers.qmix_central_no_hyper import QMixerCentralFF
from utils.rl_utils import build_td_lambda_targets
import torch as th
from torch.optim import RMSprop, Adam
from collections import deque
from controllers import REGISTRY as mac_REGISTRY
import torch.distributions as D
from utils.th_utils import get_parameters_num

class MAXQLearner:
    def __init__(self, mac, scheme, logger, args):
        self.args = args
        self.mac = mac
        self.logger = logger

        self.mac_params = list(mac.parameters())

        self.last_target_update_episode = 0
        groups = None
        self.critic_mac = mac_REGISTRY[args.critic_mac](scheme, groups, args)
        # mac_REGISTRY[args.central_mac]
        self.s_mu = th.zeros(1)
        self.s_sigma = th.ones(1)
        self.log_alpha =th.tensor(args.alpha_init, dtype=th.float32,requires_grad=True)
        self.log_alpha.requires_grad=True
        # self.alpha = th.ones(1, dtype=th.float32)
        self.alpha = th.tensor(0.2, dtype=th.float32) # just for init use
        self.alpha_optimizer = th.optim.Adam([self.log_alpha], lr=1e-4)

        self.params = list(self.critic_mac.parameters())
        self.mixer = None
        assert args.mixer is not None
        if args.mixer is not None:
            if args.mixer == "vdn":
                self.mixer = VDNMixer()
            elif args.mixer == "qmix":
                self.mixer = QMixer(args)
            else:
                raise ValueError("Mixer {} not recognised.".format(args.mixer))
            self.mixer_params = list(self.mixer.parameters())
            self.params += list(self.mixer.parameters())
            self.target_mixer = copy.deepcopy(self.mixer)

        # a little wasteful to deepcopy (e.g. duplicates action selector), but should work for any MAC
        self.target_mac = copy.deepcopy(mac)

        # Central Q
        self.central_mac = None
        if self.args.central_mixer in ["ff", "atten"]:
            if self.args.central_loss == 0:
                self.central_mixer = self.mixer
                self.central_mac = self.mac
                self.target_central_mac = self.target_mac
            else:
                if self.args.central_mixer == "ff":
                    self.central_mixer = QMixerCentralFF(args) # Feedforward network that takes state and agent utils as input
                # elif self.args.central_mixer == "atten":
                    # self.central_mixer = QMixerCentralAtten(args)
                else:
                    raise Exception("Error with central_mixer")

                assert args.central_mac == "basic_central_mac"
                self.central_mac = mac_REGISTRY[args.central_mac](scheme, args) # Groups aren't used in the CentralBasicController. Little hacky
                self.target_central_mac = copy.deepcopy(self.central_mac)
                self.params += list(self.central_mac.parameters())
        else:
            raise Exception("Error with qCentral")
        self.params += list(self.central_mixer.parameters())
        self.target_central_mixer = copy.deepcopy(self.central_mixer)

        print('Mixer Size: ')
        print(get_parameters_num(list(self.mixer.parameters()) + list(self.central_mixer.parameters())))
        print(args)

        self.optimiser = Adam(params=self.params, lr=args.lr)
        self.policy_optimiser = Adam(params=self.mac_params, lr=args.lr)
        self.log_stats_t = -self.args.learner_log_interval - 1

        self.grad_norm = 1
        self.mixer_norm = 1
        self.mixer_norms = deque([1], maxlen=100)

    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int):
        # Get the relevant quantities
        rewards = batch["reward"][:, :-1]
        actions = batch["actions"][:, :-1]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        avail_actions = batch["avail_actions"]

        critic_mac_out = []
        mu_out = []
        sigma_out = []
        logits_out = []
        m_sample_out = []
        actions_logprobs = []
        # Calculate estimated Q-Values
        mac_out = []
        self.mac.init_hidden(batch.batch_size)

        for t in range(batch.max_seq_length):
            agent_outs = self.mac.forward(batch, t=t)
            mac_out.append(agent_outs)
        mac_out = th.stack(mac_out, dim=1)  # Concat over time

        z = mac_out == 0.0
        z = z.float() * 1e-8
        actions_logprobs = (th.log(mac_out + z))

        self.critic_mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            if self.args.comm and self.args.use_IB:
                critic_agent_outs, (mu, sigma), logits, m_sample = self.critic_mac.forward(batch, t=t)
                mu_out.append(mu)
                sigma_out.append(sigma)
                logits_out.append(logits)
                m_sample_out.append(m_sample)
            else:
                critic_agent_outs = self.critic_mac.forward(batch, t=t)
            critic_mac_out.append(critic_agent_outs)
        critic_mac_out = th.stack(critic_mac_out, dim=1)  # Concat over time
        if self.args.use_IB:
            mu_out = th.stack(mu_out, dim=1)[:, :-1]  # Concat over time
            sigma_out = th.stack(sigma_out, dim=1)[:, :-1]  # Concat over time
            logits_out = th.stack(logits_out, dim=1)[:, :-1]
            m_sample_out = th.stack(m_sample_out, dim=1)[:, :-1]



        # Pick the Q-Values for the actions taken by each agent
        chosen_action_qvals_agents = th.gather(critic_mac_out[:, :-1], dim=3, index=actions).squeeze(3)  # Remove the last dim
        chosen_action_qvals = chosen_action_qvals_agents

        # Calculate the Q-Values necessary for the target
        target_mac_out = []
        self.target_mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            target_agent_outs = self.target_mac.forward(batch, t=t)
            target_mac_out.append(target_agent_outs)
        label_target_max_out = th.stack(target_mac_out[:-1], dim=1) 
        # We don't need the first timesteps Q-Value estimate for calculating targets
        target_mac_out = th.stack(target_mac_out[:], dim=1)  # Concat across time

        # Mask out unavailable actions
        target_mac_out[avail_actions[:, :] == 0] = -9999999  # From OG deepmarl

        # Max over target Q-Values
        if self.args.double_q:
            # Get actions that maximise live Q (for double q-learning)
            mac_out_detach = mac_out.clone().detach()
            mac_out_detach[avail_actions == 0] = -9999999
            cur_max_action_targets, cur_max_actions = mac_out_detach[:, :].max(dim=3, keepdim=True)
            target_max_agent_qvals = th.gather(target_mac_out[:,:], 3, cur_max_actions[:,:]).squeeze(3)
        else:
            raise Exception("Use double q")


        # Central MAC stuff
        central_mac_out = []
        self.central_mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            agent_outs = self.central_mac.forward(batch, t=t)
            central_mac_out.append(agent_outs)
        central_mac_out = th.stack(central_mac_out, dim=1)  # Concat over time
        central_chosen_action_qvals_agents = th.gather(central_mac_out[:, :-1], dim=3, index=actions.unsqueeze(4).repeat(1,1,1,1,self.args.central_action_embed)).squeeze(3)  # Remove the last dim

        central_target_mac_out = []
        self.target_central_mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            target_agent_outs = self.target_central_mac.forward(batch, t=t)
            central_target_mac_out.append(target_agent_outs)
        central_target_mac_out = th.stack(central_target_mac_out[:], dim=1)  # Concat across time
        # Mask out unavailable actions
        central_target_mac_out[avail_actions[:, :] == 0] = -9999999  # From OG deepmarl
        # Use the Qmix max actions
        central_target_max_agent_qvals = th.gather(central_target_mac_out[:,:], 3, cur_max_actions[:,:].unsqueeze(4).repeat(1,1,1,1,self.args.central_action_embed)).squeeze(3)


        # Mix
        chosen_action_qvals = self.mixer(chosen_action_qvals, batch["state"][:, :-1])
        target_max_qvals = self.target_central_mixer(central_target_max_agent_qvals, batch["state"])

        # We use the calculation function of sarsa lambda to approximate q star lambda
        targets = build_td_lambda_targets(rewards, terminated, mask, target_max_qvals,
                                    self.args.n_agents, self.args.gamma, self.args.td_lambda)

        # Td-error
        td_error = (chosen_action_qvals - (targets.detach()))

        mask = mask.expand_as(td_error)

        # 0-out the targets that came from padded data
        masked_td_error = td_error * mask

        # ActorLoss
        q_i_mean_negi_mean = th.sum(mac_out * (self.alpha * actions_logprobs - critic_mac_out), dim=-1)  # (1,60,3) # TODO
        Q_i_mean_negi_mean = self.mixer(q_i_mean_negi_mean, batch["state"])[:,:-1]

        policy_loss = (Q_i_mean_negi_mean * mask).sum() / mask.sum()

        target_entropy = -1. * self.args.n_actions
        # Training central Q
        central_chosen_action_qvals = self.central_mixer(central_chosen_action_qvals_agents, batch["state"][:, :-1])
        central_td_error = (central_chosen_action_qvals - targets.detach())
        central_mask = mask.expand_as(central_td_error)
        central_masked_td_error = central_td_error * central_mask
        central_loss = 0.5 * (central_masked_td_error ** 2).sum() / mask.sum()

        # QMIX loss with weighting
        ws = th.ones_like(td_error) * self.args.w
        if self.args.hysteretic_qmix: # OW-QMIX
            ws = th.where(td_error < 0, th.ones_like(td_error)*1, ws) # Target is greater than current max
            w_to_use = ws.mean().item() # For logging
        else: # CW-QMIX
            is_max_action = (actions == cur_max_actions[:, :-1]).min(dim=2)[0]
            max_action_qtot = self.target_central_mixer(central_target_max_agent_qvals[:, :-1], batch["state"][:, :-1])
            qtot_larger = targets > max_action_qtot
            ws = th.where(is_max_action | qtot_larger, th.ones_like(td_error)*1, ws) # Target is greater than current max
            w_to_use = ws.mean().item() # Average of ws for logging

        qmix_loss = (ws.detach()*(masked_td_error ** 2)).sum() / mask.sum()

        #
        # target_max_qvals_c = copy.copy(target_max_qvals)
        cur_max_actions_cg = cur_max_actions.clone().detach()
        target_max_qvals_cMax = target_max_qvals.clone().detach()
        cur_max_actions_cGlobalMax = cur_max_actions.clone().detach()
        with th.no_grad():
            for agentn in range(self.args.n_agents):
                cur_max_actions_c = cur_max_actions_cg.clone()
                for actionn in range(self.args.n_actions):
                    cur_max_actions_c[:,:,agentn].mul_(0).add_(actionn)
                    central_target_max_agent_qvals_c = th.gather(central_target_mac_out[:,:], 3, cur_max_actions_c[:,:].unsqueeze(4).repeat(1,1,1,1,self.args.central_action_embed)).squeeze(3)
                    target_max_qvals_c = self.target_central_mixer(central_target_max_agent_qvals_c, batch["state"])
                    condition = (target_max_qvals_c > target_max_qvals_cMax)
                    target_max_qvals_cMax = th.where(condition, target_max_qvals_c, target_max_qvals_cMax)
                    cur_max_actions_cGlobalMax = th.where(condition.repeat(1,1,self.args.n_agents).unsqueeze(-1), cur_max_actions_c, cur_max_actions_cGlobalMax)
                    # print("agentn: ", agentn)
                    # print("actionn: ", actionn)

            # th.cuda.empty_cache()
        label_target_actions =  cur_max_actions_cGlobalMax[:,:-1]
        expressiveness_loss = 0
        label_prob = th.gather(logits_out, 3, label_target_actions).squeeze(3)
        expressiveness_loss += (-th.log(label_prob + 1e-6)).sum() / mask.sum()
        # Compute KL divergence
        compactness_loss = D.kl_divergence(D.Normal(mu_out, sigma_out), D.Normal(self.s_mu, self.s_sigma)).sum() / \
                           mask.sum()
        entropy_loss = -D.Normal(self.s_mu, self.s_sigma).log_prob(m_sample_out).sum() / mask.sum()
        comm_beta = 0.001
        comm_entropy_beta= 1e-6
        comm_loss = expressiveness_loss + comm_beta * compactness_loss + comm_entropy_beta * entropy_loss
        comm_loss *= self.args.c_beta *0.01

        self.policy_optimiser.zero_grad()
        policy_loss.backward(retain_graph=True)
        self.policy_optimiser.step()

        loss =  self.args.qmix_loss * qmix_loss + self.args.central_loss * central_loss + comm_loss



        # Optimise
        self.optimiser.zero_grad()
        loss.backward(retain_graph=True)
        # Logging

        mixer_norm = 0
        for p in self.mixer_params:
            param_norm = p.grad.data.norm(2)
            mixer_norm += param_norm.item() ** 2
        mixer_norm = mixer_norm ** (1. / 2)
        self.mixer_norm = mixer_norm
        self.mixer_norms.append(mixer_norm)

        grad_norm = th.nn.utils.clip_grad_norm_(self.params, self.args.grad_norm_clip)
        self.grad_norm = grad_norm

        self.optimiser.step()
        #Alpha_loss
        target_entropy = -1. * self.args.n_actions

        alpha_loss = (th.sum(mac_out.detach() * (-self.log_alpha * (actions_logprobs + target_entropy).detach()), dim=-1)[:,:-1] * mask).sum() / mask.sum()
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        # self.alpha = self.log_alpha.exp()
        self.alpha = self.alpha_optimizer.param_groups[0]['params'][0].exp()

        if t_env < 2000000:
            self.alpha = th.clamp(self.alpha, max = 0.999)
        # if 2000000< t_env < 3500000:
        #     self.alpha = th.clamp(self.alpha, min = 0.49)
        if t_env > 3000000:        
            self.alpha = th.clamp(self.alpha, min = 0.19)
        if (episode_num - self.last_target_update_episode) / self.args.target_update_interval >= 1.0:
            self._update_targets()
            self.last_target_update_episode = episode_num

        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            self.logger.log_stat("loss", loss.item(), t_env)
            self.logger.log_stat("qmix_loss", qmix_loss.item(), t_env)
            self.logger.log_stat("comm_loss", comm_loss.item(), t_env)
            self.logger.log_stat("policy_loss", policy_loss.item(), t_env)
            self.logger.log_stat("central_loss", central_loss.item(), t_env)
            self.logger.log_stat("grad_norm", grad_norm, t_env)
            self.logger.log_stat("mixer_norm", mixer_norm, t_env)
            # self.logger.log_stat("agent_norm", agent_norm, t_env)
            mask_elems = mask.sum().item()
            self.logger.log_stat("td_error_abs", (masked_td_error.abs().sum().item()/mask_elems), t_env)
            self.logger.log_stat("q_taken_mean", (chosen_action_qvals * mask).sum().item()/(mask_elems * self.args.n_agents), t_env)
            self.logger.log_stat("alpha", self.alpha.item(), t_env)
            self.logger.log_stat("alpha_loss", alpha_loss.item(), t_env)
            self.logger.log_stat("target_mean", (targets * mask).sum().item()/(mask_elems * self.args.n_agents), t_env)
            self.logger.log_stat("w_to_use", w_to_use, t_env)
            self.log_stats_t = t_env
        th.cuda.empty_cache()

    def _update_targets(self):
        self.target_mac.load_state(self.mac)
        if self.mixer is not None:
            self.target_mixer.load_state_dict(self.mixer.state_dict())
        if self.central_mac is not None:
            self.target_central_mac.load_state(self.central_mac)
        self.target_central_mixer.load_state_dict(self.central_mixer.state_dict())
        self.logger.console_logger.info("Updated target network")

    def cuda(self):
        self.mac.cuda()
        self.target_mac.cuda()
        if self.mixer is not None:
            self.mixer.cuda()
            self.target_mixer.cuda()
        if self.central_mac is not None:
            self.central_mac.cuda()
            self.target_central_mac.cuda()
        self.central_mixer.cuda()
        self.target_central_mixer.cuda()
        self.s_mu = self.s_mu.cuda()
        self.s_sigma = self.s_sigma.cuda()
        self.critic_mac.cuda()
        self.log_alpha = self.log_alpha.cuda()
        self.alpha = self.alpha.cuda()
        # TODO: Model saving/loading is out of date!
    def save_models(self, path):
        self.mac.save_models(path)
        if self.mixer is not None:
            th.save(self.mixer.state_dict(), "{}/mixer.th".format(path))
        th.save(self.optimiser.state_dict(), "{}/opt.th".format(path))

    def load_models(self, path):
        self.mac.load_models(path)
        # Not quite right but I don't want to save target networks
        self.target_mac.load_models(path)
        if self.mixer is not None:
            self.mixer.load_state_dict(th.load("{}/mixer.th".format(path), map_location=lambda storage, loc: storage))
        self.optimiser.load_state_dict(th.load("{}/opt.th".format(path), map_location=lambda storage, loc: storage))
