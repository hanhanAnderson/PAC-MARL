# PAC: Assisted Value Factorisation with Counterfactual Predictions in Multi-Agent Reinforcement Learning (NeurIPS 2022)


This is the implementation of the paper "PAC: Assisted Value Factorisation with Counterfactual Predictions in Multi-Agent Reinforcement Learning" (NeurIPS 2022). 

## Installation
Install dependencies :

```
conda create -n pymarl python=3.8 -y
conda activate pymarl
bash install_dependencies.sh
```

Install SC2 :

```
bash install_sc2.sh
```
This will download SC2.4.10 into the 3rdparty folder and copy the maps necessary to run over.

## Run the experiments

Run all experiments
```
bash FULL_run.sh
```

or run single experiment

```
python src/main.py --config=ow_qmix --env-config=sc2 with env_args.map_name=6h_vs_8z w=0.5 epsilon_anneal_time=500000 t_max=5005000
```

All results will be stored in the ``results`` folder.

## Kill running and dead process

```
bash clean.sh
```


## Acknowledgement

This code base is implemented based on pymarl2(https://github.com/hijkzzz/pymarl2)
