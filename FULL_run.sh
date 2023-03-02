python src/main.py --config=ow_qmix --env-config=sc2 with env_args.map_name=6h_vs_8z w=0.5 epsilon_anneal_time=500000 t_max=5005000
sleep 30

python src/main.py --config=ow_qmix --env-config=sc2 with env_args.map_name=corridor w=0.5 epsilon_anneal_time=100000 t_max=5005000 
sleep 30

python src/main.py --config=ow_qmix --env-config=sc2 with env_args.map_name=3s_vs_5z w=0.5 epsilon_anneal_time=100000 t_max=3005000
sleep 30

python src/main.py --config=ow_qmix --env-config=sc2 with env_args.map_name=27m_vs_30m w=0.5 epsilon_anneal_time=100000 t_max=2005000
sleep 30

python src/main.py --config=ow_qmix --env-config=sc2 with env_args.map_name=5m_vs_6m w=0.5 epsilon_anneal_time=100000 t_max=4005000
sleep 30

python src/main.py --config=ow_qmix --env-config=sc2 with env_args.map_name=MMM2 w=0.5 epsilon_anneal_time=100000 t_max=2005000

