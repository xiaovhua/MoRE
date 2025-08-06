export CUDA_VISIBLE_DEVICES=2

# # CNN
# load_model=/home/sunzhe/vhua/project/mprotonet/results/models/CNN_51a11baf
# layers=6

# # ProtoPNet
# load_model=/home/sunzhe/vhua/project/mprotonet/results/models/MProtoNet3D_pm1_51a11baf 
# layers=6

# # XProtoNet
# load_model=/home/sunzhe/vhua/project/mprotonet/results/models/MProtoNet3D_pm2_51a11baf
# layers=6

# # MProtoNet
# load_model=/home/sunzhe/vhua/project/mprotonet/results/models/mprotonet/l5_1/MProtoNet3D_pm5_c885bfda
# layers=5

# # MoRE-Net
layers=5
# # 1. tsmambas + aux + al
load_model=/home/sunzhe/vhua/project/mprotonet/results/models/mamba/4_no_mambas/MProtoNet3D_pm6_7797d809
# # 2. tsmambas + al
# load_model=/home/sunzhe/vhua/project/mprotonet/results/models/mamba/5_no_mambas_no_aux/MProtoNet3D_pm6_7797d809
# # 3. tsmambas
# load_model=/home/sunzhe/vhua/project/mprotonet/results/models/mamba/6_no_mambas_no_aux_no_al/MProtoNet3D_pm6_7797d809

mamba_dim=64
align_mode=0.5

python ./test.py -m MProtoNet3D_pm6 -n 100 -p "{'batch_size': [8], 'lr': [0.001], 'wd': [0.01], 'features': ['resnet152_ri'], 'n_layers': [$layers], 'prototype_shape': [(30, 128, 1, 1, 1)], 'f_dist': ['cos'], 'topk_p': [1], 'coefs': [{'cls': 1, 'clst': 0.8, 'sep': -0.08, 'L1': 0.01, 'map': 0.5, 'OC': 0.05, 'AL': 0.0}]}" --save-model 1 --load-model $load_model --mamba_dim $mamba_dim --align_mode $align_mode

