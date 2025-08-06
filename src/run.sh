export CUDA_VISIBLE_DEVICES=2

mamba_dim=64

# # CNN
# python ./tumor_cls.py -m CNN -n 100 -p "{'batch_size': [32], 'lr': [0.001], 'wd': [0.01], 'features': ['resnet152_ri'], 'n_layers': [6], 'prototype_shape': [(30, 128, 1, 1, 1)], 'f_dist': ['cos'], 'topk_p': [1], 'coefs': [{'cls': 1, 'clst': 0.8, 'sep': -0.08, 'L1': 0.01, 'map': 0.5, 'OC': 0.05}]}" --save-model 1 --mamba_dim $mamba_dim

# # ProtoPNet, XProtoNet
# python ./tumor_cls.py -m MProtoNet3D_pm2 -n 100 -p "{'batch_size': [32], 'lr': [0.001], 'wd': [0.01], 'features': ['resnet152_ri'], 'n_layers': [6], 'prototype_shape': [(30, 128, 1, 1, 1)], 'f_dist': ['cos'], 'topk_p': [1], 'coefs': [{'cls': 1, 'clst': 0.8, 'sep': -0.08, 'L1': 0.01, 'map': 0.5, 'OC': 0.05}]}" --save-model 1 --mamba_dim $mamba_dim

# # MProtoNet
# python ./tumor_cls.py -m MProtoNet3D_pm5 -n 100 -p "{'batch_size': [32], 'lr': [0.001], 'wd': [0.01], 'features': ['resnet152_ri'], 'n_layers': [6], 'prototype_shape': [(30, 128, 1, 1, 1)], 'f_dist': ['cos'], 'topk_p': [1], 'coefs': [{'cls': 1, 'clst': 0.8, 'sep': -0.08, 'L1': 0.01, 'map': 0.5, 'OC': 0.05}]}" --save-model 1 --mamba_dim $mamba_dim

# # MoRENet 
python ./tumor_cls.py -m MProtoNet3D_pm6 -n 100 -p "{'batch_size': [8], 'lr': [0.001], 'wd': [0.01], 'features': ['resnet152_ri'], 'n_layers': [5], 'prototype_shape': [(30, 128, 1, 1, 1)], 'f_dist': ['cos'], 'topk_p': [1], 'coefs': [{'cls': 1, 'clst': 0.8, 'sep': -0.08, 'L1': 0.01, 'map': 0.5, 'OC': 0.05}]}" --save-model 1 --mamba_dim $mamba_dim
