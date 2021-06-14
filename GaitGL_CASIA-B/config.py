conf = {
        "WORK_PATH": "./work_angle_lt_mix_05cls_8_8_02lr_500/",
        "CUDA_VISIBLE_DEVICES": "0,1,6,7",

        "data": {
            'dataset_path': "/mnt/data/ctr/gait/CASIA_B/",
            'resolution': '64',
            'dataset': 'CASIA-B',
            'pid_num': 73,
            'pid_shuffle': False,
        },
        "model": {
            'lr': 3e-4,
            'hard_or_full_trip': 'full',
            'batch_size': (8, 8),
            'restore_iter': 0,
            'total_iter': 60000,
            'margin': 0.2,
            'num_workers': 0,
            'frame_num': 30,
            
            
            'hidden_dim':256,
            'model_name': 'GaitGL_CASIA'  



        }
    }
