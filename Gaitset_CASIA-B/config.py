
conf = {
    "WORK_PATH": "./work_angle_8_16/",
    "CUDA_VISIBLE_DEVICES": "0,1,6",
    "data": {
        'dataset_path': "/mnt/data/ctr/gait/CASIA_B",
        'resolution': '64',
        'dataset': 'CASIA-B',
        # In CASIA-B, data of subject #5 is incomplete.
        # Thus, we ignore it in training.
        # For more detail, please refer to
        # function: utils.data_loader.load_data
        'pid_num': 73,
        'pid_shuffle': False,
    },
    "model": {
        'hidden_dim': 256,
        'lr': 1e-4,
        'hard_or_full_trip': 'full',
        'batch_size': (8, 16),
        'restore_iter': 0,
        'total_iter': 100000,
        'margin': 0.2,
        'num_workers': 3,
        'frame_num': 30,
        'model_name': 'GaitSet', #'GaitSet', # 'GaitPart'
    },
}
'''

conf = {
    "WORK_PATH": "./work_p1_lstm_8_16/",
    "CUDA_VISIBLE_DEVICES": "6,7",
    "data": {
        'dataset_path': "/mnt/data/ctr/gait/CASIA_B/",
        'resolution': '64',
        'dataset': 'CASIA-B',
        # In CASIA-B, data of subject #5 is incomplete.
        # Thus, we ignore it in training.
        # For more detail, please refer to
        # function: utils.data_loader.load_data
        'pid_num': 73,
        'pid_shuffle': False,
    },
    "model": {
        'hidden_dim': 256,
        'lr': 1e-4,
        'hard_or_full_trip': 'full',
        'batch_size': (8, 16),
        'restore_iter': 0,
        'total_iter': 200000,
        'margin': 0.2,
        'num_workers': 3,
        'frame_num': 30,
        'model_name': 'GaitSet_lstm', #'GaitSet', # 'GaitPart'
    },
}

conf = {
    "WORK_PATH": "./work_p1_lstm256_8_16_08cls/",
    "CUDA_VISIBLE_DEVICES": "8",
    "data": {
        'dataset_path': "/mnt/data/ctr/gait/CASIA_B/",
        'resolution': '64',
        'dataset': 'CASIA-B',
        # In CASIA-B, data of subject #5 is incomplete.
        # Thus, we ignore it in training.
        # For more detail, please refer to
        # function: utils.data_loader.load_data
        'pid_num': 73,
        'pid_shuffle': False,
    },
    "model": {
        'hidden_dim': 256,
        'lr': 1e-4,
        'hard_or_full_trip': 'full',
        'batch_size': (8, 16),
        'restore_iter': 0,
        'total_iter': 300000,
        'margin': 0.2,
        'num_workers': 3,
        'frame_num': 30,
        'model_name': 'GaitSet_lstm', #'GaitSet', # 'GaitPart'
    },
}
'''

