conf = {
    "WORK_PATH": "./work_oumvlp",
    "CUDA_VISIBLE_DEVICES": "0,1",
    "data": {
        'dataset_path': "/data-tmp/OUMVLP/",
        'resolution': '64',
        'dataset': 'OUMVLP',
        # In CASIA-B, data of subject #5 is incomplete.
        # Thus, we ignore it in training.
        # For more detail, please refer to
        # function: utils.data_loader.load_data
        'pid_num': 5153,
        'pid_shuffle': False,
    },
    "model": {
        'hidden_dim': 256,
        'lr': 1e-4,
        'hard_or_full_trip': 'full',
        'batch_size': (24, 16),
        'restore_iter': 86000,
        'total_iter': 350000,
        'margin': 0.2,
        'num_workers': 3,
        'frame_num': 30,
        'model_name': 'GaitSet',
    },
}
