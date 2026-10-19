args = {
    'data': {
        # 'dataset': 'Chengdu',
        # 'traj_path1': '/xxxxxxx',
        # 'head_path2': '/xxxxxxx',
        # 'traj_length': 200,
        # Change for DiffTraj (NYC): run name (enriched moves, train split only; no space, so paths stay simple),
        # the files made by prepare_data.py, and 256 points per trajectory
        'dataset': 'NYC_moves_train',
        'traj_path1': '/traj.npy',
        'head_path2': '/head.npy',
        'traj_length': 256,
        'channels': 2,
        'uniform_dequantization': False,
        'gaussian_dequantization': False,
        'num_workers': True,
    },
    'model': {
        'type': "simple",
        'attr_dim': 8,
        'guidance_scale': 3,
        'in_channels': 2,
        'out_ch': 2,
        'ch': 128,
        'ch_mult': [1, 2, 2, 2],
        'num_res_blocks': 2,
        'attn_resolutions': [16],
        'dropout': 0.1,
        'var_type': 'fixedlarge',
        'ema_rate': 0.9999,
        'ema': True,
        'resamp_with_conv': True,
    },
    'diffusion': {
        'beta_schedule': 'linear',
        'beta_start': 0.0001,
        'beta_end': 0.05,
        'num_diffusion_timesteps': 500,
    },
    'training': {
        # 'batch_size': 1024,
        # 'n_epochs': 200,
        # Change for DiffTraj (NYC): settings of the best NYC run (05-04-22-31-43)
        'batch_size': 256,
        'n_epochs': 2000,
        'n_iters': 5000000,
        'snapshot_freq': 5000,
        'validation_freq': 2000,
    },
    'sampling': {
        'batch_size': 64,
        'last_only': True,
    }
}
