'''
* @name: opts.py
* @description: Hyperparameter configuration. Note: For hyperparameter settings, please refer to the appendix of the paper.
'''


import argparse

def parse_opts():
    parser = argparse.ArgumentParser()
    arguments = {
        'dataset': [
            dict(name='--datasetName',        
                 type=str,
                 default='sims',
                 help='mosi, mosei or sims'),
            dict(name='--dataPath',
                 default="D:/code/SDHM/pkl汇总/sims/unaligned_39_with_bert.pkl",
                 type=str,
                 help=' '),
            dict(name='--seq_lens',     
                 default=[50,50, 50],
                 type=list,
                 help=' '),
            dict(name='--num_workers',
                 default=8,
                 type=int,
                 help=' '),
           dict(name='--train_mode',
                 default="regression",
                 type=str,
                 help=' '),
            dict(name='--test_checkpoint',
                 default="./checkpoint/test/SIMS_Acc7_Best.pth",
                 type=str,
                 help=' '),
            dict(name='--load_checkpoint',
                 default=None,
                 type=str,
                 help='Path to load model checkpoint for training'),
        ],
        'network': [
            dict(name='--CUDA_VISIBLE_DEVICES',        
                 default='6',#6
                 type=str),
            dict(name='--fusion_layer_depth',
                 default=2,
                 type=int),
            dict(name='--sce_diffusion',
                 default=True,
                 type=bool,     
                 help='Whether to use diffusion enhancement for sce'),
            dict(name='--diffusion_beta_start',
                 default=0.0001,
                 type=float,
                 help='Initial beta for diffusion'),
            dict(name='--diffusion_beta_end',
                 default=0.02,
                 type=float,
                 help='Final beta for diffusion'),
            dict(name='--diffusion_noise_schedule',
                 default='linear',
                 type=str,
                 choices=['linear', 'cosine'],
                 help='Noise schedule for diffusion')
        ],

        'common': [
            dict(name='--project_name',    
                 default='SDHM_Demo',
                 type=str
                 ),
           dict(name='--is_test',    
                 default=1,
                 type=int
                 ),
            dict(name='--seed',  # try different seeds
                 default=18,
                 type=int
                 ),
            dict(name='--models_save_root',
                 default='./diffusion——withoutmoe',
                 type=str
                 ),
            dict(name='--batch_size',
                 default=64,
                 type=int,
                 help=' '),
            dict(
                name='--n_threads',
                default=3,#3
                type=int,
                help='Number of threads for multi-thread loading',
            ),
            dict(name='--lr',
                 type=float,
                 default=1e-4),
            dict(name='--weight_decay',
                 type=float,
                 default=1e-4),
            dict(
                name='--n_epochs',
                default=80,
                type=int,
                help='Number of total epochs to run',
            )
        ]
    }

    for group in arguments.values():
        for argument in group:
            name = argument['name']
            del argument['name']
            parser.add_argument(name, **argument)

    # Use parse_known_args to ignore unknown arguments from other scripts
    args, unknown = parser.parse_known_args()
    if unknown:
        print(f"Warning: opts.py ignored unknown arguments: {unknown}")
    return args
