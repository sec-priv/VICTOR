import os

import argparse

import sys
new_path = './'
sys.path.append(new_path)

from data.utils import inject_video_noise



def parse_args():
    """Arguments"""
    parser = argparse.ArgumentParser(description="Obtain our method's performance.")
    parser.add_argument('--path', nargs='?', default='./data/',
                        help='Input data path.')
    parser.add_argument('--dataset', nargs='?', default='hmdb51',
                        help='Choose a dataset.')
    
    parser.add_argument('--category', nargs='?', default='jump',
                        help='The corresponding category of the video.')
    parser.add_argument('--name', nargs='?', default='THE_PROTECTOR_jump_f_nm_np1_fr_bad_96',
                        help='The specific video name.')
    
    parser.add_argument('--epsilon', type=float, default=10,
                        help='Perturbation budget.')
    
    parser.add_argument('--rx', type=int, default=32,
                        help='lambda x.')
    parser.add_argument('--ry', type=int, default=32,
                        help='lambda y.')
    parser.add_argument('--tf', type=float, default=0.2,
                        help='temporal freq.')
    parser.add_argument('--octaves', type=int, default=2,
                        help='Octaves.')
    parser.add_argument('--sine', type=float, default=1.0,
                        help='Sine frequency.')
    
    parser.add_argument('--seed_n', type=int, default=1,
                        help='Seed.')
    

    
    return parser.parse_args()



if __name__ == '__main__':


    args = parse_args()

    root_dir = args.path
    dataset_name = args.dataset
    
    category = args.category
    video_name = args.name
    
    epsilon = args.epsilon
    rx = args.rx
    ry = args.ry
    tf = args.tf
    octaves = args.octaves
    sine = args.sine
    seed_n = args.seed_n

    
    root_dir = args.path 
    root_path = root_dir + '%s/' %(dataset_name)


    print("params: %s " % (args))
    
    mod_params = f'e{epsilon}_rx{rx}_ry{ry}_tf{tf}_oc{octaves}_sine{sine}_s{seed_n}'
    
    if dataset_name in ['hmdb51','ucf101']:
    
        input_video_path = root_path + f'videos/{category}/{video_name}.avi'
        
        output_video_path = root_path + f'videos/modified/{video_name}_{mod_params}.avi'
        
    elif dataset_name == 'sthv2':
        
        input_video_path = root_path + f'videos/{video_name}.webm'
        
        output_video_path = root_path + f'videos/modified/{video_name}_{mod_params}.webm'
    
    inject_video_noise(input_video_path, output_video_path, epsilon, tf, ry, rx, octaves, sine, seed_n, print_flag=True)
    
