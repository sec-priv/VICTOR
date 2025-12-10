import os

import argparse

from mmaction.apis import inference_recognizer, init_recognizer

from utils import load_class_index, load_class_index_v2

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
                        help='lambda t.')
    parser.add_argument('--octaves', type=int, default=2,
                        help='Octaves.')
    parser.add_argument('--sine', type=float, default=1.0,
                        help='Sine frequency.')
    
    parser.add_argument('--seed_n', type=int, default=1,
                        help='Seed.')
    
    
    parser.add_argument('--device', nargs='?', default='cuda:0',
                        help='The specific video name.')
    parser.add_argument('--model_type', nargs='?', default='i3d',
                        help='The evaluation model type.')
    parser.add_argument('--config_name', nargs='?', default='i3d_imagenet-pretrained-r50_8xb8-32x2x1-100e_hmdb51-rgb_d1',
                        help='The config file name.')
    parser.add_argument('--weight_dir', nargs='?', default='i3d_imagenet-pretrained-r50_8xb8-32x2x1-100e_hmdb51-rgb_d1_train',
                        help='The weight dir.')
    parser.add_argument('--weight_name', nargs='?', default='epoch_150',
                        help='The weight name.')
    
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
    
    device = args.device
    model_type = args.model_type
    config_name = args.config_name
    
    weight_dir = args.weight_dir
    weight_name = args.weight_name


    print("params: %s " % (args))
    
    mod_params = f'e{epsilon}_rx{rx}_ry{ry}_tf{tf}_oc{octaves}_sine{sine}_s{seed_n}'
    
    config_file = f'configs/recognition/{model_type}/{config_name}.py'
    weight_file = f'work_dirs/{weight_dir}/{weight_name}.pth'
    
    print(f'config:{config_file}, weight:{weight_file}')
    
    if dataset_name in ['hmdb51','ucf101']:
        
        orig_video_path = root_path + f'videos/{category}/{video_name}.avi'
        
        noise_video_path = root_path + f'videos/modified/{video_name}_{mod_params}.avi'
        
        class_ind_path = 'data/%s/annotations/classInd.txt' %(dataset_name)

        class_dic, act_dic = load_class_index(class_ind_path)
        
        act_ind = class_dic[category]
        
    elif dataset_name == 'sthv2':
        
        orig_video_path = root_path + f'videos/{video_name}.webm'
        
        noise_video_path = root_path + f'videos/modified/{video_name}_{mod_params}.webm'
        
        sel_path = 'dataset1_train0' # The selected path can be adjusted according to the actual requirement
        
        label_path = 'data/%s/label_path/%s.txt' %(dataset_name,sel_path)
        
        class_dic, act_dic = load_class_index_v2(label_path)
        
        act_ind = class_dic[video_name]
        
        
    
    model = init_recognizer(config_file, weight_file, device=device)
    
    prob1 = inference_recognizer(model, orig_video_path).pred_score[act_ind].item()
    
    prob2 = inference_recognizer(model, noise_video_path).pred_score[act_ind].item()
    
    delta_prob = prob1 - prob2
    
    print(f'[prob1:{prob1}] orig_video_path:{orig_video_path}')
    print(f'[prob2:{prob2}] noise_video_path:{noise_video_path}')
    print(f'delta prob:{delta_prob}')