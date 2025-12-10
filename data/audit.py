import os

import numpy as np

import argparse

from mmaction.apis import inference_recognizer, init_recognizer

from utils import load_class_index, load_class_index_v2, cal_delta_prob, cal_delta_prob_v2

from scipy.stats import wilcoxon

def parse_args():
    """Arguments"""
    parser = argparse.ArgumentParser(description="Obtain our method's performance.")
    parser.add_argument('--path', nargs='?', default='./data/',
                        help='Input data path.')
    parser.add_argument('--dataset', nargs='?', default='hmdb51',
                        help='Choose a dataset.')
    
    # parser.add_argument('--category', nargs='?', default='jump',
    #                     help='The corresponding category of the video.')
    # parser.add_argument('--name', nargs='?', default='THE_PROTECTOR_jump_f_nm_np1_fr_bad_96',
    #                     help='The specific video name.')
    
    # parser.add_argument('--epsilon', type=float, default=10,
    #                     help='Perturbation budget.')
    
    # parser.add_argument('--rx', type=int, default=32,
    #                     help='lambda x.')
    # parser.add_argument('--ry', type=int, default=32,
    #                     help='lambda y.')
    # parser.add_argument('--tf', type=float, default=0.2,
    #                     help='lambda t.')
    # parser.add_argument('--octaves', type=int, default=2,
    #                     help='Octaves.')
    # parser.add_argument('--sine', type=float, default=1.0,
    #                     help='Sine frequency.')
    
    # parser.add_argument('--seed_n', type=int, default=1,
    #                     help='Seed.')
    
    
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
    
    parser.add_argument('--hb', type=float, default=0.05,
                        help='The upper bound.')
    parser.add_argument('--using_clip', nargs='?', default='True',
                        help='Whether to clip the h.')
    parser.add_argument('--using_pp', nargs='?', default='True',
                        help='Whether to post processing.')
    
    
    return parser.parse_args()


if __name__ == '__main__':


    args = parse_args()

   
    dataset_name = args.dataset
    
    root_dir = args.path 
    root_path = root_dir + '%s/' %(dataset_name)
    
    device = args.device
    model_type = args.model_type
    config_name = args.config_name
    
    weight_dir = args.weight_dir
    weight_name = args.weight_name
    
    hb = args.hb
    using_clip = eval(args.using_clip)
    using_pp = eval(args.using_pp)


    print("params: %s " % (args))

    
    config_file = f'configs/recognition/{model_type}/{config_name}.py'
    weight_file = f'work_dirs/{weight_dir}/{weight_name}.pth'
    
    print(f'config:{config_file}, weight:{weight_file}')
    
    if dataset_name in ['hmdb51','ucf101']:
    
        class_ind_path = 'data/%s/annotations/classInd.txt' %(dataset_name)

        class_dic, act_dic = load_class_index(class_ind_path)
        
    elif dataset_name == 'sthv2':
        
        sel_path = 'dataset1_train0' # The selected path can be adjusted according to the actual requirement
        
        label_path = 'data/%s/label_path/%s.txt' %(dataset_name,sel_path)
        
        class_dic, act_dic = load_class_index_v2(label_path)
    
    model = init_recognizer(config_file, weight_file, device=device)
    
    
    # Step 1: Obtain the threshold based on the reference samples
    
    ref_act_list = ['jump',...]
    
    ref_orig_list = ['THE_PROTECTOR_jump_f_nm_np1_fr_bad_96',...]
    
    ref_param_list = ['e10_rx32_ry32_tf0.2_oc2_sine1.0_s1',...]
    
    root_path1 = root_path + 'videos/'
    
    
    ref_all_delta1, ref_all_prob1, ref_all_prob2 = cal_delta_prob(ref_act_list,ref_orig_list,ref_param_list,class_dic,model,root_path1,modify_dir='modified',print_flag=False)
    
    
    
    # dataset: sthv2
    # ref_orig_list = ['1',...]
    
    # ref_param_list = ['e10_rx32_ry32_tf0.2_oc2_sine1.0_s1',...]
    
    # root_path1 = root_path + 'videos/'
    
    # ref_all_delta1, ref_all_prob1, ref_all_prob2 = cal_delta_prob_v2(ref_orig_list,ref_param_list,class_dic,model,root_path1,modify_dir='modified',print_flag=False)
    
    
    
    h_mean = np.mean(ref_all_delta1)
    if using_clip:
        h = np.clip(h_mean,-hb,hb)
        
        
    # Step 2: Calculate the probability difference for the modified samples
    sel_act_list = ['kiss',...]
    
    sel_orig_list = ['TVs_Best_Kisses_Top_50-_(40_to_31)_kiss_h_cm_np2_fr_goo_9',...]
    
    sel_param_list = ['e10_rx32_ry32_tf0.2_oc2_sine1.0_s1',...]
    
    sel_all_delta1, sel_all_prob1, sel_all_prob2 = cal_delta_prob(sel_act_list,sel_orig_list,sel_param_list,class_dic,model,root_path1,modify_dir='modified',print_flag=False)
    
    # dataset: sthv2
    # sel_orig_list = ['2',...]
    
    # sel_param_list = ['e10_rx32_ry32_tf0.2_oc2_sine1.0_s1',...]
    
    # root_path1 = root_path + 'videos/'
    
    # sel_all_delta1, sel_all_prob1, sel_all_prob2 = cal_delta_prob_v2(sel_orig_list,sel_param_list,class_dic,model,root_path1,modify_dir='modified',print_flag=False)
    
    
    if dataset_name == 'hmdb51':
        small_th = 1/51
    elif dataset_name == 'ucf101':
        small_th = 1/101
        
    r1 = 0.01
    
    if using_pp:
        indices = np.where((sel_all_prob1 < small_th) & (sel_all_prob2 < small_th))[0]
        if len(indices) > 0:
            sel_all_delta1[indices] = (1+r1) * h_mean
            
    h_arr = np.zeros_like(sel_all_delta1) + h
    
    print(f'h:{h}, h_mean:{h_mean}')
            
    # Step 3: Hypothesis Test
    w_stat, p_value_w = wilcoxon(h_arr-sel_all_delta1, alternative='greater')
    
    p_val_th = 0.01
    
    if p_value_w >= p_val_th:
        final_output_w = 0
    else:
        final_output_w = 1
        
    print(f'final output:{final_output_w}, w_stat:{w_stat}, p_value:{p_value_w}')