import numpy as np
import cv2

from mmaction.apis import inference_recognizer, init_recognizer


def cal_delta_prob(act_list,orig_list,modify_list,class_dic,model,root_path1,modify_dir='modified',print_flag=False,video_type='avi'):
    act_len = len(act_list)
    
    all_delta1 = np.zeros([act_len])
    all_prob1 = np.zeros_like(all_delta1)
    all_prob2 = np.zeros_like(all_delta1)
    
    for i in range(act_len):
        act = act_list[i]
        act_ind = class_dic[act] 
        orig_name = orig_list[i]
        modify_name = orig_name + '_' + modify_list[i]
        
        video1_path = root_path1 + '%s/%s.%s' %(act,orig_name,video_type)
        
        video2_path = root_path1 + modify_dir + '%s/%s.%s' %(act,modify_name,video_type)
        
        prob1 = inference_recognizer(model, video1_path).pred_score[act_ind].item()

        prob2 = inference_recognizer(model, video2_path).pred_score[act_ind].item()
        
        delta_prob = prob1 - prob2
        
        all_delta1[i] = delta_prob
      
        all_prob1[i] = prob1
        all_prob2[i] = prob2
        
        if print_flag:
            print(f'*****act:{act}, orig:{orig_name}, modify:{modify_list[i]}*****')
            print(f'delta prob:{delta_prob}, prob1:{prob1}, prob2:{prob2}')
        
    return all_delta1, all_prob1, all_prob2
    
        
def cal_delta_prob_v2(orig_list,modify_list,class_dic,model,root_path1,modify_dir='modified',print_flag=False,video_type='webm'):
    act_len = len(orig_list)
    
    all_delta1 = np.zeros([act_len])
    all_prob1 = np.zeros_like(all_delta1)
    all_prob2 = np.zeros_like(all_delta1)
    
    for i in range(act_len):
        
        orig_name = orig_list[i]
        act_ind = class_dic[orig_name] 
        modify_name = orig_name + '_' + modify_list[i]
        
        video1_path = root_path1 + '%s.%s' %(orig_name,video_type)
        
        video2_path = root_path1 + modify_dir + '%s.%s' %(modify_name,video_type)
        
        prob1 = inference_recognizer(model, video1_path).pred_score[act_ind].item()

        prob2 = inference_recognizer(model, video2_path).pred_score[act_ind].item()
        
        delta_prob = prob1 - prob2
        
        all_delta1[i] = delta_prob
      
        all_prob1[i] = prob1
        all_prob2[i] = prob2
        
        if print_flag:
            print(f'*****orig:{orig_name}, modify:{modify_list[i]}*****')
            print(f'delta prob:{delta_prob}, prob1:{prob1}, prob2:{prob2}')
        
    return all_delta1, all_prob1, all_prob2
    



def load_class_index(file_path):
    class_dict = {}
    act_dict = {}
    i = 0
    
    with open(file_path, 'r') as file:
        for line in file:
            parts = line.strip().split()
            if len(parts) == 2:
                index, action = parts
                class_dict[action] = int(index) - 1
                act_dict[i] = action
                i = i + 1
    return class_dict, act_dict


def load_class_index_v2(file_path):
    class_dict = {}
    act_dict = {}
    i = 0
    
    with open(file_path, 'r') as file:
        for line in file:
            parts = line.strip().split()
            if len(parts) == 2:
                video_name, index = parts
                action = video_name.strip().split()[0]
                class_dict[action] = int(index)
                act_dict[i] = action
                i = i + 1
    return class_dict, act_dict


def inject_video_noise(video1_path, output_path, epsilon=10, tf=0.2, res_y=32, res_x=32, octaves=2, sine_freq=1, seed=0, print_flag=False):

    cap1 = cv2.VideoCapture(video1_path)
 
    width = int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap1.get(cv2.CAP_PROP_FPS)
    frame_count1 = int(cap1.get(cv2.CAP_PROP_FRAME_COUNT))

  
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    all_frames1 = []
   
    while True:
        ret1, frame1 = cap1.read()
        if not ret1:
            break
        all_frames1.append(frame1)
        
    frame1_len = len(all_frames1)
    
    all_frames1 = np.array(all_frames1,dtype=np.uint8)
    
    all_noise_frames = add_perlin_noise_video(all_frames1, epsilon, tf, res_y, res_x, octaves, sine_freq, seed)
    
    fc = min(frame1_len,frame_count1)
    for fi in range(fc):
        frame = all_noise_frames[fi]
        out.write(frame)
        
    cap1.release()
    out.release()
    if print_flag:
        print(f"Save: {output_path} done.")
    
        
        
        
def add_perlin_noise_video(video, epsilon=10, tf=0.2, res_y=32, res_x=32, octaves=2, sine_freq=1, seed=0):

    T, H, W, C = video.shape
   
    noise = perlin_noise_3d((T, H, W), tf, res_y, res_x, octaves, sine_freq, seed)
    noise = noise[..., np.newaxis]  
    noise = (noise - 0.5) * 2 * epsilon  

    noisy_video = video.astype(np.float32) + noise
    return np.clip(noisy_video, 0, 255).astype(np.uint8)



def perlin_noise_3d(shape, tf=0.2, res_y=32, res_x=32, octaves=2, sine_freq=1, seed=0):

    def fade(t):
        return 6 * t**5 - 15 * t**4 + 10 * t**3

    T, H, W = shape

    res_t = res_x
    t = np.linspace(0, res_t*tf , T) 
    y = np.linspace(0, res_y, H)
    x = np.linspace(0, res_x, W)
    
    t_grid, y_grid, x_grid = np.meshgrid(t, y, x, indexing='ij')

  
    np.random.seed(seed)
    gradients = np.random.randn(res_x, res_y, res_x, 3) 
    gx, gy, gt = gradients[..., 0], gradients[..., 1], gradients[..., 2]


    ix, iy, it = (x_grid.astype(int) % res_x, y_grid.astype(int) % res_y, t_grid.astype(int) % res_t)
    ixp1, iyp1, itp1 = (ix + 1) % res_x, (iy + 1) % res_y, (it + 1) % res_t


    dx, dy, dt = x_grid - ix, y_grid - iy, t_grid - it


    dot_tl = gx[ix, iy, it] * dx + gy[ix, iy, it] * dy + gt[ix, iy, it] * dt
    dot_tr = gx[ixp1, iy, it] * (dx - 1) + gy[ixp1, iy, it] * dy + gt[ixp1, iy, it] * dt
    dot_bl = gx[ix, iyp1, it] * dx + gy[ix, iyp1, it] * (dy - 1) + gt[ix, iyp1, it] * dt
    dot_br = gx[ixp1, iyp1, it] * (dx - 1) + gy[ixp1, iyp1, it] * (dy - 1) + gt[ixp1, iyp1, it] * dt


    u, v, w = fade(dx), fade(dy), fade(dt)
    noise = (1 - u) * ((1 - v) * dot_tl + v * dot_bl) + u * ((1 - v) * dot_tr + v * dot_br)
    

    perlin = np.zeros_like(noise)
    freq = 1
    for _ in range(octaves):
        perlin += noise / freq
        freq *= 2


    perlin = np.sin(perlin * 2 * np.pi * sine_freq)
    
    return (perlin - perlin.min()) / (perlin.max() - perlin.min()) 

