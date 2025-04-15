import os
import sys
import numpy as np
import time
import cv2
import pickle
import faulthandler
faulthandler.enable()
from collections import OrderedDict
import torch.distributions.multivariate_normal as torchdist
from toolkits.dataset import LaSOT
from Sot_STGCNN.utils import * 
from Sot_STGCNN.metrics import * 
from Sot_STGCNN.model import sot_stgcnn_raw


otetrack_path = os.path.join(os.path.dirname(__file__), '..','OTETrack')
if otetrack_path not in sys.path:
    sys.path.append(otetrack_path)

unicorn_path = os.path.join(os.path.dirname(__file__), '..','Unicorn')
if unicorn_path not in sys.path:
    sys.path.append(unicorn_path)
unicorn_path = os.path.join(os.path.dirname(__file__), '..','Unicorn',"external_2")
if unicorn_path not in sys.path:
    sys.path.append(unicorn_path)
unicorn_path = os.path.join(os.path.dirname(__file__), '..','Unicorn',"exps","default")
if unicorn_path not in sys.path:
    sys.path.append(unicorn_path)


from OTETrack.lib.test.evaluation.tracker import Tracker as Local_Tracker
from Unicorn.external_2.lib.test.evaluation.tracker import Tracker as Global_Tracker
def _intersection(rects1, rects2):
    assert rects1.shape == rects2.shape
    x1 = np.maximum(rects1[..., 0], rects2[..., 0])
    y1 = np.maximum(rects1[..., 1], rects2[..., 1])
    x2 = np.minimum(rects1[..., 0] + rects1[..., 2],
                    rects2[..., 0] + rects2[..., 2])
    y2 = np.minimum(rects1[..., 1] + rects1[..., 3],
                    rects2[..., 1] + rects2[..., 3])

    w = np.maximum(x2 - x1, 0)
    h = np.maximum(y2 - y1, 0)

    return np.stack([x1, y1, w, h]).T

def rect_iou(rects1, rects2):
    r"""Intersection over union.

    Args:
        rects1 (numpy.ndarray): An N x 4 numpy array, each line represent a rectangle
            (left, top, width, height).
        rects2 (numpy.ndarray): An N x 4 numpy array, each line represent a rectangle
            (left, top, width, height).
        bound (numpy.ndarray): A 4 dimensional array, denotes the bound
            (min_left, min_top, max_width, max_height) for ``rects1`` and ``rects2``.
    """
    assert rects1.shape == rects2.shape
    rects_inter = _intersection(rects1, rects2)
    areas_inter = np.prod(rects_inter[..., 2:], axis=-1)

    areas1 = np.prod(rects1[..., 2:], axis=-1)
    areas2 = np.prod(rects2[..., 2:], axis=-1)
    areas_union = areas1 + areas2 - areas_inter

    eps = np.finfo(float).eps
    ious = areas_inter / (areas_union + eps)
    ious = np.clip(ious, 0.0, 1.0)

    return ious
def c_e_a_r_c(x1, y1, w, h, W, H):
    x_min = x1 - w / 2
    y_min = y1 - h / 2
    x_max = x1 + w / 2
    y_max = y1 + h / 2
    

    clipped_x_min = max(0, x_min)  
    clipped_y_min = max(0, y_min)  
    clipped_x_max = min(W, x_max)  
    clipped_y_max = min(H, y_max)  
    

    clipped_w = max(0, clipped_x_max - clipped_x_min)
    clipped_h = max(0, clipped_y_max - clipped_y_min)
    

    effective_area = clipped_w * clipped_h
    

    original_area = w * h
    

    if original_area == 0:
        return 0
    

    effective_area_ratio = effective_area / original_area
    
    return effective_area_ratio
def clip_box(box: list, H, W, margin=0):
    x1, y1, w, h = box
    x2, y2 = x1 + w, y1 + h
    x1 = min(max(0, x1), W-margin)
    x2 = min(max(margin, x2), W)
    y1 = min(max(0, y1), H-margin)
    y2 = min(max(margin, y2), H)
    w = max(margin, x2-x1)
    h = max(margin, y2-y1)
    return [x1, y1, w, h]
class LimitedStack:
    def __init__(self, max_size=8):
        self.stack = []
        self.max_size = max_size
    def push(self, item):
        if isinstance(item, list):
            if len(self.stack) >= self.max_size:
                self.stack.pop(0)  # 移除栈底元素
            self.stack.append(item)
        else:
            raise ValueError("Only lists are allowed as elements in the stack.")
    def pop(self):
        if self.stack:
            return self.stack.pop()
        else:
            return None
    def size(self):
        return len(self.stack)
    def to_list(self):
        return self.stack

def get_local_tracker():
    local_tracker_name='otetrack'
    local_tracker_param='otetrack_256_full'
    local_dataset_name='lasot'
    local_runid=None
    test_checkpoint = './OTETrack/test_checkpoint/OTETrack_all.pth.tar'
    update_intervals =None
    update_threshold =None
    hanning_size =None
    pre_seq_number =None
    std_weight =None
    local_tracker_raw = Local_Tracker(local_tracker_name, local_tracker_param, local_dataset_name, local_runid,
                                      test_checkpoint,update_intervals,update_threshold,hanning_size,
                                      pre_seq_number,std_weight)
    params = local_tracker_raw.get_parameters()
    params.debug = 0
    return local_tracker_raw.create_tracker(params)

def get_global_tracker():
    global_tracker_name="unicorn_sot"
    global_tracker_param="unicorn_track_tiny_sot_only"
    global_dataset_name="trackingnet"
    global_run_id=None
    global_tracker_raw = Global_Tracker(global_tracker_name, global_tracker_param, global_dataset_name, global_run_id)
    params = global_tracker_raw.get_parameters()
    params.debug = 0
    return global_tracker_raw.create_tracker(params)


def _lasot_otb_record(record_file, boxes, times):
    # record bounding boxes
    record_dir = os.path.dirname(record_file)
    if not os.path.isdir(record_dir):
        os.makedirs(record_dir)
    np.savetxt(record_file, boxes, fmt='%.3f', delimiter=',')
    while not os.path.exists(record_file):
        print('warning: recording failed, retrying...')
        np.savetxt(record_file, boxes, fmt='%.3f', delimiter=',')
    print('  Results recorded at', record_file)

    # record running times
    time_dir = os.path.join(record_dir, 'times')
    if not os.path.isdir(time_dir):
        os.makedirs(time_dir)
    time_file = os.path.join(time_dir, os.path.basename(
        record_file).replace('.txt', '_time.txt'))
    np.savetxt(time_file, times, fmt='%.8f')




def _lasot_otb_record_iou_scores(record_file, ious):
    # record bounding boxes
    ious_file = record_file.replace('.txt', '_iou_score.txt')

    np.savetxt(ious_file, ious, fmt='%.8f', delimiter=',')





dataset = LaSOT(root_dir="datasets/LaSOT", subset="test")
len_dataset = len(dataset)
print("len_dataset : ",len_dataset)
local_tracker  = get_local_tracker()
global_tracker = get_global_tracker()

print("global_tracker running")
KSTEPS=20
print("*"*50)
print('Number of samples:',KSTEPS)
print("*"*50)
ade_ls = [] 
fde_ls = [] 
exp_path='./Sot_STGCNN/checkpoint/sot-stgcnn-lasot_1_300_nr_6_4'
print("*"*50)
print("Evaluating model:",exp_path)
model_path = exp_path+'/val_best.pth'
args_path = exp_path+'/args.pkl'
with open(args_path,'rb') as f: 
    args = pickle.load(f)
#Data prep     
obs_seq_len = args.obs_seq_len
pred_seq_len = args.pred_seq_len
#Defining the model 
model = sot_stgcnn_raw(n_stgcnn =args.n_stgcnn,n_txpcnn=args.n_txpcnn,
output_feat=args.output_size,seq_len=args.obs_seq_len,
kernel_size=args.kernel_size,pred_seq_len=args.pred_seq_len).cuda()
model.load_state_dict(torch.load(model_path))
H = np.array([
    [2.8128700e-02, 2.0091900e-03, -4.6693600e+00],
    [8.0625700e-04, 2.5195500e-02, -5.0608800e+00],
    [3.4555400e-04, 9.2512200e-05, 4.6255300e-01]
])
H_inv = np.linalg.inv(H)
# print("H : ",H.shape)
# print("H_inv : ",H_inv.shape)
# # 显示图片
device = torch.device('cuda:0')
T1 = args.obs_seq_len #6
T2 = args.pred_seq_len #4
for idx in range(len_dataset):
    img_files, anno = dataset[idx]
    seq_name = dataset.seq_names[idx]
    print("idx :",idx," seq_name : ",seq_name)
    frame_num = len(img_files)
    print("frame_num : ",frame_num)
    boxes = np.zeros((frame_num, 4))
    iou= np.zeros(frame_num)
    scores= np.zeros(frame_num)
    iou_scores= np.zeros((frame_num, 2))
    times= np.zeros(frame_num)
    boxes[0] = anno[0, :]
    iou[0]=1.0
    record_file = os.path.join("results/LaSOT", "etp_ot", '%s.txt' % seq_name)
    record_dir =  os.path.join("results/LaSOT", "etp_ot", seq_name)




    if os.path.exists(record_dir) is False:
        os.makedirs(record_dir)

    history_trajectories_1 =LimitedStack(max_size=6)
    history_trajectories_2 =LimitedStack(max_size=6)

    pred_traj =None
    first_flag = True
    edge_flag = False
    ofv_flag = False
    edge_free = 0
    score_sum=0.0
    score_num=0
    score_list=[]
    for frame in range(0,frame_num):
        image = cv2.imread(img_files[frame])#, cv2.IMREAD_COLOR)
        image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        start_time = time.time()
        if frame == 0 :
            linit_info = {}
            linit_info['init_bbox'] = boxes[frame]
            x, y, w, h = boxes[frame]
            out = local_tracker.initialize(image, linit_info)
            times[frame] = time.time() - start_time
            if out is None:
                out = {}
            prev_output = OrderedDict(out)

            ginit_info = {}
            ginit_info['init_bbox'] = boxes[frame]
            gout = global_tracker.initialize(image, ginit_info)

            if w<10:
                w=10
            if h<10:
                h=10
            HH,W,C = image.shape
            x1 = x + w/2
            y1 = y + h/2
            x2 = w
            y2 = h

            history_trajectories_1.push([x1, y1, 1])
            history_trajectories_2.push([x2, y2, 1])
        else:
            linit_info = {}
            linit_info['previous_output'] = prev_output
            linit_info['gt_bbox'] = boxes[frame]

            out = local_tracker.track(image, frame, info=linit_info)
            x, y, w, h = out["target_bbox"]
            box = out["target_bbox"]

            if w<10:
                w=10
            if h<10:
                h=10
            # score = out["conf_score"]
            x1 = x + w/2
            y1 = y + h/2
            x2 = w
            y2 = h
            history_trajectories_1.push([x1, y1, 1])
            history_trajectories_2.push([x2, y2, 1])




            e_a_r = c_e_a_r_c(x1, y1, w, h, W*0.95, HH*0.95)


            if  e_a_r < 0.99  and ofv_flag == False and edge_free == 0:
                edge_flag = True
            else: 
                edge_flag = False

            if edge_free >0:
                edge_free = edge_free-1

            if history_trajectories_2.size() == T1:
                if edge_flag == True:
                    print(seq_name," ",frame, "ready to traj", "e_a_r: ",e_a_r,"ofv_flag : ",ofv_flag,"edge_free : ",edge_free)
                    # print("x1 y1 : ",x1, y1)

                    pixel_pos=np.array([history_trajectories_1.to_list(),history_trajectories_2.to_list()])
                    world_pos_ = np.einsum('mn, cdnk -> cdmk',H,pixel_pos[:,:,:,np.newaxis])
                    world_pos_=world_pos_[:,:,:,0]
                    world_pos= world_pos_[:,:,:2] / world_pos_[:,:,2:]
                    nones = np.ones((world_pos.shape[0],world_pos.shape[1],1))
                    world_pos = np.concatenate((world_pos,nones),axis=2)

                    a_p_j = world_pos[:,:,:2]
                    h_p_j = a_p_j.transpose(1,0,2)
                    a_p_j=a_p_j.transpose(0,2,1)
                    ra_p_j = np.zeros(a_p_j.shape)
                    ra_p_j[:,:, 1:] = a_p_j[:,:, 1:] - a_p_j[:,:, :-1]
                    a_p_j = torch.from_numpy(a_p_j).type(torch.float)
                    ra_p_j = torch.from_numpy(ra_p_j).type(torch.float)
                    v_,a_ = seq_to_graph(a_p_j,ra_p_j,True)
                    V_obs = v_.unsqueeze(dim=0)
                    V_obs_tmp = V_obs.permute(0,3,1,2)
                    obs_traj = a_p_j.unsqueeze(dim=0)
                    obs_traj_rel= ra_p_j.unsqueeze(dim=0)
                    V_obs_tmp= V_obs_tmp.to(device)
                    a_ = a_.to(device)
                    V_pred,_ = model(V_obs_tmp,a_.squeeze())

                    V_pred = V_pred.permute(0,2,3,1)
                    V_pred = V_pred.squeeze()
                    num_of_objs = obs_traj_rel.shape[1] #2
                    V_pred =  V_pred[:,:num_of_objs,:]
                    sx = torch.exp(V_pred[:,:,2]) #sx
                    sy = torch.exp(V_pred[:,:,3]) #sy
                    corr = torch.tanh(V_pred[:,:,4]) #corr
                    cov = torch.zeros(V_pred.shape[0],V_pred.shape[1],2,2).cuda()
                    cov[:,:,0,0]= sx*sx
                    cov[:,:,0,1]= corr*sx*sy
                    cov[:,:,1,0]= corr*sx*sy
                    cov[:,:,1,1]= sy*sy
                    mean = V_pred[:,:,0:2]
                    mean=mean.cpu()
                    cov=cov.cpu()
                    mvnormal = torchdist.MultivariateNormal(mean,cov)
                    V_x = seq_to_nodes(obs_traj.data.cpu().numpy().copy())
                    V_x_rel_to_abs = nodes_rel_to_nodes_abs(V_obs.data.cpu().numpy().squeeze().copy(),
                                                            V_x[0,:,:].copy())
                    pred_trj = []
                    V_pred=[]
                    for _ in range(KSTEPS):
                        V_pred.append(mvnormal.sample())
                    V_pred = torch.mean(torch.stack(V_pred),dim=0)
                    V_pred_rel_to_abs = nodes_rel_to_nodes_abs(V_pred.data.cpu().numpy().squeeze().copy(),
                                                            V_x[-1,:,:].copy())
                    nones = np.ones((V_pred_rel_to_abs.shape[0],V_pred_rel_to_abs.shape[1],1))
                    pred_world_pos = np.concatenate((V_pred_rel_to_abs,nones),axis=2)
                    pixel_pos = np.einsum('mn, cdnk -> cdmk', H_inv, pred_world_pos[:,:,:,np.newaxis])
                    pixel_pos_ = pixel_pos[:,:,:2, :] / pixel_pos[:,:,2:, :]
                    nones = np.ones((h_p_j.shape[0],h_p_j.shape[1],1))
                    his_world_pos = np.concatenate((h_p_j,nones),axis=2)
                    his_pos = np.einsum('mn, cdnk -> cdmk', H_inv, his_world_pos[:,:,:,np.newaxis])
                    his_pos_ = his_pos[:,:,:2, :] / his_pos[:,:,2:, :]
                    his_pos_ = np.squeeze(his_pos_, axis=-1)
                    point_num = pixel_pos_.shape[0]
                    peds_num = pixel_pos_.shape[1]
                    pixel_pos_ = np.squeeze(pixel_pos_, axis=-1)
                    pred_traj = pixel_pos_[:,0,:].tolist()
                    pred_wh = pixel_pos_[:,1,:].tolist()
                    # print("wh list: ", history_trajectories_2.to_list())
                    for pdx in range(T2):
                        x_c = pred_traj[pdx][0]
                        y_c = pred_traj[pdx][1]
                        p_w = pred_wh[pdx][0]
                        p_h = pred_wh[pdx][1]
                        if p_w <= 0 or p_h <= 0: 
                            p_w = w
                            p_h = h
                        # print("p_w : ",p_w, "w : ",w,"p_h : ",p_h, "h : ",h)
                        e_a_r = c_e_a_r_c(x_c, y_c, p_w, p_h, W, HH)
                        if e_a_r < 0.99 :
                            ofv_flag = True
                            break
                    if ofv_flag == False:
                        edge_free = T2


            box = clip_box(box, HH, W, margin=10)
            local_tracker.state = box
            box =  np.array(box)

            if ofv_flag == True :
                gout= global_tracker.track(image, info=None)
                iou_score_data = gout["conf_score"]
                global_truth_data = np.array(gout["target_bbox"])
                print(seq_name," ",frame," two trackers running ", "score : ",iou_score_data, "iou : ", rect_iou(box, global_truth_data))
                if iou_score_data > 0.6:
                    if rect_iou(box, global_truth_data) <= 0.5 :
                        print(seq_name," ",frame," use global ")
                        box = global_truth_data
                        local_tracker.state = box
                        x, y, w, h = box
                        x1 = x + w/2
                        y1 = y + h/2
                        x2 = w
                        y2 = h
                        history_trajectories_1.pop()
                        history_trajectories_2.pop()
                        history_trajectories_1.push([x1, y1, 1])
                        history_trajectories_2.push([x2, y2, 1])
                    else:
                        ofv_flag = False


            out["target_bbox"] = box
            prev_output = OrderedDict(out)
            boxes[frame, :]=out["target_bbox"]
            times[frame] = time.time() - start_time
            # scores[frame] = out["conf_score"]
            iou[frame] = rect_iou(boxes[frame, :], anno[frame,:])
            iou_scores[frame] =np.array([iou[frame], scores[frame]])
    print('FPS: {}'.format(len(times)/sum(times)))
    _lasot_otb_record(record_file, boxes, times)
    _lasot_otb_record_iou_scores(record_file, iou_scores)



