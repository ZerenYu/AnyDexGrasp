
import os
import shutil
from tqdm import tqdm
import numpy as np
import math
import copy
import json
import os
import sys
import time
import datetime
import argparse
import torch
import numpy as np
import cv2
from PIL import Image
import open3d as o3d
import MinkowskiEngine as ME
from graspnetAPI import GraspGroup

from multiprocessing import shared_memory
from collections import OrderedDict
import matplotlib.pyplot as plt
import pickle

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))
sys.path.append(os.path.join(ROOT_DIR, 'dataset'))
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
# from generate_mesh_and_pointcloud.Allegro_grasp import AllegroGraspGroup
from minkowski_graspnet import MinkowskiGraspNet
from np_utils import create_point_cloud_from_depth_image
from pt_utils import batch_viewpoint_params_to_matrix
# from collision_detector import ModelFreeCollisionDetectorAllegro
import queue
from itertools import count
from threading import Thread
from queue import Queue
import multiprocessing as mp

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint_path', default='log/checkpoint.tar', help='Model checkpoint path')
parser.add_argument('--use_graspnet_v2', action='store_true', help='Whether to use graspnet v2 format')
parser.add_argument('--half_views', action='store_true', help='Use only half views in network.')
parser.add_argument('--inspire_mesh_json_path', default='generate_mesh_and_pointcloud/Allegro_mesh/mesh', 
                    help='Inspire mesh json path.')
cfgs = parser.parse_args()

MAX_GRASP_WIDTH = 0.119
MIN_GRASP_WIDTH = 0.01
BATCH_SIZE = 1
DEBUG = False
Allegro_VOXElGRID = 0.003

def parse_preds(end_points, use_v2=False):
    ## load preds
    before_generator = end_points['before_generator']  # (B, Ns, 256)
    point_features = end_points['point_features']  # (B, Ns, 512)
    coords = end_points['sinput'].C  # (\Sigma Ni, 4)
    objectness_pred = end_points['stage1_objectness_pred']  # (Sigma Ni, 2)
    objectness_mask = torch.argmax(objectness_pred, dim=1).bool()  # (\Sigma Ni,)
    # objectness_mask = (objectness_pred[:,1]>-1)
    seed_xyz = end_points['stage2_seed_xyz']  # (B, Ns, 3)
    seed_inds = end_points['stage2_seed_inds']  # (B, Ns)
    grasp_view_xyz = end_points['stage2_view_xyz']  # (B, Ns, 3)
    grasp_view_inds = end_points['stage2_view_inds']
    grasp_view_scores = end_points['stage2_view_scores']
    grasp_scores = end_points['stage3_grasp_scores']  # (B, Ns, A, D)
    grasp_features_two_finger = end_points['stage3_grasp_features'].view(grasp_scores.size()[0], grasp_scores.size()[1], -1) # (B, Ns, 3 + C)
    grasp_widths = MAX_GRASP_WIDTH * end_points['stage3_normalized_grasp_widths']  # (B, Ns, A, D)
    grasp_widths[grasp_widths > MAX_GRASP_WIDTH] = MAX_GRASP_WIDTH

    grasp_preds = []
    grasp_features = []
    grasp_vdistance_list = []
    for i in range(BATCH_SIZE):
        
        cloud_mask_i = (coords[:, 0] == i)
        seed_inds_i = seed_inds[i]
        # print('seed_inds_i: ', seed_inds_i.shape, seed_inds_i)
        objectness_mask_i = objectness_mask[cloud_mask_i][seed_inds_i]  # (Ns,)

        if objectness_mask_i.any() == False:
            continue

        ## remove background
        seed_xyz_i = seed_xyz[i] # [objectness_mask_i]  # (Ns', 3)
        point_features_i = point_features[i] # [objectness_mask_i]
        
        seed_inds_i = seed_inds_i # [objectness_mask_i]
        before_generator_i = before_generator[i] # [objectness_mask_i]
        grasp_view_xyz_i = grasp_view_xyz[i] # [objectness_mask_i]  # (Ns', 3)
        grasp_view_inds_i = grasp_view_inds[i] # [objectness_mask_i]
        grasp_view_scores_i = grasp_view_scores[i] # [objectness_mask_i]
        # print('shape: ', grasp_view_inds_i.shape, grasp_view_scores.shape, grasp_view_xyz_i.shape)
        grasp_scores_i = grasp_scores[i] # [objectness_mask_i]  # (Ns', A, D)
        grasp_widths_i = grasp_widths[i] # [objectness_mask_i] # (Ns', A, D)
        
        Ns, A, D = grasp_scores_i.size()
        grasp_features_two_finger_i = grasp_features_two_finger[i] # [objectness_mask_i] # (Ns', 3 + C)
        grasp_scores_i_A_D = copy.deepcopy(grasp_scores_i).view(Ns, -1)

        grasp_scores_i = torch.minimum(grasp_scores_i[:,:24,:], grasp_scores_i[:,24:,:])
        # print('grasp_scores_i_A_D: ', grasp_scores_i_A_D.size(), grasp_features_two_finger_i.size())
        ## slice preds by topk grasp score/angle
        # grasp score & angle
        
        # grasp_view_xyz_i = grasp_view_xyz_i.unsqueeze(1).expand(-1, topk, -1).contiguous().view(Ns * topk, 3)
        # grasp_view_inds_i = grasp_view_inds_i.unsqueeze(1).expand(-1, topk).contiguous().view(Ns * topk, 1)
        # grasp_view_scores_i = grasp_view_scores_i.unsqueeze(1).expand(-1, topk).contiguous().view(Ns * topk, 1)

        # seed_xyz_i = seed_xyz_i.unsqueeze(1).expand(-1, topk, -1).contiguous().view(Ns * topk, 3)
        # seed_inds_i = seed_inds_i.unsqueeze(1).expand(-1, topk).contiguous().view(Ns * topk, 1)
        seed_inds_i = seed_inds_i.view(Ns, -1)
        grasp_view_inds_i = grasp_view_inds_i.view(Ns, -1)
        grasp_view_scores_i = grasp_view_scores_i.view(Ns, -1)
        # point_features_i = point_features_i.unsqueeze(1).expand(-1, topk, -1).contiguous().view(Ns * topk, 512)
        # before_generator_i = before_generator_i.unsqueeze(1).expand(-1, topk, -1).contiguous().view(Ns * topk, 512)

        grasp_scores_i, grasp_angles_class_i = torch.max(grasp_scores_i, dim=1) # (Ns', D), (Ns', D)
        grasp_angles_i = (grasp_angles_class_i.float()-12) / 24 * np.pi  # (Ns', topk, D)
        # grasp_angles_i = grasp_angles_i.view(Ns * topk, D)  # (Ns', topk, D)
        # grasp_scores_i = grasp_scores_i.view(Ns * topk, D)  # (Ns'*topk, D)
        # grasp width & vdistance
        grasp_angles_class_i = grasp_angles_class_i.unsqueeze(1) # (Ns', 1, D)
        grasp_widths_pos_i = torch.gather(grasp_widths_i, 1, grasp_angles_class_i).squeeze(1) # (Ns', D)
        grasp_widths_neg_i = torch.gather(grasp_widths_i, 1, grasp_angles_class_i+24).squeeze(1) # (Ns', D)

        ## slice preds by grasp score/depth
        # grasp score & depth
        grasp_scores_i, grasp_depths_class_i = torch.max(grasp_scores_i, dim=1, keepdims=True) # (Ns', 1), (Ns', 1)
        grasp_depths_i = (grasp_depths_class_i.float() + 1) * 0.01  # (Ns'*topk, 1)
        # if use_v2:
        #     if month in [10, 11, 12]:
        #         grasp_depths_i[grasp_depths_class_i == 0] = 0.0065 #2023.1.6, 0.0065 previously
        #         grasp_depths_i[grasp_depths_class_i != 0] -= 0.008 #2023.1.6, 0.008 previously
        #     else:
        #         grasp_depths_i[grasp_depths_class_i == 0] = 0.005 #2023.1.6, 0.0065 previously
        #         grasp_depths_i[grasp_depths_class_i != 0] -= 0.01 #2023.1.6, 0.008 previously
        grasp_depths_i -= 0.01
        grasp_depths_i[grasp_depths_class_i==0] = 0.005
        # grasp angle & width & vdistance
        # grasp_angles_i = torch.gather(grasp_angles_i, 1, grasp_depths_class_i)  # (Ns'*topk, 1)
        # grasp_widths_i = torch.gather(grasp_widths_i, 1, grasp_depths_class_i)  # (Ns'*topk, 1)
        grasp_angles_i = torch.gather(grasp_angles_i, 1, grasp_depths_class_i) # (Ns', 1)
        grasp_widths_pos_i = torch.gather(grasp_widths_pos_i, 1, grasp_depths_class_i) # (Ns', 1)
        grasp_widths_neg_i = torch.gather(grasp_widths_neg_i, 1, grasp_depths_class_i) # (Ns', 1)

        # convert to rotation matrix
        rotation_matrices_i = batch_viewpoint_params_to_matrix(-grasp_view_xyz_i, grasp_angles_i.squeeze(1))
        # get offsets
        # offsets = torch.zeros(seed_xyz_i.shape, dtype=seed_xyz_i.dtype).to(seed_xyz_i.device)
        # offsets[:,1:2] = (grasp_widths_pos_i - grasp_widths_neg_i) / 2
        # # adjust gripper centers
        grasp_widths_i = grasp_widths_pos_i + grasp_widths_neg_i
        # seed_xyz_i += torch.matmul(rotation_matrices_i.view(Ns,3,3), offsets[:,:,np.newaxis]).squeeze(2)
        rotation_matrices_i = rotation_matrices_i.view(Ns, 9)
        # rotation_matrices_i = rotation_matrices_i.view(Ns * topk, 9)

        # merge preds
        # print(grasp_widths_i.shape, grasp_widths_i.shape, grasp_depths_i.shape, rotation_matrices_i.shape, seed_xyz_i.shape)
        grasp_preds.append(torch.cat([grasp_scores_i, grasp_widths_i, grasp_depths_i, rotation_matrices_i, seed_xyz_i],axis=1))  # (Ns, 15)
        # print(' shape: ', grasp_scores_i_A_D.shape, grasp_features_two_finger_i.shape, before_generator_i.shape, point_features_i.shape, grasp_view_inds_i.shape, grasp_view_scores_i.shape, seed_inds_i.shape, grasp_angles_i.shape, grasp_depths_i.shape)
        # print('\ngrasp_features_two_finger_i: ', grasp_features_two_finger_i.size())
        grasp_features.append(torch.cat([grasp_scores_i_A_D, grasp_features_two_finger_i, before_generator_i, point_features_i, grasp_view_inds_i, grasp_view_scores_i, seed_inds_i, grasp_angles_i*24/np.pi+12, grasp_depths_i], axis=1)) # (Ns'*3, A, D)
        
    return grasp_preds, grasp_features

def get_net(checkpoint_path, use_v2=False):
    if use_v2:
        net = MinkowskiGraspNet(num_depth=5, num_seed=2048, is_training=False, half_views=cfgs.half_views)
    else:
        net = MinkowskiGraspNet(num_depth=4, num_seed=2048, is_training=False, half_views=cfgs.half_views)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net.to(device)
    net.eval()
    checkpoint = torch.load(checkpoint_path)
    net.load_state_dict(checkpoint['model_state_dict'])
    return net

def get_depth(existing_shm_depth):
    time.sleep(0.1)
    depths_1 = np.copy(np.ndarray((720, 1280), dtype=np.uint16, buffer=existing_shm_depth.buf))
    # depths_2 = np.copy( np.ndarray((720,1280), dtype=np.uint16, buffer=existing_shm_depth.buf) )
    # depths_3 = np.copy( np.ndarray((720,1280), dtype=np.uint16, buffer=existing_shm_depth.buf) )
    # depths_4 = np.copy( np.ndarray((720,1280), dtype=np.uint16, buffer=existing_shm_depth.buf) )
    # depths_5 = np.copy( np.ndarray((720,1280), dtype=np.uint16, buffer=existing_shm_depth.buf) )

    # depths = (depths_1+depths_2+depths_3+depths_4+depths_5)/5
    return depths_1

def get_grasp(net, depths, existing_shm_color, date, point_id_origin, voxel_size=0.005):
    # iros Inspire
    # fx, fy = 908.435, 908.679
    # cx, cy = 650.366, 367.277
    # s = 1000.0

    # fx, fy = 912.898, 912.258
    # cx, cy = 629.536, 351.637
    # s = 1000.0
    # sr Inspire
    # fx, fy = 919.835, 919.61
    # cx, cy = 631.119, 363.884
    # s = 1000.0
    # sr Allegro
    fx, fy = 913.232, 912.452
    cx, cy = 628.847, 350.771
    s = 1000.0

    xmap, ymap = np.arange(depths.shape[1]), np.arange(depths.shape[0])
    xmap, ymap = np.meshgrid(xmap, ymap)

    points_z = depths / s
    points_x = (xmap - cx) / fx * points_z
    points_y = (ymap - cy) / fy * points_z
    # mask = (points_z > 0.45) & (points_z < 0.88) & (points_x > -0.21) & (points_x < 0.21) & (points_y > -0.08) & (points_y < 0.2)
    mask = (points_z > 0.35) & (points_z < 0.55)
    # mask[:] = True
    points = np.stack([points_x, points_y, points_z], axis=-1)
    points = points[mask].astype(np.float32)
    cloud = None
    
    points = torch.from_numpy(points)
    # print('points: ', points)
    coords = np.ascontiguousarray(points / voxel_size, dtype=int)
    # Upd Note. API change.
    _, idxs = ME.utils.sparse_quantize(coords, return_index=True)
    coords = coords[idxs]
    points = points[idxs]
    
    coords_batch, points_batch = ME.utils.sparse_collate([coords], [points])

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    sinput = ME.SparseTensor(points_batch, coords_batch, device=device)
    # print('sinput: ', sinput)
    end_points = {'sinput': sinput, 'point_clouds': [sinput.F], 'point_id_origin': point_id_origin}
    t1 = time.time()
    with torch.no_grad():
        end_points = net(end_points)
        preds, grasp_features = parse_preds(end_points, use_v2=cfgs.use_graspnet_v2)
        # grasp_angles_index, grasp_depths_index, grasp_features, grasp_A_D = grasp_information
        if len(preds) == 0:
            print('No grasp detected')
            return None, cloud, points.cuda(), None
        else:
            preds = preds[0]
            grasp_features = grasp_features[0]
    # print('111preds grasp features: ', preds.size(), grasp_features.size()) 
    # mask = (preds[:,9] > 0.9) & (preds[:,1] < MAX_GRASP_WIDTH) & (preds[:,1] > MIN_GRASP_WIDTH)
    date = date.replace('_', '').replace('-','')
    
    # print(date)
    if date < '20221123150500':
        workspace_mask = (preds[:,12] > -0.2) & (preds[:,12] < 0.2) & (preds[:,13] > -0.25) & (preds[:,13] < 0.1) # 2022.11.23.15.05
    elif date < '20221216223300':
        workspace_mask = (preds[:,12] > -0.25) & (preds[:,12] < 0.25) & (preds[:,13] > -0.205) & (preds[:,13] < 0.08) # 2022.12.16.22.33
    else:
        workspace_mask = (preds[:,12] > -0.25) & (preds[:,12] < 0.25) & (preds[:,13] > -0.205) & (preds[:,13] < 0.03)
        # id = np.where(grasp_features.cpu().numpy()[:,-3]==8261)[0][0]
        # print('test workspace mask: ', preds[id])
    # if date < '20221123150500':
    #     workspace_mask = (preds[:,12] > -0.2) & (preds[:,12] < 0.2) & (preds[:,13] > -0.25) & (preds[:,13] < 0.1) # 2022.11.23.15.05
    # elif date < '20221216223300':
    #     workspace_mask = (preds[:,12] > -0.25) & (preds[:,12] < 0.25) & (preds[:,13] > -0.205) & (preds[:,13] < 0.08) # 2022.12.16.22.33
    # else:
    #     workspace_mask = (preds[:,12] > -0.25) & (preds[:,12] < 0.25) & (preds[:,13] > -0.205) & (preds[:,13] < 0.03)
    preds = preds[workspace_mask]
    grasp_features = grasp_features[workspace_mask]
    # print('111preds grasp features: ', preds.size(), grasp_features.size())
    if len(preds) == 0:
        print('No grasp detected after masking')
        return None, cloud, points.cuda(), None

    points = points.cuda()
    heights = 0.03 * torch.ones([preds.shape[0], 1]).cuda()
    object_ids = -1 * torch.ones([preds.shape[0], 1]).cuda()
    ggarray = torch.cat([preds[:, 0:2], heights, preds[:, 2:15], preds[:, 15:16]], axis=-1)
    
    return ggarray, cloud, points, grasp_features

def read_file(path, cfg):
    net = get_net(cfgs.checkpoint_path, use_v2=cfgs.use_graspnet_v2)
    # meshes_pcls = load_meshes_pointcloud(cfgs.inspire_mesh_json_path)
    x_min = [1 for _ in range(17)]
    x_max = [0 for _ in range(17)]
    all_data_json = dict()
    for label in tqdm(os.listdir(path)):
        every_label = os.path.join(path, label)
        for file in tqdm(os.listdir(every_label)):
            every_file = os.path.join(every_label, file)
            color_path = os.path.join(every_file, 'color.png')
            depth_path = os.path.join(every_file, 'depth.png')
            information_path = os.path.join(every_file, 'information.json')
            # month = int(file.split('_')[0].split('-')[1])
            # print(month)
            try:
                with open(information_path, 'r') as f:
                    information = json.load(f)
            
            # twgg = information['two_fingers_ggarray_source']
            # print(twg)
            # for twg in twgg:
            #     for idx, digital in enumerate(twg):
            #         if idx == 10:
            #             digital = abs(digital)
            #         if digital > x_max[idx]:
            #             x_max[idx] = digital
            #         if digital < x_min[idx]:
            #             x_min[idx] = digital
            except:
                print(' error open : ', every_file)

            source_informations = os.path.join(every_file, 'two_fingers_ggarray_informations_source.npy')
            colors = np.array(Image.open(color_path), dtype=np.float32) / 255.0
            depths = np.array(Image.open(depth_path))
            source_informations = np.load(source_informations)
            grasp_feature = update_data(colors, depths, information, every_file, source_informations, cfg, file, net)  
            all_data_json[file] = grasp_feature 

    json_file = json.dumps(all_data_json, indent=4)
    with open(path+'_single_point.json', 'w') as handle:
        handle.write(json_file)

def remove_file(path):
    shutil.rmtree(path)
    name = path.split('/')[-1]
    dense_path = os.path.join('/disk2/hengxu/yhx/Allegro_data/data_dense_extend', name)
    if os.path.exists(dense_path):
        shutil.rmtree(dense_path)
    augment_path = os.path.join('/disk2/hengxu/yhx/Allegro_data/data_augment_extend', name)
    if os.path.exists(augment_path):
        shutil.rmtree(augment_path)
    augment100_path = os.path.join('/disk2/hengxu/yhx/Allegro_data/data_augment_extend_100', name)
    if os.path.exists(augment100_path):
        shutil.rmtree(augment100_path)

def flip_ggarray(ggarray):
    ggarray_rotations = ggarray[:, 4:13].reshape((-1, 3, 3))
    tcp_x_axis_on_base_frame = ggarray_rotations[:, 1, 1]
    # normal_vector = np.array((- 1 / 2, np.sqrt(3) / 2, 0))
    if_flip = [False for _ in range(len(ggarray))]
    for ids, y_x in enumerate(tcp_x_axis_on_base_frame):
        if y_x < 0:
            ggarray_rotations[ids, :3, 1:3] = -ggarray_rotations[ids, :3, 1:3]
            if_flip[ids] = True
    ggarray[:, 4:13] = ggarray_rotations.reshape((-1, 9))
    return ggarray, if_flip

def load_meshes_pointcloud(path):
    meshes_pcls = dict()
    meshes_pcl_path = os.path.join(path, 'source_pointclouds/voxel_size_' + str(int(Allegro_VOXElGRID * 1000)))
    for type in os.listdir(meshes_pcl_path):
        type_path = os.path.join(meshes_pcl_path, type)
        for name in os.listdir(type_path):
            width = name[:-4]
            name_path = os.path.join(type_path, name)
            meshes_pcl = o3d.io.read_point_cloud(name_path)
            meshes_pcls[type + '_' + width] = meshes_pcl
    return meshes_pcls

def get_grasp_features(grasp_features_array, grasp_features):
    # grasp_features['grasp_angles'] = int(grasp_features_array[-3]+0.1)
    # grasp_features['grasp_depths'] = int(grasp_features_array[-2]*100+0.1)
    grasp_features['stage3_grasp_scores'] = grasp_features_array[:240].tolist()
    grasp_features['grasp_preds_features'] = grasp_features_array[240:240+480].tolist()
    grasp_features['stage3_grasp_features'] = grasp_features_array[240+480:240+480+512].tolist()
    grasp_features['before_generator'] = grasp_features_array[240+480+512:240+480+512+512].tolist()
    grasp_features['point_features'] = grasp_features_array[240+480+512+512:240+480+512+512+512].tolist()
    # grasp_features['point_id'] = int(grasp_features_array[-4])
    # grasp_features['if_flip'] = bool(int(grasp_features_array[-1]))
    return grasp_features

def get_grasp_idx(grasp_features, information):
    point_ids = grasp_features[:, -3]
    select_point_id = information['point_id']
    point_idx = np.where(point_ids==select_point_id) 
    
    if len(point_idx[0]) != 0:
        return point_idx[0][0]
    else:
        print('point_idx: ', point_idx, point_ids, select_point_id)
        return 'false'
            

def update_data(colors, depths, information, save_path, source_informations, cfgs, date, net):
    point_id_origin = information['point_id']
    # try:
    ggarray, cloud, points_down, grasp_features = get_grasp(net, depths, colors, date, point_id_origin)
    # except:
    #     print('fail path: ', save_path)
    
    # grasp_informations = source_informations
    ########## PROCESS GRASPS ##########
    ggarray = ggarray.cpu().numpy()
    grasp_features = grasp_features.cpu().numpy()
    # ggarray = ggarray[source_index][::-1][:500]
    
    # # Prevent the robot arm from crossing the border, 
    # ggarray, if_flip = flip_ggarray(ggarray)
    # grasp_features = grasp_features.cpu().numpy()
    # two_fingers_source_grasp_features = copy.deepcopy(grasp_features)
    # grasp_features = grasp_features[source_index][::-1][:500]
    # grasp_features = np.c_[grasp_features, if_flip]

    # print('grasp_informations point_features source : ', source_informations[:500, -1]-two_fingers_source_grasp_features[:500, -1])
    # Allegro_types = np.array(information['Allegro_ggarray_source_saved'])[:,2]
    # Allegro_depths = np.array(information['Allegro_ggarray_source_saved'])[:,1]

    # for ids, grasp_type in enumerate(Allegro_types):
    #     if grasp_type == 4 or grasp_type == 5:
    #         Allegro_depths[ids] = Allegro_depths[ids] - 0.02 

    two_fingers_ggarray = GraspGroup(ggarray)
    # two_fingers_ggarray_object_ids_source = ggarray[:, 16]

    # Allegro_ggarray = AllegroGraspGroup()
    # Allegro_ggarray.set_grasp_min_width(MIN_GRASP_WIDTH)
    # try:
    #     Allegro_ggarray.from_graspgroup(two_fingers_ggarray, Allegro_types, cfgs.inspire_mesh_json_path)
    #     Allegro_ggarray.object_ids = two_fingers_ggarray_object_ids_source
    #     Allegro_ggarray.depths = Allegro_depths
    #     t5 = time.time()
    #     # print('two to five time: ', t5-t4)
    #     index_filter_by_z_axis = Allegro_ggarray.filter_grasp_group_by_z_axis(0.4)
    #     two_fingers_ggarray = two_fingers_ggarray[index_filter_by_z_axis]
    #     two_fingers_ggarray_object_ids = two_fingers_ggarray_object_ids_source[index_filter_by_z_axis]
    #     grasp_features = grasp_features[index_filter_by_z_axis]
    # except:
    #     print('warning from_graspgroup len(Allegro_types)!=len(ggarray) : ', save_path)
        # return
    

    # np.set_printoptions(threshold=np.inf) 
    # if np.sum(np.array(information['before_collision'][0])-Allegro_ggarray.grasp_group_array) > 0.001:
    #     print('before collision fail path: ', save_path, np.sum(np.array(information['before_collision'][0])-Allegro_ggarray.grasp_group_array))
    #     # remove_file(save_path)
    #     return
    
    grasp_idx = get_grasp_idx(grasp_features,  information)
    if grasp_idx == 'false':
        print('can not get_grasp_idx, fail path: ', save_path)
        # remove_file(save_path)
        return
    grasp_idx = int(grasp_idx)

    g = two_fingers_ggarray[grasp_idx]
    g = [(g.score), float(g.width), float(g.height), float(g.depth)] + np.array(g.rotation_matrix).reshape((-1)).tolist() + list(g.  translation.tolist()) 
    grasp_feature = dict()
    grasp_feature['result'] = information['result']
    grasp_feature['two_fingers_pose_single_point'] = g
    grasp_feature['two_fingers_pose_two_points'] = information['two_fingers_pose']
    grasp_feature['Allegro_pose'] = information['Allegro_pose']
    grasp_feature['two_fingers_pose_angle_type'] = information['two_fingers_pose_angle_type']
    grasp_feature['two_fingers_pose_depth_type'] = information['two_fingers_pose_depth_type']
    grasp_feature['Allegro_pose_finger_type'] = information['Allegro_pose_finger_type']
    grasp_feature['Allegro_pose_depth_type'] = information['Allegro_pose_depth_type']
    grasp_feature['point_id'] = information['point_id']
    grasp_feature['if_flip'] = information['if_flip']
    information['camera_internal'] = [[628.847, 350.771], [913.232, 912.452]]
    grasp_feature['view_inds'] = int(grasp_features[grasp_idx, -5]+0.1)
    grasp_feature['view_score'] = float(grasp_features[grasp_idx, -4])
    grasp_feature = get_grasp_features(grasp_features[grasp_idx], grasp_feature)
    # print('before collision: ', np.array(information['before_collision'][0][:30])-Allegro_ggarray.grasp_group_array[:30], np.array(information['before_collision'][2])[21, -4])
    # if len(two_fingers_ggarray) == 0:
    #     print('fail path: ', save_path)
    #     # remove_file(save_path)
    #     return
    t6 = time.time()
    # print('angle filter time: ', t6-t5)
    # print('path: ', save_path, len(two_fingers_ggarray), len(Allegro_ggarray))
    # collision detection
    # mfcdetector = ModelFreeCollisionDetectorAllegro(points_down.cpu().numpy(), voxel_size=0.001)
    # Allegro_ggarray, two_fingers_ggarray, empty_mask = mfcdetector.detect(Allegro_ggarray,
    #                                                                            two_fingers_ggarray,
    #                                                                            path_mesh_json, VoxelGrid=0.002,
    #                                                                            DEBUG=False, approach_dist=0.04,
    #                                                                            collision_thresh=2,
    #                                                                            adjust_gripper_centers=True)
    # Allegro_ggarray, two_fingers_ggarray, empty_mask, min_width_index = mfcdetector.detect(
    #                                                             Allegro_ggarray, two_fingers_ggarray, 
    #                                                             cfgs.inspire_mesh_json_path, meshes_pcls, min_grasp_width=MIN_GRASP_WIDTH,
    #                                                             VoxelGrid=Allegro_VOXElGRID, DEBUG=False, approach_dist=0.04,
    #                                                             collision_thresh=2,adjust_gripper_centers=True,)                                      

          

    # t7 = time.time()
    # # print('collision time: ', t7-t6)
    # # proposals
    # # print(min_width_index, empty_mask)
    # Allegro_ggarray = Allegro_ggarray[empty_mask]
    # two_fingers_ggarray = two_fingers_ggarray[empty_mask]
    # two_fingers_ggarray_object_ids = two_fingers_ggarray_object_ids[min_width_index][empty_mask]
    # grasp_features = grasp_features[min_width_index][empty_mask]
    

    # print(len(Allegro_ggarray), len(grasp_features), len(two_fingers_ggarray), sum(min_width_index), sum(empty_mask))
    # print(save_path)
    # print('after_collision: ', np.array(information['after_collision'][0][1]), Allegro_ggarray.grasp_group_array[1])
    # print('after collision:', np.array(information['after_collision'][2])[:30, -4], grasp_features[:30, -4]) 
    # print('after collision:', np.array(information['after_collision'][2])[:30, -4]-grasp_features[:30, -4]) 

    # if len(two_fingers_ggarray) == 0:
    #     print('error path: ', save_path)
    #     # remove_file(save_path)
    #     return
    # # sort
    # index_score = np.argsort(Allegro_ggarray.scores)[::-1]
    # two_fingers_ggarray_object_ids = two_fingers_ggarray_object_ids[index_score][0:10]
    # Allegro_ggarray = Allegro_ggarray[index_score][0:10]
    # two_fingers_ggarray = two_fingers_ggarray[index_score][0:10]
    # grasp_features = grasp_features[index_score][0:10]

    # Allegro_grasp_used = Allegro_ggarray[0]
    # two_fingers_grasp_used = two_fingers_ggarray[0]
    # grasp_features_used = grasp_features[0]
    # grasp_features_used = get_grasp_features(grasp_features_used)
    # grasp_features_used['point_features']
    # print(np.array(Allegro_grasp_used.get_array_grasp()) - np.array(information['InspiredHandR_pose']))
    # print('point_features:', sum(np.array(information['point_features'])-np.array(grasp_features_used['point_features'])))
    # print('before_generator:', sum(np.array(information['two_fingers_pose_features_before_generator'])-np.array(grasp_features_used['before_generator'])))
    # print('stage3_grasp_features:', sum(np.array(information['two_fingers_pose_features'])-np.array(grasp_features_used['stage3_grasp_features'])))
    # print('grasp_preds_features:', sum(np.array(information['two_fingers_pose_features_grasp_preds'])-np.array(grasp_features_used['grasp_preds_features'])))
    # print('stage3_grasp_scores:', sum(np.array(information['two_fingers_pose_AD'])-np.array(grasp_features_used['stage3_grasp_scores'])))
    # information['camera_internal'] = [[631.119, 363.884], [919.835, 919.61]]
    # print('point id: ', information['point_id'], grasp_features_used['point_id'])
    
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    json_path = os.path.join(save_path, 'information_single_point.json')
    json_file = json.dumps(grasp_feature, indent=4)
    with open(json_path, 'w') as handle:
        handle.write(json_file)
    
    del information['before_collision']
    del information['after_collision']
    del information['base_2_tcp1']
    del information['base_2_tcp1_backup']
    del information['tcp_2_gripper']
    del information['base_2_TwoFingersGripper_pose']
    del information['tcp_2_camera']
    del information['base_2_tcp_ready']
    # del information['camera_internal']Allegro_pose
    del information['two_fingers_ggarray_proposals']
    del information['Allegro_ggarray_proposals']
    del information['two_fingers_ggarray_informations_proposals']
    del information['two_fingers_ggarray_source']
    del information['Allegro_ggarray_source_saved']

    return grasp_feature

if __name__ == '__main__':
    path = '/disk2/hengxu/yhx/InspireHandR_data/Allegro_data/Allegro_information'
    # path = '/disk2/hengxu/yhx/InspireHandR_data/Allegro_data/1'

    read_file(path, cfgs)
    # collision(path, cfgs.extend)