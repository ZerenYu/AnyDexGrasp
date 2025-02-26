import copy
import json
import os
import sys
import pdb
import time
import datetime
import random
import argparse
import torch
import numpy as np
import cv2
import open3d as o3d

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
from ur_toolbox.robot import UR_Camera_Gripper
from ur_toolbox.robot.Allegro.Allegro_grasp import AllegroGraspGroup, grasp_types
from np_utils import transform_point_cloud
from pt_utils import batch_viewpoint_params_to_matrix
from collision_detector import ModelFreeCollisionDetectorMultifinger
import queue
from itertools import count
from threading import Thread
from queue import Queue
import multiprocessing as mp
import MinkowskiEngine as ME
from minkowski_graspnet_single_point import MinkowskiGraspNet, MinkowskiGraspNetMultifingerType1Inference

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint_path', required=True, help='Model checkpoint path')
parser.add_argument('--allegro_model_path', default='logs/model/allegro_model/allegro_obj140', help='allegro model checkpoint path')
parser.add_argument('--save_information_path', default='logs/data/allegro/allegro_test/model_obj140trials', help='Allegro model result information path')
parser.add_argument('--Allegro_mesh_json_path', default='generate_mesh_and_pointcloud/allegro_urdf', help='Allegro meshes and json path')
parser.add_argument('--robot_ip', required=True, help='Robot IP')
parser.add_argument('--use_graspnet_v2', action='store_true', help='Whether to use graspnet v2 format')
parser.add_argument('--half_views', action='store_true', help='Use only half views in network.')
parser.add_argument('--global_camera', action='store_true', help='Use the settings for global camera.')
cfgs = parser.parse_args()

MAX_GRASP_WIDTH = 0.11
MIN_GRASP_WIDTH = 0.04
BATCH_SIZE = 1
DEBUG = True
NUM_OF_ALLEGRO_DEPTH = 4
NUM_OF_ALLEGRO_TYPE = 10
Allegro_VOXElGRID = 0.003
POINTCLOUD_AUGMENT_NUM = 10
DEFAULT_DEPTH = 0.00
RANDOM = False

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

def flip_ggarray(ggarray):
    ggarray_rotations = ggarray[:, 4:13].reshape((-1, 3, 3))
    tcp_x_axis_on_base_frame = ggarray_rotations[:, 1, 1]
    # normal_vector = np.array((- 1 / 2, np.sqrt(3) / 2, 0))
    if_flip = [False for _ in range(len(ggarray))]
    for ids, y_x in enumerate(tcp_x_axis_on_base_frame):
        if y_x < 0:
            ggarray_rotations[ids, :3, 1:3] = -ggarray_rotations[ids, :3, 1:3]
            if_flip[ids] = True
    return ggarray, if_flip

def flip_z_ggarray(ggarray, allegro_types):
    ggarray_rotations = ggarray[:, 4:13].reshape((-1, 3, 3))
    tcp_z_axis_on_base_frame = ggarray_rotations[:, 0, 2]
    tcp_x_axis_on_base_frame = ggarray_rotations[:, 1, 1]
    if_flip = [False for _ in range(len(ggarray))]
    for ids, y_z in enumerate(tcp_z_axis_on_base_frame):
        if y_z < 0 and allegro_types[ids] in [9]:
            ggarray_rotations[ids, :3, 1:3] = -ggarray_rotations[ids, :3, 1:3]

    ggarray[:, 4:13] = ggarray_rotations.reshape((-1, 9))
    return ggarray, if_flip

def get_grasp_features(grasp_features_array):
    grasp_features = dict()
    grasp_features['grasp_angles'] = int(grasp_features_array[-3]+0.1)
    grasp_features['grasp_depths'] = int(grasp_features_array[-2]*100+0.1)
    grasp_features['stage3_grasp_scores'] = grasp_features_array[:240].tolist()
    grasp_features['grasp_preds_features'] = grasp_features_array[240:240+480].tolist()
    grasp_features['stage3_grasp_features'] = grasp_features_array[240+480:240+480+512].tolist()
    grasp_features['before_generator'] = grasp_features_array[240+480+512:240+480+512+512].tolist()
    grasp_features['point_features'] = grasp_features_array[240+480+512+512:240+480+512+512+512].tolist()
    grasp_features['point_id'] = int(grasp_features_array[-4])
    grasp_features['if_flip'] = bool(int(grasp_features_array[-1]))
    grasp_features['view_inds'] = bool(int(grasp_features_array[-6]))
    grasp_features['view_score'] = bool(int(grasp_features_array[-5]))
    return grasp_features

def get_graspgroup_features(grasp_features_array, sinput):
    grasp_features = dict()
    grasp_features['grasp_angles'] = (grasp_features_array[:, -3]+0.1).astype(int)
    grasp_features['grasp_depths'] = (grasp_features_array[:, -2]*100+0.1).astype(int)
    grasp_features['stage3_grasp_scores'] = grasp_features_array[:, :240]
    grasp_features['grasp_preds_features'] = grasp_features_array[:, 240:240+480]
    grasp_features['stage3_grasp_features'] = grasp_features_array[:, 240+480:240+480+512]
    grasp_features['before_generator'] = grasp_features_array[:, 240+480+512:240+480+512+512]
    grasp_features['point_features'] = grasp_features_array[:, 240+480+512+512:240+480+512+512+512]
    grasp_features['point_id'] = (grasp_features_array[:, -4]+0.1).astype(int)
    grasp_features['if_flip'] = grasp_features_array[:, -1]
    grasp_features['view_inds'] = (grasp_features_array[:, -6]+0.1).astype(int)
    grasp_features['view_score'] = grasp_features_array[:, -5]
    grasp_features['sinput'] = sinput

    grasp_preds_features_rot = np.zeros(grasp_features['grasp_preds_features'].shape, dtype = np.float32)
    new_type = grasp_features['grasp_angles']
    for idx, if_flip in enumerate(grasp_features['if_flip']):
        if if_flip:
            first_half_scores = copy.deepcopy(grasp_features['grasp_preds_features'][idx, :120])
            last_half_scores = copy.deepcopy(grasp_features['grasp_preds_features'][idx, 120:240])
            first_half_widths = copy.deepcopy(grasp_features['grasp_preds_features'][idx, 240:360])
            last_half_widths = copy.deepcopy(grasp_features['grasp_preds_features'][idx, 360:480])
            grasp_features['grasp_preds_features'][idx, :120] = last_half_scores
            grasp_features['grasp_preds_features'][idx, 120:240] = first_half_scores
            grasp_features['grasp_preds_features'][idx, 240:360] = last_half_widths
            grasp_features['grasp_preds_features'][idx, 360:480] = first_half_widths
        grasp_preds_features_rot[idx, :240-new_type[idx]*5] = grasp_features['grasp_preds_features'][idx, new_type[idx]*5:240]
        grasp_preds_features_rot[idx, 240-new_type[idx]*5:240] = grasp_features['grasp_preds_features'][idx, 0:new_type[idx]*5]
        grasp_preds_features_rot[idx, 240:480-new_type[idx]*5] = grasp_features['grasp_preds_features'][idx, 240+new_type[idx]*5:480]
        grasp_preds_features_rot[idx, 480-new_type[idx]*5:480] = grasp_features['grasp_preds_features'][idx, 240:240+new_type[idx]*5] 
    grasp_features['grasp_preds_features'] = grasp_preds_features_rot

    return grasp_features

def get_robot(robot_ip="192.168.2.102", use_rt=False, robot_debug=True, gripper_type='robotiq',
              gripper_port='192.168.1.29', global_cam=False):
    robot = UR_Camera_Gripper(robot_ip, use_rt, camera=None, robot_debug=robot_debug, gripper_type=gripper_type,
                              gripper_port=gripper_port, global_cam=global_cam)
    robot.set_tcp((0, 0, 0.0, 0, 0, 0))
    robot.set_payload(2.7, (0, 0, 0.12))
    return robot

def get_depth(existing_shm_depth):
    time.sleep(0.1)
    depths_1 = np.copy(np.ndarray((720, 1280), dtype=np.uint16, buffer=existing_shm_depth.buf))
    return depths_1

def save_grasp_information(two_fingers_ggarray, two_fingers_ggarray_object_ids, Allegro_ggarray, 
                            two_fingers_ggarray_source, Allegro_ggarray_source, two_fingers_ggarray_object_ids_source,
                            two_fingers_grasp_used, Allegro_grasp_used, grasp_features_used, 
                            mat_pose, colors_saved, depths_saved, grasp_features, two_fingers_source_grasp_features,
                            before_collision, after_collision, Allegro_grasp_type):
    save_path = cfgs.save_information_path

    timeStamp = datetime.datetime.now().timestamp()
    timeArray = time.localtime(timeStamp)
    otherStyleTime = time.strftime("%Y-%m-%d_%H-%M-%S", timeArray)
    save_path = os.path.join(save_path, grasp_types[str(int(Allegro_grasp_type))]['name'], otherStyleTime)

    information = OrderedDict()
    two_fingers_ggarray_proposals = []
    Allegro_ggarray_proposals = []
    for idx, tfg in enumerate(two_fingers_ggarray):
        two_fingers_ggarray_proposals.append(
            [float(tfg.score), float(tfg.width), float(tfg.height), float(tfg.depth)] +
            np.array(tfg.rotation_matrix).reshape((-1)).tolist() + list(tfg.translation.tolist()) +
            [float(two_fingers_ggarray_object_ids[idx])])
        Allegro_ggarray_proposals.append(list(Allegro_ggarray[idx].get_array_grasp()))


    two_fingers_ggarray_source_saved = []
    Allegro_ggarray_source_saved = []
    for idx, tfg in enumerate(two_fingers_ggarray_source):
        two_fingers_ggarray_source_saved.append(
            [float(tfg.score), float(tfg.width), float(tfg.height), float(tfg.depth)] +
            np.array(tfg.rotation_matrix).reshape((-1)).tolist() + list(tfg.translation.tolist()) +
            [float(two_fingers_ggarray_object_ids_source[idx])])
        Allegro_ggarray_source_saved.append(list(Allegro_ggarray_source[idx].get_array_grasp()))

    tfg = two_fingers_grasp_used
    two_fingers_array = [float(tfg.score), float(tfg.width), float(tfg.height), float(tfg.depth)] + \
                        np.array(tfg.rotation_matrix).reshape((-1)).tolist() + \
                        list(tfg.translation.reshape(-1).tolist()) + \
                        [float(Allegro_grasp_used.object_id)]
    grasp_features_used = get_grasp_features(grasp_features_used)
    restart = False
    print('Is the grasping successful? press 1 successfully, press 2 failed, restart grasping and press 3, exit press 4\n')
    if_success = input('The result is: ')
    while True:
        if if_success == '1':
            information['result'] = True
            break
        elif if_success == '2':
            information['result'] = False
            break
        elif if_success == '3':
                restart = True
                break
        elif if_success == '4':
            exit()
        else:
            if_success = input('Re-enter the result: ')
    if restart:
        return
    information['two_fingers_pose'] = list(two_fingers_array)
    information['Allegro_pose'] = list(Allegro_grasp_used.get_array_grasp())
    information['two_fingers_pose_angle_type'] = grasp_features_used['grasp_angles']
    information['two_fingers_pose_depth_type'] = grasp_features_used['grasp_depths']

    information['two_fingers_pose_AD'] = grasp_features_used['stage3_grasp_scores']
    information['two_fingers_pose_features_grasp_preds'] = grasp_features_used['grasp_preds_features']
    information['two_fingers_pose_features'] = grasp_features_used['stage3_grasp_features']
    information['two_fingers_pose_features_before_generator'] = grasp_features_used['before_generator']
    information['point_features'] = grasp_features_used['point_features']
    
    information['point_id'] = grasp_features_used['point_id']
    information['if_flip'] = grasp_features_used['if_flip']
    information['Allegro_pose_finger_type'] = int(Allegro_grasp_used.grasp_type + 0.1)
    information['Allegro_pose_depth_type'] = int(Allegro_grasp_used.depth*100 + 0.1) - grasp_features_used['grasp_depths']
    information['base_2_tcp1'] = np.array(mat_pose[0]).tolist()
    information['base_2_tcp1_backup'] = np.array(mat_pose[1]).tolist()
    information['tcp_2_gripper'] = np.array(mat_pose[2]).tolist()
    information['base_2_TwoFingersGripper_pose'] = np.array(mat_pose[3]).tolist()
    information['tcp_2_camera'] = np.array(mat_pose[4]).tolist()
    information['base_2_tcp_ready'] = np.array(mat_pose[5]).tolist()

    information['camera_internal'] = [[631.119, 363.884], [919.835, 919.61]]

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    color_path = os.path.join(save_path, 'color.png')
    depth_path = os.path.join(save_path, 'depth.png')
    from PIL import Image
    cv2.imwrite(color_path, (cv2.cvtColor(colors_saved, cv2.COLOR_RGB2BGR) * 255.0).astype(np.float32))
    cv2.imwrite(depth_path, depths_saved)
    json_path = os.path.join(save_path, 'information.json')

    json_file = json.dumps(information, indent=4)
    with open(json_path, 'w') as handle:
        handle.write(json_file)
    print('Saved successfully')

def get_grasp(net, depths, existing_shm_color, augment_mat=np.eye(4), flip=False, voxel_size=0.005):
    # fx, fy = 908.435, 908.679
    # cx, cy = 650.366, 367.277
    # s = 1000.0
    # D415, id=104122064489
    # fx, fy = 912.898, 912.258
    # cx, cy = 629.536, 351.637
    # s = 1000.0
    # fx, fy = 919.835, 919.61
    # cx, cy = 631.119, 363.884
    # s = 1000.0
    # D415, id=104122062823
    fx, fy = 913.232, 912.452
    cx, cy = 628.847, 350.771
    s = 1000.0

    xmap, ymap = np.arange(depths.shape[1]), np.arange(depths.shape[0])
    xmap, ymap = np.meshgrid(xmap, ymap)

    points_z = depths / s
    points_x = (xmap - cx) / fx * points_z
    points_y = (ymap - cy) / fy * points_z
    # mask = (points_z > 0.45) & (points_z < 0.88) & (points_x > -0.21) & (points_x < 0.21) & (points_y > -0.08) & (points_y < 0.2)

    # mask = (points_z > 0.35) & (points_z < 0.55)
    mask = (points_z > 0.2) & (points_z < 0.65) # 23.4.26
    # mask[:] = True
    points = np.stack([points_x, points_y, points_z], axis=-1)
    points = points[mask].astype(np.float32)

    # print(points[:, 0].max(), points[:, 0].min())
    # print(points[:, 1].max(), points[:, 1].min())
    if DEBUG:
        colors = np.copy(np.ndarray((720, 1280, 3), dtype=np.float32, buffer=existing_shm_color.buf))
        # plt.subplot(2, 1, 1)
        # plt.imshow(colors)
        # plt.subplot(2, 1, 2)
        # plt.imshow(depths)
        # plt.show()
        colors = colors[mask].astype(np.float32)

    cloud = None
    if DEBUG:
        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(points)
        cloud.colors = o3d.utility.Vector3dVector(colors)

    points = transform_point_cloud(points, augment_mat).astype(np.float32)
    points = torch.from_numpy(points)
    coords = np.ascontiguousarray(points / voxel_size, dtype=int)
    # Upd Note. API change.
    _, idxs = ME.utils.sparse_quantize(coords, return_index=True)
    coords = coords[idxs]
    points = points[idxs]
    coords_batch, points_batch = ME.utils.sparse_collate([coords], [points])

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    sinput = ME.SparseTensor(points_batch, coords_batch, device=device)

    end_points = {'sinput': sinput, 'point_clouds': [sinput.F]}
    with torch.no_grad():
        end_points = net(end_points)
        preds, grasp_features = parse_preds(end_points, use_v2=cfgs.use_graspnet_v2)
        if len(preds) == 0:
            print('No grasp detected')
            return None, cloud, points.cuda(), None, [sinput]
        else:
            preds = preds[0]
    # filter
    if flip:
        augment_mat[:, 0] = -augment_mat[:, 0]
    augment_mat_tensor = torch.tensor(copy.deepcopy(np.linalg.inv(augment_mat).astype(np.float32)), device=device)
    rotation = augment_mat_tensor[:3, :3].reshape((-1)).repeat((preds.size()[0], 1)).view((preds.size()[0], 3, 3))
    translation = augment_mat_tensor[:3, 3]

    preds[:,12:15] = torch.matmul(rotation, preds[:,12:15].view((-1, 3, 1))).view(-1, 3) + translation
    pose_rotation = torch.matmul(rotation, preds[:,3:12].view((-1, 3, 3)))
    if flip:
        preds[:, 12] = -preds[:, 12]
        pose_rotation[:, 0, :] = -pose_rotation[:, 0, :]
        pose_rotation[:, :, 1] = -pose_rotation[:, :, 1]
    preds[:, 3:12] = pose_rotation.view((-1, 9))

    mask = (preds[:,9] > 0.93) & (preds[:,1] < MAX_GRASP_WIDTH) & (preds[:,1] > MIN_GRASP_WIDTH)
    # workspace_mask = (preds[:,12] > -0.2) & (preds[:,12] < 0.2) & (preds[:,13] > -0.25) & (preds[:,13] < 0.1) 2022.11.23.15.05
    #workspace_mask = (preds[:,12] > -0.25) & (preds[:,12] < 0.25) & (preds[:,13] > -0.205) & (preds[:,13] < 0.08) 2022.12.16.22.33
    # workspace_mask = (preds[:,12] > -0.25) & (preds[:,12] < 0.25) & (preds[:,13] > -0.205) & (preds[:,13] < 0.03)
    workspace_mask = (preds[:,12] > -0.2) & (preds[:,12] < 0.2) & (preds[:,13] > -0.20) & (preds[:,13] < 0.07) # 23.4.26

    # workspace_mask = (preds[:,12] > -0.21) & (preds[:,12] < 0.21) & (preds[:,13] > -0.02) & (preds[:,13] < 0.19) & (preds[:,14] > 0.45) & (preds[:,14] < 0.88) 
    # print('000preds grasp features: ', len(preds))

    preds = preds[workspace_mask & mask]
    grasp_features = grasp_features[0][workspace_mask & mask]
    # print('111preds grasp features: ', preds.size(), grasp_features.size())
    if len(preds) == 0:
        print('No grasp detected after masking')
        return None, cloud, points.cuda(), None, [sinput]

    points = points.cuda()
    heights = 0.03 * torch.ones([preds.shape[0], 1]).cuda()
    object_ids = -1 * torch.ones([preds.shape[0], 1]).cuda()
    ggarray = torch.cat([preds[:, 0:2], heights, preds[:, 2:15], preds[:, 15:16], object_ids], axis=-1)

    return ggarray, cloud, points, grasp_features, [sinput]

def load_meshes_pointcloud(path):
    meshes_pcls = dict()
    meshes_pcl_path = os.path.join(path, 'meshes/source_pointclouds/voxel_size_' + str(int(Allegro_VOXElGRID * 1000)))
    for type in os.listdir(meshes_pcl_path):
        type_path = os.path.join(meshes_pcl_path, type)
        for name in os.listdir(type_path):
            width = name[:-4]
            name_path = os.path.join(type_path, name)
            meshes_pcl = o3d.io.read_point_cloud(name_path)
            meshes_pcls[type + '_' + width] = meshes_pcl
    return meshes_pcls

def create_tale_pointcloud(width=0.5, height=0.005, depth=0.4, dx=-0.25, dy=-0.15, dz=-0.55, grid_size=0.005):
    xmap = np.linspace(0, width, int(width/grid_size))
    ymap = np.linspace(0, depth, int(depth/grid_size))
    zmap = np.linspace(0, height, int(height/grid_size))
    xmap, ymap, zmap = np.meshgrid(xmap, ymap, zmap, indexing='xy')
    xmap += dx
    ymap += dy
    zmap += dz
    points = np.stack([xmap, -ymap, -zmap], axis=-1)
    points = points.reshape([-1, 3])
    cloud = o3d.geometry.PointCloud()
    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(0.01)
    cloud.points = o3d.utility.Vector3dVector(points)
    # o3d.visualization.draw_geometries([cloud, frame])
    return points

def get_allegro_model(allegro_models_path):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    allegro_models = dict()
    print('model path: ', allegro_models_path)
    for model_type in os.listdir(allegro_models_path):
        allegro_model_path = os.path.join(allegro_models_path, model_type)
        for model_class in os.listdir(allegro_model_path):
            allegro_models[model_type] = dict()
            model_classs_type = os.path.join(allegro_model_path, model_class)
            for model in os.listdir(model_classs_type):
                allegro_model_type_path = os.path.join(model_classs_type, model)
                # print(allegro_model_type_path)
                allegro_model = MinkowskiGraspNetMultifingerType1Inference(input_num=int(model_type))
                allegro_net = torch.load(allegro_model_type_path)
                allegro_model.load_state_dict(allegro_net.state_dict())
                # allegro_model = MinkowskiGraspNetMultifingerType1Inference(input_num=int(model_type))
                # allegro_net = torch.load(allegro_model_type_path)
                # allegro_model.load_state_dict(allegro_net['model_state_dict'])
                allegro_model.to(device)
                allegro_model.eval()
                if model_class not in allegro_models.keys():
                    allegro_models[model_type][model_class] = [allegro_model]
                else:
                    allegro_models[model_type][model_class].append(allegro_model)
    return allegro_models

def get_allegro_depth_type(allegro_models, grasp_features_dic, ggarray, grasp_features):
    if RANDOM:
        allegro_depth = (np.random.randint(0, NUM_OF_ALLEGRO_DEPTH, (len(ggarray),))) * 0.01
        allegro_type = (np.random.randint(0, NUM_OF_ALLEGRO_TYPE, (len(ggarray),)))
        scores = ggarray[:, 0] 
        return allegro_depth, allegro_type, scores, ggarray, grasp_features
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    for k, v in grasp_features_dic.items():
        if k == 'point_id' or k == 'sinput':
            continue
        grasp_features_dic[k] = torch.tensor(copy.deepcopy(v), device=device)
    allegro_depth_type_scores = []
    for model_type, allegro_model in allegro_models.items():
        if model_type == '240':
            continue
            model_input = torch.cat([grasp_features_dic["stage3_grasp_scores"]], dim=1)
        elif model_type == '480':
            print('use final model: ', model_type)
            model_input = torch.cat([grasp_features_dic["grasp_preds_features"]], dim=1)

        allegro_model_sorted = OrderedDict(sorted(allegro_model.items(), key = lambda t : t[0]))
        for model_class, sub_allegro_model in allegro_model_sorted.items():
            allegro_depth_types = torch.tensor(0, device=device)
            for sub_sub_allegro_model in sub_allegro_model:
                with torch.no_grad():
                    grasp_pred, _ = sub_sub_allegro_model(model_input) # (B, 1， NUM_OF_TWO_FINGER_DEPTH*NUM_OF_ALLEGRO_DEPTH)
                    grasp_pred = grasp_pred.view(grasp_pred.shape[0], 5*NUM_OF_ALLEGRO_DEPTH)
                two_fingers_depth = grasp_features_dic['grasp_depths'] # (B, )
                base = torch.tensor(np.array([[i for i in range(NUM_OF_ALLEGRO_DEPTH)]
                                    for _ in range(grasp_pred.size()[0])]), device=device) # (B, NUM_OF_ALLEGRO_DEPTH)
                select_index = (two_fingers_depth).view(-1, 1) * NUM_OF_ALLEGRO_DEPTH + base
                # print(select_index.shape, grasp_pred.shape)
                grasp_pred = grasp_pred.gather(1, select_index)
                allegro_depth_types = allegro_depth_types + grasp_pred
            allegro_depth_types = (allegro_depth_types / len(sub_allegro_model)).squeeze(1)
            allegro_depth_type_scores.append(allegro_depth_types)
                
    print('allegro_depth_type_scores: ', torch.cat(allegro_depth_type_scores, axis=1).shape)
    allegro_depth_type_scores = torch.cat(allegro_depth_type_scores, axis=1).view(-1) # (B, NUM_OF_ALLEGRO_DEPTH*NUM_OF_ALLEGRO_TYPE)
    
    # allegro_depth_types = (allegro_depth_types / len(allegro_models)).squeeze(1)
    # two_fingers_angle = grasp_features['grasp_angles']
    
    # print('allegro_depth_type_scores > 0.85: ',allegro_depth_type_scores.shape, torch.sum(allegro_depth_type_scores >= 0.85, dim=0), allegro_depth_type_scores[:-10])
    scores, index = allegro_depth_type_scores.topk(min(3000, allegro_depth_type_scores.size()[0]))
    pose_index = (index / (NUM_OF_ALLEGRO_DEPTH * NUM_OF_ALLEGRO_TYPE)).long()
    ggarray = torch.tensor(copy.deepcopy(ggarray), device=device)[pose_index]
    grasp_features = torch.tensor(copy.deepcopy(grasp_features), device=device)[pose_index]
    allegro_depth = ((index % (NUM_OF_ALLEGRO_DEPTH * NUM_OF_ALLEGRO_TYPE)) % NUM_OF_ALLEGRO_DEPTH).int() * 0.01
    allegro_type = ((index % (NUM_OF_ALLEGRO_DEPTH * NUM_OF_ALLEGRO_TYPE)) / NUM_OF_ALLEGRO_DEPTH).int()
    # for i in range(4):
    #     print('\n\n depth {}: '.format(i), sum(allegro_depth*100<i+1))
    # print('\n\n\nscores {}: '.format(i), sum(scores>i*0.1))    
    # scores = scores + allegro_depth * 2
    # scores = scores + allegro_type * 0.1 - allegro_depth * 10
    # scores[allegro_type==3] -= 0.05
    return allegro_depth.cpu().numpy(), allegro_type.cpu().numpy() + 1, scores.detach().cpu().numpy(), \
                ggarray.cpu().numpy(), grasp_features.cpu().numpy()

def augment_data(flip=False):
    flip_mat = np.identity(4)
    # Flipping along the YZ plane
    if flip:
        flip_mat = np.array([[-1, 0, 0, 0],
                             [0, 1, 0, 0],
                             [0, 0, 1, 0],
                             [0, 0, 0, 1]])

    # Rotation along up-axis/Z-axis
    rot_angle = (np.random.random() * np.pi / 3) - np.pi / 6  # -30 ~ +30 degree
    c, s = np.cos(rot_angle), np.sin(rot_angle)
    rot_mat = np.array([[c, -s, 0, 0],
                        [s, c, 0, 0],
                        [0, 0, 1, 0],
                        [0, 0, 0, 1]])

    # Translation along X/Y/Z-axis
    offset_x = np.random.random() * 0.1 - 0.05  # -0.05 ~ 0.05
    offset_y = np.random.random() * 0.1 - 0.05  # -0.05 ~ 0.05
    # offset_z = np.random.random() * 0.3 - 0.1  # -0.1 ~ 0.2
    trans_mat = np.array([[1, 0, 0, offset_x],
                          [0, 1, 0, offset_y],
                          [0, 0, 1, 0],
                          [0, 0, 0, 1]])

    aug_mat = np.dot(trans_mat, np.dot(rot_mat, flip_mat).astype(np.float32)).astype(np.float32)
    # print('aug_mat: ', aug_mat, trans_mat, rot_mat, flip_mat)
    return aug_mat

def get_ggarray_features(existing_shm_depth, existing_shm_color, net):
    depths = get_depth(existing_shm_depth)
    augment_mat1 = np.eye(4)
    augment_mats = []
    for i in range(POINTCLOUD_AUGMENT_NUM):
        if i % 2 == 0:
            augment_mat = augment_data()
        else:
            augment_mat = augment_data(flip=True)
        augment_mats.append(augment_mat)

    ggarray, cloud, points_down, grasp_features, sinput = get_grasp(net, depths, existing_shm_color, augment_mat=augment_mat1)
    for i in range(POINTCLOUD_AUGMENT_NUM):
        if i % 2 == 0:
            ggarray2, _, _, grasp_features2, sinput2 = get_grasp(net, depths, existing_shm_color, augment_mat=augment_mats[i])
        else:
            ggarray2, _, _, grasp_features2, sinput2 = get_grasp(net, depths, existing_shm_color, augment_mat=augment_mats[i], flip=True)
        if ggarray2 is None:
            continue
        if ggarray is None:
            ggarray = ggarray2
            grasp_features = grasp_features2
            sinput = sinput2
        else:
            sinput.append(sinput2[0])
            ggarray = torch.cat([ggarray, ggarray2], axis=0)
            grasp_features = torch.cat([grasp_features, grasp_features2], axis=0)
    return ggarray, cloud, points_down, grasp_features, sinput

def select_grasp_type(allegro_gg):
    select_gg_types = [[] for _ in range(NUM_OF_ALLEGRO_TYPE)]
    for idx, score in enumerate(allegro_gg.scores):
        grasp_type = allegro_gg.grasp_types[idx]
        max_num = 50
        if grasp_type in [4, 6]:
            max_num = 5
        # else:
        #     max_num = 50
        if len(select_gg_types[int(grasp_type)-1]) < max_num:
            select_gg_types[int(grasp_type)-1].append(idx)
    print('select grasp type shape: ', [len(i) for i in select_gg_types])
    gg_type_id = []
    for gg_type in select_gg_types:
        gg_type_id += gg_type
    return gg_type_id

def robot_grasp(cfgs):
    robot = get_robot(cfgs.robot_ip, robot_debug=True, gripper_type='Allegro', global_cam=cfgs.global_camera)
    net = get_net(cfgs.checkpoint_path, use_v2=cfgs.use_graspnet_v2)
    allegro_models = get_allegro_model(cfgs.allegro_model_path)
    fail = 0
    existing_shm_color = shared_memory.SharedMemory(name='realsense_color')
    existing_shm_depth = shared_memory.SharedMemory(name='realsense_depth')
    meshes_pcls = load_meshes_pointcloud(cfgs.Allegro_mesh_json_path)
    table_pointcloud = create_tale_pointcloud()
    try:
        v = 0.01
        a = 0.01
        if cfgs.global_camera:
            robot.movej(robot.throwj2, acc=a*2,
                        vel=v*3)  # this v and a are anguler, so it should be larger than translational
            t1 = time.time()
            depths = get_depth(existing_shm_depth)
            depths_saved = copy.deepcopy(depths)
            colors_saved = np.copy(np.ndarray((720, 1280, 3), dtype=np.float32, buffer=existing_shm_color.buf))
            # ggarray, cloud, points_down, grasp_features = get_grasp(net, depths, existing_shm_color)
            ggarray, cloud, points_down, grasp_features, sinput = get_ggarray_features(existing_shm_depth, existing_shm_color, net)

            t3 = time.time()
            print(f'Net Time:{t3 - t1}')

        else:
            robot.movel(robot.ready_pose(), acc=a*2,
                        vel=v*3)  # this v and a are anguler, so it should be larger than translational

        while True:
            if not cfgs.global_camera:
                t1 = time.time()
                robot.movel(robot.ready_pose(), acc=a * 10, vel=v * 10,
                            wait=True)  # this v and a are anguler, so it should be larger than translational
                time.sleep(0.5)
                print('movel')
                robot.gripper_home()
                print('gripper home')
                time.sleep(0.7)
                depths = get_depth(existing_shm_depth)
                depths_saved = copy.deepcopy(depths)
                colors_saved = np.copy(np.ndarray((720, 1280, 3), dtype=np.float32, buffer=existing_shm_color.buf))
                # ggarray, cloud, points_down, grasp_features = get_grasp(net, depths, existing_shm_color)
                ggarray, cloud, points_down, grasp_features, sinput = get_ggarray_features(existing_shm_depth, existing_shm_color, net)

                t3 = time.time()
                print(f'Net Time:{t3 - t1}')

            if ggarray is None:
                fail = fail + 1
                if not cfgs.global_camera:
                    while robot.is_program_running():
                        robot.stopj(acc=10.0 * a)
                    robot.movel(robot.ready_pose(), acc=a*10,
                                vel=v*10)  # this v and a are anguler, so it should be larger than translational
                    time.sleep(0.1)
                else:
                    depths = get_depth(existing_shm_depth)
                    # ggarray, cloud, points_down, grasp_features = get_grasp(net, depths, existing_shm_color)
                    ggarray, cloud, points_down, grasp_features, sinput = get_ggarray_features(existing_shm_depth,
                                                                                       existing_shm_color, net)
                    time.sleep(0.1)
                if DEBUG:
                    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(0.1)
                    sphere = o3d.geometry.TriangleMesh.create_sphere(0.002, 20).translate([0, 0, 0.490])
                    o3d.visualization.draw_geometries([cloud, frame, sphere])
                continue

            ########## PROCESS GRASPS ##########
            # collision detection
            ggarray = ggarray.cpu().numpy()
            print('init ggroup', ggarray.shape)
            # Prevent the robot arm from crossing the border, 
            grasp_features = grasp_features.cpu().numpy()
            two_fingers_source_grasp_features = copy.deepcopy(grasp_features)

            ggarray, if_flip = flip_ggarray(ggarray)
            grasp_features = np.c_[grasp_features, if_flip]

            source_index = ggarray[:, 0].argsort()
            ggarray = ggarray[source_index][::-1][:2000]
            grasp_features = grasp_features[source_index][::-1][:2000]

            grasp_features_dic = get_graspgroup_features(grasp_features, sinput)
            allegro_depths, allegro_types, scores, ggarray, grasp_features = \
                                            get_allegro_depth_type(allegro_models, grasp_features_dic,
                                                                    ggarray, grasp_features=grasp_features)
            print('before allegro hand score > 0.85: ', sum(scores > 0.85), len(ggarray), len(allegro_depths), len(allegro_types), len(grasp_features))
            if not RANDOM:
                score_thresh = 0.7
                mask = (scores > score_thresh)
                ggarray = ggarray[mask]
                grasp_features = grasp_features[mask]
                allegro_depths = allegro_depths[mask]
                allegro_types = allegro_types[mask]
                scores = scores[mask]
            
            # allegro_types = np.ones(len(allegro_types)) * 2
            ggarray, if_flip = flip_z_ggarray(ggarray, allegro_types)

            print('after allegro hand score > 0.7: ', len(ggarray))
            if len(ggarray) == 0:
                print('There is no grasp that score greater than 0.9 ')
                if cfgs.global_camera:
                    depths = get_depth(existing_shm_depth)
                    depths_saved = copy.deepcopy(depths)
                    colors_saved = np.copy(np.ndarray((720, 1280, 3), dtype=np.float32, buffer=existing_shm_color.buf))
                    ggarray, cloud, points_down, grasp_features, sinput = get_ggarray_features(existing_shm_depth, existing_shm_color, net)
                if DEBUG:
                    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(0.1)
                    sphere = o3d.geometry.TriangleMesh.create_sphere(0.002, 20).translate([0, 0, 0.490])
                    o3d.visualization.draw_geometries([cloud, frame, sphere])
                continue

            two_fingers_ggarray = GraspGroup(ggarray)
            two_fingers_ggarray_object_ids_source = ggarray[:, 16]
            two_fingers_ggarray_source = copy.deepcopy(two_fingers_ggarray)

            Allegro_ggarray = AllegroGraspGroup() 
            Allegro_ggarray.from_graspgroup(two_fingers_ggarray, allegro_types, cfgs.Allegro_mesh_json_path)
            Allegro_ggarray.object_ids = two_fingers_ggarray_object_ids_source
            Allegro_ggarray.scores = scores
            Allegro_ggarray.depths = Allegro_ggarray.depths + allegro_depths + DEFAULT_DEPTH
            Allegro_ggarray_source = copy.deepcopy(Allegro_ggarray)

            two_fingers_ggarray_object_ids = two_fingers_ggarray_object_ids_source
            # import pdb
            # pdb.set_trace()
            before_collision = [copy.deepcopy(np.array(Allegro_ggarray.grasp_group_array).tolist()), 
                                copy.deepcopy(np.array(two_fingers_ggarray.grasp_group_array).tolist()),
                                copy.deepcopy(np.array(grasp_features).tolist())]
            # print('ggarray object_id2: ', len(two_fingers_ggarray), two_fingers_ggarray_object_ids.shape)
            if len(Allegro_ggarray) == 0:
                print('No grasp detected after filter')
                if cfgs.global_camera:
                    ggarray, cloud, points_down, grasp_features, sinput = get_ggarray_features(existing_shm_depth,
                                                                                       existing_shm_color, net)
                if DEBUG:
                    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(0.1)
                    sphere = o3d.geometry.TriangleMesh.create_sphere(0.002, 20).translate([0, 0, 0.490])
                    o3d.visualization.draw_geometries([cloud, frame, sphere])
                continue
            
            index_score = np.argsort(Allegro_ggarray.scores)[::-1]
            Allegro_ggarray = Allegro_ggarray[index_score]
            two_fingers_ggarray = two_fingers_ggarray[index_score]
            grasp_features = grasp_features[index_score]
            two_fingers_ggarray_object_ids = two_fingers_ggarray_object_ids[index_score]

            index_type = select_grasp_type(Allegro_ggarray)
            Allegro_ggarray = Allegro_ggarray[index_type]
            two_fingers_ggarray = two_fingers_ggarray[index_type]
            grasp_features = grasp_features[index_type]
            two_fingers_ggarray_object_ids = two_fingers_ggarray_object_ids[index_type]

            print('\n\nafter filter*********', len(Allegro_ggarray), len(two_fingers_ggarray), len(two_fingers_ggarray_object_ids))
            if len(Allegro_ggarray) == 0:
                print('No grasp detected after filter')
                if cfgs.global_camera:
                    ggarray, cloud, points_down, grasp_features, sinput = get_ggarray_features(existing_shm_depth,
                                                                                       existing_shm_color, net)
                if DEBUG:
                    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(0.1)
                    sphere = o3d.geometry.TriangleMesh.create_sphere(0.002, 20).translate([0, 0, 0.490])
                    o3d.visualization.draw_geometries([cloud, frame, sphere])
                continue
            approach_distance = 0.08

            start_time = time.time()
            # print('points down shape: ', points_down.shape, np.vstack((points_down.cpu().numpy(), table_pointcloud)).shape)
            mfcdetector = ModelFreeCollisionDetectorMultifinger(points_down.cpu().numpy(), voxel_size=0.001)
            Allegro_ggarray, two_fingers_ggarray, empty_mask, min_width_index = mfcdetector.detect(Allegro_ggarray, two_fingers_ggarray,
                                                                  cfgs.Allegro_mesh_json_path, meshes_pcls, min_grasp_width=MIN_GRASP_WIDTH,
                                                                  VoxelGrid=Allegro_VOXElGRID, DEBUG=False, approach_dist=approach_distance,
                                                                  collision_thresh=1, adjust_gripper_centers=True,)
            print('collision time: ', time.time()-start_time)
            print('ggarray object_id2: ', len(two_fingers_ggarray), two_fingers_ggarray_object_ids.shape)

            # proposals
            Allegro_ggarray = Allegro_ggarray[empty_mask]
            two_fingers_ggarray = two_fingers_ggarray[empty_mask]
            two_fingers_ggarray_object_ids = two_fingers_ggarray_object_ids[min_width_index][empty_mask]
            grasp_features = grasp_features[min_width_index][empty_mask]

            after_collision = [copy.deepcopy(np.array(Allegro_ggarray.grasp_group_array).tolist()), 
                                copy.deepcopy(np.array(two_fingers_ggarray.grasp_group_array).tolist()),
                                copy.deepcopy(np.array(grasp_features).tolist())]

            print('\n\nafter collision num *********', len(Allegro_ggarray), len(two_fingers_ggarray), len(two_fingers_ggarray_object_ids))

            if len(Allegro_ggarray) == 0:
                print('No Grasp detected after collision detection!')
                fail = fail + 1
                if DEBUG:
                    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(0.1)
                    sphere = o3d.geometry.TriangleMesh.create_sphere(0.002, 20).translate([0, 0, 0.490])
                    o3d.visualization.draw_geometries([cloud, frame, sphere])
                if not cfgs.global_camera:
                    while robot.is_program_running():
                        robot.stopj(acc=10.0 * a)
                    robot.movel(robot.ready_pose(), acc=a*10,
                                vel=v*10)  # this v and a are anguler, so it should be larger than translational
                else:
                    t1 = time.time()
                    depths = get_depth(existing_shm_depth)
                    depths_saved = copy.deepcopy(depths)
                    colors_saved = np.copy(np.ndarray((720, 1280, 3), dtype=np.float32, buffer=existing_shm_color.buf))
                    ggarray, cloud, points_down, grasp_features, sinput = get_ggarray_features(existing_shm_depth,
                                                                                       existing_shm_color, net)
                    t3 = time.time()
                    print(f'Net Time:{t3 - t1}')
                continue

            # sort
            index_score = np.argsort(Allegro_ggarray.scores)[::-1][0:10]
            two_fingers_ggarray_object_ids = two_fingers_ggarray_object_ids[index_score]
            Allegro_ggarray = Allegro_ggarray[index_score]
            two_fingers_ggarray = two_fingers_ggarray[index_score]
            grasp_features = grasp_features[index_score]

            idx = random.randint(0, len(index_score)-1)
            Allegro_grasp_used = Allegro_ggarray[idx]
            two_fingers_grasp_used = two_fingers_ggarray[idx]
            grasp_features_used = grasp_features[idx]


            print('picked by scores rotations, translations: ', Allegro_grasp_used.rotation_matrix,
                  Allegro_grasp_used.translation, two_fingers_grasp_used.translation, two_fingers_grasp_used.rotation_matrix)
            print('grasp score:', Allegro_grasp_used.score, two_fingers_grasp_used.score)
            print('grasp width:', Allegro_grasp_used.width, two_fingers_grasp_used.width)
            print('grasp depth:', Allegro_grasp_used.depth, two_fingers_grasp_used.depth)
            print('grasp type:', Allegro_grasp_used.grasp_type)
            print('grasp angle:', Allegro_grasp_used.angle)

            t4 = time.time()
            print(f'Collision Processing Time:{t4 - t3}')
            ####################################
            if DEBUG:
                Allegro_pose = Allegro_grasp_used.load_mesh(cfgs.Allegro_mesh_json_path, two_fingers_grasp_used)
                frame = o3d.geometry.TriangleMesh.create_coordinate_frame(0.1)
                sphere = o3d.geometry.TriangleMesh.create_sphere(0.002, 20).translate([0, 0, 0.490])
                Allegro_pose.paint_uniform_color([1, 0, 0])
                meshes_pointclouds = Allegro_grasp_used.load_mesh_pointclouds(cfgs.Allegro_mesh_json_path, two_fingers_grasp_used, voxel_size=0.002)
                voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(input=meshes_pointclouds,
                                                                            voxel_size=0.002)
                scene_cloud = o3d.geometry.PointCloud()
                scene_cloud.points = o3d.utility.Vector3dVector(points_down.cpu().numpy())

                # scene_cloud.points = o3d.utility.Vector3dVector(np.vstack((points_down.cpu().numpy(), table_pointcloud)))
                scene_cloud = scene_cloud.voxel_down_sample(0.001)
                ps = scene_cloud.points
                ps = o3d.utility.Vector3dVector(ps)
                output = voxel_grid.check_if_included(ps)
                print('\n\n\nnp.array(output).astype(int).sum(): ', np.array(output).astype(int).sum(), '\n\n\n')

                o3d.visualization.draw_geometries(
                    [Allegro_pose, cloud, sphere, frame, two_fingers_grasp_used.to_open3d_geometry()])

            gripper_time = 0.8
            print('angle: ', Allegro_grasp_used.angle)
            if Allegro_grasp_used.grasp_type == 8:
                allegro_ready_pose = np.array([[0, 1.4, 1.4, 1.4], [0, 1.4, 1.4, 1.4], [0, 1.4, 1.4, 1.4], [0.5, 0, 0.2, 0]]).reshape(16)
                robot.open_gripper(allegro_ready_pose, sleep_time = gripper_time)

            robot.open_gripper(Allegro_grasp_used.angle, sleep_time = gripper_time)
            print(Allegro_grasp_used.rotation_matrix, Allegro_grasp_used.translation)
            mat_pose = robot.grasp_and_throw(Allegro_grasp_used, two_fingers_grasp_used, cloud, cfgs.Allegro_mesh_json_path,
                                             acc=a*2, vel=v*3, approach_dist=approach_distance,
                                             execute_grasp=True, use_ready_pose=True, gripper_time=gripper_time)

            while robot.is_program_running():
                pass

            t45 = time.time()

            save_grasp_information(two_fingers_ggarray, two_fingers_ggarray_object_ids, Allegro_ggarray, 
                            two_fingers_ggarray_source, Allegro_ggarray_source, two_fingers_ggarray_object_ids_source,
                            two_fingers_grasp_used, Allegro_grasp_used, grasp_features_used, 
                            mat_pose, colors_saved, depths_saved, grasp_features, two_fingers_source_grasp_features,
                            before_collision, after_collision, Allegro_grasp_used.grasp_type)
            if cfgs.global_camera:
                t1 = time.time()
                depths = get_depth(existing_shm_depth)
                depths_saved = copy.deepcopy(depths)
                colors_saved = np.copy(np.ndarray((720, 1280, 3), dtype=np.float32, buffer=existing_shm_color.buf))
                ggarray, cloud, points_down, grasp_features = get_grasp(net, depths, existing_shm_color)
                t3 = time.time()
                print(f'Net Time:{t3 - t1}')

            t5 = time.time()
            print(f'ready time:{t5 - t45}')
            print(f'Exec Time:{t5 - t4}')
            mpph = 3600 / (t5 - t1)
            print(f'\033[1;31mMPPH:{mpph}\033[0m\n--------------------')
    finally:
        robot.close()
        # print('save')
        existing_shm_depth.close()
        if DEBUG:
            existing_shm_color.close()


if __name__ == '__main__':

    t0 = time.time()
    try:
        robot_grasp(cfgs)
    finally:
        tn = time.time()
        print(f'total time:{tn - t0}')