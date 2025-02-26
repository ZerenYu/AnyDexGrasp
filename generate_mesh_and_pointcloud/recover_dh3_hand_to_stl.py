import copy
import os
import json
import xlrd2
import numpy as np
import open3d as o3
from transforms3d.euler import euler2mat, quat2mat
import pybullet as p
import pybullet_data
import math
from tqdm import tqdm
import sys
import time
from angles2stl import DH3Angles2STLs
from ur_toolbox.robot.DH3.DH3_grasp import grasp_types

joint_mesh_mapping = {"base": "base_Link.STL",
                      "0": "finger1_base_Link.STL",
                      "1": "finger1_Link.STL",
                      "2": "finger1_outlet_Link.STL",
                      "3": "finger1_tip_Link.STL",
                      "4": "finger2_base_Link.STL",
                      "5": "finger2_Link.STL",
                      "6": "finger2_outlet_Link.STL",
                      "7": "finger2_tip_Link.STL",
                      "8": "finger3_base_Link.STL",
                      "9": "finger3_Link.STL",
                      "10": "finger3_outlet_Link.STL",
                      "11": "finger3_tip_Link.STL"}
# grasp_types = {'1':{'name': 'pose1', 'facenet_thumb': [60388], 'facenet_index': [69638, 51138]},
#                 '2':{'name': 'pose2', 'facenet_thumb': [60388], 'facenet_index': [69638, 51138]},
#                 '3':{'name': 'pose3', 'facenet_thumb': [60388], 'facenet_index': [69638, 51138]},
#                 '4':{'name': 'pose4', 'facenet_thumb': [60388], 'facenet_index': [69641, 51169]},
#                 }

def vis_pybullet(urdf_path):
    clid = p.connect(p.SHARED_MEMORY)
    if (clid < 0):
        p.connect(p.GUI)
        p.connect(p.SHARED_MEMORY_GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    fps = 240
    timeStep = 1./fps
    i=0
    p.setPhysicsEngineParameter(solverResidualThreshold=0, maxNumCmdPer1ms=1000)
    p.setTimeStep(timeStep)
    # p.setGravity(0,0,-9.8	)
    p.resetDebugVisualizerCamera(cameraDistance=0.7, cameraYaw=2, cameraPitch=2, cameraTargetPosition=[-0.35,-0.0,-0.0])

    flags = p.URDF_ENABLE_CACHED_GRAPHICS_SHAPES

    base_orn = [0,0,0]
    base_orn = p.getQuaternionFromEuler(base_orn)

    hand = p.loadURDF(urdf_path, [0.0,0.0,0.0], base_orn, useFixedBase=True)
    limit = []
    for i in range(p.getNumJoints(hand)):
        info = p.getJointInfo(hand, i)
        # print(info)
        limit.append([info[8], info[9]])
    print(limit)
    finger1_1 = p.addUserDebugParameter("finger1-1", -1.0, 0.54, 0.01)
    finger1_2 = p.addUserDebugParameter("finger1-2", 0, 1.16, 0.01)
    finger1_3 = p.addUserDebugParameter("finger1-3", -1, 1, 0.01)
    finger1_4 = p.addUserDebugParameter("finger1-4", 0, 0, 0.01)

    finger2_1 = p.addUserDebugParameter("f inger2-1", -1.0, 0.54, 0.01)
    finger2_2 = p.addUserDebugParameter("finger2-2", 0, 1.16, 0.01)
    finger2_3 = p.addUserDebugParameter("finger2-3", -1, 1, 0.01)
    finger2_4 = p.addUserDebugParameter("finger2-4", 0, 0, 0.01)

    finger3_1 = p.addUserDebugParameter("finger3-1", -1.0, 0.54, 0.01)
    finger3_2 = p.addUserDebugParameter("finger3-2", 0, 1.16, 0.01)
    finger3_3 = p.addUserDebugParameter("finger3-3", -1, 1, 0.01)
    finger3_4 = p.addUserDebugParameter("finger3-4", 0, 0, 0.01)
    # while True:
    p.resetJointState(hand, 0, p.readUserDebugParameter(finger1_1))
    p.resetJointState(hand, 1, p.readUserDebugParameter(finger1_2))
    p.resetJointState(hand, 2, p.readUserDebugParameter(finger1_3))
    p.resetJointState(hand, 3, p.readUserDebugParameter(finger1_4))

    p.resetJointState(hand, 4, p.readUserDebugParameter(finger2_1))
    p.resetJointState(hand, 5, p.readUserDebugParameter(finger2_2))
    p.resetJointState(hand, 6, p.readUserDebugParameter(finger2_3))
    p.resetJointState(hand, 7, p.readUserDebugParameter(finger2_4))

    p.resetJointState(hand, 8, p.readUserDebugParameter(finger3_1))
    p.resetJointState(hand, 9, p.readUserDebugParameter(finger3_2))
    p.resetJointState(hand, 10, p.readUserDebugParameter(finger3_3))
    p.resetJointState(hand, 11, p.readUserDebugParameter(finger3_4))
    p.stepSimulation()
    time.sleep(10000)

def four_meta_to_matrix(x):
    xyz = [x[0], x[1], x[2]]
    rpy = [x[6], x[3], x[4], x[5]]

    rot_mat = quat2mat(rpy)
    return np.array([[rot_mat[0][0], rot_mat[0][1], rot_mat[0][2], xyz[0]],
                     [rot_mat[1][0], rot_mat[1][1], rot_mat[1][2], xyz[1]],
                     [rot_mat[2][0], rot_mat[2][1], rot_mat[2][2], xyz[2]],
                     [0, 0, 0, 1]], dtype=np.float32)

def rate(rate1, rate2, num1):
    return (num1-rate1[0])/(rate1[1]-rate1[0])*(rate2[1]-rate2[0])+rate2[0]

def read_excel_2D_angle_to_12D_angle(path_2d):
    angles = [[] for _ in range(len(grasp_types))]
    pos_scope = [(-1.08, 0.43), (1.333, 0.), (0.06, 0.01), (0.1, 0),
                (1, 1),          (1.333, 0.), (0.06, 0.01), (0.1, 0),
                (1.08, -0.43), (1.333, 0.), (0.06, 0.01), (0.1, 0)] 
    wb_6d = xlrd2.open_workbook(path_2d) 
    for pose_id in range(len(wb_6d.sheets())):
        # if pose_id != 0:
        #     continue
        sheet_6d = wb_6d.sheet_by_index(pose_id)

        width = sheet_6d.col_values(0)[1:]
        width = np.array(width) * 0.1

        positions = sheet_6d.col_values(1)[1:]
        rotations = sheet_6d.col_values(2)[1:]
        for ids, _ in enumerate(positions):
            # if ids < 60:
            #     continue
            if positions[ids] == -1:
                continue
            rotation0 = rate((0, 95), pos_scope[0], rotations[ids])
            position1 = rate((0, 95), pos_scope[1], positions[ids])
            position2 = rate((0, 95), pos_scope[2], positions[ids])
            position3 = rate((0, 95), pos_scope[3], positions[ids])

            rotation4 = rate((0, 95), pos_scope[4], rotations[ids])
            position5 = rate((0, 95), pos_scope[5], positions[ids])
            position6 = rate((0, 95), pos_scope[6], positions[ids])
            position7 = rate((0, 95), pos_scope[7], positions[ids])

            rotation8 = rate((0, 95), pos_scope[8], rotations[ids])
            position9 = rate((0, 95), pos_scope[9], positions[ids])
            position10 = rate((0, 95), pos_scope[10], positions[ids])
            position11 = rate((0, 95), pos_scope[11], positions[ids])

            angles[pose_id].append([rotation0, position1, position2, position3, rotation4, position5, position6, position7,
                            rotation8, position9, position10, position11,
                            width[ids],
                            positions[ids], rotations[ids]])
    return angles

def open_pybullet(urdf_path):
    clid = p.connect(p.SHARED_MEMORY)
    if (clid < 0):
        p.connect(p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setPhysicsEngineParameter(solverResidualThreshold=0, maxNumCmdPer1ms=1000)
    fps = 240
    timeStep = 1. / fps
    p.setTimeStep(timeStep)
    p.resetDebugVisualizerCamera(cameraDistance=1.3, cameraYaw=38, cameraPitch=-22,
                                 cameraTargetPosition=[0.35, -0.13, 0.5])

    flags = p.URDF_ENABLE_CACHED_GRAPHICS_SHAPES

    base_orn = [0, 0, 0]
    base_orn = p.getQuaternionFromEuler(base_orn)

    hand = p.loadURDF(urdf_path, [0.0, 0.0, 0.0], base_orn, useFixedBase=True)
    return p, hand

def get_mesh(po, id, PATH):
    t_angle = four_meta_to_matrix(po)
    
    path_file = PATH + joint_mesh_mapping[str(id)]
    link = o3.io.read_triangle_mesh(path_file)
    lingk_mesh = copy.deepcopy(link)

    link = link.transform(t_angle)
    # if id in [3, 7, 11]:
    #     t_angle[2,3] = t_angle[2,3] + 0.008
    link_mesh = lingk_mesh.transform(t_angle)    
    return link, link_mesh

def save_stl_and_pointcloud(name, angle, mesh, output_path):
    if not os.path.exists(output_path + 'source/' + name):
        os.makedirs(output_path + 'source/' + name)
    save_stl_path = output_path + 'source/' + name + '/' + str(np.round(angle[12], 1)) + '.STL'
    o3.io.write_triangle_mesh(save_stl_path, mesh)

    pcd = o3.geometry.TriangleMesh.sample_points_uniformly(mesh, 300000)
    pcd.estimate_normals(search_param=o3.geometry.KDTreeSearchParamHybrid(radius=0.01, max_nn=30))
    points = np.array(pcd.points)
    for i in range(1, 6):
        points = np.vstack((points, np.array(pcd.points) - np.array(pcd.normals) * i * 1.5 * 0.0005))
    for i in range(6, 16):
        points = np.vstack(
            (points, np.array(pcd.points)[:170000 * 1] - np.array(pcd.normals)[:170000 * 1] * i * 1.5 * 0.0005))
    pcd_extend = o3.geometry.PointCloud()
    pcd_extend.points = o3.utility.Vector3dVector(points)
    for i in range(1, 6):
        pcd_saved = pcd_extend.voxel_down_sample(0.001 * i)
        if not os.path.exists(output_path + 'source_pointclouds/voxel_size_' + str(i) + '/' + name):
            os.makedirs(output_path + 'source_pointclouds/voxel_size_' + str(i) + '/' + name)
        save_pcd_path = output_path + 'source_pointclouds/voxel_size_' + str(i) + '/' + name + '/' + str(np.round(angle[12], 1)) + '.ply'
        o3.io.write_point_cloud(save_pcd_path, pcd_saved)


def get_meshes(angles, stl_path, output_path, width_12D_angle_2D_angle_json, urdf_path, if_source, vis):
    p, hand = open_pybullet(urdf_path)
    # p.stepSimulatiograsp_n()
    angles_to_stls = DH3Angles2STLs(grasp_types)
    width_12D_angle_2D_angle = dict()
    for grasp_type, angle8 in enumerate(angles):
        for ids in tqdm([i for i in range(len(angle8))]):
            angle = angle8[ids]
            # print(angle)
            width = np.round(angle[12], 1)

            path_file = stl_path + joint_mesh_mapping['base']
            base = o3.io.read_triangle_mesh(path_file)
            base_mesh = copy.deepcopy(base)
            frame = o3.geometry.TriangleMesh.create_coordinate_frame(0.1)
            pose = [[] for _ in range(12)]
            for joint_id in range(12):
                p.resetJointState(hand, joint_id, angle[joint_id])
            
            for joint_id in range(12):
                po = p.getLinkState(hand, joint_id)
                pose[joint_id] = [po[4][0], po[4][1], po[4][2], po[5][0], po[5][1],po[5][2],po[5][3]]
            
            for joint_id in range(12):
                if joint_id % 4 == 2:
                    x, y, z = pose[joint_id][0], pose[joint_id][1], pose[joint_id][2]
                    pose[joint_id][0], pose[joint_id][1], pose[joint_id][2] = pose[joint_id+1][0], pose[joint_id+1][1], pose[joint_id+1][2]
                    pose[joint_id+1][0], pose[joint_id+1][1], pose[joint_id+1][2] = x, y, z
                
                mesh, link_mesh = get_mesh(pose[joint_id], joint_id, stl_path)
                base = base + mesh
                base_mesh = base_mesh + link_mesh
            base.compute_triangle_normals()
            base_mesh.compute_triangle_normals()
            name = grasp_types[str(grasp_type+1)]['name']
            save_stl_and_pointcloud(name, angle, base_mesh, output_path)
            if if_source:
                translation, rotation = angles_to_stls.get_pose_information(base, grasp_type, vis=vis)
                translation, rotation = translation.tolist(), rotation.tolist()
                if str(name) not in width_12D_angle_2D_angle.keys():
                    width_12D_angle_2D_angle[name] = dict()
                width_12D_angle_2D_angle[name][str(width)] = {'12d': angles[grasp_type][ids][:12], '2d': angles[grasp_type][ids][13:],
                                                        'translation': translation,  'rotation': rotation}
                # print(translation, rotation)

    if if_source:
        json_str = json.dumps(width_12D_angle_2D_angle, indent=4)
        with open(width_12D_angle_2D_angle_json, 'w') as json_file:
            json_file.write(json_str)

def modify_width(output_path, json_path, pose, distance=0.5):
    for po in pose:
        mesh_path = os.path.join(output_path, 'source', po)
        pointcloud_path = os.path.join(output_path, 'source_pointclouds')
        file_path = os.listdir(mesh_path)
        file_path.sort(key=lambda x:float(x[:-4]))
        total_file = len(file_path)
        with open(json_path, 'r') as f:
            information = json.load(f)
        for name in file_path:
            width = round(float(name.split('.STL')[0]) - distance, 1)
            # print(width, name)
            old_mesh_name = os.path.join(mesh_path, name)
            new_mesh_name = os.path.join(mesh_path, str(width) + '.STL')
            
            if width < 0:
                os.remove(old_mesh_name)
                for voxel_size in os.listdir(pointcloud_path):
                    pc_path = os.path.join(pointcloud_path, voxel_size, po)
                    old_pc_name = os.path.join(pc_path, name.split('.STL')[0] + '.ply')
                    os.remove(old_pc_name)
            else:
                os.rename(old_mesh_name, new_mesh_name)
                information[po][str(width)] = information[po][str(round(float(name.split('.STL')[0]), 1))]
                if float(name.split('.STL')[0]) * 10 > total_file-distance*10-0.1:
                    del(information[po][str(round(float(name.split('.STL')[0]), 1))])
                for voxel_size in os.listdir(pointcloud_path):
                    pc_path = os.path.join(pointcloud_path, voxel_size, po)
                    old_pc_name = os.path.join(pc_path, name.split('.STL')[0] + '.ply')
                    new_pc_name = os.path.join(pc_path, str(width) + '.ply')
                    os.rename(old_pc_name, new_pc_name)
    json_str = json.dumps(information, indent=4)
    with open(json_path, 'w') as json_file:
        json_file.write(json_str)
            


if __name__ == '__main__':
    path_6d = './generate_mesh_and_pointcloud/dh3_urdf/dh3_routine_to_angle-use.xlsx'

    source_stl_path = './generate_mesh_and_pointcloud/dh3_urdf/dh3_urdf/meshes/'
    output_path = './generate_mesh_and_pointcloud/dh3_urdf/meshes/'
    json_path = './generate_mesh_and_pointcloud/dh3_urdf/width_12D_angle_2D_angle.json'

    urdf_path = './generate_mesh_and_pointcloud/dh3_urdf/dh3_urdf/urdf/dh3_urdf.urdf'
    # vis_pybullet(urdf_path)

    angles = read_excel_2D_angle_to_12D_angle(path_6d)
    get_meshes(angles, source_stl_path, output_path, json_path, urdf_path, if_source=True, vis=False)
    modify_width(output_path, json_path, ['pose1', 'pose3'])




