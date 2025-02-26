import numpy as np
import time
import sys

from .src.allegro_hand.src.allegro_hand.controller import AllegroController

# cd /home/ubuntu/data/hengxu/git/universal-grasp/ur_toolbox/ur_toolbox/robot/Allegro/drivers/peak-linux-driver-8.12.0
# make clean
# make NET=NO_NETDEV_SUPPORT
# sudo make install
# sudo modprobe pcan
# cd ../PCAN-Basic_Linux-4.6.2.36/libpcanbasic
# make
# sudo make install
# cat /proc/pcan
# ls -l /dev/pcan*
# sudo apt-get install ros-noetic-libpcan 
# roslaunch allegro_hand allegro_hand.launch VISUALIZE:=true
class Allegro(object):
    def __init__(self):
        self.allegro_controller = AllegroController()
        self.allegro_ready_pose = np.array([[0, 1.4, 1.4, 1.4], [0, 1.4, 1.4, 1.4], [0, 1.4, 1.4, 1.4], [1.496, 0, 0.2, 0]]).reshape(16)
        time.sleep(0.5)
        # self.set_ready_pose()
        # time.sleep(1)
        # self.set_ready_pose()

    def set_torque(self, torque = np.ones(16) * 0.1):
        self.allegro_controller.apply_joint_torque(torque)

    def set_ready_pose(self, ):
        self.open_gripper(self.allegro_ready_pose, True)

    def set_pose(self, pose):
        self.allegro_controller.hand_pose(pose, True)

    def open_gripper(self, pose=np.zeros(16), sleep_time=0.2):
        current_pose = np.array(self.allegro_controller.current_joint_pose.position)
        error_pose = (current_pose - pose) / 20
        for i in range(20):
            current_pose = current_pose - error_pose
            self.set_pose(current_pose.reshape(16))
            time.sleep(0.1)
        # self.set_pose(pose.reshape(16))
        time.sleep(sleep_time)

    def get_current_angle(self):
        return self.allegro_controller.current_joint_pose.position

    def get_current_effort(self):
        return self.allegro_controller.current_joint_pose.effort

    
    def get_angle_contact(self, pre_angle, current_angle):
        pre_angle = np.array(pre_angle).reshape((4, 4))
        current_angle = np.array(current_angle).reshape((4, 4))
        contact = np.array([0,0,0,0])
        for i in range(4):
            if sum(current_angle[i][1:3] - pre_angle[i][1:3]) > 0.08:
                contact[i] = contact[i] + 1
        return contact


    def close_gripper(self, close_pose, multifinger_grasp_used=None, tactile=None, angle=0.3, sleep_time=0.2, step=10):
        # print("Close gripper")        
        close_pose = np.array(close_pose).reshape(16)
        all_close_poses = []
        all_close_effort = []
        target_angle = []
        current_pose = np.array(self.allegro_controller.current_joint_pose.position)
        all_close_poses.append(list(current_pose))
        all_close_effort.append(self.allegro_controller.current_joint_pose.effort)
        target_angle.append(list(current_pose))

        when_contact = np.array([0, 0, 0, 0])
        pre_angle = self.get_current_angle()
        error_pose = (current_pose - close_pose) / step
        for i in range(step):
            current_pose = current_pose - error_pose
            self.set_pose(current_pose.reshape(16))

            target_angle.append(list(current_pose))
            all_close_poses.append(self.allegro_controller.current_joint_pose.position)
            all_close_effort.append(self.allegro_controller.current_joint_pose.effort)
            time.sleep(0.1)
        target_angle.append(list(close_pose))
        time.sleep(sleep_time)
        

if __name__ == '__main__':
    import xlrd2

    allegro = Allegro()
    time.sleep(2)
    # allegro.set_ready_pose()
    # allegro.set_torque()
    path = '/home/ubuntu/data/hengxu/git/MG/minkowski-grasp_inspire_inference/generate_mesh_and_pointcloud/allegro_urdf/meshes/allegro_width_to_angle.xls'
    wb_6d = xlrd2.open_workbook(path) 
    pose_id = 12
    
    sheet_6d = wb_6d.sheet_by_index(pose_id)

    widths = sheet_6d.col_values(0)[1:]
    widths = np.array(widths) * 0.1

    thumb0 = sheet_6d.col_values(1)[1:]
    thumb1 = sheet_6d.col_values(2)[1:]
    thumb2 = sheet_6d.col_values(3)[1:]
    thumb3 = sheet_6d.col_values(4)[1:]

    index0 = sheet_6d.col_values(5)[1:]
    index1 = sheet_6d.col_values(6)[1:]
    index2 = sheet_6d.col_values(7)[1:]
    index3 = sheet_6d.col_values(8)[1:]

    middle0 = sheet_6d.col_values(9)[1:] 
    middle1 = sheet_6d.col_values(10)[1:]
    middle2 = sheet_6d.col_values(11)[1:]
    middle3 = sheet_6d.col_values(12)[1:]

    ring0 = sheet_6d.col_values(13)[1:]
    ring1 = sheet_6d.col_values(14)[1:]
    ring2 = sheet_6d.col_values(15)[1:]
    ring3 = sheet_6d.col_values(16)[1:]
    ids = 40
    pose = np.array([index0[ids], index1[ids], index2[ids], index3[ids], middle0[ids], middle1[ids], middle2[ids], middle3[ids],
                            ring0[ids], ring1[ids], ring2[ids], ring3[ids],  thumb0[ids], thumb1[ids], thumb2[ids], thumb3[ids]]).reshape(16)
    print(pose)
    
    allegro.open_gripper(pose)
    # time.sleep(2)
    # current = allegro.get_current_angle()
    # time.sleep(2)
    # print(current)
    # allegro.open_gripper(current)
    # ids = 70
    # pose = np.array([index0[ids], index1[ids], index2[ids], index3[ids], middle0[ids], middle1[ids], middle2[ids], middle3[ids],
    #                         ring0[ids], ring1[ids], ring2[ids], ring3[ids],  thumb0[ids], thumb1[ids], thumb2[ids], thumb3[ids]]).reshape(16)
    # allegro.open_gripper(pose)
    # print(pose)