import pdb

from ur_toolbox.camera import RealSense
from multiprocessing import shared_memory
import numpy as np
import cv2
import os

DEBUG = True
save_path = "capture_b06"

# function to display the coordinates of
# of the points clicked on the image 
def click_event(event, x, y, flags, params):
    # checking for left mouse clicks

    fx, fy = 910.673, 908.948
    cx, cy = 655.339, 371.053
    s = 1000.0

    xmap, ymap = np.arange(depths.shape[1]), np.arange(depths.shape[0])
    xmap, ymap = np.meshgrid(xmap, ymap)

    points_z = depths / s
    points_x = (xmap - cx) / fx * points_z
    points_y = (ymap - cy) / fy * points_z
    points = np.stack([points_x, points_y, points_z], axis=-1)
    # displaying the coordinates
    # on the image window
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(colors, str(int(points[y][x][0]*1000)) + ',' +
                str(int(points[y][x][1]*1000)) + ',' +
                str(int(points[y][x][2]*1000)), (x,y), font,
                1, (255, 0, 0), 2)
    cv2.imshow('image', colors)




camera = RealSense(serial = '035622060973', frame_rate = 30)
if DEBUG:
    for i in range(10):
        colors, depths = camera.get_rgbd_image()
else:
    for i in range(10):
        depths = camera.get_depth_image()
        

shm_depth = shared_memory.SharedMemory(name='realsense_depth', create=True, size=depths.nbytes)
depthbuf = np.ndarray(depths.shape, dtype=depths.dtype, buffer=shm_depth.buf)

if DEBUG:
    colors = (cv2.cvtColor(colors, cv2.COLOR_BGR2RGB) / 255.0).astype(np.float32)
    shm_color = shared_memory.SharedMemory(name='realsense_color', create=True, size=colors.nbytes)
    colorbuf = np.ndarray(colors.shape, dtype=colors.dtype, buffer=shm_color.buf)

try:
    cnt = 0
    # if not os.path.exists(save_path):
    #     os.mkdir(save_path)

    while True:
        colors = None
        
        if DEBUG:
            colors, depths = camera.get_rgbd_image()
            colors = (cv2.cvtColor(colors, cv2.COLOR_BGR2RGB) / 255.0).astype(np.float32)
            # cv2.imshow("image", colors)
            # cv2.setMouseCallback('image', click_event)
            # cv2.waitKey(1)
            # cv2.imwrite(os.path.join(save_path, 'color_%03d.png'%cnt), colors)
            # np.save(os.path.join(save_path, 'depth_%03d.npy'%cnt), depths)
            # print(depths.dtype)

            cnt += 1
            
            colorbuf[:] = colors[:]
            depthbuf[:] = depths[:]
        else:
            print(shm_depth.name)
            depths = camera.get_depth_image()

            depthbuf[:] = depths[:]
except KeyboardInterrupt:
    shm_depth.unlink()
    shm_color.unlink()