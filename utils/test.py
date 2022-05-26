from ctypes import sizeof
from stringprep import b1_set
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import utils

# 点云读取
pointcloud = np.fromfile(str("/data/KITTI_Data/temprory_files_by_guanfeiyu/data_odometry_velodyne/dataset/sequences/00/velodyne/000010.bin"), dtype=np.float32, count=-1).reshape([-1, 4])

a,b = utils.BEV_projection(pointcloud)
print(a.shape)
print(b.shape)
plt.imshow(a)
