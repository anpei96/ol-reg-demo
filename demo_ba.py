from __future__ import print_function
import urllib
import urllib.request
import bz2
import os
import numpy as np
np.set_printoptions(suppress=True)
 
 
# import pcl
# import pcl.pcl_visualization
import random
# def vis_pair(cloud1, cloud2, rdm=False):
#     color1 = [255, 0, 0]
#     color2 = [0, 255, 0]
#     if rdm:
#         color1 = [255, 0, 0]
#         color2 = [random.randint(0, 255) for _ in range(3)]
#     visualcolor1 = pcl.pcl_visualization.PointCloudColorHandleringCustom(cloud1, color1[0], color1[1], color1[2])
#     visualcolor2 = pcl.pcl_visualization.PointCloudColorHandleringCustom(cloud2, color2[0], color2[1], color2[2])
#     vs = pcl.pcl_visualization.PCLVisualizering
#     vss1 = pcl.pcl_visualization.PCLVisualizering()  # 初始化一个对象，这里是很重要的一步
#     vs.AddPointCloud_ColorHandler(vss1, cloud1, visualcolor1, id=b'cloud', viewport=0)
#     vs.AddPointCloud_ColorHandler(vss1, cloud2, visualcolor2, id=b'cloud1', viewport=0)
#     vs.SetBackgroundColor(vss1, 0, 0, 0)
#     #vs.InitCameraParameters(vss1)
#     #vs.SetFullScreen(vss1, True)
#     # v = True
#     while not vs.WasStopped(vss1):
#         vs.Spin(vss1)
 
BASE_URL = "http://grail.cs.washington.edu/projects/bal/data/ladybug/"
FILE_NAME = "problem-49-7776-pre.txt.bz2"
URL = BASE_URL + FILE_NAME
 
if not os.path.isfile(FILE_NAME):
    urllib.request.urlretrieve(URL, FILE_NAME)
 
def read_bal_data(file_name):
    with bz2.open(file_name, "rt") as file:
        n_cameras, n_points, n_observations = map(
            int, file.readline().split())
        print("相机pose数目(image数目) n_cameras: {}".format(n_cameras))
        print("重构出的3D点数目 n_points: {}".format(n_points))
        print("所有图像中2D特征点数目 n_observations: {}".format(n_observations))
 
        camera_indices = np.empty(n_observations, dtype=int)
        point_indices = np.empty(n_observations, dtype=int)
        points_2d = np.empty((n_observations, 2))
 
        # 读取每个特征点xy,及其对应的相机索引，重构的3D点索引
        for i in range(n_observations):
            camera_index, point_index, x, y = file.readline().split()
            camera_indices[i] = int(camera_index)
            point_indices[i] = int(point_index)
            points_2d[i] = [float(x), float(y)]
 
        # 读取每个相机的内参 0,1,2是R的旋转向量 3,4,5是平移向量
        # 6是焦距，7,8是畸变系数k1k2
        camera_params = np.empty(n_cameras * 9)
        for i in range(n_cameras * 9):
            camera_params[i] = float(file.readline())
        camera_params = camera_params.reshape((n_cameras, -1))
 
        # 读取所有重构出的3D点，他们的list索引就是自身的索引
        points_3d = np.empty(n_points * 3)
        for i in range(n_points * 3):
            points_3d[i] = float(file.readline())
        points_3d = points_3d.reshape((n_points, -1))
 
    return camera_params, points_3d, camera_indices, point_indices, points_2d
 
camera_params, points_3d, camera_indices, point_indices, points_2d = read_bal_data(FILE_NAME)
 
n_cameras = camera_params.shape[0]
n_points = points_3d.shape[0]
 
n = 9 * n_cameras + 3 * n_points
m = 2 * points_2d.shape[0]
 
print("n_cameras: {}".format(n_cameras))
print("n_points: {}".format(n_points))
print("Total number of parameters: {}".format(n))
print("Total number of residuals: {}".format(m))
 
# points_3d_pcl_ori = pcl.PointCloud(points_3d.astype(np.float32))
# vis_pair(points_3d_pcl_ori, points_3d_pcl_ori)
 
 
def rotate(points, rot_vecs):
    """Rotate points by given rotation vectors.
    Rodrigues' rotation formula is used.
    """
    # 参考: https://blog.csdn.net/qq_42658249/article/details/114494198
    #      https://zhuanlan.zhihu.com/p/113299607
    #      https://zhuanlan.zhihu.com/p/298128519
    # 旋转向量转换为旋转矩阵
 
    # 二范数
    theta = np.linalg.norm(rot_vecs, axis=1)[:, np.newaxis]
    with np.errstate(invalid='ignore'):
        v = rot_vecs / theta
        v = np.nan_to_num(v)
    dot = np.sum(points * v, axis=1)[:, np.newaxis]
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
 
    return cos_theta * points + sin_theta * np.cross(v, points) + dot * (1 - cos_theta) * v
 
def project(points, camera_params):
    """Convert 3-D points to 2-D by projecting onto images."""
    #参考：https://www.comp.nus.edu.sg/~cs4243/lecture/camera.pdf
    #     https://blog.csdn.net/waeceo/article/details/50580607
    #     https://blog.csdn.net/qq_42615787/article/details/102485890
    # x =K[RX+T]
    # 若存在畸变
    # x_d = x(1 + k1*r^2 + k1*r^4 + k3*r^6)
    # y_d = y(1 + k1*r^2 + k1*r^4 + k3*r^6)
    # 其中r^2 = x^2 + y^2
    # x_d y_d是畸变坐标系中的坐标  注:我们的特征点处于畸变坐标系中,因为拍摄出来的图像都是畸变图像
    # x,y 是矫畸坐标系中的坐标
    points_proj = rotate(points, camera_params[:, :3])
    points_proj += camera_params[:, 3:6]
    # 齐次坐标系转换到欧式坐标系
    points_proj = -points_proj[:, :2] / points_proj[:, 2, np.newaxis]
    f = camera_params[:, 6]
    k1 = camera_params[:, 7]
    k2 = camera_params[:, 8]
    n = np.sum(points_proj**2, axis=1)
    r = 1 + k1 * n + k2 * n**2
    # 原本K= [fx, -fx * cot(@), cx]
    #       [0,  fy / sin(@), cx]
    #       [0,  0, 1]
    # 简化后
    #    K= [fx, s, cx]
    #       [0, fy, cy]
    #       [0,  0, 1]
    # 现代工艺成熟忽略后
    #    K= [fx, 0, cx]
    #       [0, fy, cy]
    #       [0,  0, 1]
    # 其中：
    #    K= [fx=f/dx,     0,      cx]
    #       [0,        fy=f/dy,   cy]
    #       [0,  0, 1]
    # dxdy表示感光芯片上像素的实际大小，用于连接像素坐标系和真实尺寸坐标系
    # 表示每像素对应真实世界的mm
    # 继续简化 令f^ = fx(f/dx) = fy(f/dy)
    #    K= [f^, 0, cx]
    #       [0, f^, cy]
    #       [0, 0, 1]
    # 我们这里的坐标都是以光心为原点,不再是左上角
    # 继续简化
    #    K= [f^, 0, 0]
    #       [0, f^, 0]
    #       [0, 0, 1]
    points_proj *= (r * f)[:, np.newaxis]
    return points_proj
 
 
def fun(params, n_cameras, n_points, camera_indices, point_indices, points_2d):
    """Compute residuals.
    `params` contains camera parameters and 3-D coordinates.
    """
    camera_params = params[:n_cameras * 9].reshape((n_cameras, 9))
    points_3d = params[n_cameras * 9:].reshape((n_points, 3))
    points_proj = project(points_3d[point_indices], camera_params[camera_indices])
    return (points_proj - points_2d).ravel()
 
from scipy.sparse import lil_matrix
 
def bundle_adjustment_sparsity(n_cameras, n_points, camera_indices, point_indices):
    # 计算雅可比矩阵比较麻烦，我们进行有限差分近似
    # 计算雅可比矩阵比较麻烦，我们进行有限差分近似
    # 构造雅可比稀疏结构(i. e. mark elements which are known to be non-zero)
    # 标记已知的非0元素
    m = camera_indices.size * 2
    n = n_cameras * 9 + n_points * 3
    A = lil_matrix((m, n), dtype=int)
 
    i = np.arange(camera_indices.size)
    for s in range(9):
        A[2 * i, camera_indices * 9 + s] = 1
        A[2 * i + 1, camera_indices * 9 + s] = 1
 
    for s in range(3):
        A[2 * i, n_cameras * 9 + point_indices * 3 + s] = 1
        A[2 * i + 1, n_cameras * 9 + point_indices * 3 + s] = 1
 
    return A
 
import matplotlib.pyplot as plt
 
# 相机pose数目(image数目) n_cameras: 49
# 重构出的3D点数目 n_points: 7776
# 所有图像中2D特征点数目 n_observations: 31843
# points_2d(2D点[x,y]存储-) :31843*2
 
# camera_indices(该2D点[x,y]对应的相机索引) :31843
# point_indices(该2D点[x,y]对应的重构3D点索引) :31843
# points_3d(构出的3D点,list索引就是自身的索引) :7776
# camera_params(相机参数) :49*9
 
# ravel多维数组转换为一维数组
# 23769=49*9 + 7776*3
x0 = np.hstack((camera_params.ravel(), points_3d.ravel()))
f0 = fun(x0, n_cameras, n_points, camera_indices, point_indices, points_2d)
plt.plot(f0)
plt.show()
 
# 63686(31843*2) * 23769(49*9 + 7776*3)
A = bundle_adjustment_sparsity(n_cameras, n_points, camera_indices, point_indices)
import time
from scipy.optimize import least_squares
t0 = time.time()
res = least_squares(fun, x0, jac_sparsity=A, verbose=2, x_scale='jac', ftol=1e-4, method='trf',
                    args=(n_cameras, n_points, camera_indices, point_indices, points_2d))
'''
Setting scaling='jac' was done to automatically scale the variables and equalize 
their influence on the cost function (clearly the camera parameters and coordinates 
of the points are very different entities). This option turned out to be crucial 
for successfull bundle adjustment.
相机参数和三维点坐标来源不同实体
scaling='jac' 自动缩放变量，均衡他们对cost损失的影响
'''
 
t1 = time.time()
print("Optimization took {0:.0f} seconds".format(t1 - t0))
plt.plot(res.fun)
plt.show()
 
'''
We see much better picture of residuals now, with the mean being very close to zero. 
There are some spikes left. It can be explained by outliers in the data, or, 
possibly, the algorithm found a local minimum (very good one though) or didn't converged enough. 
Note that the algorithm worked with Jacobian finite difference aproximate, 
which can potentially block the progress near the minimum because of insufficient accuracy 
(but again, computing exact Jacobian for this problem is quite difficult).
大部分均值已经接近0,剩下的一些峰值可用异常值来解释，or收敛不够，or找到了局部最优
因为使用了雅可比矩阵的有限差分 可能会由于精度不足处在最小值附近徘徊
'''
 
# 原始点云和BA优化后点云
# new_camera_params = res.x[:n_cameras * 9].reshape((n_cameras, 9))
# new_points_3d = res.x[n_cameras * 9:].reshape((n_points, 3))
# points_3d_pcl_target = pcl.PointCloud(new_points_3d.astype(np.float32))
# vis_pair(points_3d_pcl_target, points_3d_pcl_target)
# vis_pair(points_3d_pcl_ori, points_3d_pcl_target)
