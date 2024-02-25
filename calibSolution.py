#
# Project: lidar-camera system calibration based on
#          object-level 3d-2d correspondence
# Author:  anpei
# Data:    2023.03.07
# Email:   anpei@wit.edu.cn
#

import os
import math
import torch
import cv2 as cv
import numpy as np
import open3d as o3d

from utilsCalib import projection
from utils3dVisual import show_pcd
from utilsOpt import bundle_adjustment, point_to_line_solver

def change_order_array(x):
    '''
        x   is [8,2] 4+4
        x_c is [8,2]
    '''
    x_c = x.copy()
    for i in range(2):
        # x_c[0+4*i,:] = x[1+4*i,:]
        # x_c[1+4*i,:] = x[2+4*i,:]
        # x_c[2+4*i,:] = x[3+4*i,:]
        # x_c[3+4*i,:] = x[0+4*i,:]

        # x_c[0+4*i,:] = x[2+4*i,:]
        # x_c[1+4*i,:] = x[3+4*i,:]
        # x_c[2+4*i,:] = x[0+4*i,:]
        # x_c[3+4*i,:] = x[1+4*i,:]

        x_c[0+4*i,:] = x[3+4*i,:]
        x_c[1+4*i,:] = x[0+4*i,:]
        x_c[2+4*i,:] = x[1+4*i,:]
        x_c[3+4*i,:] = x[2+4*i,:]

    return x_c

def rotationMatrixToEulerAngles(R) :
    sy = math.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])
    singular = sy < 1e-6
 
    if  not singular :
        x = math.atan2(R[2,1] , R[2,2])
        y = math.atan2(-R[2,0], sy)
        z = math.atan2(R[1,0], R[0,0])
    else :
        x = math.atan2(-R[1,2], R[1,1])
        y = math.atan2(-R[2,0], sy)
        z = 0

    return np.array([x, y, z])

def eva_err_rmat(rmat_pd, rmat_gt):
    error_rmat = np.matmul(rmat_pd.T, rmat_gt)
    error_vec  = rotationMatrixToEulerAngles(error_rmat)
    error_vec *= 180/np.pi
    return error_vec

def eva_err_tvec(tvec_pd, tvec_gt):
    error_vec = np.abs(tvec_pd - tvec_gt)[:,0].T
    # err = np.linalg.norm(tvec_pd - tvec_gt)
    return error_vec

def bb_intersection_over_union(boxA, boxB):
    '''
        box type [xmin,ymin,xmax,ymax]
        https://zhuanlan.zhihu.com/p/51680715
    '''
    boxA = [int(x) for x in boxA]
    boxB = [int(x) for x in boxB]

    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou

# implememtation of object-level 3d-2d registration
class object_3d_2d_match:
    '''
        3d-2d objects matching from rgb and lidar point cloud
    '''
    def __init__(self, pts, rgb, kmat):
        self.pts  = pts
        self.rgb  = rgb
        self.kmat = kmat
        self.iou_threshold = 0.50
    
    def load_3d_object_from_pts(self, obj_3d_pts):
        self.obj_3d_pts = obj_3d_pts.reshape((-1,8,3))

    def load_3d_object_from_rgb(self, obj_2d_pts):
        self.obj_2d_pts = obj_2d_pts.reshape((-1,8,2))

    def coarse_iou_compute(self, box_a, box_b):
        a_xmin, a_ymin = np.min(box_a[:,0]), np.min(box_a[:,1])
        a_xmax, a_ymax = np.max(box_a[:,0]), np.max(box_a[:,1])
        b_xmin, b_ymin = np.min(box_b[:,0]), np.min(box_b[:,1])
        b_xmax, b_ymax = np.max(box_b[:,0]), np.max(box_b[:,1])
        iou = bb_intersection_over_union(
            boxA=[a_xmin,a_ymin,a_xmax,a_ymax],
            boxB=[b_xmin,b_ymin,b_xmax,b_ymax])
        
        # print("box_a: ", box_a.shape)
        # print("box_b: ", box_b.shape)
        # print("iou: ", iou)
        return iou

    def match_evaluate(self, obj_3d_pts, obj_2d_pts, rmat, tvec):
        '''
            using iou to measure the matching results
        '''
        num_3d, num_2d = obj_3d_pts.shape[0], obj_2d_pts.shape[0]
        obj_3d_pts_    = obj_3d_pts.reshape((-1,3))
        can_2d_pts, _  = projection(obj_3d_pts_, self.kmat, rmat, tvec)
        
        can_2d_pts = can_2d_pts.reshape((-1,8,2))
        match_mat  = np.zeros((num_3d, num_2d)) # store iou value
        for id_3d in range(num_3d):
            box_a = can_2d_pts[id_3d]
            for id_2d in range(num_2d):
                box_b  = obj_2d_pts[id_2d]
                iou_ab = self.coarse_iou_compute(box_a, box_b)
                match_mat[id_3d, id_2d] = iou_ab
                # if iou_ab >= self.iou_threshold:
                #     match_mat[id_3d, id_2d] = iou_ab

        # for debug
        # print("can_2d_pts: ", can_2d_pts.shape)
        # print("obj_2d_pts: ", obj_2d_pts.shape)
        # print("match_mat")
        # print(match_mat)
        return match_mat

    def object_match(self, tunc_val=0.5, is_has_prior=False, is_debug=True):
        '''
            return matching results and coarse calib result
        '''
        num_obj_3d  = self.obj_3d_pts.shape[0]
        num_obj_2d  = self.obj_2d_pts.shape[0]
        dist_coeffs = np.zeros((4, 1)) 
        
        best_match_mat = None # [num_obj_3d, num_obj_2d] matrix
        best_match_sc  = 0.0
        best_match_res = {'rmat': None, 'tvec': None}

        for idx_3d in range(num_obj_3d):
            obj_3d = self.obj_3d_pts[idx_3d]
            for idx_2d in range(num_obj_2d):
                obj_2d = self.obj_2d_pts[idx_2d]
                # obtain calib result
                pts_3d  = obj_3d.reshape((-1,3))
                pix_2d  = obj_2d.reshape((-1,2))
                if is_has_prior == False:
                    success, rvec, tvec = cv.solvePnP(
                        pts_3d, pix_2d, self.kmat, dist_coeffs, flags=cv.SOLVEPNP_ITERATIVE)
                else:
                    tvec_ini = np.zeros((3,1))
                    rmat_ini = np.zeros((3,3))
                    rmat_ini[2,0] = 1
                    rmat_ini[0,1] = -1
                    rmat_ini[1,2] = -1
                    success, rvec, tvec = cv.solvePnP(
                        pts_3d, pix_2d, self.kmat, dist_coeffs,
                        rvec=cv.Rodrigues(rmat_ini)[0], 
                        tvec=tvec_ini.astype(np.double),
                        useExtrinsicGuess=True,
                        flags=cv.SOLVEPNP_ITERATIVE)
                    # rvec=cv.Rodrigues(rmat_ini)[0]
                    # tvec=tvec_ini.astype(np.double)
                rmat = cv.Rodrigues(rvec)[0]
                # print("try match (3d-2d): ", idx_3d, " -- ", idx_2d)
                match_mat = self.match_evaluate(
                    self.obj_3d_pts, self.obj_2d_pts, rmat, tvec)
                match_sc  = np.sum(match_mat)

                # find the best match
                if match_sc >= best_match_sc:
                    best_match_sc  = match_sc
                    best_match_mat = match_mat
                    best_match_res['rmat'] = rmat
                    best_match_res['tvec'] = tvec

        # analysis the best match
        sort_obj_3d_pts = []
        sort_obj_2d_pts = []
        sort_obj_scores = []
        for idx_3d in range(num_obj_3d):
            sc_vec = best_match_mat[idx_3d, :]
            id_max = np.argmax(sc_vec)
            # print(idx_3d, " --- ", sc_vec[id_max])
            if sc_vec[id_max] >= self.iou_threshold:
                sort_obj_3d_pts.append(self.obj_3d_pts[idx_3d])
                sort_obj_2d_pts.append(self.obj_2d_pts[id_max])
                sort_obj_scores.append(sc_vec)

        sort_obj_3d_pts = np.array(sort_obj_3d_pts)
        sort_obj_2d_pts = np.array(sort_obj_2d_pts)

        '''
            note-0410 

            to improve the initail calibration accuracy,
            we add a refinement
        '''
        # optimization again with sort_obj_3d_pts and sort_obj_2d_pts
        success, rvec, tvec = cv.solvePnP(
            sort_obj_3d_pts.reshape((-1,3)), 
            sort_obj_2d_pts.reshape((-1,2)), 
            self.kmat, dist_coeffs, 
            rvec=cv.Rodrigues(best_match_res['rmat'])[0], 
            tvec=best_match_res['tvec'].astype(np.double),
            useExtrinsicGuess=True,
            flags=cv.SOLVEPNP_ITERATIVE)
        rmat = cv.Rodrigues(rvec)[0]
        best_match_res['rmat'] = rmat
        best_match_res['tvec'] = tvec

        # truncate operation for tvec
        tvec_sign = np.sign(tvec)
        val = np.abs(tvec)
        invalid_id = (val >= tunc_val)
        tvec[invalid_id] = tunc_val*tvec_sign[invalid_id]
        best_match_res['tvec'] = tvec
        # print(tvec)
        # tvec = aaaaaaaaa

        if is_debug == True:
            print("===== 3d-2d match result =====")
            print("=> best_match_mat")
            print(best_match_mat)
            print("=> best_match_sc: ", best_match_sc)
            # print("sort_obj_3d_pts: ", sort_obj_3d_pts.shape)
            # print("sort_obj_2d_pts: ", sort_obj_2d_pts.shape)

        return best_match_res, sort_obj_3d_pts, sort_obj_2d_pts, sort_obj_scores

class object_3d_2d_registration:
    '''
        3d-2d point registration from the matched objects
    '''
    def __init__(self, pts, rgb, kmat):
        self.pts  = pts
        self.rgb  = rgb
        self.kmat = kmat

    def load_sort_obj_info(
        self, sort_obj_3d_pts, sort_obj_2d_pts, sort_obj_scores):
        '''
            from 3d-2d object match
        '''
        self.sort_obj_3d_pts = sort_obj_3d_pts
        self.sort_obj_2d_pts = sort_obj_2d_pts
        self.sort_obj_scores = sort_obj_scores

    def load_coarse_calib_res(self, rmat, tvec):
        self.rmat_ini = rmat
        self.tvec_ini = tvec

    def obtain_3d_region(self, obj_3d_pts):
        x_min, x_max = np.min(obj_3d_pts[:,0]), np.max(obj_3d_pts[:,0])
        y_min, y_max = np.min(obj_3d_pts[:,1]), np.max(obj_3d_pts[:,1])
        z_min, z_max = np.min(obj_3d_pts[:,2]), np.max(obj_3d_pts[:,2])
        
        id_x = (self.pts[:,0] >= x_min) & (self.pts[:,0] <= x_max)
        id_y = (self.pts[:,1] >= y_min) & (self.pts[:,1] <= y_max)
        id_z = (self.pts[:,2] >= z_min) & (self.pts[:,2] <= z_max)
        id_xyz = id_x & id_y & id_z

        pts = self.pts[id_xyz, :]
        return pts

    def obtain_2d_region(self, obj_2d_pts):
        u_min, u_max = np.min(obj_2d_pts[:,0]), np.max(obj_2d_pts[:,0])
        v_min, v_max = np.min(obj_2d_pts[:,1]), np.max(obj_2d_pts[:,1])
        u_min, u_max = int(u_min), int(u_max)
        v_min, v_max = int(v_min), int(v_max)

        img = np.zeros_like(self.rgb)
        # canny = cv.Canny(img, 50, 150)
        # img[v_min:v_max, u_min:u_max, :] = \
        #     self.rgb[v_min:v_max, u_min:u_max, :]

        # add sobel image
        gray = cv.cvtColor(self.rgb, cv.COLOR_BGR2GRAY)
        x = cv.Sobel(gray, cv.CV_16S, 1, 0)
        y = cv.Sobel(gray, cv.CV_16S, 0, 1)
        Scale_absX = cv.convertScaleAbs(x)  # 格式转换函数
        Scale_absY = cv.convertScaleAbs(y)
        sobel = cv.addWeighted(Scale_absX, 0.5, Scale_absY, 0.5, 0)
        canny = cv.Canny(gray, 50, 150)

        img = np.zeros_like(gray)
        img[v_min:v_max, u_min:u_max] = \
            canny[v_min:v_max, u_min:u_max]
        
        return img

    def comp_distance_lines_points(self, lines, points):
        '''
            suppose N lines and M points
                    lines  [N, 3]  (normalized vector)
                    points [M, 3]
            compute the distance of line L to point P
            return distance matrix of N*M
        '''
        n = lines.shape[0]
        m = points.shape[0]
        line_dis_mat = np.zeros((n,m), dtype=np.double)
        proj_pts_mat = np.zeros((n,m,3), dtype=np.double)
    
        # obtain the projected point on each line
        for i in range(m):
            pt = points[i,:3] # [1,3]
            delta_t_pt = np.matmul(lines, pt.T) # [N,1]
            delta_t_pt = delta_t_pt.reshape((-1,1))
            proj_pt    = delta_t_pt * lines     # [N,3]
            proj_pts_mat[:,i] = proj_pt

        # line-point distance = norm_2(point, projected point) 
        for i in range(n):
            proj_all_pts_one_line = proj_pts_mat[i,:,:] # [M,3]
            dis = (points - proj_all_pts_one_line) # [M,3]
            dis = np.linalg.norm(dis, axis=1) # [M,1]
            line_dis_mat[i] = dis
        
        return line_dis_mat

    def extract_2d_contour_single(self, img, pts):
        '''
            a improved and stable version of extract_2d_contour
            (avoid one-to-many match)
        '''
        # gather the edge pixels
        idx = (img > 0)
        u_array = np.zeros_like(img, dtype=np.uint64)
        v_array = np.zeros_like(img, dtype=np.uint64)
        for u_idx in range(img.shape[0]):
            u_array[u_idx,:] = u_idx
        for v_idx in range(img.shape[1]):
            v_array[:,v_idx] = v_idx
        u_set = u_array[idx]
        v_set = v_array[idx]

        # generate 3d line via back-projection
        num_pix = u_set.shape[0]
        pixels  = np.zeros((num_pix, 3))
        pixels[:,0] = v_set
        pixels[:,1] = u_set
        pixels[:,2] = 1
        pts_tf  = np.matmul(self.rmat_ini, pts[:,:3].T)
        pts_tf += pts_tf + self.tvec_ini
        pts_tf  = pts_tf.T
        
        lines   = np.matmul(np.linalg.inv(self.kmat), pixels.T).T
        tmp_nor = np.linalg.norm(lines, axis=1)
        tmp_nor = tmp_nor.reshape((-1,1))
        lines  /= tmp_nor  # line normalization

        # compute distance of lines and points
        # line_dis_mat is [lines, points] matrix
        # please note that point number is smaller than line number
        # one-to-one match (single point v.s single lines)
        line_dis_mat = self.comp_distance_lines_points(lines, pts_tf)
        min_line_dis_vec = np.min(line_dis_mat, axis=0)
        min_line_dis_index = np.argmin(line_dis_mat, axis=0)

        match_num = min_line_dis_vec.shape[0]
        r_match_set_3d_2d = np.zeros((match_num, 6)) # (uv1+xyz)
        r_match_set_3d_2d[:,0:3] = pixels[min_line_dis_index,0:3]
        r_match_set_3d_2d[:,3:6] = pts[:, 0:3]

        # filter the incorrect match via distance threshold
        min_dis = np.min(min_line_dis_vec)
        max_dis = np.max(min_line_dis_vec)
        dis_w   = 0.75
        dis_th  = min_dis*dis_w + max_dis*(1.0-dis_w)
        can_idx = (min_line_dis_vec <= dis_th) # unit: meter
        match_set_3d_2d = r_match_set_3d_2d[can_idx,:]

        return match_set_3d_2d

    def extract_2d_contour(self, img, pts):
        '''
            filter the true object contour using 3d points
            and the coarse calibration results

            also obtain the coarse 3d-2d match
        '''
        # gather the edge pixels
        idx = (img > 0)
        u_array = np.zeros_like(img, dtype=np.uint64)
        v_array = np.zeros_like(img, dtype=np.uint64)
        for u_idx in range(img.shape[0]):
            u_array[u_idx,:] = u_idx
        for v_idx in range(img.shape[1]):
            v_array[:,v_idx] = v_idx
        u_set = u_array[idx]
        v_set = v_array[idx]

        # generate 3d line via back-projection
        num_pix = u_set.shape[0]
        pixels  = np.zeros((num_pix, 3))
        pixels[:,0] = v_set
        pixels[:,1] = u_set
        pixels[:,2] = 1
        pts_tf  = np.matmul(self.rmat_ini, pts[:,:3].T)
        pts_tf += pts_tf + self.tvec_ini
        pts_tf  = pts_tf.T

        lines   = np.matmul(np.linalg.inv(self.kmat), pixels.T).T
        tmp_nor = np.linalg.norm(lines, axis=1)
        tmp_nor = tmp_nor.reshape((-1,1))
        lines  /= tmp_nor  # line normalization

        # compute distance of lines and points
        # line_dis_mat is [lines, points] matrix
        # please note that point number is smaller than line number
        # one-to-many match (single point v.s multiple lines)
        line_dis_mat = self.comp_distance_lines_points(lines, pts_tf)
        min_line_dis_vec = np.min(line_dis_mat, axis=1)
        min_line_dis_index = np.argmin(line_dis_mat, axis=1)

        # obtain coarse 3d-2d match
        match_num = min_line_dis_vec.shape[0]
        r_match_set_3d_2d = np.zeros((match_num, 6)) # (uv1+xyz)
        r_match_set_3d_2d[:,0:3] = pixels[:,0:3]
        r_match_set_3d_2d[:,3:6] = pts[min_line_dis_index, 0:3]
        # r_match_set_3d_2d[:,3:6] = pts_tf[min_line_dis_index, 0:3]

        # pixels_ ,_ = projection(
        #     pts[min_line_dis_index,:3], self.kmat, self.rmat_ini, self.tvec_ini)
        # r_match_set_3d_2d[:,0:2] = pixels_[:,0:2]

        # filter the incorrect match via distance threshold
        min_dis = np.min(min_line_dis_vec)
        max_dis = np.max(min_line_dis_vec)
        dis_w   = 0.75
        # dis_w   = 0.90
        dis_th  = min_dis*dis_w + max_dis*(1.0-dis_w)
        can_idx = (min_line_dis_vec <= dis_th) # unit: meter
        match_set_3d_2d = r_match_set_3d_2d[can_idx,:]

        # pixels_ = pixels_[can_idx,:]
        # pixels  = pixels[can_idx,:]

        # print("r_match_set_3d_2d: ", r_match_set_3d_2d.shape)
        # print("match_set_3d_2d: ", match_set_3d_2d.shape)

        # filter over-dense pixels
        # todo

        # visulization matching results (1) ===
        # vis_img = np.zeros((img.shape[0]*2, img.shape[1], 3))
        # vis_img[:img.shape[0],:,0] = img
        # vis_img[:img.shape[0],:,1] = img
        # vis_img[:img.shape[0],:,2] = img

        # pts_img = np.zeros((img.shape[0], img.shape[1], 3))
        # pixels ,_ = projection(
        #     pts[min_line_dis_index,:3], self.kmat, self.rmat_ini, self.tvec_ini)

        # for id in range(match_set_3d_2d.shape[0]):
        #     # vis_img = cv.line(
        #     #     vis_img, (int(match_set_3d_2d[id,0]), int(match_set_3d_2d[id,1])),
        #     #     (int(pixels[id,0]), int(pixels[id,1]+img.shape[0])),
        #     #     color=(255,255,0), thickness=1)

        #     vis_img = cv.circle(
        #         vis_img, (int(match_set_3d_2d[id,0]), int(match_set_3d_2d[id,1])), 
        #         radius=2, color=(0,255,0), thickness=-1)
        
        #     vis_img = cv.circle(
        #         vis_img, (int(pixels[id,0]), int(pixels[id,1]+img.shape[0])), 
        #         radius=2, color=(0,0,255), thickness=-1)

        # cv.imshow("vis_img", vis_img)
        # cv.waitKey()
        # =================

        # visulization matching results (2) ===
        # vis_img = np.zeros((img.shape[0]*2, img.shape[1], 3))
        # vis_img[:img.shape[0],:,0] = img
        # vis_img[:img.shape[0],:,1] = img
        # vis_img[:img.shape[0],:,2] = img

        # for id in range(pixels.shape[0]):
        #     # vis_img = cv.line(
        #     #     vis_img, (int(match_set_3d_2d[id,0]), int(match_set_3d_2d[id,1])),
        #     #     (int(pixels[id,0]), int(pixels[id,1]+img.shape[0])),
        #     #     color=(255,255,0), thickness=1)

        #     vis_img = cv.circle(
        #         vis_img, (int(pixels[id,0]), int(pixels[id,1])), 
        #         radius=2, color=(0,255,0), thickness=-1)
        
        #     vis_img = cv.circle(
        #         vis_img, (int(pixels_[id,0]), int(pixels_[id,1])), 
        #         radius=2, color=(0,0,255), thickness=-1)
        
        # cv.imshow("vis_img", vis_img)
        # cv.waitKey()
        # =================

        # print("lines: ", lines.shape)
        # print("pts: ", pts.shape)
        # print("line_dis_mat: ", line_dis_mat.shape)
        # print("min_line_dis_vec: ", min_line_dis_vec.shape)
        # print("min_line_dis_index: ", min_line_dis_index.shape)
        # print(min_line_dis_index)

        return match_set_3d_2d

    def proj_3d_2d_icp(self, region_3d, region_2d, match_set):
        '''
            extend 3d-3d icp to 3d-2d case, 
            we propose proj-icp for extrinsic calibration

            (a simple version with epnp)

            In match_set, each element is match_set_3d_2d
            which is [N_match,6] matrix where (uv1+xyz)
        '''
        num_obj = len(region_3d)
        num_obj = len(match_set)
        pts_3d  = np.zeros((0,3), dtype=np.float32)
        pix_2d  = np.zeros((0,2), dtype=np.float32)
        for id in range(num_obj):
            pts_3d_ = match_set[id][:,3:6]
            pix_2d_ = match_set[id][:,0:2]
            pts_3d_ = pts_3d_.astype(np.float32)
            pix_2d_ = pix_2d_.astype(np.float32)
            pts_3d  = np.concatenate((pts_3d, pts_3d_), axis=0)
            pix_2d  = np.concatenate((pix_2d, pix_2d_), axis=0)


        dist_coeffs = np.zeros((4, 1)) 
        # success, rvec, tvec = cv.solvePnP(
        #     pts_3d, pix_2d, self.kmat, dist_coeffs, flags=cv.SOLVEPNP_ITERATIVE)
        success, rvec, tvec = cv.solvePnP(
            pts_3d, pix_2d, self.kmat, dist_coeffs,
            rvec=cv.Rodrigues(self.rmat_ini)[0], 
            tvec=self.tvec_ini.astype(np.double),
            useExtrinsicGuess=True,
            flags=cv.SOLVEPNP_ITERATIVE)
        
        rmat = cv.Rodrigues(rvec)[0]

        pixels_a ,_ = projection(
            pts_3d, self.kmat, self.rmat_ini, self.tvec_ini)
        pixels_ ,_ = projection(
            pts_3d, self.kmat, rmat, tvec)

        # debug visulization ===
        # col = np.ones_like(pts_3d)
        # pcd = o3d.geometry.PointCloud()
        # pcd.points = o3d.utility.Vector3dVector(pts_3d)
        # pcd.colors = o3d.utility.Vector3dVector(col)
        # show_pcd(pcd)

        # img = region_2d[0]
        # vis_img = np.zeros((img.shape[0], img.shape[1], 3))
        # vis_img[:img.shape[0],:,0] = img
        # vis_img[:img.shape[0],:,1] = img
        # vis_img[:img.shape[0],:,2] = img
        # for id in range(pix_2d.shape[0]):
        #     # vis_img = cv.circle(
        #     #     vis_img, (int(pix_2d[id,0]), int(pix_2d[id,1])), 
        #     #     radius=2, color=(0,255,0), thickness=-1)
        #     vis_img = cv.circle(
        #         vis_img, (int(pixels_[id,0]), int(pixels_[id,1])), 
        #         radius=2, color=(0,0,255), thickness=-1)
        #     vis_img = cv.circle(
        #         vis_img, (int(pixels_a[id,0]), int(pixels_a[id,1])), 
        #         radius=2, color=(0,255,0), thickness=-1)
        # cv.imshow("vis_img", vis_img)
        # cv.waitKey()
        # ======================

        return rmat, tvec

    def proj_3d_2d_icp_solver(self, region_3d, region_2d, match_set):
        '''
            extend 3d-3d icp to 3d-2d case, 
            we propose proj-icp for extrinsic calibration

            (a general version using gradient)

            In match_set, each element is match_set_3d_2d
            which is [N_match,6] matrix where (uv1+xyz)
        '''
        num_obj = len(region_3d)
        pts_3d  = np.zeros((0,3), dtype=np.float32)
        pix_2d  = np.zeros((0,2), dtype=np.float32)
        for id in range(num_obj):
            pts_3d_ = match_set[id][:,3:6]
            pix_2d_ = match_set[id][:,0:2]
            pts_3d_ = pts_3d_.astype(np.float32)
            pix_2d_ = pix_2d_.astype(np.float32)
            pts_3d  = np.concatenate((pts_3d, pts_3d_), axis=0)
            pix_2d  = np.concatenate((pix_2d, pix_2d_), axis=0)

        # scheme 1 -- using reprojection error
        # ba_solver = bundle_adjustment(self.kmat, self.rmat_ini, self.tvec_ini)
        # ba_solver.load_measure_data(pts_3d, pix_2d)
        # rmat_opt, tvec_opt = ba_solver.optmization()
        # rmat_opt, tvec_opt = ba_solver.optmization_light()

        # scheme 2 -- using point-to-line error
        pl_solver = point_to_line_solver(self.kmat, self.rmat_ini, self.tvec_ini)
        pl_solver.load_measure_data(pts_3d, pix_2d)
        # pl_solver.optmization_light()
        rmat_opt, tvec_opt = pl_solver.optmization_light()
        # rmat_opt, tvec_opt = pl_solver.optmization()

        return rmat_opt, tvec_opt

    def points_regiestration(self, is_use_pnp=False, is_use_history=False):
        '''
            return 3d-2d registration result and precise calib
        '''
        # step 1: pre-processing
        #         obtain 2d and 3d matched region
        self.num_obj = self.sort_obj_3d_pts.shape[0]
        region_3d = []
        region_2d = []
        match_set = [] # store corase match info

        if is_use_history == True:
            m0 = np.load("match_set_3d_2d_0.npy")
            m1 = np.load("match_set_3d_2d_1.npy")
            match_set.append(m0)
            match_set.append(m1)
            match_set.append(m0)
            # match_set.append(m1)

        for i in range(self.num_obj):
            obj_3d_pts = self.sort_obj_3d_pts[i]
            obj_2d_pts = self.sort_obj_2d_pts[i]
            obj_scores = self.sort_obj_scores[i]
            obj_3d_sp  = self.obtain_3d_region(obj_3d_pts)
            obj_2d_sp  = self.obtain_2d_region(obj_2d_pts)
            region_3d.append(obj_3d_sp)
            region_2d.append(obj_2d_sp)

            # step 2: extract object 2d contour
            #         and obtain coarse 3d-2d match
            # match_set_3d_2d is [N_match,6] matrix where (uv1+xyz)
            # note 0320: using one-to-one match is stable
            #            (not stable enough however)

            # match_set_3d_2d = self.extract_2d_contour(obj_2d_sp, obj_3d_sp)
            match_set_3d_2d = self.extract_2d_contour_single(obj_2d_sp, obj_3d_sp)
            match_set.append(match_set_3d_2d)

            # if i == 0:
            #     np.save("match_set_3d_2d_0.npy", match_set_3d_2d)
            # if i == 1:
            #     np.save("match_set_3d_2d_1.npy", match_set_3d_2d)
        
        # if is_use_history == True:
        #     m0 = np.load("match_set_3d_2d_0.npy")
        #     m1 = np.load("match_set_3d_2d_1.npy")
        #     match_set.append(m0)
        #     match_set.append(m1)
        #     match_set.append(m0)
        #     match_set.append(m1)
        
        # step 3: coarse-to-fine calibration via  
        #         projection based iterative closet point (proj-icp)
        if is_use_pnp == True:
            rmat, tvec = self.proj_3d_2d_icp(region_3d, region_2d, match_set)
        else:
            rmat, tvec = self.proj_3d_2d_icp_solver(region_3d, region_2d, match_set)

        # need visulization
        pts = region_3d[0][:,:3]
        col = np.ones_like(pts)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)
        pcd.colors = o3d.utility.Vector3dVector(col)
        
        # show_pcd(pcd)
        # cv.imshow("region_2d", region_2d[0])
        # cv.waitKey()

        # print("===== 3d-2d registration result =====")

        # for debug
        # print("self.sort_obj_3d_pts: ", self.sort_obj_3d_pts.shape)

        calib_res = {'rmat': None, 'tvec': None}
        calib_res['rmat'] = rmat
        calib_res['tvec'] = tvec 

        return calib_res

    def points_regiestration_global(self):
        '''
            obtain accurate and robust calibration results

            this function can be used only if
            the iterative 2d-3d registration is conducted.
        '''
        # step 1: obtain point cloud and canny contour image
        gray  = cv.cvtColor(self.rgb, cv.COLOR_BGR2GRAY)
        canny = cv.Canny(gray, 50, 150)
        lidar = self.pts

        # step 1*: sub-sample canny image and point cloud
        #          due to the limited cpu memory
        # note-0329
        #   only preserve the point cloud in the FoV
        h, w = canny.shape[0], canny.shape[1]
        # canny[:int(h/4)]  = 0
        # canny[-int(h/4):] = 0
        # canny[:int(h/2)] = 0
        # canny[-int(h/4):] = 0

        # tmp = canny.copy()
        # tmp[180-120:180+120, 600-400:600+400] = 0
        # canny = canny - tmp

        pixels, depths = projection(lidar, self.kmat, self.rmat_ini, self.tvec_ini)
        valid_idx_0 = ((depths >= 10) & (depths <= 30)).reshape((-1))
        valid_idx_1 = (pixels[:,0] >= 10) & (pixels[:,0] <= 1200)
        valid_idx_2 = (pixels[:,1] >= 10) & (pixels[:,1] <= 370)
        valid_idx_1 = (pixels[:,0] >= 600-300) & (pixels[:,0] <= 600+300)
        valid_idx_2 = (pixels[:,1] >= 20) & (pixels[:,1] <= 200)
        valid_idx = valid_idx_0 & valid_idx_1 & valid_idx_2
        sub_lidar = lidar[valid_idx,:]

        n = sub_lidar.shape[0]
        # sub_lidar = sub_lidar[0:2:n, :]
        
        # step 2: obtain coarse global 2d-3d match result
        match_set = []
        match_set_3d_2d = self.extract_2d_contour_single(canny, sub_lidar)
        match_set.append(match_set_3d_2d)

        # step 2a: pre-processing
        #          obtain 2d and 3d matched region
        self.num_obj = self.sort_obj_3d_pts.shape[0]
        region_3d = []
        region_2d = []
        for i in range(self.num_obj):
            obj_3d_pts = self.sort_obj_3d_pts[i]
            obj_2d_pts = self.sort_obj_2d_pts[i]
            obj_scores = self.sort_obj_scores[i]
            obj_3d_sp  = self.obtain_3d_region(obj_3d_pts)
            obj_2d_sp  = self.obtain_2d_region(obj_2d_pts)
            region_3d.append(obj_3d_sp)
            region_2d.append(obj_2d_sp)

            # step 2: extract object 2d contour
            #         and obtain coarse 3d-2d match
            # match_set_3d_2d is [N_match,6] matrix where (uv1+xyz)
            # note 0320: using one-to-one match is stable
            #            (not stable enough however)

            # match_set_3d_2d = self.extract_2d_contour(obj_2d_sp, obj_3d_sp)
            match_set_3d_2d = self.extract_2d_contour_single(obj_2d_sp, obj_3d_sp)
            match_set.append(match_set_3d_2d)

        # step 3: 
        # rmat, tvec = self.proj_3d_2d_icp_solver(match_set, match_set, match_set)
        rmat, tvec = self.proj_3d_2d_icp(match_set, match_set, match_set)
        
        calib_res = {'rmat': None, 'tvec': None}
        calib_res['rmat'] = rmat
        calib_res['tvec'] = tvec 

        return calib_res