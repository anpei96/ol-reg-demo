#
# Project: lidar-camera system calibration based on
#          object-level 3d-2d correspondence
# Author:  anpei
# Data:    2023.03.07
# Email:   anpei@wit.edu.cn
#

import numpy as np
import scipy

from scipy.sparse import lil_matrix
from scipy.optimize import least_squares
from scipy.linalg import expm,logm

from numpy import pi,sin,cos,tan,arccos,matmul
from numpy.linalg import norm

deg = pi/180

class se_group_tf:
    '''
        a simple se(3)-SE(3) transformation api
        ref: https://blog.csdn.net/weixin_41855010/article/details/118972833
    '''
    def __init__(self):
        pass

    def vec2sM(self, vec):
        return np.array([
                [0,-vec[2],vec[1]],
                [vec[2],0,-vec[0]],
                [-vec[1],vec[0],0]
            ])

    def sM2vec(self, sM):
        return np.array([sM[2][1],sM[0][2],sM[1][0]])

    def upgradeStoE(self, Screw):
        '''
        规定：Screw=(S,S0),单位旋量screw=(s,s0)，Screw=theta*screw
        规定：omg为三维的单位向量，OMG = theta*omg,
        把运动旋量Screw∈se(3)，转换（升维）成T的矩阵对数E矩阵
        #E=Eu*theta，其中theta为转角，Eu为单位螺旋对应的矩阵对数
        '''
        theta = norm(Screw[:3])
        screw   = Screw.reshape((6,1))/theta
        omg,vel = screw[:3],screw[-3:]   
        sMomg   = self.vec2sM(omg)
        bottom  = np.array([[0,0,0,0]])
        Eu = np.r_[np.c_[sMomg,vel],bottom]
        return Eu.astype(float)*theta

    def degradeEtoS(self, Ematrix):
        '''
        输入：李代数的4×4标准表示E矩阵
        输出：对应的李代数向量形式Screw
        '''
        sMOMG = Ematrix[:3,:3]
        VEL = Ematrix[:3,3].reshape(3,1)
        OMG = self.sM2vec(sMOMG).reshape(3,1)
        theta = np.linalg.norm(OMG)
        omg = OMG/theta
        vel = VEL/theta
        screw = np.vstack((omg,vel))
        # sMomg = sMOMG/theta
        # Ginv = 1/theta*I-1/2*sMomg+(1/theta-0.5/tan(theta/2))*np.matmul(sMomg,sMomg)
        return (screw*theta).reshape(1,6)

    def logm_and_degrade(self, Tmat):
        '''
            from SE(3) to se(3)
        '''
        EM   = logm(Tmat)
        Svec = self.degradeEtoS(EM)
        return Svec
    
    def expm_and_upgrade(self, Svec):
        '''
            from se(3) to SE(3)
        '''
        EM   = self.upgradeStoE(Svec)
        Tmat = expm(EM)
        return Tmat

def trans_Tmat(rmat, tvec):
    Tmat = np.zeros((4,4), dtype=np.float32)
    Tmat[3,3]   = 1.0
    Tmat[:3,:3] = rmat
    Tmat[:3,3:4] = tvec 
    return Tmat

def trans_rmat_tvec(Tmat):
    rmat = Tmat[:3,:3]
    tvec = Tmat[:3,3:4]
    return rmat, tvec

class bundle_adjustment:
    '''
        a simple bundle adjustment with se(3) optimization 
        (single view version)
        ref: https://blog.csdn.net/baidu_40840693/article/details/115554682
        ref: https://notes.andrewtorgesen.com/doku.php?id=public:ceres
    '''
    def __init__(self, kmat, rmat_ini, tvec_ini):
        self.kmat = kmat
        self.rmat_ini = rmat_ini
        self.tvec_ini = tvec_ini
        self.Tmat_ini = trans_Tmat(rmat_ini, tvec_ini)

        self.tf = se_group_tf()
        self.Svec_ini = self.tf.logm_and_degrade(self.Tmat_ini)
    
    def load_measure_data(self, pts_3d, pix_2d):
        '''
            pts_3d [N,3] matrix
            pix_2d [N,2] matrix
        '''
        self.pts_3d = pts_3d
        self.pix_2d = pix_2d

    def projection(self, pts_3d, Svec):
        Tmat = self.tf.expm_and_upgrade(Svec)
        rmat, tvec = trans_rmat_tvec(Tmat)
        tmp  = np.matmul(rmat, pts_3d.T) + tvec # [3,N]
        tmp  = np.matmul(self.kmat, tmp) # [3,N]
        tmp /= (tmp[2,:]+1e-5)
        tmp  = tmp[:2, :]
        return tmp.T

    def cost(self, x, pts_2d):
        Svec    = x[:6].reshape((-1,1))
        pts_3d  = x[6:].reshape((-1,3))
        pix_2d_ = self.projection(pts_3d, Svec)
        return (pix_2d_ - pts_2d).ravel()

    def fun(self, x):
        Svec    = x[:6].reshape((-1,1))
        pts_3d  = x[6:].reshape((-1,3))
        pix_2d_ = self.projection(pts_3d, Svec)
        return (pix_2d_ - self.pix_2d).ravel()

    def fun_light(self, x):
        Svec    = x[:6].reshape((-1,1))
        pix_2d_ = self.projection(self.pts_3d, Svec)
        return (pix_2d_ - self.pix_2d).ravel()

    def bundle_adjustment_sparsity(self, pts_num):
        '''
            construct a sparse matrix from single view
        '''
        m = 1*2
        n = 6*1 + pts_num*3
        A = lil_matrix((m, n), dtype=int)
        A[:,:] = 1

    def optmization(self):
        '''
            use scipy least_squares to solve problem
            ref: https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html
        '''
        # step-1  define parameters x0 ([6+3N] vector)
        #         prepare initial values
        num_x = 6 + self.pts_3d.shape[0]*3
        x0 = np.zeros((num_x))
        x0[:6] = self.Svec_ini[0,:6]
        x0[6:] = self.pts_3d.reshape((-1))

        # step-1* test cost function --- ok
        # loss = self.cost(x0, self.pix_2d)
        # print("loss: ", loss)
        # print("x0: ", x0.shape)

        # step-2 construct the sparse matrix
        A = self.bundle_adjustment_sparsity(self.pts_3d.shape[0])

        # step-3 optimzation with least-square
        # res = least_squares(
        #     self.cost, x0, 
        #     jac_sparsity=A, verbose=2, x_scale='jac', 
        #     ftol=1e-4, method='trf', args=(self.pix_2d))
        res = least_squares(
            self.fun, x0, 
            jac_sparsity=A, verbose=2, x_scale='jac', 
            ftol=1e-4, method='trf')

        # step-3* using gradient (jacobian matrix)
        # todo
        
        # step-4 extract optimization reuslts
        Svec_opt   = res.x[:6].reshape((-1,1))
        pts_3d_opt = res.x[6:].reshape((-1,3))
        Tmat_opt   = self.tf.expm_and_upgrade(Svec_opt)
        rmat_opt, tvec_opt = trans_rmat_tvec(Tmat_opt)

        return rmat_opt, tvec_opt

    def optmization_light(self):
        '''
            use scipy least_squares to solve problem
            (only optimize the pose)
        '''
        # step-1  define parameters x0 ([6+3N] vector)
        #         prepare initial values
        num_x = 6
        x0 = np.zeros((num_x))
        x0[:6] = self.Svec_ini[0,:6]

        # step-2 optimzation with least-square
        res = least_squares(
            self.fun_light, x0, 
            verbose=0, x_scale='jac', 
            ftol=1e-5, method='trf')

        # step-2* using gradient (jacobian matrix)
        # todo
        
        # step-3 extract optimization reuslts
        Svec_opt   = res.x[:6].reshape((-1,1))
        Tmat_opt   = self.tf.expm_and_upgrade(Svec_opt)
        rmat_opt, tvec_opt = trans_rmat_tvec(Tmat_opt)

        return rmat_opt, tvec_opt

class point_to_line_solver:
    '''
        a simple solver with se(3) optimization 
        by minimizing the line-to-point loss
        (single view version)
    '''
    def __init__(self, kmat, rmat_ini, tvec_ini):
        self.kmat = kmat
        self.rmat_ini = rmat_ini
        self.tvec_ini = tvec_ini
        self.Tmat_ini = trans_Tmat(rmat_ini, tvec_ini)

        self.tf = se_group_tf()
        self.Svec_ini = self.tf.logm_and_degrade(self.Tmat_ini)
    
    def load_measure_data(self, pts_3d, pix_2d):
        '''
            pts_3d [N,3] matrix
            pix_2d [N,2] matrix
        '''
        self.pts_3d = pts_3d
        self.pix_2d = pix_2d

        # pre-computing
        num = self.pts_3d.shape[0]
        pix_2d_ = np.ones((num, 3))
        pix_2d_[:,:2] = self.pix_2d
        line_3d = np.matmul(np.linalg.inv(self.kmat), pix_2d_.T) # [3,N]
        line_3d_nor = line_3d/np.linalg.norm(line_3d, axis=0).reshape((1,-1))
        
        self.num = num
        self.line_3d_nor = line_3d_nor

    def fun_light(self, x):
        '''
            compute point-to-line error
        '''
        Svec = x[:6].reshape((-1,1))
        Tmat = self.tf.expm_and_upgrade(Svec)
        rmat, tvec = trans_rmat_tvec(Tmat)
        pts_3d_tf = np.matmul(rmat, self.pts_3d.T) + tvec # [3,N]

        # num = self.pts_3d.shape[0]
        # pix_2d_ = np.ones((num, 3))
        # pix_2d_[:,:2] = self.pix_2d
        # line_3d = np.matmul(np.linalg.inv(self.kmat), pix_2d_.T) # [3,N]
        # line_3d_nor = line_3d/np.linalg.norm(line_3d, axis=0).reshape((1,-1))
        mat = np.matmul(pts_3d_tf.T, self.line_3d_nor)
        vec = mat.diagonal()
        err = pts_3d_tf - vec*self.line_3d_nor # [3,N]

        return err.ravel()
        # return err.reshape((-1))
        # return np.linalg.norm(err, axis=0).reshape((-1))

    def sub_jacobian_light(self):
        '''
            compute the sub-jacobian matrix 
            it is 3*6 matrix

            ref: https://blog.csdn.net/qq_43256088/article/details/123550314
        '''
        # it is merged in jacobian_light
        pass

    def jacobian_light(self, x0):
        '''
            compute the jacobian matrix 

            N    is point number 
            m    is pose number (m=6)
            loss is 1*(3N) vector (xyz is flatten)
            x0   is 1*m    vector
            jacobian matrix is (3N)*m matrix
            it can be decomposed as N m*3 sub-jacobian matrix
        '''
        Svec = x0[:6].reshape((-1,1))
        Tmat = self.tf.expm_and_upgrade(Svec)
        rmat, tvec = trans_rmat_tvec(Tmat)
        pts_3d_tf = np.matmul(rmat, self.pts_3d.T) + tvec # [3,N]

        N = self.num
        jac_mat = np.zeros((N*3, 6))
        for i in range(N):
            l = self.line_3d_nor[:,i].reshape((-1,1))
            p = pts_3d_tf[:,i].reshape((-1,1))
            l_mat = np.matmul(l, l.T)
            p_mat = self.tf.vec2sM(p)
            # p_mat = np.concatenate((np.eye(3), -p_mat), axis=1)
            p_mat = np.concatenate((-p_mat, np.eye(3)), axis=1)
            j_mat = np.matmul(l_mat, p_mat) # [3,6]
            jac_mat[3*i:3*(i+1), :] = j_mat
        
        return jac_mat

    def optmization_light(self):
        '''
            use scipy least_squares to solve problem
            (only optimize the pose)
        '''
        # step-1  define parameters x0 ([6+3N] vector)
        #         prepare initial values
        num_x = 6
        x0 = np.zeros((num_x))
        x0[:6] = self.Svec_ini[0,:6]

        # step-1* test cost function --- ok
        # loss = self.fun_light(x0)
        # print("loss: ", loss)
        # print("x0: ", x0.shape)

        # step-2 optimzation with least-square
        res = least_squares(
            self.fun_light, x0, jac='3-point',
            verbose=0, x_scale='jac', loss='huber',
            ftol=1e-5, method='trf')

        # step-2* using gradient (jacobian matrix)
        # x0[:6] = res.x[:6]
        # res = least_squares(
        #     self.fun_light, x0, jac=self.jacobian_light,
        #     verbose=2, x_scale='jac', 
        #     ftol=1e-8, method='trf')
        
        # # step-3 extract optimization reuslts
        Svec_opt   = res.x[:6].reshape((-1,1))
        Tmat_opt   = self.tf.expm_and_upgrade(Svec_opt)
        rmat_opt, tvec_opt = trans_rmat_tvec(Tmat_opt)

        return rmat_opt, tvec_opt

    def fun(self, x):
        '''
            compute point-to-line error
        '''
        Svec = x[:6].reshape((-1,1))
        pts_3d  = x[6:].reshape((-1,3))
        Tmat = self.tf.expm_and_upgrade(Svec)
        rmat, tvec = trans_rmat_tvec(Tmat)
        pts_3d_tf = np.matmul(rmat, pts_3d.T) + tvec # [3,N]

        num = pts_3d.shape[0]
        pix_2d_ = np.ones((num, 3))
        pix_2d_[:,:2] = self.pix_2d
        line_3d = np.matmul(np.linalg.inv(self.kmat), pix_2d_.T) # [3,N]
        line_3d_nor = line_3d/np.linalg.norm(line_3d, axis=0).reshape((1,-1))
        mat = np.matmul(pts_3d_tf.T, line_3d_nor)
        vec = mat.diagonal()
        err = pts_3d_tf - vec*line_3d_nor # [3,N]

        return err.ravel()

    def optmization(self):
        '''
            use scipy least_squares to solve problem
            (optimize the pose and points)

            not good and also very slow 
            has the risk of over-fiting
        '''
        # step-1  define parameters x0 ([6+3N] vector)
        #         prepare initial values
        num_x = 6 + self.pts_3d.shape[0]*3
        x0 = np.zeros((num_x))
        x0[:6] = self.Svec_ini[0,:6]
        x0[6:] = self.pts_3d.reshape((-1))

        # step-1* test cost function --- ok
        # loss = self.fun(x0)
        # print("loss: ", loss)
        # print("x0: ", x0.shape)

        # step-2 optimzation with least-square
        res = least_squares(
            self.fun, x0, 
            verbose=2, x_scale='jac', 
            ftol=1e-1, method='trf')

        # # step-2* using gradient (jacobian matrix)
        # # todo
        
        # # step-3 extract optimization reuslts
        Svec_opt   = res.x[:6].reshape((-1,1))
        Tmat_opt   = self.tf.expm_and_upgrade(Svec_opt)
        rmat_opt, tvec_opt = trans_rmat_tvec(Tmat_opt)

        return rmat_opt, tvec_opt

# some unit test
# if __name__ == '__main__':
#     pass