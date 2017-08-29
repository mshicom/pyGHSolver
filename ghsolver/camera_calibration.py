#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  7 16:53:29 2017

@author: kaihong
"""
import numpy as np
import matplotlib.pyplot as plt
import cv2

from ghsolver import *
import pickle

class Camera(object):
  def __init__(self, K, dist, frame_size):
    '''
    dist: k1,k2,p1,p2[,k3,k4,k5,k6], compatible with opencv
    see http://docs.opencv.org/2.4/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html
    '''
    self.K_raw = K
    self.dist = np.zeros(8)
    self.dist[:len(dist)] = dist
    self.frame_size = frame_size

    self.K_rect = None
    self.rectifier = None

  def __call__(self, img):
    if self.rectifier is None:
      self.MakeRectifier()
    return self.rectifier(img)

  def MakeRectifier(self, alpha=0, frame_size=None):
    '''alpha: 0:no black area, 1:retained all source pixels
       frame_size: height, width
    '''
    if frame_size is None:
      frame_size = self.frame_size
    frame_size = tuple(reversed(frame_size)) # h,w -> w,h

    # Compute optimal new matrix, etc
    new_K, valid_roi = cv2.getOptimalNewCameraMatrix(self.K_raw, self.dist, frame_size, alpha)
    # Calculate undistort maps
    map1, map2 = cv2.initUndistortRectifyMap(self.K_raw, self.dist, None, new_K, frame_size, cv2.CV_32FC1)

    # Undistort
    self.rectifier = lambda img : cv2.remap(img, map1, map2, cv2.INTER_CUBIC)
    self.K_rect    = new_K
    return self

  def ProjectRaw(self, pw_3d, Tcw):
    pw_3d = np.atleast_2d(pw_3d)
    # 1. transform
    R, t  = Tcw[:3,:3], Tcw[:3,3]   # T: cam <-world
    pc_3d = R.dot(pw_3d.T) + t[:, np.newaxis]   # Nx3 -> 3xN
    p_uv  = pc_3d[:2]/pc_3d[2]

    # 2. distort
    u, v = np.copy(p_uv)
    r2 = u**2 + v**2
    k1,k2,p1,p2,k3,k4,k5,k6 = self.dist[:5]
    a = (1 + k1*r2 + k2*r2**2 + k3*r2**3)/(1 + k4*r2 + k5*r2**2 + k6*r2**3)
    p_uv[0] = a*u + p1*2*u*v + p2*(r2+2*u**2)
    p_uv[1] = a*v + p1*(r2+2*v**2) + p2*2*u*v

    # 3. project
    p_uv[0] = K[0,0] * p_uv[0] + K[0,2]
    p_uv[1] = K[1,1] * p_uv[1] + K[1,2]
    return p_uv.T # 3xN -> Nx3

  def Save(self, file_name):
    self.from_where = __file__
    pickle.dump(self, open(file_name, 'w'))
    return self

  @classmethod
  def Load(cls, file_name):
    cam = pickle.load(open(file_name, 'r'))
    assert isinstance(cam, Camera)
    print('class from' + cam.from_where)
    return cam

def ProjectPoints(point_3d, T, K, distCoeffs=None):
  '''http://docs.opencv.org/2.4/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html'''
  if 0:
    rvec = cv2.Rodrigues(T[:3,:3])[0]
    tvec = T[:3, 3]
    point_2d = cv2.projectPoints(point_3d[:,np.newaxis,:], rvec, tvec, K, distCoeffs)[0] # ignore jacobian
    return point_2d.squeeze()
  else:
    point_3d = np.atleast_2d(point_3d)
    R,t = T[:3,:3], T[:3,3]   # T: cam <-world
    p_xyz = R.dot(point_3d.T) + t[:, np.newaxis]   # Nx3 -> 3xN
    p_uv  = p_xyz[:2]/p_xyz[2]
    if not distCoeffs is None:
      u, v = np.copy(p_uv)
      r2 = u**2 + v**2

      k1,k2,p1,p2,k3 = distCoeffs[:5]
      a = (1 + k1*r2 + k2*r2**2 + k3*r2**3)
      if len(distCoeffs)==8:
        k4,k5,k6 = distCoeffs[5:8]
        a /= (1 + k4*r2 + k5*r2**2 + k6*r2**3)

      p_uv[0] = a*u + p1*2*u*v + p2*(r2+2*u**2)
      p_uv[1] = a*v + p1*(r2+2*v**2) + p2*2*u*v
    p_uv[0] = K[0,0] * p_uv[0] + K[0,2]
    p_uv[1] = K[1,1] * p_uv[1] + K[1,2]
    return p_uv.T # 3xN -> Nx3

def ComputeCameraPose(point_3d, point_2d, K, distCoeffs=None, with_refine=False):
  """
  ITERATIVE: Iterative method is based on Levenberg-Marquardt optimization
  DLS      : Method based-on "A Direct Least-Squares (DLS) Method for PnP".
  EPNP     : Method based-on "EPnP: Efficient Perspective-n-Point Camera Pose Estimation".
  P3P      : Method based-on "Complete Solution Classification for the Perspective-Three-Point Problem".
             Requires exactly four object and image points
  UPNP     : Method based-on "Exhaustive Linearization for Robust Camera Pose and Focal Length Estimation".
             The focal length is also estimated and updated in K, assuming f_x = f_y
  """
  # formats the input points from Nx3 to Nx1x3
  point_3d = point_3d[:,np.newaxis,:]
  point_2d = point_2d[:,np.newaxis,:]

  retval, rvec, tvec = cv2.solvePnP(point_3d,  point_2d,
                                    K, distCoeffs,
                                    None, None, True,
                                    cv2.SOLVEPNP_EPNP)
  if with_refine:
    retval, rvec, tvec = cv2.solvePnP(point_3d, point_2d,
                                      K, distCoeffs,
                                      rvec, tvec, True,
                                      cv2.SOLVEPNP_ITERATIVE)
  T = np.eye(4)
  T[:3,3] = tvec.ravel()
  cv2.Rodrigues(rvec, T[:3,:3], jacobian=None)
  return T

def MakeK(fx,fy,cx,cy):
  return np.array([[fx, 0, cx],
                   [0, fy, cy],
                   [0,  0,  1]])


class CameraCalibrationProblem(object):

  class Camera(object):

    class SnapShot(object):

      def __init__(self, p_img, p_obj, img=None):
        assert np.shape(p_img)[1]==2    # Nx2
        assert np.shape(p_obj)[1]==3    # Nx3
        assert len(p_img) == len(p_obj)
        self.p_img = p_img
        self.p_obj = p_obj
        self.img   = img
        self.pose_qt = None
        self.p_img_est = [None]*len(p_img)

      @property
      def Tcw(self):
        if self.pose_qt is None:
          raise RuntimeError('pose not known yet')
        return QTParameterization.ToM(self.pose_qt)

      def CalculateInitialPose(self, K, dist=None):
        Tcw = ComputeCameraPose(self.p_obj, self.p_img, K, dist, with_refine=True)
        self.pose_qt = QTParameterization.FromM(Tcw)
        return self

      def Plot(self, K, dist=None, ax=None):
        if self.img is None:
          raise RuntimeError("no image supplied")

        if ax is None:
          f, ax = plt.subplots()

        p_img_est = ProjectPoints(self.p_obj, self.Tcw, K, dist)

        ax.imshow(self.img)
        ax.plot(self.p_img[:,0], self.p_img[:,1],'r.')
        ax.plot( p_img_est[:,0],  p_img_est[:,1],'b.')
        return self

    # ----------------------- Camera ----------------------------
    def __init__(self, K, dist):
      self.fx, self.fy, self.cx, self.cy = np.atleast_1d(K[0,0], K[1,1], K[0,2], K[1,2])
      self.dist = np.zeros(8)
      self.dist[:len(dist)] = dist
      self.snapshots = []
    @property
    def K(self):
      K = np.eye(3)
      K[0,0], K[1,1], K[0,2], K[1,2] = self.fx, self.fy, self.cx, self.cy
      return K

    def AddSnapShot(self, p2d, p3d, img=None):
      self.snapshots.append(self.SnapShot(p2d, p3d, img))
      return self

    def AddToProblem(self, problem, fix_3d_point=True, fix_f=True, fix_dist=True):
      #define constraint function
      def G(fx, fy, cx, cy, dist, qt_cw, p_uv, p_xyz):
        T = MfromQT(qt_cw[:4], qt_cw[4:] )
        p_uv_est  = ProjectPoints(p_xyz, T, MakeK(fx,fy,cx,cy), dist).ravel()
        return p_uv_est - p_uv

      fxfycxcy_id, _ = problem.AddParameter([self.fx, self.fy, self.cx, self.cy])
      dist_id, _ = problem.AddParameter([self.dist])

      if fix_f:
        problem.SetVarFixedWithID(fxfycxcy_id[0], 0)
        problem.SetVarFixedWithID(fxfycxcy_id[1], 0)
      if fix_dist:
        problem.SetVarFixedWithID(dist_id[0], 0)

      # for each image
      for shot in self.snapshots:
        if shot.pose_qt is None:
          shot.CalculateInitialPose(self.K, self.dist)
        qt_id, _ = problem.AddParameter([shot.pose_qt])
        problem.SetParameterizationWithID(qt_id[0], QTParameterization())

        # for each point
        num_pts = len(shot.p_img)
        for i in xrange(num_pts):
          pt_id, arr = problem.AddObservation([shot.p_img[i,:], shot.p_obj[i,:]])
          shot.p_img_est[i] = arr[0]
          if fix_3d_point:
            problem.SetVarFixedWithID(pt_id[1], 1) # set 3d point fixed

          problem.AddConstraintWithID(G, fxfycxcy_id + dist_id + qt_id, pt_id)

      return problem

    def Plot(self, ax=None, ind=0):
      if ax is None:
        num_shots = len(self.snapshots)
        self.s = Scroller(self.Plot, num_shots)
      else:
        ax.cla()
        self.snapshots[ind].Plot(self.K, self.dist, ax)
      return self

    def Save(self, filename):
      Camera(self.K, self.dist, self.snapshots[0].img.shape[:2]).Save(filename)
      return self

  # ----------------------- PnPProblem ----------------------------
  def __init__(self):
    self.cameras = []

  def AddCamera(self, K, dist=[]):
    self.cameras.append(self.Camera(K, dist))
    return self.cameras[-1]

  def AddObservations(self, p2d, p3d, img, which=-1):
    self.cameras[which].AddSnapShot(p2d, p3d, img)
    return self.cameras[which].snapshots[-1]

  def BuildProblem(self, fix_3d_point=True, fix_f=True, fix_dist=True):
    # build problem
    problem = GaussHelmertProblem()
    for cam in self.cameras:
      cam.AddToProblem(problem, fix_3d_point, fix_f, fix_dist)
    return problem



#%%
if __name__ == '__main__':

  import glob  #Unix style pathname pattern expansion

  if 1: # calbibration for bumblebee
    base_path = '/mnt/workbench/2018icra/calib/'

    # termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((6*8,3), np.float32)
    objp[:,:2] = np.mgrid[0:8,0:6].T.reshape(-1,2)

    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.
    imagefiles = glob.glob(base_path+'left-*.png')
    images    = []

    for fname in imagefiles:
      img = cv2.imread(fname)
      gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

      # Find the chess board corners
      ret, corners = cv2.findChessboardCorners(gray, (8,6),None)

      # If found, add object points, image points (after refining them)
      if ret == True:
        images.append(gray)
        objpoints.append(objp)
    #    cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
        imgpoints.append( np.squeeze(corners) )

        # Draw and display the corners
        cv2.drawChessboardCorners(img, (8,6), corners,ret)
        cv2.imshow('img',img)
        cv2.waitKey(500)
      else:
        print("image %s failed"% fname)
    cv2.destroyAllWindows()

    K = np.array([[ 548.213,    0.   ,  536.257],
                  [   0.   ,  547.499,  397.527],
                  [   0.   ,    0.   ,    1.   ]])
    dist = np.array([  1.402e-01, -6.360e-02,  4.821e-04, -2.870e-04, 1.917e-02,
                       5.421e-01, -1.272e-01,  3.468e-02]) # k1,k2,p1,p2,k3,k4,k5,k6

    cal = CameraCalibrationProblem()
    cam1 = cal.AddCamera(K, dist)
    for img,p2d,p3d in zip(images, imgpoints, objpoints):
      cal.AddObservations( p2d, p3d, img)
    problem = cal.BuildProblem(fix_f=False, fix_dist=False)
    x, lc = SolveWithGESparse(problem)
    problem.UpdateXL()
    cam1.Plot()
    cam1.Save(base_path+"bumblebee_11123872_left.pkl")

    #
    cam = Camera.Load(base_path+"bumblebee_11123872_left.pkl")
    def ShowRectImage(ax,ind):
      ax.cla()
      ax.imshow(cam(images[ind]))
    s = Scroller(ShowRectImage, len(images))

