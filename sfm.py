#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Thu May 25 22:41:30 2017

@author: nubot
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D, art3d
#from matplotlib.path import Path
#from matplotlib.patches import PathPatch
from contracts import contract,new_contract


import sys
sys.path.append('/home/nubot/data/workspace/hand-eye/')
#from batch_optimize import *

import pycppad
from solver2 import *

K = np.array([[100, 0,   250],
              [0,   100, 250],
              [0,   0,      1]],'d')
K_inv = np.linalg.inv(K)
def skew(v):
    return np.array([[   0, -v[2],  v[1]],
                     [ v[2],    0, -v[0]],
                     [-v[1], v[0],    0 ]])

def ax2Rot(r):
    p = np.linalg.norm(r)
    if np.abs(p) < 1e-12:
        return np.eye(3)
    else:
        S = skew(r/p)
        return np.eye(3) + np.sin(p)*S + (1.0-np.cos(p))*S.dot(S)

def Rot2ax(R):
    tr = np.trace(R)
    a  = np.array( [R[2,1]-R[1,2], R[0,2]-R[2,0], R[1,0]-R[0,1]] )
    an = np.linalg.norm(a)
    phi= np.arctan2(an, tr-1)
    if np.abs(phi) < 1e-12:
        return np.zeros(3,'d')
    else:
        return phi/an*a

def MfromRT(r,t):
    T = np.eye(4)
    T[:3,:3] = ax2Rot(r)
    T[:3, 3] = t
    return T

def invT(T):
    R, t = T[:3, :3], T[:3, 3]
    Ti = np.eye(4, dtype='d')
    Ti[:3, :3] = R.T
    Ti[:3, 3]  = -R.T.dot(t)
    return Ti

def Transform(P1, M21):
  P2 = M21[:3,:3].dot(P1) + M21[:3,3][:, np.newaxis]
  return P2

def TransformInv(P1, M12):
  P2 = M12[:3,:3].T.dot(P1 - M21[:3,3][:, np.newaxis])
  return P2

def TransformRT(P1, r21, t21):
  P2 = ax2Rot(r21).dot(P1) + t21[:, np.newaxis]
  return P2

def Project3Dto2D(Pw, Mcw):
  pc = K.dot( Transform(Pw, Mcw) )
  return pc[:2]/pc[2]

def lookAt(eyePoint, targetPoint=None, upVector=None, isInverted=False):
  if targetPoint is None:
    targetPoint = np.array([0, 0, 0],'d')
  if upVector is None:
    upVector = np.array([0, 1, 0],'d')

  # step one: generate a rotation matrix
  z = targetPoint - eyePoint
  if np.all(z==0):
      z = np.array([0,0,1])
  x = np.cross(upVector, z)   # cross product
  y = np.cross(z, x)   # cross product

  normalize = lambda x: x/np.linalg.norm(x)
  x = normalize(x)
  y = normalize(y)
  z = normalize(z)
  eRo =  np.vstack([x, y, z])

  eMo = np.identity(4,dtype='f')
  eMo[0:3,0:3] = eRo
  eMo[0:3,3] = -eRo.dot(eyePoint)
  return eMo if not isInverted else np.linalg.inv(eMo)

def GenerateCameraPoseOnSphere(num_pose, radius):
  eyePoint = 2*np.random.rand(3, num_pose) - 1
  eyePoint /= np.linalg.norm(eyePoint, axis=0)/radius # on sphere of radius 2
  return [lookAt(p) for p in eyePoint.T]



def DrawCamera(Twc, scale=0.1):
  vertex = np.array([[ 0, 0, 0],
                     [-2,-1.5, 1],
                     [ 2,-1.5, 1],
                     [ 2, 1.5, 1],
                     [-2, 1.5, 1]],'d').T * scale
  if not isinstance(Twc, list):
    Twc = [Twc]

  for T in Twc:
    v = T[:3,:3].dot(vertex) + T[:3,3].reshape((-1, 1))
    p0,p1,p2,p3,p4 = v.T

    line1_data = np.c_[p1,p2,p3,p4,p1,p0,p3]
    line2_data = np.c_[p2,p0,p4]
    plt.gca(projection='3d').plot(line1_data[0], line1_data[1], line1_data[2], 'b')
    plt.gca(projection='3d').plot(line2_data[0], line2_data[1], line2_data[2], 'b')

#new_contract('Camera', lambda obj: isinstance(obj, Camera))
#new_contract('KeyFrame', lambda obj: isinstance(obj, KeyFrame))
#new_contract('Mappoint', lambda obj: isinstance(obj, Mappoint))
import networkx as nx
from itertools import product
class SLAMSystem(object):
  def __init__(self, K, baseline=None):
    self.graph = nx.Graph()
    self.cam = Camera(K) if baseline is None else StereoCamera(K,baseline)

  @property
  def KFs(self):
    for v in self.graph:
      if isinstance(v, KeyFrame):
        yield v

  @property
  def MPs(self):
    for v in self.graph:
      if isinstance(v, Mappoint):
        yield v

  def NewKF(self, **kwargs):
    new_kf = KeyFrame( cam=self.cam, **kwargs)
    self.graph.add_node(new_kf)
    return new_kf

  def NewMP(self, **kwargs):
    new_mp = Mappoint(**kwargs)
    self.graph.add_node(new_mp)
    return new_mp

  def Simulation(self, num_point = 100, num_pose=10, radius=2):
    for r in np.linspace(0, 1.6*np.pi, num_pose):
      self.NewKF(t_cw=np.array([0,0,radius],'d'), r_cw=np.array([0, r,0],'d'))

    for i in xrange(num_point):
      self.NewMP(xyz_w = 2*np.random.rand(3)-1 )


  def Draw(self):
    fig = plt.figure(figsize=(11,11), num='slam' )
    ax = fig.add_subplot(111, projection='3d')
    Pw = np.vstack(mp.xyz_w for mp in self.MPs).T
    ax.scatter(Pw[0],Pw[1],Pw[2], 'r' )
    ax.set_xlim3d(-3,3)
    ax.set_ylim3d(-3,3)
    ax.set_zlim3d(-3,3)

    M = [invT(MfromRT(kf.r_cw, kf.t_cw)) for kf in self.KFs ]
    DrawCamera(M)



class Mappoint(object):
  def __init__(self, id=None, xyz_w=None):
    self.id = id
    self.xyz_w = xyz_w if not xyz_w is None else np.empty(3,'d')

  def __repr__(self):
    return "Mappoint(%s:%s)" % (self.id, self.xyz_w)


class KeyFrame(object):
  def __init__(self, id=None, cam=None, t_cw=None, r_cw=None):
    self.cam  = cam
    self.t_cw = t_cw if not t_cw is None else np.empty(3,'d')
    self.r_cw = r_cw if not r_cw is None else np.empty(3,'d')
    self.id = id

  def __repr__(self):
    return "KeyFrame(%s:%s,%s)" % (self.id, self.t_cw, self.r_cw)

  @contract(xyz_w='array[3] | array[3xN] | list[3](number)')
  def Project(self, xyz_w):
    xyz_w = np.atleast_2d(xyz_w)
    if xyz_w.shape[0] != 3:
      xyz_w = xyz_w.T

    xyz_c = ax2Rot(self.r_cw).dot(xyz_w) + self.t_cw.reshape(3,1)
    return self.cam.Project(xyz_c)

  @contract(u='array[N]', v='array[N]',returns='array[3xN]')
  def BackProject(self, u, v):
    xyz_c = self.cam.BackProject(u,v)
    xyz_w = ax2Rot(self.r_cw).T.dot(xyz_c) - self.t_cw.reshape(3,1)
    return xyz_w

class Camera(object):
  @contract(K='array[3x3]')
  def __init__(self, K):
    self.K = K.copy()
    self.fx = K[0,0]
    self.fy = K[1,1]
    self.cx = K[0,2]
    self.cy = K[1,2]
    self.Kinv = np.linalg.inv(K)

  @contract(p='array[3] | array[3xN] | list[3](number)')
  def Project(self, p):
    p = np.atleast_2d(p)
    if p.shape[0] != 3:
      p = p.T

    u = self.fx * p[0]/p[2] + self.cx
    v = self.fy * p[1]/p[2] + self.cy
    return u, v

  @contract(u='array[N]', v='array[N]',returns='array[3xN]')
  def BackProject(self, u,v):
    xyz = np.ones((3, u.shape[0]),'d')
    xyz[0] = (u - self.cx)/self.fx
    xyz[1] = (v - self.cy)/self.fy
    return xyz

class StereoCamera(Camera):
  @contract(K='array[3x3]')
  def __init__(self, K, baseline):
    super(StereoCamera, self).__init__(K)
    self.baseline = baseline
    self.bf = baseline*self.fx

  @contract(p='array[3] | array[3xN] | list[3](number)')
  def Project(self, p):
    u,v = super(StereoCamera, self).Project(p)
    du = self.bf/p[2] # du=u_l - u_r and  du/fx=baseline/Z
    return u, v, du

  @contract(u='array[N]', v='array[N]',returns='array[3xN]')
  def BackProject(self, u, v, du):
    xyz = super(StereoCamera, self).BackProject(u, v)
    xyz *= self.bf/du
    return xyz

K = np.array([[100, 0,   250],
             [0,   100, 250],
             [0,   0,      1]],'d')
baseline = 0.5
slam = SLAMSystem(K, baseline)
slam.Simulation(10,3)
slam.Draw()
#%%
def ProjectError(kf_r_cw, kf_t_cw, mp_xyz_w, kf_u, kf_v):
  Pc = ax2Rot(kf_r_cw).dot(mp_xyz_w) + kf_t_cw
  p  = K.dot(Pc)
  uv_predict = p[:2]/p[2]
  err_u = kf_u - uv_predict[0]
  err_v = kf_v - uv_predict[1]
  return np.hstack([err_u,err_v])

if 0:
  problem = GaussHelmertProblem()
  for kf, mp in product(slam.KFs, slam.MPs):
    if kf.id is None:
      kf.id, _ = problem.AddParameter([kf.r_cw, kf.t_cw])

    if mp.id is None:
      mp.id, _ = problem.AddParameter([mp.xyz_w])
    u,v = kf.Project(mp.xyz_w)
    uv_id, _ = problem.AddObservation([u,v])
    slam.graph.add_edge(kf, mp, obs_u=u, obs_v=v)

    problem.AddConstraintWithKnownBlocks(ProjectError, kf.id + mp.id, uv_id)
  problem.SetVarFixed(kf.r_cw)
  problem.SetVarFixed(kf.t_cw)
  x,le,cov = SolveWithCVX(problem, cov=True)

#%%
def StereoProjectError(kf_r_cw, kf_t_cw, mp_xyz_w, kf_u, kf_v, kf_du):
  Pc = ax2Rot(kf_r_cw).dot(mp_xyz_w) + kf_t_cw
  p  = K.dot(Pc)
  uv_predict = p[:2]/p[2]
  err_u = kf_u - uv_predict[0]
  err_v = kf_v - uv_predict[1]
  err_du= kf_du - baseline*K[0,0]/Pc[2]
  return np.hstack([err_u,err_v,err_du])
if 0:
  problem = GaussHelmertProblem()
  for kf, mp in product(slam.KFs, slam.MPs):
    if kf.id is None:
      kf.id, _ = problem.AddParameter([kf.r_cw, kf.t_cw])

    if mp.id is None:
      mp.id, _ = problem.AddParameter([mp.xyz_w])
    u,v,du = kf.Project(mp.xyz_w)
    uvd_id, _ = problem.AddObservation([u,v,du])
    slam.graph.add_edge(kf, mp, obs_u=u, obs_v=v, obs_du=du)

    problem.AddConstraintWithKnownBlocks(StereoProjectError, kf.id + mp.id, uvd_id)
  problem.SetVarFixed(kf.r_cw)
  problem.SetVarFixed(kf.t_cw)
  x,le,cov = SolveWithCVX(problem, cov=True)
#%% 2D-2D ego motion
#def EpipolarConstraint(r12, t12, p1x, p1y, p2x, p2y):
#  E = skew(t12).dot( ax2Rot(r12) )
#  F = K_inv.T.dot(E).dot(K_inv)
#  P1 = np.vstack( [ p1x, p1y, np.ones( len(p1x) ) ] )
#  P2 = np.vstack( [ p2x, p2y, np.ones( len(p2x) ) ] )
#  err = [ p1.dot(F).dot(p2) for p1, p2 in zip(P1.T, P2.T)]
#  return np.hstack(err)
#
#def test_EpipolarConstraint():
#  e = EpipolarConstraint(dr_true[0], dt_true[0],
#                         p2d_true[0][0], p2d_true[0][1], p2d_true[1][0], p2d_true[1][1] )
#  x_list = [ dr_true[0].copy(), dt_true[0].copy() ]
#  l_list = [ p2d_noisy[0][0].copy(), p2d_noisy[0][1].copy(),
#                                 p2d_noisy[1][0].copy(), p2d_noisy[1][1].copy()]
#  f = GenerateJacobianFunction(EpipolarConstraint, x_list, l_list)
#  Js = f( np.hstack(x_list),  np.hstack(l_list) )
#
#def GenErrorFuncFor3DBatchRelativeProjection(Pw, Mcw):
#  Pc = Transform(Pw, Mcw)
#  def RelativeProjectionConstraint(r12, t12, p2x, p2y):
#    P2 = ax2Rot(r12).T.dot( Pc - t12[:, np.newaxis] )
#    p2 = K.dot( P2 )
#    p2 = p2[:2]/p2[2]
#    err = [ p2[0] - p2x, p2[1] - p2y ]
#    return np.hstack(err)
#  return RelativeProjectionConstraint
#def test_RelativeProjectionConstraint():
#  M12ProjectionConstraint = GenErrorFuncFor3DBatchRelativeProjection(Pw, Mcw[0])
#  M12ProjectionConstraint(dr_true[0], dt_true[0], p2d_true[1][0], p2d_true[1][1])
#
## test data
#loop_closing = 0
#sigma = 0.5
#
#dM = [ Ms.dot( invT(Me) ) for Ms,Me in zip(Mcw[:-1], Mcw[1:])] # Msw_we
#if loop_closing:
#  dM.append( Mcw[-1].dot( invT( Mcw[0] ) ) ) # loop closing
#dr_true = [ Rot2ax(M[:3,:3])  for M in dM  ]
#dt_true = [        M[:3, 3]   for M in dM  ]
#
#
#sigma_null = []
#for it in range(1):
#  p2d_noisy_x = [p2d[0] + sigma*np.random.randn(num_point)  for p2d in p2d_true]
#  p2d_noisy_y = [p2d[1] + sigma*np.random.randn(num_point)  for p2d in p2d_true]
#
#
#  dr_noisy = [ r + 0.1*np.random.randn(3)  for r in dr_true  ]
#  dt_noisy = [ t + 0.2*np.random.randn(3)  for t in dt_true  ]

#  problem = GaussHelmertProblem()
#
#  M01ProjectionConstraint = GenErrorFuncFor3DBatchRelativeProjection(Pw, Mcw[0])
#  problem.AddConstraintUsingAD( M01ProjectionConstraint,
#                                [ dr_noisy[0],       dt_noisy[0] ],
#                                [ p2d_noisy_x[1],    p2d_noisy_y[1] ])
#  problem.SetSigma(p2d_noisy_x[1], S_p2d)
#  problem.SetSigma(p2d_noisy_y[1], S_p2d)

#  for i in range( len(dM) - loop_closing ):
#    problem.AddConstraintUsingAD( EpipolarConstraint,
#                                 [ dr_noisy[i],       dt_noisy[i] ],
#                                 [ p2d_noisy_x[i  ], p2d_noisy_y[i  ],
#                                   p2d_noisy_x[i+1], p2d_noisy_y[i+1]],
#                                 [ np.diag( np.full(num_point, 1./sigma**2 ) ) ]*4 )
#  if loop_closing:
#    problem.AddConstraintUsingAD( EpipolarConstraint,
#                                 [ dr_noisy[-1],       dt_noisy[-1] ],
#                                 [ p2d_noisy_x[-1],   p2d_noisy_y[-1],
#                                   p2d_noisy_x[0 ],   p2d_noisy_y[0 ]],
#                                 [ np.diag( np.full(num_point, 1./sigma**2 ) ) ]*4 )
#
#  x, le = problem.SolveGaussEliminateDense()
#  sigma_null.append( problem.variance_factor )
#plt.figure()
#plt.hist( sigma_null )
#dr_true
#Mest = [ MfromRT(x_[:3], x_[3:]) for x_ in np.split(x,  len(dM)) ]


#%% stereo
def GenErrorFuncForStereoConstraint(M12, K):
  R12, t12 = M12[:3,:3], M12[:3,3][:, np.newaxis]
  Kinv = np.linalg.inv(K)
  def StereoConstraint(d1, p1x, p1y, p2x, p2y):
    p1  = Kinv.dot( np.vstack([d1*p1x, d1*p1y, d1]) )
    p21 = K.dot( R12.T.dot(p1 - t12) )
    p21 = p21[:2]/p21[2]
    err_x = p2x - p21[0,:]
    err_y = p2y - p21[1,:]
    return np.hstack([err_x,err_y])
  return StereoConstraint

def StereoMotionConstraint(d1, r12, t12, p1x, p1y, p2x, p2y):
  p1  = K_inv.dot( np.vstack([d1*p1x, d1*p1y, d1]) )
  p21 = K.dot( ax2Rot(r12).T.dot(p1 - t12[:, np.newaxis]) )
  p21 = p21[:2]/p21[2]
  err_x = p2x - p21[0,:]
  err_y = p2y - p21[1,:]
  return np.hstack([err_x, err_y])

if 0:
  Msc = MfromRT(np.zeros(3), np.r_[-2.5, 0, 0])
  d =  [ Transform(Pw, M)[2].copy() for M in Mcw]

  p2d_stereo = [Project3Dto2D(Pw, Msc.dot(M) ) for M in Mcw]
  p2d_stereo_noisy_x = [p2d[0] + sigma*np.random.randn(num_point)  for p2d in p2d_stereo]
  p2d_stereo_noisy_y = [p2d[1] + sigma*np.random.randn(num_point)  for p2d in p2d_stereo]

  StereoConstraint = GenErrorFuncForStereoConstraint(Msc, K)
  StereoConstraint(d[0], p2d_stereo[0][0], p2d_stereo[0][1], p2d_true[0][0], p2d_true[0][1])

  problem = GaussHelmertProblem()

  for i in range( len(dM)):
    problem.AddConstraintUsingAD( StereoConstraint,
                               [ d[i] ],
                               [ p2d_stereo_noisy_x[i],   p2d_stereo_noisy_y[i],
                                 p2d_noisy_x[i],     p2d_noisy_y[i]],
                               [ np.diag( np.full(num_point, 1./sigma**2 ) ) ]*4 )

    problem.AddConstraintUsingAD( StereoMotionConstraint,
                                 [ d[i], dr_noisy[i],    dt_noisy[i] ],
                                 [ p2d_noisy_x[i  ], p2d_noisy_y[i  ],
                                   p2d_noisy_x[i+1], p2d_noisy_y[i+1]],
                                 [ np.diag( np.full(num_point, 1./sigma**2 ) ) ]*4 )

  x, le = problem.SolveGaussEliminateDense()

#%%
if 0:
  import cv2
  kitti_data_dir = '/media/kaihong/2ADA2A32DA29FAA9/work/calibrate/kitti/2011_09_26_drive_0046_sync/image_00/data/'

  img1 = cv2.imread(kitti_data_dir+'0000000000.png',0)          # queryImage
  img2 = cv2.imread(kitti_data_dir+'0000000001.png',0) # trainImage

  # Initiate SIFT detector
  orb = cv2.ORB_create(edgeThreshold=15,
                       patchSize=31,
                       nlevels=8,
                       fastThreshold=20,
                       scaleFactor=1.2,
                       WTA_K=2,
                       scoreType=cv2.ORB_FAST_SCORE,
                       firstLevel=0,
                       nfeatures=1000)

  # find the keypoints and descriptors with orb
  kp1, des1 = orb.detectAndCompute(img1,None)
  kp2, des2 = orb.detectAndCompute(img2,None)
  out_im = cv2.drawKeypoints(img1, kp1, None, color=(0,255,0), flags=0)
  plt.imshow(out_im)

  # create BFMatcher object
  bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

  # Match descriptors.
  matches = bf.match(des1,des2)

  # Sort them in the order of their distance.
  matches = sorted(matches, key = lambda x:x.distance)

  # Draw first 10 matches.
  out_im = cv2.drawMatches(img1,kp1, img2,kp2, matches[:20], None, flags=2)
  plt.figure()
  plt.imshow(out_im)