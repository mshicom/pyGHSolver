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


import sys
sys.path.append('/home/nubot/data/workspace/hand-eye/')
from batch_optimize import *

import pycppad
from solver import *

K = np.array([[100, 0,   250],
              [0,   100, 250],
              [0,   0,      1]],'d')
K_inv = np.linalg.inv(K)

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

# generate camera pose
num_pose = 2
#Mcw = GenerateCameraPoseOnSphere(num_pose, 2)
Mcw = [MfromRT(np.array([0, r,0],'d') , np.array([0,0,2],'d')) for r in np.linspace(0, 0.5*np.pi, num_pose) ]

# generate point cloud
num_point = 10
Pw = 2*np.random.rand(3, num_point)-1
Pw /= np.maximum(1, np.linalg.norm(Pw, axis=0)) # within unit sphere

fig = plt.figure(figsize=(11,11) )
ax = fig.add_subplot(111, projection='3d')
ax.scatter(Pw[0],Pw[1],Pw[2], 'r' )
ax.set_xlim3d(-3,3)
ax.set_ylim3d(-3,3)
ax.set_zlim3d(-3,3)

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


DrawCamera( [invT(M) for M in Mcw] , 0.2)

p2d_true = [Project3Dto2D(Pw, M) for M in Mcw]
p2d_noisy = [p2d + 0.5*np.random.randn(2, num_point)  for p2d in p2d_true]

#%% 2D-2D ego motion
def EpipolarConstraint(r12, t12, p1x, p1y, p2x, p2y):
  E = skew(t12).dot( ax2Rot(r12) )
  F = K_inv.T.dot(E).dot(K_inv)
  P1 = np.vstack( [ p1x, p1y, np.ones( len(p1x) ) ] )
  P2 = np.vstack( [ p2x, p2y, np.ones( len(p2x) ) ] )
  err = [ p1.dot(F).dot(p2) for p1, p2 in zip(P1.T, P2.T)]
  return np.hstack(err)

def test_EpipolarConstraint():
  e = EpipolarConstraint(dr_true[0], dt_true[0],
                         p2d_true[0][0], p2d_true[0][1], p2d_true[1][0], p2d_true[1][1] )
  x_list = [ dr_true[0].copy(), dt_true[0].copy() ]
  l_list = [ p2d_noisy[0][0].copy(), p2d_noisy[0][1].copy(),
                                 p2d_noisy[1][0].copy(), p2d_noisy[1][1].copy()]
  f = GenerateJacobianFunction(EpipolarConstraint, x_list, l_list)
  Js = f( np.hstack(x_list),  np.hstack(l_list) )

def GenErrorFuncFor3DBatchRelativeProjection(Pw, Mcw):
  Pc = Transform(Pw, Mcw)
  def RelativeProjectionConstraint(r12, t12, p2x, p2y):
    P2 = ax2Rot(r12).T.dot( Pc - t12[:, np.newaxis] )
    p2 = K.dot( P2 )
    p2 = p2[:2]/p2[2]
    err = [ p2[0] - p2x, p2[1] - p2y ]
    return np.hstack(err)
  return RelativeProjectionConstraint
def test_RelativeProjectionConstraint():
  M12ProjectionConstraint = GenErrorFuncFor3DBatchRelativeProjection(Pw, Mcw[0])
  M12ProjectionConstraint(dr_true[0], dt_true[0], p2d_true[1][0], p2d_true[1][1])

# test data
loop_closing = 0
sigma = 1

dM = [ Ms.dot( invT(Me) ) for Ms,Me in zip(Mcw[:-1], Mcw[1:])] # Msw_we
if loop_closing:
  dM.append( Mcw[-1].dot( invT( Mcw[0] ) ) ) # loop closing
dr_true = [ Rot2ax(M[:3,:3])  for M in dM  ]
dt_true = [        M[:3, 3]   for M in dM  ]


sigma_null = []
for it in range(1):
  p2d_noisy_x = [p2d[0] + sigma*np.random.randn(num_point)  for p2d in p2d_true]
  p2d_noisy_y = [p2d[1] + sigma*np.random.randn(num_point)  for p2d in p2d_true]
  dr_noisy = [ r + 0.1*np.random.randn(3)  for r in dr_true  ]
  dt_noisy = [ t + 0.2*np.random.randn(3)  for t in dt_true  ]

  problem = GaussHelmertProblem()

  M01ProjectionConstraint = GenErrorFuncFor3DBatchRelativeProjection(Pw, Mcw[0])
  problem.AddConstraintUsingAD( M01ProjectionConstraint,
                                [ dr_noisy[0],       dt_noisy[0] ],
                                [ p2d_noisy_x[1],    p2d_noisy_y[1] ],
                                [ np.diag( np.full(num_point, 1./sigma**2 ) ) ]*2 )

#  for i in range( len(dM) - loop_closing ):
#    problem.AddConstraintUsingAD( EpipolarConstraint,
#                                 [ dr_noisy[i],       dt_noisy[i] ],
#                                 [ p2d_noisy_x[i  ], p2d_noisy_y[i  ],
#                                   p2d_noisy_x[i+1], p2d_noisy_y[i+1]],
#                                 [ np.diag( np.full(num_point, 1./sigma**2 ) ) ]*4 )
  if loop_closing:
    problem.AddConstraintUsingAD( EpipolarConstraint,
                                 [ dr_noisy[-1],       dt_noisy[-1] ],
                                 [ p2d_noisy_x[-1],   p2d_noisy_y[-1],
                                   p2d_noisy_x[0 ],   p2d_noisy_y[0 ]],
                                 [ np.diag( np.full(num_point, 1./sigma**2 ) ) ]*4 )

  x, le = problem.SolveGaussEliminateDense()
  sigma_null.append( problem.variance_factor )
plt.figure()
plt.hist( sigma_null )
dr_true
Mest = [ MfromRT(x_[:3], x_[3:]) for x_ in np.split(x,  len(dM)) ]

#%% 3D-2D absolute pose
def Project3Dto2D_rt(Pw, r_cw, t_cw):
  Pc = ax2Rot(r_cw).dot(Pw) + t_cw[:, np.newaxis]
  p = K.dot(Pc)
  return p[:2]/p[2]

if 0:
  r_true  = [ Rot2ax(M[:3,:3])  for M in Mcw  ]
  t_true  = [        M[:3, 3]   for M in Mcw  ]
  r_noisy = [ r + 0.05*np.random.randn(3)  for r in r_true  ]
  t_noisy = [ t + 0.05*np.random.randn(3)  for t in t_true  ]

  def GenErrorFuncFor3DBatchProjection(P3d):
    def ProjectError(r, t, p2d_x, p2d_y):
      p2d_predict = Project3Dto2D_rt(P3d, r, t)
      err_x = p2d_x - p2d_predict[0,:]
      err_y = p2d_y - p2d_predict[1,:]
      return np.hstack([err_x,err_y])
    return ProjectError

  BatchProjection3D = GenErrorFuncFor3DBatchProjection(Pw)
  e = BatchProjection3D(r_true[0], t_true[0], p2d_true[0][0], p2d_true[0][1])

  #
  problem = GaussHelmertProblem()
  for i in range(num_pose):
    problem.AddConstraintUsingAD( BatchProjection3D,
                                 [ r_noisy[i].copy(), t_noisy[i].copy() ],
                                 [ p2d_noisy[i][0].copy(), p2d_noisy[i][1].copy() ] )
  x, le = problem.SolveGaussEliminateDense()
  print MfromRT(x[:3], x[3:6])
  print Mcw[0]
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