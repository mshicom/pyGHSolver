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
from contracts import contract,new_contract,disable_all
disable_all()

import sys
sys.path.append('/home/nubot/data/workspace/hand-eye/')
#from batch_optimize import *

import pycppad
from solver2 import *

K = np.array([[100, 0,   250],
              [0,   100, 250],
              [0,   0,      1]],'d')
K_inv = np.linalg.inv(K)

def ax2Rot(r):
  phi = np.linalg.norm(r)
  if np.abs(phi) > 1e-8:
    sinp_div_p             = np.sin(phi)/phi
    one_minus_cos_p_div_pp = (1.0-np.cos(phi))/(phi**2)
  else:
    sinp_div_p             = 1. - phi**2/6.0 + phi**4/120.0
    one_minus_cos_p_div_pp = 0.5 - phi**2/24.0 + phi**4/720.0

  S = np.array([[   0, -r[2],  r[1]],
                [ r[2],    0, -r[0]],
                [-r[1], r[0],    0 ]])

  return np.eye(3) + sinp_div_p*S + one_minus_cos_p_div_pp*S.dot(S)


def Rot2ax(R):
    tr = 0.5*(np.trace(R)-1)
    phi= np.arccos(tr)
    if np.abs(phi) > 1e-8:
      p_div_sinp = phi/np.sin(phi)
    else:
      p_div_sinp = 1 + phi**2 / 6.0 + 7.0/360 * phi**4

    ln = (0.5*p_div_sinp)*(R-R.T)
    return np.array([ln[2,1], ln[0,2], ln[1,0]])


def MfromRT(r,t):
    T = np.eye(4)
    T[:3,:3] = ax2Rot(r)
    T[:3, 3] = t
    return T

def invT(T):
    R, t = T[:3, :3], T[:3, 3]

    if T.dtype==object:
      Ti = pycppad.a_float(1) * np.eye(4, dtype='d')
    else:
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



def DrawCamera(Twc, scale=0.1, color='b'):
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
    plt.gca(projection='3d').plot(line1_data[0], line1_data[1], line1_data[2], color)
    plt.gca(projection='3d').plot(line2_data[0], line2_data[1], line2_data[2], color)

#new_contract('Camera', lambda obj: isinstance(obj, Camera))
#new_contract('KeyFrame', lambda obj: isinstance(obj, KeyFrame))
#new_contract('Mappoint', lambda obj: isinstance(obj, Mappoint))

Mat = SE3Parameterization.Mat44
Vec = SE3Parameterization.Vec12
import networkx as nx
from itertools import product
class SLAMSystem(object):
  def __init__(self, K, baseline=None):
    self.cam = Camera(K) if baseline is None else StereoCamera(K,baseline)
    self.offset = SE3Offset( r_ab=np.array([0.0, 0.5, 0]), t_ab=np.array([0, 0, 1]) )
    self.KFs = []
    self.MPs = []
    self.ODs = []

  def NewKF(self, **kwargs):
    new_kf = KeyFrame( cam=self.cam, **kwargs)
    self.KFs.append(new_kf)
    return new_kf

  def NewMP(self, **kwargs):
    new_mp = Mappoint(**kwargs)
    self.MPs.append(new_mp)
    return new_mp

  def NewOD(self, **kwargs):
    new_od = OdometryFrame(se3_oc=self.offset, **kwargs)
    self.ODs.append(new_od)
    return new_od

  def Simulation(self, num_point = 100, num_pose=10, radius=2):
    for r in np.linspace(0.01, 1.6*np.pi, num_pose):
      t_cw = np.array([0, 0, radius],'d')
      r_cw = np.array([np.random.rand(1), r, np.random.rand(1)],'d')

      kf = self.NewKF(r_cw=r_cw, t_cw=t_cw)
      kf.vT_cw = Vec( MfromRT(r_cw, t_cw) )

      r_ow, t_ow = self.offset.FromB(r_cw, t_cw)
      od = self.NewOD(r_ow=r_ow, t_ow=t_ow)
      od.se3_oc = self.offset

    for i in xrange(num_point):
      self.NewMP(xyz_w = 2*np.random.rand(3)-1 )


  def Draw(self, point=True, cam=True, color='b'):

    fig = plt.figure(figsize=(11,11))
    ax = fig.add_subplot(111, projection='3d')
    if point:
      Pw = np.vstack(mp.xyz_w for mp in self.MPs).T
      ax.scatter(Pw[0],Pw[1],Pw[2])
      ax.set_xlim3d(-3,3)
      ax.set_ylim3d(-3,3)
      ax.set_zlim3d(-3,3)
    if cam:
      M = [invT(MfromRT(kf.r_cw, kf.t_cw)) for kf in self.KFs ]
      DrawCamera(M, color=color)



class Mappoint(object):
  def __init__(self, id=None, xyz_w=None):
    self.id = id
    self.xyz_w = xyz_w if not xyz_w is None else np.empty(3,'d')

  def __repr__(self):
    return "Mappoint(%s:%s)" % (self.id, self.xyz_w)


class KeyFrame(object):
  def __init__(self, id=None, cam=None, r_cw=None, t_cw=None):
    self.cam  = cam
    self.t_cw = t_cw if not t_cw is None else np.empty(3,'d')
    self.r_cw = r_cw if not r_cw is None else np.empty(3,'d')
    self.vT_cw  = Vec(np.eye(4))
    self.id = id

  def __repr__(self):
    return "KeyFrame(%s:%s,%s)" % (self.id, self.t_cw, self.r_cw)

  @contract(xyz_w='array[3] | array[3xN] | list[3](number)')
  def Project(self, xyz_w):
    xyz_w = np.atleast_2d(xyz_w)
    if xyz_w.shape[0] != 3:
      xyz_w = xyz_w.T
    M = Mat(self.vT_cw)
    xyz_c = M[:3,:3].dot(xyz_w) + M[:3,3].reshape(3,1)
#    xyz_c = ax2Rot(self.r_cw).dot(xyz_w) + self.t_cw.reshape(3,1)
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

class SE3Offset(object):
  def __init__(self, r_ab=None, t_ab=None):
    self.id = None
    self.t_ab = t_ab if not t_ab is None else np.zeros(3,'d')
    self.r_ab = r_ab if not r_ab is None else np.zeros(3,'d')
    self.vT_ab = Vec(MfromRT(self.r_ab, self.t_ab))

  def FromB(self, r_bw, t_bw):
    R_ab = ax2Rot(self.r_ab)
    r_aw = Rot2ax( R_ab.dot( ax2Rot(r_bw) ) )
    t_aw = self.t_ab + R_ab.dot(t_bw)
    return r_aw, t_aw

  def FromA(self, r_aw, t_aw):
    R_ba = ax2Rot(self.r_ab).T
    r_bw = Rot2ax( R_ba.dot( ax2Rot(r_aw) ) )
    t_bw = R_ba.dot(t_aw - self.t_ab)
    return r_bw, t_bw

  def SE3FromB(self, vT_bw):
    T_aw = Mat( self.vT_ab ).dot( Mat(vT_bw) )
    return Vec( T_aw )

  def SE3FromA(self, vT_aw):
    T_bw = invT( Mat(self.vT_ab) ).dot( Mat(vT_aw) )
    return Vec( T_bw )

class OdometryFrame(object):
  def __init__(self, se3_oc=None, r_ow=None, t_ow=None):
    self.id = None
    self.t_ow = t_ow if not t_ow is None else np.empty(3,'d')
    self.r_ow = r_ow if not r_ow is None else np.empty(3,'d')
    self.vT_ow = Vec( MfromRT(self.r_ow, self.t_ow) )
    self.se3_oc = se3_oc

  def __repr__(self):
    return "OdometryFrame(%s:%s,%s)" % (self.id, self.t_ow, self.r_ow)
#%
K = np.array([[100, 0,   250],
              [0,   100, 250],
              [0,   0,     1]],'d')
baseline = None#0.5
slam = SLAMSystem(K, baseline)
slam.Simulation(10,3)
#slam.Draw()
#%% ProjectError

if 0:
  def ProjectError(kf_r_cw, kf_t_cw, mp_xyz_w, kf_u, kf_v):
    Pc = ax2Rot(kf_r_cw).dot(mp_xyz_w) + kf_t_cw
    p  = K.dot(Pc)
    uv_predict = p[:2]/p[2]
    err_u = kf_u - uv_predict[0]
    err_v = kf_v - uv_predict[1]
    return np.hstack([err_u,err_v])

  problem = GaussHelmertProblem()
  for kf, mp in product(slam.KFs, slam.MPs):
    if kf.id is None:
      kf.id, _ = problem.AddParameter([kf.r_cw, kf.t_cw])

    if mp.id is None:
      mp.id, _ = problem.AddParameter([mp.xyz_w])
    u,v = kf.Project(mp.xyz_w)
    u += np.random.rand(1)
    v += np.random.rand(1)
    uv_id, _ = problem.AddObservation([u,v])

    problem.AddConstraintWithKnownBlocks(ProjectError, kf.id + mp.id, uv_id)
  problem.SetVarFixed(kf.r_cw)
  problem.SetVarFixed(kf.t_cw)
  x,le,cov = SolveWithGESparse(problem, cov=True)

#%% ProjectErrorSE3
if 0:

  def ProjectErrorSE3(kf_se3, mp_xyz_w, kf_u, kf_v):
    M = Mat(kf_se3)
    Pc = M[:3,:3].dot(mp_xyz_w) + M[:3,3]
    p  = K.dot(Pc)
    uv_predict = p[:2]/p[2]
    err_u = kf_u - uv_predict[0]
    err_v = kf_v - uv_predict[1]
    return np.hstack([err_u,err_v])

  problem = GaussHelmertProblem()
  for kf, mp in product(slam.KFs, slam.MPs):
    if kf.id is None:
      kf.id, _ = problem.AddParameter([kf.vT_cw])
      problem.SetParameterization(kf.vT_cw, SE3Parameterization())

    if mp.id is None:
      mp.id, _ = problem.AddParameter([mp.xyz_w])
    u,v = kf.Project(mp.xyz_w)
    u += np.random.rand(1)
    v += np.random.rand(1)
    uv_id, _ = problem.AddObservation([u,v])

    problem.AddConstraintWithKnownBlocks(ProjectErrorSE3, kf.id + mp.id, uv_id)
  problem.SetVarFixed(kf.vT_cw)
  x,le,fac,cov = SolveWithGESparse(problem, fac=True, cov=True)
  print fac


#%% StereoProjectError
if 0:
  def StereoProjectError(kf_r_cw, kf_t_cw, mp_xyz_w, kf_u, kf_v, kf_du):
    Pc = ax2Rot(kf_r_cw).dot(mp_xyz_w) + kf_t_cw
    p  = K.dot(Pc)
    uv_predict = p[:2]/p[2]
    err_u = kf_u - uv_predict[0]
    err_v = kf_v - uv_predict[1]
    err_du= kf_du - baseline*K[0,0]/Pc[2]
    return np.hstack([err_u,err_v,err_du])

  problem = GaussHelmertProblem()
  for kf, mp in product(slam.KFs, slam.MPs):
    if kf.id is None:
      kf.id, _ = problem.AddParameter([kf.r_cw, kf.t_cw])

    if mp.id is None:
      mp.id, _ = problem.AddParameter([mp.xyz_w])
    u,v,du = kf.Project(mp.xyz_w)
    u += np.random.rand(1)
    v += np.random.rand(1)
    du+= np.random.rand(1)
    uvd_id, _ = problem.AddObservation([u,v,du])

    problem.AddConstraintWithKnownBlocks(StereoProjectError, kf.id + mp.id, uvd_id)
  problem.SetVarFixed(kf.r_cw)
  problem.SetVarFixed(kf.t_cw)
  x,le,cov = SolveWithGESparse(problem, cov=True)

#%% ExtrinsicError
if 0:
  def ExtrinsicError(r_cw, t_cw, r_oc, t_oc, r_ow, t_ow):
    R_co = ax2Rot(r_oc).T
    r_cw_est = Rot2ax( R_co.dot(ax2Rot(r_ow)) )
    t_cw_est = R_co.dot(t_ow - t_oc)
    return np.hstack([t_cw - t_cw_est, r_cw - r_cw_est])

    def test():
      r_oc, t_oc, r_ow, t_ow = np.random.rand(4,3)

      T_cw = invT(MfromRT(r_oc, t_oc)).dot( MfromRT(r_ow, t_ow) )
      r_cw, t_cw = Rot2ax(T_cw[:3,:3]), T_cw[:3,3]

      R_co = ax2Rot(r_oc).T
      r_cw_est = Rot2ax( R_co.dot(ax2Rot(r_ow)) )#R_co.dot(r_ow)#
      t_cw_est = R_co.dot(t_ow - t_oc)
      assert_array_equal( r_cw, r_cw_est )
      assert_array_equal( t_cw, t_cw_est )


  fig = plt.figure(figsize=(11,11), num='cal')
  ax = fig.add_subplot(111, projection='3d')
  Pw = np.vstack(mp.xyz_w for mp in slam.MPs).T
  ax.scatter(Pw[0],Pw[1],Pw[2])
  ax.set_xlim3d(-3,3)
  ax.set_ylim3d(-3,3)
  ax.set_zlim3d(-3,3)
  cam = [invT(MfromRT(kf.r_cw, kf.t_cw)) for kf in slam.KFs ]
  DrawCamera(cam, color='b')
  od = [invT(MfromRT(od.r_ow, od.t_ow)) for od in slam.ODs ]
  DrawCamera(od, color='r')


  problem = GaussHelmertProblem()
  se3_id, se3_vec = problem.AddParameter([slam.offset.r_ab, slam.offset.t_ab])
  for kf, od in zip(slam.KFs, slam.ODs):
    kf.id, _ = problem.AddParameter([kf.r_cw, kf.t_cw])
    od.id, _ = problem.AddObservation([od.r_ow, od.t_ow])
    problem.AddConstraintWithKnownBlocks(ExtrinsicError, kf.id + se3_id, od.id)

  for kf, mp in product(slam.KFs, slam.MPs):
    if mp.id is None:
      mp.id, _ = problem.AddParameter([mp.xyz_w])
    u,v = kf.Project(mp.xyz_w)
    u += np.random.rand(1)
    v += np.random.rand(1)
    uv_id, _ = problem.AddObservation([u,v])
    problem.AddConstraintWithKnownBlocks(ProjectError, kf.id + mp.id, uv_id)

  x,le,fac = SolveWithGESparse(problem, maxit=20, fac=True)
#  problem.ViewJacobianPattern()
  print se3_vec[0].array, se3_vec[1].array
  print slam.offset.r_ab, slam.offset.t_ab
#%% ExtrinsicErrorSE3
if 1:
  def ExtrinsicErrorSE3(vT_cw, vT_oc, vT_ow):
    T_cw_est = invT( Mat(vT_oc) ).dot( Mat(vT_ow) )
    return Vec(T_cw_est) - vT_cw

  fig = plt.figure(figsize=(11,11), num='cal')
  ax = fig.add_subplot(111, projection='3d')
  Pw = np.vstack(mp.xyz_w for mp in slam.MPs).T
  ax.scatter(Pw[0],Pw[1],Pw[2])
  ax.set_xlim3d(-3,3)
  ax.set_ylim3d(-3,3)
  ax.set_zlim3d(-3,3)
  cam = [invT( Mat(kf.vT_cw) ) for kf in slam.KFs ]
  DrawCamera(cam, color='b')
  od = [invT( Mat(od.vT_ow) ) for od in slam.ODs ]
  DrawCamera(od, color='r')


  problem = GaussHelmertProblem()
  se3_id, se3_vec = problem.AddParameter([slam.offset.vT_ab])
  problem.SetParameterization(slam.offset.vT_ab, SE3Parameterization())
  for kf, od in zip(slam.KFs, slam.ODs):
    kf.id, _ = problem.AddParameter([kf.vT_cw])
    problem.SetParameterization(kf.vT_cw, SE3Parameterization())

    od.id, _ = problem.AddObservation([od.vT_ow])
    problem.SetParameterization(od.vT_ow, SE3Parameterization())
    problem.AddConstraintWithKnownBlocks(ExtrinsicErrorSE3, kf.id + se3_id, od.id)

  for kf, mp in product(slam.KFs, slam.MPs):
    if mp.id is None:
      mp.id, _ = problem.AddParameter([mp.xyz_w])
    u,v = kf.Project(mp.xyz_w)
    u += np.random.rand(1)
    v += np.random.rand(1)
    uv_id, _ = problem.AddObservation([u,v])
    problem.AddConstraintWithKnownBlocks(ProjectErrorSE3, kf.id + mp.id, uv_id)

  x,le,fac = SolveWithGESparse(problem, maxit=20, fac=True)
#  problem.ViewJacobianPattern()
  print Mat(se3_vec[0].array)
  print Mat(slam.offset.vT_ab)

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