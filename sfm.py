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

def randsp(n=3):
  v = np.random.uniform(size=n)
  return v/norm(v)

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

def axToRodriguez(r):
  theta_half = 0.5*np.linalg.norm(r)
  return np.tan(theta_half) / theta_half * r

def axFromRodriguez(m):
  norm_half = 0.5*np.linalg.norm(m)
  theta = np.arctan(norm_half)
  return theta/norm_half * m

def axToCayley(r):
  theta = np.linalg.norm(r)
  if theta==0.0:
    return r
  return np.tan(0.5*theta) / theta * r

def axFromCayley(u):
  norm = np.linalg.norm(u)
  if norm==0.0:
    return u
  return 2.0*np.arctan(norm) / norm * u

def fromRodriguez(m):
  norm_half = 0.5*np.linalg.norm(m)
  theta = np.arctan(norm_half)
  return theta/norm_half * m

def axAdd(r1, r2):
#  m1 = toRodriguez(r1)
#  m2 = toRodriguez(r2)
#  m12 = ( 4.0*(m1+m2) + 2.0*skew(m1).dot(m2) ) / (4.0 - m1.dot(m2))
#  return  fromRodriguez(m12)
  u1 = axToCayley(r1)
  u2 = axToCayley(r2)
  u12 = ( u1+u2 + skew(u1).dot(u2) ) / (1 - u1.dot(u2))
  return axFromCayley(u12)

def test():
  r1,r2 = 0.5*np.random.rand(2,3)
  assert_array_almost_equal( Rot2ax( ax2Rot(r1).dot(ax2Rot(r2)) ),
                      axAdd( r1, r2) )

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
def check_magnitude(r):
  if np.linalg.norm(r) > np.pi:
    raise ValueError("rotation magnitude larger than pi, will cause problems during optimizatoin")
  return r

Mat = SE3Parameterization.Mat44
Vec = SE3Parameterization.Vec12
import networkx as nx
from itertools import product
class SLAMSystem(object):
  def __init__(self, K, baseline=None):
    self.cam = Camera(K) if baseline is None else StereoCamera(K,baseline)
    self.offset = SE3Offset( r_ab=np.array([0.0, 0.5, 0.]), t_ab=np.array([0., 0., 1.]) )
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
      t_cw = np.array([0, 0, r+3],'d')
      r_cw = 0.02*np.ones(3) #np.array( 0.9* np.random.rand(1)*np.pi*randsp(),'d')

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
    self.homo_w = np.r_[self.xyz_w, 1.0]

  def __repr__(self):
    return "Mappoint(%s:%s)" % (self.id, self.xyz_w)


class KeyFrame(object):
  def __init__(self, id=None, cam=None, r_cw=None, t_cw=None):
    self.cam  = cam
    self.t_cw = t_cw if not t_cw is None else np.empty(3,'d')
    self.r_cw = check_magnitude(r_cw) if not r_cw is None else np.empty(3,'d')
    self.vT_cw  = Vec(np.eye(4))
    self.id = id

  def __repr__(self):
    return "KeyFrame(%s:%s,%s)" % (self.id, self.t_cw, self.r_cw)

  @contract(xyz_w='array[3] | array[3xN] | list[3](number)')
  def Project(self, xyz_w, noK=False):
    xyz_w = np.atleast_2d(xyz_w)
    if xyz_w.shape[0] != 3:
      xyz_w = xyz_w.T
    M = Mat(self.vT_cw)
    xyz_c = M[:3,:3].dot(xyz_w) + M[:3,3].reshape(3,1)
#    xyz_c = ax2Rot(self.r_cw).dot(xyz_w) + self.t_cw.reshape(3,1)
    if noK:
      return xyz_c[0]/xyz_c[-1], xyz_c[1]/xyz_c[-1]
    else:
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
    self.r_ab = check_magnitude(r_ab) if not r_ab is None else np.empty(3,'d')
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
    self.r_ow = check_magnitude(r_ow) if not r_ow is None else np.empty(3,'d')

    self.vT_ow = Vec( MfromRT(self.r_ow, self.t_ow) )
    self.se3_oc = se3_oc

  def __repr__(self):
    return "OdometryFrame(%s:%s,%s)" % (self.id, self.t_ow, self.r_ow)
#%
K = np.array([[100, 0,   250],
              [0,   100, 250],
              [0,   0,     1]],'d')
baseline = None#0.5#
slam = SLAMSystem(K, baseline)
slam.Simulation(15,2)
#slam.Draw()
#%% ProjectError
def ProjectError(kf_r_cw, kf_t_cw, mp_xyz_w, kf_u, kf_v):
  check_magnitude(kf_r_cw)

  Pc = ax2Rot(kf_r_cw).dot(mp_xyz_w) + kf_t_cw
  p  = K.dot(Pc)
  uv_predict = p[:2]/p[2]
  err_u = kf_u - uv_predict[0]
  err_v = kf_v - uv_predict[1]
  return np.hstack([err_u,err_v])

if 0 and baseline is None:
  fig = plt.figure(figsize=(11,11), num='ba')
  ax = fig.add_subplot(111, projection='3d')
  Pw = np.vstack(mp.xyz_w for mp in slam.MPs).T
  ax.scatter(Pw[0],Pw[1],Pw[2])
  ax.set_xlim3d(-3,3)
  ax.set_ylim3d(-3,3)
  ax.set_zlim3d(-3,3)
  cam = [invT(MfromRT(kf.r_cw, kf.t_cw)) for kf in slam.KFs ]
  DrawCamera(cam, color='b')

  problem = GaussHelmertProblem()
  for kf, mp in product(slam.KFs, slam.MPs):
    if kf.id is None:
      kf.id, _ = problem.AddParameter([kf.r_cw, kf.t_cw])

    if mp.id is None:
      mp.id, _ = problem.AddParameter([mp.xyz_w])
    u,v = kf.Project(mp.xyz_w)
    u += np.random.randn(1)
    v += np.random.randn(1)
    uv_id, _ = problem.AddObservation([u,v])

    problem.AddConstraintWithID(ProjectError, kf.id + mp.id, uv_id)
  problem.SetVarFixed(kf.r_cw)
  problem.SetVarFixed(kf.t_cw)
  x,le,fac = SolveWithGESparse(problem, fac=True)
  print 'variance factor:%f' % fac
  problem.cv_x.OverWriteOrigin()
  cam = [invT(MfromRT(kf.r_cw, kf.t_cw)) for kf in slam.KFs ]
  DrawCamera(cam, color='r')

#%% ProjectErrorSE3
def ProjectErrorSE3(kf_se3, mp_xyz_w, kf_u, kf_v):
  M = Mat(kf_se3)
  Pc = M[:3,:3].dot(mp_xyz_w) + M[:3,3]
  p  = K.dot(Pc)
  uv_predict = p[:2]/p[2]
  err_u = kf_u - uv_predict[0]
  err_v = kf_v - uv_predict[1]
  return np.hstack([err_u,err_v])

if 0 and baseline is None:
  problem = GaussHelmertProblem()
  for kf, mp in product(slam.KFs, slam.MPs):
    if kf.id is None:
      kf.id, _ = problem.AddParameter([kf.vT_cw])
      problem.SetParameterization(kf.vT_cw, SE3Parameterization())

    if mp.id is None:
      mp.id, _ = problem.AddParameter([mp.xyz_w])
    u,v = kf.Project(mp.xyz_w)
    u += np.random.randn(1)
    v += np.random.randn(1)
    uv_id, _ = problem.AddObservation([u,v])

    problem.AddConstraintWithID(ProjectErrorSE3, kf.id + mp.id, uv_id)
  problem.SetVarFixed(kf.vT_cw)
  x,le,fac,cov = SolveWithGESparse(problem, fac=True, cov=True)
  print 'variance factor:%f' % fac

#%% ProjectErrorHomo
h4 = HomogeneousParameterization(4)
s4 = SphereParameterization(4)
h3 = HomogeneousParameterization(3)

#def ProjectErrorHomo(kf_r_cw, kf_t_cw, mp_homo_w, kf_u, kf_v):
#  check_magnitude(kf_r_cw)
#  ray_obs = np.hstack([kf_u, kf_v])
#  ray_est = ax2Rot(kf_r_cw).dot(mp_homo_w[:3]) + kf_t_cw * mp_homo_w[3]
#  return ray_est[:2]/ray_est[-1] - ray_obs

def ProjectErrorHomo(kf_r_cw, kf_t_cw, mp_homo_w, kf_u, kf_v):
  check_magnitude(kf_r_cw)
  ray_obs = np.hstack([kf_u, kf_v, 1.])
  ray_est = ax2Rot(kf_r_cw).dot(mp_homo_w[:3]) + kf_t_cw * mp_homo_w[3]
  return HomoVectorCollinearError(ray_obs, ray_est)

if 1 and baseline is None:
  fig = plt.figure(figsize=(11,11), num='ba')
  ax = fig.add_subplot(111, projection='3d')
  Pw = np.vstack(mp.xyz_w for mp in slam.MPs).T
  ax.scatter(Pw[0],Pw[1],Pw[2])
  ax.set_xlim3d(-3,3)
  ax.set_ylim3d(-3,3)
  ax.set_zlim3d(-3,3)
  cam = [invT(MfromRT(kf.r_cw, kf.t_cw)) for kf in slam.KFs ]
  DrawCamera(cam, color='b')

  problem = GaussHelmertProblem()
  for kf, mp in product(slam.KFs, slam.MPs):
    if kf.id is None:
      kf.id, _ = problem.AddParameter([kf.r_cw, kf.t_cw])

    if mp.id is None:
      mp.id, _ = problem.AddParameter([h4.ToHomoSphere(mp.xyz_w)])
      problem.SetParameterizationWithID(mp.id[0], h4)
    u,v = kf.Project(mp.xyz_w, True)
    u += 0.1*np.random.randn(1)
    v += 0.1*np.random.randn(1)
    uv_id, _ = problem.AddObservation([u,v])

    problem.AddConstraintWithID(ProjectErrorHomo, kf.id + mp.id, uv_id)
  problem.SetVarFixed(kf.r_cw)
  problem.SetVarFixed(kf.t_cw)
  x,le,fac = SolveWithGESparse(problem, maxit=20, fac=True)

  problem.cv_x.OverWriteOrigin()
  cam = [invT(MfromRT(kf.r_cw, kf.t_cw)) for kf in slam.KFs ]
  DrawCamera(cam, color='r')

#%% StereoProjectError

def StereoProjectError(kf_r_cw, kf_t_cw, mp_xyz_w, kf_u, kf_v, kf_du):
  check_magnitude(kf_r_cw)

  Pc = ax2Rot(kf_r_cw).dot(mp_xyz_w) + kf_t_cw
  p  = K.dot(Pc)
  z_inv = 1/p[2]
  uv_predict = p[:2]*z_inv
  err_u = kf_u - uv_predict[0]
  err_v = kf_v - uv_predict[1]
  err_du= kf_du - baseline*K[0,0]*z_inv
  return np.hstack([err_u,err_v,err_du])

if 0 and not baseline is None:
  problem = GaussHelmertProblem()
  for kf in slam.KFs:
    if kf.id is None:
      kf.id, _ = problem.AddParameter([kf.r_cw, kf.t_cw])

    for mp in slam.MPs:
      if np.random.rand(1)>0.5: # random connection with kf
        continue

      if mp.id is None:
        mp.id, _ = problem.AddParameter([mp.xyz_w])
      u,v,du = kf.Project(mp.xyz_w)
      u += np.random.randn(1)
      v += np.random.randn(1)
      du+= np.random.randn(1)
      uvd_id, _ = problem.AddObservation([u,v,du])

      problem.AddConstraintWithID(StereoProjectError, kf.id + mp.id, uvd_id)
  # anchor the whole system
  problem.SetVarFixed(kf.r_cw)
  problem.SetVarFixed(kf.t_cw)
  x,le,fac = SolveWithGESparse(problem, fac=True)
  print 'variance factor:%f' % fac

#%% ExtrinsicError
def ExtrinsicError(r_cw, t_cw, r_oc, t_oc, r_ow, t_ow):
  check_magnitude(r_cw)
  check_magnitude(r_oc)
  check_magnitude(r_ow)

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

if 0 and baseline is None:

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
    od.id, (r,t) = problem.AddObservation([od.r_ow, od.t_ow])
    r.array += 0.02*np.random.randn(3)
    t.array += 0.1*np.random.randn(3)
    problem.AddConstraintWithID(ExtrinsicError, kf.id + se3_id, od.id)
    problem.SetSigma(od.r_ow, 0.02**2*np.eye(3))
    problem.SetSigma(od.t_ow,  0.1**2*np.eye(3))

  for kf, mp in product(slam.KFs, slam.MPs):
    if mp.id is None:
      mp.id, _ = problem.AddParameter([mp.xyz_w])
    u,v = kf.Project(mp.xyz_w)
    u += np.random.randn(1)
    v += np.random.randn(1)
    uv_id, _ = problem.AddObservation([u,v])
    problem.AddConstraintWithID(ProjectError, kf.id + mp.id, uv_id)
  problem.SetVarFixed(slam.KFs[0].r_cw )
  problem.SetVarFixed(slam.KFs[0].t_cw )

  x,le,fac = SolveWithGESparse(problem, maxit=20, fac=True)
  print 'variance factor:%f' % fac

#  problem.ViewJacobianPattern()
  print se3_vec[0].array, se3_vec[1].array
  print slam.offset.r_ab, slam.offset.t_ab
#%% ExtrinsicErrorSE3
def ExtrinsicErrorSE3(vT_cw, vT_oc, vT_ow):
  T_cw_est = invT( Mat(vT_oc) ).dot( Mat(vT_ow) )
  return Vec(T_cw_est) - vT_cw

if 0 and baseline is None:

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
    problem.AddConstraintWithID(ExtrinsicErrorSE3, kf.id + se3_id, od.id)
  problem.SetVarFixed(slam.KFs[0].vT_cw )

  for kf, mp in product(slam.KFs, slam.MPs):
    if mp.id is None:
      mp.id, _ = problem.AddParameter([mp.xyz_w])
    u,v = kf.Project(mp.xyz_w)
    u += np.random.randn(1)
    v += np.random.randn(1)
    uv_id, _ = problem.AddObservation([u,v])
    problem.AddConstraintWithID(ProjectErrorSE3, kf.id + mp.id, uv_id)

#  x,le,fac = SolveWithGESparse(problem, maxit=20, fac=True)
#  problem.ViewJacobianPattern()
#  print Mat(se3_vec[0].array)
#  print Mat(slam.offset.vT_ab)

#%% orbslam
if 0:

  if not 'tracker' in vars() or 0:
    import sys
    sys.path.append("/home/nubot/data/workspace/ORB_SLAM2/src/swig")
    sys.path.append("/home/kaihong/workspace/ORB_SLAM2/src/swig")
    import orbslam
    import cv2

    base_dir = "/home/nubot/data/workspace/ORB_SLAM2/"
    pic_l_path = "/home/nubot/data/Kitti/image_0/%06d.png"
    pic_r_path = "/home/nubot/data/Kitti/image_1/%06d.png"
#    base_dir = "/home/kaihong/workspace/ORB_SLAM2/"
#    pic_l_path = "/media/kaihong/2ADA2A32DA29FAA9/work/calibrate/kitti/sequences/00/image_0/%06d.png"
#    pic_r_path = "/media/kaihong/2ADA2A32DA29FAA9/work/calibrate/kitti/sequences/00/image_1/%06d.png"
    plt.imread(pic_l_path % 0)

    tracker = orbslam.System( base_dir + "Vocabulary/ORBvoc.bin",
                              base_dir + "Examples/Stereo/KITTI00-02.yaml",
                              orbslam.System.STEREO,
                              False)
    for i in range(10):
      im_l = cv2.imread( pic_l_path % i )
      im_r = cv2.imread( pic_r_path % i )
      tracker.TrackStereo(im_l, im_r, i)
      print i

    kfs = [kf for kf in tracker.mpMap.GetAllKeyFrames() if not kf.isBad()]
    mps = [mp for mp in tracker.mpMap.GetAllMapPoints() if not mp is None and not mp.isBad() ]

    fig = plt.figure(figsize=(11,11))
    ax = fig.add_subplot(111, projection='3d')
    Pw = np.hstack(mp.GetWorldPos() for mp in mps)
    ax.scatter(Pw[0],Pw[1],Pw[2])
    ax.set_xlim3d(-3,3)
    ax.set_ylim3d(-3,3)
    ax.set_zlim3d(-3,3)

    Mwc = [kf.GetPoseInverse() for kf in kfs ]
    DrawCamera(Mwc, scale=0.6, color='b')
#%%
  """
  kf.mvuRight = uR;
  disparity = uL-uR
  kf.mvDepth = kf.mbf/disparity
  kf.mbf = baseline * f_x
  """


  def GenEdgeStereoSE3ProjectXYZ(fx,fy,cx,cy,bf):
    def EdgeStereoSE3ProjectXYZ(pw_xyz, r_cw, t_cw, pc_xyd ):
      check_magnitude(r_cw)

      pc_xyz = ax2Rot(r_cw).dot(pw_xyz) + t_cw
      invZ = 1.0/pc_xyz[2]
#      if invZ > 0:
#        print "neg inv"

      xl = pc_xyz[0]*invZ*fx + cx
      yl = pc_xyz[1]*invZ*fy + cy
      dx = bf*invZ    # bf = baseline * f_x
      return pc_xyd - np.hstack([xl,yl,dx])
    return EdgeStereoSE3ProjectXYZ
  EdgeStereoSE3ProjectXYZ = GenEdgeStereoSE3ProjectXYZ(718.86, 718.86, 607.19, 185.21, 386.14)

  class OrbSLAMProblem(GaussHelmertProblem):
    def __init__(self):
      self.kf_dict = {}
      self.mp_dict = {}
      super(OrbSLAMProblem, self).__init__()

    def FindOrAddKeyFramePosParameter(self, kf):
      id = kf.mnId
      kf_xid = self.kf_dict.get(id, None)
      if not kf_xid is None:
        return kf_xid

      kf_pos = kf.GetPose()
      r, t   = Rot2ax(kf_pos[:3,:3]), kf_pos[:3,3].ravel()
      kf_xid, _ = self.AddParameter([ r,t ])
      self.kf_dict[id] = kf_xid
      return kf_xid

    def FindOrAddMapPointParameter(self, mp):
      id = mp.mnId
      mp_xid = self.mp_dict.get(id, None)

      if not mp_xid is None:
        return mp_xid

      mp_pos = mp.GetWorldPos().ravel()
      mp_xid, _ = self.AddParameter([mp_pos])
      self.mp_dict[id] = mp_xid
      return mp_xid

  def GatherCommonMP(kf_list, problem):
    first_set = None
    for kf in kf_list:
      for kp_id, mp in enumerate(kf.GetMapPointMatches()):
        if mp is None or mp.isBad() or kf.mvuRight[kp_id]<0:
          continue
        kf_xid = problem.FindOrAddKeyFramePosParameter(kf) # not eariler because there might not be any valid mp
        if first_set is None:
          first_set = copy(kf_xid)

        mp_xid  = problem.FindOrAddMapPointParameter(mp)
        keypoint= kf.mvKeysUn[kp_id]
        kp_dx   = 386.14/kf.mvDepth[kp_id]
        pc_xyd = np.r_[keypoint.pt.x, keypoint.pt.y, kp_dx]
        l_xid, l_vec = problem.AddObservation([ pc_xyd ] )
        sigma = np.eye(3)/kf.mvInvLevelSigma2[keypoint.octave]
        problem.AddConstraintWithID(EdgeStereoSE3ProjectXYZ, mp_xid+kf_xid, l_xid)
        problem.SetSigma(pc_xyd, sigma)
    map(problem.SetVarFixedWithID, first_set)
    return



#  kf0 = kfs[2]
#  # Local KeyFrames: First Breath Search from Current Keyframe
#  local_kfs = { kf.mnId: kf for kf in kf0.GetBestCovisibilityKeyFrames(2) if not kf.isBad() }
#
#  # Local MapPoints seen in Local KeyFrames
#  local_mps = {}
#  for kf in local_kfs.values():
#    mp_dict = { mp.mnId : mp for mp in kf.GetMapPoints()} # all ready filtered mp
#    local_mps.update( mp_dict )
#
#  # Fixed Keyframes. Keyframes that see Local MapPoints but that are not Local Keyframes
#  all_kfs = {}
#  for mp in local_mps.values():
#    all_kfs.update( { kf.mnId: kf for kf, _ in mp.GetObservations().items() if not kf.isBad() } )
#  fixed_kfs = set(all_kfs) - set(local_kfs)
#  print "In total mp:%d, kf:%d (%d fixed)" % (len(local_mps), len(all_kfs), len(fixed_kfs))


  problem = OrbSLAMProblem()
  GatherCommonMP(kfs[:5], problem)
#  kf_xid = problem.FindOrAddKeyFramePosParameter(kf0)
#  map(problem.SetVarFixedWithID, kf_xid)
#  for kp_id, mp in enumerate(kf0.GetMapPointMatches()):
#    if mp is None or mp.isBad() or kf0.mvuRight[kp_id]<0:
#      continue
#
#    mp_xid  = problem.FindOrAddMapPointParameter(mp)
#    keypoint= kf0.mvKeysUn[kp_id]
#    kp_x_r  = kf0.mvuRight[kp_id]
#    kp_dx   = 386.14/kf0.mvDepth[kp_id]
#    pc_xyd = np.r_[keypoint.pt.x, keypoint.pt.y, kp_dx]
#    l_xid, l_vec = problem.AddObservation([ pc_xyd ] )
#    sigma = np.eye(3)/kf.mvInvLevelSigma2[keypoint.octave]
#
#    problem.AddConstraintWithID(EdgeStereoSE3ProjectXYZ, mp_xid+kf_xid, l_xid)
#    problem.SetSigma(pc_xyd, sigma)

#  for kfid, kf in all_kfs.items():
#    if kf.isBad():
#      continue
#    kf_xid = problem.FindOrAddKeyFramePosParameter(kf)
#    if kfid in fixed_kfs:
#      map(problem.SetVarFixedWithID, kf_xid)
#
#  for mp in local_mps.values():
#    mp_xid = problem.FindOrAddMapPointParameter(mp)
#
#    for kf, kp_id in mp.GetObservations().items():
#      keypoint = kf.mvKeysUn[kp_id]
#      kp_x_r = kf.mvuRight[kp_id]
#      kp_dx  = 386.14/kf.mvDepth[kp_id]
#      if kp_dx < 0:
#        continue
#      kf_xid = problem.FindOrAddKeyFramePosParameter(kf)
#
#      pc_xyx = np.r_[keypoint.pt.x, keypoint.pt.y, kp_x_r]
#      pc_xyd = np.r_[keypoint.pt.x, keypoint.pt.y, kp_dx]
#
#      l_xid, _ = problem.AddObservation([ pc_xyd ] )
#
#      sigma = np.eye(3)/kf.mvInvLevelSigma2[keypoint.octave]
#
#      problem.AddConstraintWithID(EdgeStereoSE3ProjectXYZ, mp_xid+kf_xid, l_xid)
#      problem.SetSigma(pc_xyd, sigma)

  x, err,fac  = SolveWithGESparse(problem, maxit=8, fac=True)
#  problem.ViewJacobianPattern()

