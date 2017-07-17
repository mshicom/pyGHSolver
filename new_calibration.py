#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 13 13:38:09 2017

@author: kaihong
"""
import numpy as np
import scipy
import matplotlib.pyplot as plt

import sys
sys.path.append('/home/kaihong/workspace/pyGHSolver/')
from solver2 import *


tFromT = lambda T: T[:3,3].copy()
rFromT = lambda T: Rot2ax(T[:3,:3])
rtFromT= lambda T: ( Rot2ax(T[:3,:3]), T[:3,3].copy() )

from collections import  defaultdict, OrderedDict
from geometry import  SE3


class Pose(object):
  __slots__ = 'id','T','r','t','cov_r','cov_t','rt_id','rt_blk'
  def __init__(self, T, id=None,  cov_r=None, cov_t=None):
    self.id = id
#    self.T = np.copy(T)
    self.r, self.t = rtFromT(T)
    self.cov_r, self.cov_t  = cov_r, cov_t
    self.rt_id, self.rt_blk = None,None
    if not np.isfinite(self.r).all():
      self.r = np.zeros(3)

  def __repr__(self):
    return "Pose %s:(%s,%s)" % (self.id, self.r, self.t)

  def AddToProblemAsObservation(self, problem):
    self.rt_id, self.rt_blk = problem.AddObservation([self.r, self.t])
    if not self.cov_r is None:
      problem.SetSigmaWithID(self.rt_id[0], self.cov_r)
    if not self.cov_t is None:
      problem.SetSigmaWithID(self.rt_id[1], self.cov_t)

  def AddToProblemAsParameter(self, problem):
    self.rt_id, self.rt_blk = problem.AddParameter([self.r, self.t])

  @property
  def R(self):
    return ax2Rot(self.r)

  @property
  def T(self):
    return MfromRT(self.r, self.t)

  @staticmethod
  def Interpolate(a, b, time):
    ratio = float(time - a.id)/( b.id - a.id)
    M0, M1 = a.T, b.T
    dm = SE3.algebra_from_group( M1.dot(invT(M0)) )
    dM = SE3.group_from_algebra( ratio*dm )
    Mt = dM.dot(M0)
    cov_r_time = (1-ratio)*a.cov_r + ratio*b.cov_r
    cov_t_time = (1-ratio)*a.cov_t + ratio*b.cov_t

    return Pose(Mt, id=time, cov_r=cov_r_time, cov_t=cov_t_time )

def test_Interpolate():
  p1 = Pose(MfromRT(np.zeros(3), np.zeros(3)), 0, np.eye(3), np.eye(3) )
  p2 = Pose(MfromRT(np.zeros(3), np.ones(3)), 1, np.eye(3), np.eye(3) )
  Pose.Interpolate(p1,p2,0)
  Pose.Interpolate(p1,p2,0.5)
  Pose.Interpolate(p1,p2,1)

from bisect import bisect_left
class Trajectory(object):
  @classmethod
  def FromPoseData(cls, T_list, cov_r_list, cov_t_list, timestamp=None):
    num_pos = len(T_list)
    if not isinstance(cov_r_list, list):
      cov_r_list = [cov_r_list]*num_pos
    if not isinstance(cov_t_list, list):
      cov_t_list = [cov_t_list]*num_pos

    trj = cls()
    if timestamp is None:
      trj.poses = [Pose(T, id, cov_r, cov_t)        \
                      for id,(T, cov_r, cov_t)                          \
                      in  enumerate(zip(T_list, cov_r_list, cov_t_list))]
    else:
      trj.poses = [Pose(T, ts, cov_r, cov_t)        \
                      for ts, T, cov_r, cov_t                          \
                      in  zip(timestamp, T_list, cov_r_list, cov_t_list)]
      trj.poses.sort(key=lambda p:p.id)
    return trj

  @classmethod
  def FromPoseList(cls, pose_list):
    assert isinstance(pose_list[0], Pose)
    trj = cls()
    trj.poses = copy(pose_list)
    trj.poses.sort(key=lambda p:p.id)
    return trj

  def __init__(self):
    self.name = "none"
    self.poses = []

  def AddPosesToProblemAsObservation(self, problem, skip_first=False):
    if skip_first:
      for p in self.poses[1:]:
        p.AddToProblemAsObservation(problem)
    else:
      for p in self.poses:
        p.AddToProblemAsObservation(problem)

  @property
  def r(self):
    return [p.r for p in self.poses]
  @property
  def t(self):
    return [p.t for p in self.poses]
  @property
  def T(self):
    return [p.T for p in self.poses]
  @property
  def id(self):
    return [p.id for p in self.poses]

  def Interpolate(self, sorted_insertion_timestamp):
    sorted_timestep = self.id
    end = len(self.poses)
    i = 0
    new_p = []
    for other in sorted_insertion_timestamp:
      i = bisect_left(sorted_timestep, other, lo=i)  #  a[:i] < x <= a[i:]
      if i == 0:
        continue
      if i == end:
        break
      p = Pose.Interpolate(self.poses[i-1], self.poses[i], other)
      new_p.append(p)
    return self.FromPoseList(new_p)

  def __repr__(self):
    return "Trajectory %s with %d poses " % (self.name, len(self.poses))

class RelTrajectory(Trajectory):
  def __init__(self):
    super(RelTrajectory, self).__init__()

  def ToAbs(self):
    raise NotImplementedError()

class AbsTrajectory(Trajectory):
  def __init__(self):
    super(AbsTrajectory, self).__init__()

  @staticmethod
  @AddJacobian
  def RelT(r1,t1,r2,t2):
    r12 = axAdd(-r1,r2)
    t12 = ax2Rot(-r1).dot(t2-t1)
    return np.hstack([r12, t12])


  def SetFirstPoseFix(self, problem):
    problem.SetVarFixedWithID(self.poses[0].rt_id[0])
    problem.SetVarFixedWithID(self.poses[0].rt_id[1])

  def ToRel(self, interval=1):
    pose_list = []

    for p_base, p_end in zip(self.poses[:-interval],self.poses[interval:]):
      drdt, J = AbsTrajectory.RelT(p_base.r, p_base.t, p_end.r, p_end.t)
      J = np.hstack(J)
      Cov_abs = scipy.linalg.block_diag(p_base.cov_r, p_base.cov_t, p_end.cov_r, p_end.cov_t)
      cov_drdt = J.dot(Cov_abs).dot(J.T)

      pose = Pose( T = MfromRT(drdt[:3],drdt[3:]),
                  id = p_end.id,
                  cov_r= cov_drdt[0:3,0:3],
                  cov_t= cov_drdt[3:6,3:6] )
      pose_list.append(pose)

    return RelTrajectory.FromPoseList(pose_list)



class CalibrationProblem(object):
  def __init__(self):
    self.trajectory = {}
    self.calibration = defaultdict(dict)

  def __setitem__(self, key, value):
    assert isinstance(value, Trajectory)
    self.trajectory[key] = value

  def __getitem__(self, key):
    return self.trajectory[key]

  def __repr__(self):
    return str(self.trajectory)

  def SolveDirect(self, base, opponent):
    r1 = np.asarray(self.trajectory[base].r)
    r2 = np.asarray(self.trajectory[opponent].r)

    H = r1.T.dot(r2)
    U, d, Vt = np.linalg.svd(H)
    R21 = Vt.T.dot(U.T)

    t1 = np.asarray(self.trajectory[base].t)
    t2 = np.asarray(self.trajectory[opponent].t)
    I = np.eye(3)

    A = np.vstack([ I - ax2Rot(r_) for r_ in r2])
    b = np.hstack( t2 - ( R21.dot(t1.T) ).T )
    t21 = np.linalg.lstsq(A, b)[0]
    T21 = np.eye(4)
    T21[:3,:3], T21[:3,3] = R21, t21
    return T21

  def MakeProblemWithAbsModel(self, base, T0_dict={}):

    def AbsoluteConstraint(r_sa, t_sa, r_wa, t_wa, r_vs, t_vs):
      check_magnitude(r_sa)
      check_magnitude(r_wa)
      check_magnitude(r_vs)
      r_vw, t_vw = r_sa, t_sa
      R_vw = ax2Rot(r_vw)
      t_vwa = t_vw + R_vw.dot(t_wa) # v -> w-> a
      t_vsa = t_vs + ax2Rot(r_vs).dot(t_sa)# v -> s-> a

      e_t = t_vwa - t_vsa
      e_r = R_vw.dot(r_wa) - r_vs
      return np.r_[e_r, e_t]

    problem = GaussHelmertProblem()
    """observation"""
    for trj in self.trajectory.values():
      assert isinstance(trj, AbsTrajectory)
      trj.AddPosesToProblemAsObservation(problem, skip_first=True)

    Pbase = self.trajectory[base].poses[1:]

    """parameter"""
    opp_key = self.trajectory.keys()
    opp_key.remove(base)
    for key in opp_key:
      if key in T0_dict:
        Tob = T0_dict[key]
      else:
        Tob = self.SolveDirect(base, key)
        print "Init guess for %s:\n%s" %(key, Tob)
      Pob = Pose(Tob)
      Pob.AddToProblemAsParameter(problem)
      self.calibration[base][key] = Pob

      Popp = self.trajectory[key].poses[1:]
      for p1, p2 in zip(Pbase, Popp):
        problem.AddConstraintWithID( AbsoluteConstraint,
                                     Pob.rt_id,
                                     p1.rt_id + p2.rt_id )
    return problem

  def MakeProblemWithRelModel(self, base, T0_dict={}):

    def RelativeConstraint(r_sa, t_sa, dr_a, dt_a, dr_s, dt_s):
        check_magnitude(r_sa)
        check_magnitude(dr_a)
        check_magnitude(dr_s)

        R_sa = ax2Rot(r_sa)
        dR_s  = ax2Rot(dr_s)
        e_t = dt_s + dR_s.dot(t_sa) - R_sa.dot(dt_a) - t_sa
        e_r = R_sa.dot(dr_a) - dr_s
        return np.r_[e_r, e_t]

    problem = GaussHelmertProblem()
    """observation"""
    for trj in self.trajectory.values():
      assert isinstance(trj, RelTrajectory)
      trj.AddPosesToProblemAsObservation(problem)

    Pbase = self.trajectory[base].poses

    """parameter"""
    opp_key = self.trajectory.keys()
    opp_key.remove(base)
    for key in opp_key:
      if key in T0_dict:
        Tob = T0_dict[key]
      else:
        Tob = self.SolveDirect(base, key)
        print "Init guess for %s:\n%s" %(key, Tob)

      Pob = Pose(Tob)
      Pob.AddToProblemAsParameter(problem)
      self.calibration[base][key] = Pob

      Popp = self.trajectory[key].poses
      for p1, p2 in zip(Pbase, Popp):
        problem.AddConstraintWithID( RelativeConstraint,
                                     Pob.rt_id,
                                     p1.rt_id + p2.rt_id )
    return problem


if __name__ == '__main__':
  #%% test
  if 1:
    num_sensor = 2
    num_seg = 300
    def add_n_noise(sigma):
      return lambda x: x + sigma*np.random.randn(3)
    def add_u_noise(scale):
      return lambda x: x + scale*np.random.rand(3)
    noise_on = 1.0
    np.random.seed(20)

    ConjugateT = lambda dT1, T21 : T21.dot(dT1).dot(invT(T21))
    d2r =  lambda deg: np.pi*deg/180
    def deep_map(function, list_of_list):
      if not isinstance(list_of_list[0], list):
        return map(function, list_of_list)
      return [ deep_map(function, l) for l in list_of_list ]

    Tob_all = [ MfromRT(randsp(), randsp()) for _ in xrange(num_sensor-1) ] # other <- base
    r_ob_all = map(rFromT, Tob_all)
    t_ob_all = map(tFromT, Tob_all)

    dT_1   = [ MfromRT( d2r(10+5*np.random.rand(1))*randsp(), 0.5*np.random.rand(1) * randsp() ) for _ in xrange(num_seg)]
    dT_all = [dT_1] + [ map(ConjugateT, dT_1, [T21]*num_seg ) for T21 in Tob_all ]

    T_all  = [ [np.eye(4)] for _ in range(num_sensor)]
    for T_trj, dT_trj in zip(T_all, dT_all):
      for dT in dT_trj:
        T_trj.append( T_trj[-1].dot( dT ) )
    r_all = deep_map(rFromT, T_all)
    r_all_noisy = deep_map(add_n_noise(noise_on*0.002), r_all)

    t_all = deep_map(tFromT, T_all)
    t_all_noisy = deep_map(add_n_noise(noise_on*0.02), t_all)
    T_all_noisy = [ map(MfromRT, r_trj, t_trj) for r_trj, t_trj in zip(r_all_noisy, t_all_noisy) ]

    calp_abs = CalibrationProblem()
    for i in xrange(num_sensor):
      calp_abs[i]= AbsTrajectory.FromPoseData( T_all_noisy[i], 0.002**2*np.eye(3), 0.02**2*np.eye(3) )
    init_guest = {i+1:Tob for i,Tob in enumerate(Tob_all)}#{}#
    problem_abs = calp_abs.MakeProblemWithAbsModel(0, init_guest)
    x_abs, le_abs, fac_abs = SolveWithGESparse(problem_abs, fac=True)
#
    calp_rel = CalibrationProblem()
    for key, trj in calp_abs.trajectory.items():
      calp_rel[key] = trj.ToRel(5)
    problem_rel = calp_rel.MakeProblemWithRelModel(0, init_guest)
    x_rel, le_rel, fac_abs = SolveWithGESparse(problem_rel, fac=True)

  #  problem.UpdateXL(True, False)
  #  print r_ob_all,t_ob_all
  #  print calp.param.values()

  #%% kitti
  if 0:
    data = np.load('kitti_abs.npz')
    T_laser        = [ T for T in data['T_laser']]
    T_cam, Cov_cam = zip(* [ (T,Cov) for T,Cov in data['T_cam'] ] )
    T_gps          = [ T for T in data['T_gps']]
    calp = CalibrationProblem({'laser': (T_laser, 0.02*np.eye(3), 0.02*np.eye(3)),
                               'camer': (T_cam,   0.02*np.eye(3), 0.02*np.eye(3)),
                               'gps'  : (T_gps,   0.02*np.eye(3), 0.02*np.eye(3))})
    calp['camer'].r
    R = calp.SolveDirect('laser','gps')
    problem = calp.MakeProblemWithAbsModel('laser')
