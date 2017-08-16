#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 13 13:38:09 2017

@author: kaihong
"""
import numpy as np
import scipy
import matplotlib.pyplot as plt

from solver import *
from parameterization import *

tFromM = lambda M: M[:3,3].copy()
rFromM = lambda M: Rot2ax(M[:3,:3])
rtFromM= lambda M: ( Rot2ax(M[:3,:3]), M[:3,3].copy() )
qFromM = lambda M: Quaternion.FromRot(M[:3,:3]).q

from collections import  defaultdict, OrderedDict
from geometry import  SE3


from scipy.linalg import block_diag

class QTParameterization(ProductParameterization):
  def __init__(self):
    super(QTParameterization, self).__init__(QuaternionParameterization(), IdentityParameterization(3))

  @staticmethod
  def ToQt(x):
    return Quaternion(x[:4]), x[4:]

  @staticmethod
  def FromMwithNoise(M, cov_q, cov_t):
    q = Quaternion(qFromM(M)).AddNoise(cov_q).q
    t = np.random.multivariate_normal(tFromM(M), cov_t)
    return MfromQT(q, t)

  @staticmethod
  def FromM(M):
    return np.hstack([qFromM(M), tFromM(M)])

  @staticmethod
  def ToM(qt):
    return MfromQT(qt[:4], qt[4:])



ToQt = QTParameterization.ToQt

class Pose(object):
  __slots__ = 'id','cov','param','param_id','err'
  def __init__(self, id=None, cov=None):
    self.id = id
    self.cov = cov
    self.param_id, self.err = None,None

  @property
  def R(self):  return self.M[:3,:3]
  @property
  def t(self):  return self.M[:3,3]

  def __repr__(self):
    return "Pose %s:\n%s" % (self.id, self.M)

  @classmethod
  def FromM(cls, M, *arg,**kwarg):
    raise NotImplementedError("")

  @property
  def M(self):
    raise NotImplementedError("")

  def ApplyTransform(self, M):
    """used in AbsTrajectory.Rebase"""
    raise NotImplementedError("")

  @staticmethod
  def Interpolate(a, b, time):
    raise NotImplementedError("")

  @staticmethod
  def Plus(a, b):
    """used in RelTrajectory.ToAbs"""
    raise NotImplementedError("")

  @staticmethod
  def Diff(a, b):
    """used in AbsTrajectory.ToRel"""
    raise NotImplementedError("")


class QuaternionPose(Pose):
  __slots__ = 'qt'
  def __init__(self, q, t, id=None, cov=None):
    super(QuaternionPose,self).__init__(id, cov)
    q *= np.sign(q[0])
    q /= np.linalg.norm(q)
    self.qt = np.hstack([q,t])
    self.param = self.qt
  @property
  def r(self):    return self.qt[:4]
  @property
  def t(self):    return self.qt[4:]

  @classmethod
  def FromM(cls, M, *arg,**kwarg):
    return cls(qFromM(M), tFromM(M), *arg, **kwarg)

  def ApplyTransform(self, M):
    M_new = np.dot(M, self.M)
    self.qt[:] = np.hstack([qFromM(M_new), tFromM(M_new)])

  def AddToProblemAsObservation(self, problem, slot):
    self.param_id, self.err = problem.AddObservation(slot, self.param, self.cov, QTParameterization())

  def AddToProblemAsParameter(self, problem, slot):
    problem.SetParameter(slot, self.param, QTParameterization())

  @property
  def M(self):
    M = np.eye(4)
    M[:3,:3] = Quaternion(self.r).ToRot()
    M[:3, 3] = self.t
    return M

  def __repr__(self):
    return "QuaternionPose %s:(%s,%s)" % (self.id, self.r, self.t)

from bisect import bisect_left
class Trajectory(object):
  @classmethod
  def FromPoseData(cls, M_list, cov_list, timestamp=None, pose_class=QuaternionPose):
    num_pos = len(M_list)
    if not isinstance(cov_list, list):
      cov_list = [cov_list]*num_pos

    trj = cls()
    if timestamp is None:
      timestamp = count()
    trj.poses = [pose_class.FromM(M, ts, cov)               \
                    for ts, M, cov                          \
                    in  zip(timestamp, M_list, cov_list)]
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
    self.slot = -1

  def AddPosesToProblemAsObservation(self, problem, skip_first=False):
    assert self.slot != -1
    start = 1 if skip_first else 0
    for p in self.poses[start:]:
      p.AddToProblemAsObservation(problem, self.slot)

  def CollectError(self):
    err = np.vstack([p.err for p in self.poses if p.param_id ])
    return np.split(err, 2, axis=1) # r,t

#  def CollectCost(self):
#    cost_func = lambda blk: blk.err.dot( np.linalg.inv(blk.sigma) ).dot(blk.err)
#
#    cost = [ cost_func(p.param_blk[0]) for p in self.poses if p.param_blk ]
#    return np.hstack(cost)
  @property
  def r(self):
    return [p.r for p in self.poses]
  @property
  def t(self):
    return [p.t for p in self.poses]
  @property
  def M(self):
    return [p.M for p in self.poses]
  @property
  def id(self):
    return [p.id for p in self.poses]

  def __len__(self):
    return len(self.poses)

  def __getitem__(self, key):
    return self.poses[key]

#  def Interpolate(self, sorted_insertion_timestamp):
#    sorted_timestep = self.id
#    end = len(self.poses)
#    i = 0
#    new_p = []
#    PoseType = type(self.poses[0])
#    for other in sorted_insertion_timestamp:
#      i = bisect_left(sorted_timestep, other, lo=i)  #  a[:i] < x <= a[i:]
#      if i == 0:
#        continue
#      if i == end:
#        break
#      p = PoseType.Interpolate(self.poses[i-1], self.poses[i], other)
#      new_p.append(p)
#    return self.FromPoseList(new_p)

  def __repr__(self):
    return "Trajectory %s with %d poses " % (self.name, len(self.poses))

class RelTrajectory(Trajectory):
  def __init__(self):
    super(RelTrajectory, self).__init__()

  @classmethod
  def FromPoseData(cls, M_list, cov_list, timestamp=None):
    return super(RelTrajectory, cls).FromPoseData(M_list, cov_list, timestamp, QuaternionPose)

#  def ToAbs(self, M0=np.eye(4) ):
#    PoseClass = type(self.poses[0])
#    pose_list = [PoseClass( M = M0, id = 0, cov = np.zeros(6) )]
#
#    for dp in self.poses:
#      p_last = pose_list[-1]
#      if np.array_equal(p_last.M, np.eye(4)): # a hack when r,t=0
#        pose = PoseClass( M = dp.M,
#                     id = len(pose_list),
#                     cov= dp.cov)
#      else:
#        pose = PoseClass.Add(p_last, dp)
#        pose.id = len(pose_list)
#      pose_list.append(pose)
#
#    return AbsTrajectory.FromPoseList(pose_list)


class AbsTrajectory(Trajectory):
  def __init__(self):
    super(AbsTrajectory, self).__init__()

  @classmethod
  def FromPoseData(cls, M_list, cov_list, timestamp=None):
    return super(AbsTrajectory, cls).FromPoseData(M_list, cov_list, timestamp, QuaternionPose)

#  @staticmethod
#  @AddJacobian
#  def RelT(r1,t1,r2,t2):
#    r12 = axAdd(-r1,r2)
#    t12 = ax2Rot(-r1).dot(t2-t1)
#    return np.hstack([r12, t12])

#  def SetFirstPoseFix(self, problem):
#    problem.SetVarFixedWithID(self.poses[0].param_id[0])

#  def ToRel(self, interval=1):
#    pose_list = []
#
#    for p_base, p_end in zip(self.poses[:-interval],self.poses[interval:]):
#      if np.allclose(p_base.M, p_end.M):
#        pass
#
#      else:
#        drdt, J = AbsTrajectory.RelT(p_base.r, p_base.t, p_end.r, p_end.t)
#        J = np.hstack(J)
#        Cov_abs = block_diag(p_base.cov, p_end.cov)
#        cov_drdt = J.dot(Cov_abs).dot(J.T)
#
#        pose = AngleAxisPose( drdt[:3], drdt[3:],
#                              id = p_end.id,
#                              cov= cov_drdt)
#      pose_list.append(pose)
#
#    return RelTrajectory.FromPoseList(pose_list)

  def Plot(self, scale=1, select=slice(None), **kwarg):
    PlotPose(self.M[select], scale, **kwarg)

  def Rebase(self, M=None):
    if M is None:
      M = invT(self.poses[0].M)
    for p in self.poses:
      p.ApplyTransform(M)


class BatchCalibrationProblem(object):
  def __init__(self):
    self.trajectory = {}
    self.calibration = defaultdict(dict)

  def __setitem__(self, key, value):
    assert isinstance(value, Trajectory)
    self.trajectory[key] = value
    value.name = str(key)

  def __getitem__(self, key):
    return self.trajectory[key]

  def __repr__(self):
    return str(self.trajectory)

  def SolveDirect(self, base, opponent):
    # 1.solve rotation
    if isinstance(self.trajectory[base][0], QuaternionPose):
      # method A with quaternion
      q_b = map(Quaternion, self.trajectory[base].r)
      q_o = map(Quaternion, self.trajectory[opponent].r)
      R_o = map(lambda q:q.ToRot(),  q_o)

      H  = [ q.ToMulMatL() - p.ToMulMatR() for q,p in zip(q_o, q_b) ] # L(q_o)*q_ob - R(q_b)*q_ob = 0, st. |q_ob|=1
      _, s, V = np.linalg.svd(np.vstack( H ), full_matrices=0)
      q_ob = V[3]   # eigen vector with smallest eigen value
      R_ob = Quaternion(q_ob).ToRot()

    else:
      # method B with angle-axis
      r_b = np.asarray(self.trajectory[base].r)
      r_o = np.asarray(self.trajectory[opponent].r)
      R_o = map(ax2Rot, r_o)

      H = r_b.T.dot(r_o)
      U, d, Vt = np.linalg.svd(H)
      R_ob = Vt.T.dot(U.T)

    # 2.solve translation given rotation
    t_b = np.asarray(self.trajectory[base].t)
    t_o = np.asarray(self.trajectory[opponent].t)
    I = np.eye(3)
    A = np.vstack([ I - R for R in R_o])
    b = np.hstack( t_o - ( R_ob.dot(t_b.T) ).T )
    t_ob = np.linalg.lstsq(A, b)[0]

    # 3.final result
    M_ob = np.eye(4)
    M_ob[:3,:3], M_ob[:3,3] = R_ob, t_ob
    return M_ob

  def MakeProblemWithAbsModel(self, base, M0_dict={}):
    # check
    for trj in self.trajectory.values():
      assert isinstance(trj, AbsTrajectory)
      if not np.allclose( trj[0].M, np.eye(4) ):
        raise RuntimeWarning('First pose is not Identity, please use trj.Rebase() to make so.')

    # define constraint function
    num_sensors = len(self.trajectory)
    num_x = num_sensors-1
    num_l = num_sensors

    @AddJacobian(split=False)
    def AbsoluteConstraint(*args):
      # separate parameters and observation from one large args
      # args = x_args + l_args
      x_args, l_args = list(args[:num_x]), list(args[num_x:])
      e = []
      # base trajectory is the first
      q_b, t_b = ToQt(l_args.pop(0))
      # loop through trajectory-pair
      for qt_ob, qt_o in zip(x_args, l_args):
        q_ob, t_ob = ToQt(qt_ob)   # other -> base
        q_o , t_o  = ToQt(qt_o)
        err_q = ( q_o * q_ob * q_b.Inv() * q_ob.Inv() ).q[1:]
        err_t = t_ob + q_ob.RotatePoint(t_b) - q_o.RotatePoint(t_ob) - t_o
        e += [err_q, err_t]
      return np.hstack(e)

    # assign slot
    self.trajectory[base].slot = 0
    opp_key = self.trajectory.keys()
    opp_key.remove(base)
    for slot, key in enumerate(opp_key):
      self.trajectory[key].slot = slot+1

    problem = BatchGaussHelmertProblem(AbsoluteConstraint, num_x, num_l)
    for key in opp_key:
      trj = self.trajectory[key]
      # parameter, get init guess
      if key in M0_dict:
        Mob = M0_dict[key]
      else:
        Mob = self.SolveDirect(base, key)
        print "Init guess for %s:\n%s" %(key, Mob)
      Pob = QuaternionPose.FromM(Mob)
      Pob.AddToProblemAsParameter(problem, trj.slot-1)
      self.calibration[key][base] = Pob     # other <- base
      # observation
      trj.AddPosesToProblemAsObservation(problem, skip_first=True)
    self.trajectory[base].AddPosesToProblemAsObservation(problem, skip_first=True)

    return problem

  def FillCalibration(self):
    from itertools import permutations
    for key1,key2 in permutations( self.trajectory.keys(), 2):
      if key1 in self.calibration and key2 in self.calibration[key1]:
        pose12 = self.calibration[key1][key2]
        self.calibration[key2][key1] = MatrixPose( invT(pose12.M) )

  def PlotErrHist(self, bins=80):
    num_trj = len(self.trajectory)
    f1,(b) = plt.subplots(3, num_trj, sharex=True, sharey=True) #num='r',
    f2,(c) = plt.subplots(3, num_trj, sharex=True, sharey=True) #num='t',
    f1.subplots_adjust(hspace=0.02)
    f1.subplots_adjust(wspace=0.01)
    f2.subplots_adjust(hspace=0.02)
    f2.subplots_adjust(wspace=0.01)

    for s, (key,trj) in enumerate(self.trajectory.items()):
      r_err, t_err = trj.CollectError()
      for j in range(3):
        b[j][s].hist( r_err[:,j], bins, edgecolor='None',color='royalblue')
        c[j][s].hist( t_err[:,j], bins, edgecolor='None',color='royalblue')
      b[0][s].set_title(str(key),fontsize=20)
      c[0][s].set_title(str(key),fontsize=20)

    # y axis lable
    for i in range(3):
      b[i][0].set_ylabel(r"$\mathbf{r}_%d$" % i,fontsize=20)
      c[i][0].set_ylabel(r"$\mathbf{t}_%d$" % i,fontsize=20)


 #%% test
if __name__ == '__main__':

  if 1:
    num_sensor = 3
    num_seg = 1000
    def add_n_noise(sigma):
      return lambda x: x + sigma*np.random.randn(3)
    def add_u_noise(scale):
      return lambda x: x + scale*np.random.rand(3)
    noise_on = 1.0
#    np.random.seed(20)

    ConjugateM = lambda dM1, M21 : M21.dot(dM1).dot(invT(M21))
    d2r =  lambda deg: np.pi*deg/180
    def deep_map(function, list_of_list):
      if not isinstance(list_of_list[0], list):
        return map(function, list_of_list)
      return [ deep_map(function, l) for l in list_of_list ]

    Mob_all = [ MfromRT(randsp(), randsp()) for _ in xrange(num_sensor-1) ] # other <- base
    print Mob_all
    dM_1   = [ MfromRT( d2r(10+5*np.random.rand(1))*randsp(), 0.5*np.random.rand(1) * randsp() ) for _ in xrange(num_seg)]
    dM_all = [dM_1] + [ map(ConjugateM, dM_1, [M21]*num_seg ) for M21 in Mob_all ]

    M_all  = [ [np.eye(4)] for _ in range(num_sensor)]
    for M_trj, dM_trj in zip(M_all, dM_all):
      for dM in dM_trj:
        M_trj.append( M_trj[-1].dot( dM ) )

    cov_q = np.diag(np.r_[0.01,0.02,0.03]**2)
    cov_t = np.diag(np.r_[0.01,0.02,0.01]**2)
    fac = []
    for i in range(1):
      M_all_noisy = [ [np.eye(4)]+[ QTParameterization.FromMwithNoise(M, cov_q, cov_t) for M in trj[1:] ] for trj in M_all ]

      print "Abs:"
      calp_abs = BatchCalibrationProblem()
      for i in xrange(num_sensor):
        calp_abs[i]= AbsTrajectory.FromPoseData( M_all_noisy[i], block_diag(cov_q, cov_t) )
      init_guest = {}#{i+1:Tob for i,Tob in enumerate(Mob_all)}#
      problem_abs = calp_abs.MakeProblemWithAbsModel(0, init_guest)
      x, Cov_xx, sigma_0, w = problem_abs.Solve()
      fac.append(sigma_0)
