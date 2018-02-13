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
  def __init__(self, fix_rot=False, fix_x=False,fix_y=False,fix_z=False):
    if fix_rot:
      rot_param = ConstantParameterization(4)
    else:
      rot_param = QuaternionParameterization()

    if fix_x or fix_y or fix_z:
      trs_param = SubsetParameterization([not fix_x, not fix_y, not fix_z])
    else:
      trs_param = IdentityParameterization(3)
    super(QTParameterization, self).__init__(rot_param, trs_param)

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
    ratio = float(time - a.id)/( b.id - a.id)
    M0, M1 = a.M, b.M
    dm = SE3.algebra_from_group( M1.dot(invT(M0)) )
    dM = SE3.group_from_algebra( ratio*dm )
    Mt = dM.dot(M0)
    cov_time = a.cov if ratio<0.5 else  b.cov
    return type(a).FromM(Mt, id=time, cov=cov_time)

  @staticmethod
  def Plus(a, b):
    """used in RelTrajectory.ToAbs"""
    raise NotImplementedError("")

  @staticmethod
  def Diff(a, b):
    """used in AbsTrajectory.ToRel"""
    raise NotImplementedError("")

  @staticmethod
  def Adjoint(M):
    R, t = M[:3,:3], M[:3,3]
    A = np.zeros((6,6))
    A[:3,:3] = A[3:,3:] = R
    A[3:,:3] = np.dot(skew(t), R)
    return A


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
    return self

  def AddToProblemAsObservation(self, problem, slot, fix_rot=False, fix_x=False,fix_y=False,fix_z=False):
    cov = self.cov
    if fix_rot:
      cov = np.delete(cov, [0,1,2],  axis=0)
      cov = np.delete(cov, [0,1,2],  axis=1)
    if fix_x:
      cov = np.delete(cov, -3,  axis=0)
      cov = np.delete(cov, -3,  axis=1)
    if fix_y:
      cov = np.delete(cov, -2,  axis=0)
      cov = np.delete(cov, -2,  axis=1)
    if fix_z:
      cov = np.delete(cov, -1,  axis=0)
      cov = np.delete(cov, -1,  axis=1)
    self.param_id, self.err = problem.AddObservation(slot, self.param, cov, QTParameterization(fix_rot, fix_x,fix_y,fix_z))
    return self

  def AddToProblemAsParameter(self, problem, slot, *args):
    problem.SetParameter(slot, self.param, QTParameterization(*args))
    return self

  @property
  def M(self):
    M = np.eye(4)
    M[:3,:3] = Quaternion(self.r).ToRot()
    M[:3, 3] = self.t
    return M

  @staticmethod
  def Plus(a, b):
    """used in RelTrajectory.ToAbs"""
    q_a, t_a = ToQt(a.qt)
    q_b, t_b = ToQt(b.qt)

    q = (q_a * q_b).q
    t = q_a.RotatePoint(t_b) + t_a

    if not a.cov is None and not b.cov is None:
      A = Pose.Adjoint(a.M)
      cov = a.cov + A.dot(b.cov).dot(A.T)
    else:
      cov = None
    return QuaternionPose(q, t, b.id, cov)

  @staticmethod
  def Diff(a, b):
    """used in AbsTrajectory.ToRel"""
    q_a, t_a = ToQt(a.qt)
    q_b, t_b = ToQt(b.qt)

    q = (q_a.Inv() * q_b).q
    t = q_a.Inv().RotatePoint(t_b - t_a)

    if not a.cov is None and not b.cov is None:
      A = Pose.Adjoint( invT(a.M) )
      cov = A.dot( a.cov + b.cov ).dot(A.T)
    else:
      cov = None
    return QuaternionPose(q, t, b.id, cov)

  def __repr__(self):
    return "QuaternionPose %s:(%s,%s)" % (self.id, self.r, self.t)



from bisect import bisect_left
class Trajectory(object):
  def __init__(self):
    self.name = "none"
    self.poses = []
    self.slot = -1
    self.fix_rot = False
    self.fix_x,self.fix_y,self.fix_z = [False]*3

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

  def SimulateNoise(self, cov=None):
    zero = np.zeros(6)
    for p in self.poses:
      if not cov is None:
        p.cov = cov
      drdt = np.random.multivariate_normal(zero, p.cov)
      p.r[:] = (Quaternion.FromAngleAxis(drdt[:3]) * Quaternion(p.r)).q
      p.t[:] += drdt[3:]

#      p.ApplyTransform(MfromRT(drdt[:3], drdt[3:]))
    return self

  def AddPosesToProblemAsObservation(self, problem, skip_first=False):
    assert self.slot != -1
    start = 1 if skip_first else 0
    for p in self.poses[start:]:
      p.AddToProblemAsObservation(problem, self.slot,
                                  self.fix_rot,
                                  self.fix_x, self.fix_y, self.fix_z)

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

  def Interpolate(self, sorted_insertion_timestamp, mask=True):
    sorted_timestep = self.id
    end = len(self.poses)
    i = 0
    new_p = []
    PoseType = type(self.poses[0])
    for other in sorted_insertion_timestamp:
      i = bisect_left(sorted_timestep, other, lo=i)  #  a[:i] < x <= a[i:]
      if i == 0:
        print("%f skipped" % other)
        continue
      if i == end:
        break
      p = PoseType.Interpolate(self.poses[i-1], self.poses[i], other)
      new_p.append(p)
    return self.FromPoseList(new_p)

  def __repr__(self):
    return "Trajectory %s with %d poses " % (self.name, len(self.poses))

class RelTrajectory(Trajectory):
  def __init__(self):
    super(RelTrajectory, self).__init__()

  @classmethod
  def FromPoseData(cls, M_list, cov_list, timestamp=None):
    return super(RelTrajectory, cls).FromPoseData(M_list, cov_list, timestamp, QuaternionPose)

  def ToAbs(self, M0=np.eye(4) ):
    PoseClass = QuaternionPose #type(self.poses[0])
    pose_list = [PoseClass.FromM( M = M0, id = 0, cov = np.zeros((6,6)) )]

    for dp in self.poses:
      p_last = pose_list[-1]
      if np.array_equal(p_last.M, np.eye(4)): # a hack when r,t=0
        pose = PoseClass.FromM( M = dp.M,
                               id = len(pose_list),
                               cov= dp.cov)
      else:
        pose = QuaternionPose.Plus(p_last, dp)
        pose.id = len(pose_list)
      pose_list.append(pose)

    return AbsTrajectory.FromPoseList(pose_list)

  def MakeConjugate(self, M):
    raise NotImplementedError("")
    pose_list = []
    M_inv = invT(M)
    for dp in self.poses:
      pass


class AbsTrajectory(Trajectory):
  def __init__(self):
    super(AbsTrajectory, self).__init__()

  @classmethod
  def FromPoseData(cls, M_list, cov_list, timestamp=None):
    return super(AbsTrajectory, cls).FromPoseData(M_list, cov_list, timestamp, QuaternionPose)

#  def SetFirstPoseFix(self, problem):
#    problem.SetVarFixedWithID(self.poses[0].param_id[0])

  def ToRel(self, interval=1):
    pose_list = []
    for p_base, p_end in zip(self.poses[:-interval],self.poses[interval:]):
      if not np.allclose(p_base.M, p_end.M):
        pose = QuaternionPose.Diff( p_base,  p_end )
        pose_list.append(pose)
    return RelTrajectory.FromPoseList(pose_list)

  def Plot(self, scale=1, select=slice(None), **kwarg):
    PlotPose(self.M[select], scale, **kwarg)
    return self

  def Rebase(self, M=None):
    if M is None:
      M = invT(self.poses[0].M)
    for p in self.poses:
      p.ApplyTransform(M)
    return self


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

  def SolveAXBDirect(self, base, opponent):
    ''' solve Awa = Bwb Xba '''
    # 1.solve rotation
    if isinstance(self.trajectory[base][0], QuaternionPose):
      # method A with quaternion
      q_a = map(Quaternion, self.trajectory[base].r)
      q_b = map(Quaternion, self.trajectory[opponent].r)

      H  = [ ( p.Inv() * q).ToMulMatL()[1:,:] for q,p in zip(q_b, q_a) ] # L(q_b)*q_ba - q_a = 0, st. |q_ba|=1
      _, s, V = np.linalg.svd( np.vstack(H) , full_matrices=0)
      q_ba = V[3]   # eigen vector with smallest eigen value
      R_ba = Quaternion(q_ba).ToRot()
    else:
      raise ValueError('Only QuaternionPose supported')

    # 2.solve translation given rotation, ta = Rb x + tb
    t_a = np.asarray(self.trajectory[base].t)
    t_b = np.asarray(self.trajectory[opponent].t)
    b = np.ravel( t_a - t_b )
    A = np.vstack([ q.ToRot() for q in q_b])
    t_ba = np.linalg.lstsq(A, b)[0]

    # 3.final result
    M_ba = np.eye(4)
    M_ba[:3,:3], M_ba[:3,3] = R_ba, t_ba
    return M_ba

  def SolveAXXBDirect(self, base, opponent):
    # 1.solve rotation
    if isinstance(self.trajectory[base][0], QuaternionPose):
      # method A with quaternion
      q_a = map(Quaternion, self.trajectory[base].r)
      q_b = map(Quaternion, self.trajectory[opponent].r)
      R_b = map(lambda q:q.ToRot(),  q_b)

      H  = [ q.ToMulMatL() - p.ToMulMatR() for q,p in zip(q_b, q_a) ] # L(q_b)*q_ba - R(q_a)*q_ba = 0, st. |q_ba|=1
      _, s, V = np.linalg.svd(np.vstack( H ), full_matrices=0)
      q_ba = V[-1]   # eigen vector with smallest eigen value
      R_ba = Quaternion(q_ba).ToRot()

    else:
      # method B with angle-axis
      r_a = np.asarray(self.trajectory[base].r)
      r_b = np.asarray(self.trajectory[opponent].r)
      R_b = map(ax2Rot, r_b)

      H = r_a.T.dot(r_b)
      U, d, Vt = np.linalg.svd(H)
      R_ba = Vt.T.dot(U.T)

    # 2.solve translation given rotation
    t_a = np.asarray(self.trajectory[base].t)
    t_b = np.asarray(self.trajectory[opponent].t)
    I = np.eye(3)
    A = np.vstack([ I - R for R in R_b])
    b = np.hstack( t_b - ( R_ba.dot(t_a.T) ).T )
    t_ba = np.linalg.lstsq(A, b)[0]

    # 3.final result
    M_ba = np.eye(4)
    M_ba[:3,:3], M_ba[:3,3] = R_ba, t_ba
    return M_ba

  def SolveAXYBDirect(self, base, opponent):
    # 1.solve rotation
    if isinstance(self.trajectory[base][0], QuaternionPose):
      # method A with quaternion
      q_a = map(Quaternion, self.trajectory[base].r)
      q_b = map(Quaternion, self.trajectory[opponent].r)
      R_b = map(lambda q:q.ToRot(),  q_b)

      H  = [ q.ToMulMatL() - p.ToMulMatR() for q,p in zip(q_b, q_a) ] # L(q_b)*q_ba - R(q_a)*q_ba = 0, st. |q_ba|=1
      _, s, V = np.linalg.svd(np.vstack( H ), full_matrices=0)
      q_ba = V[3]   # eigen vector with smallest eigen value
      R_ba = Quaternion(q_ba).ToRot()

    else:
      # method B with angle-axis
      r_a = np.asarray(self.trajectory[base].r)
      r_b = np.asarray(self.trajectory[opponent].r)
      R_b = map(ax2Rot, r_b)

      H = r_a.T.dot(r_b)
      U, d, Vt = np.linalg.svd(H)
      R_ba = Vt.T.dot(U.T)

    # 2.solve translation given rotation
    t_a = np.asarray(self.trajectory[base].t)
    t_b = np.asarray(self.trajectory[opponent].t)
    I = np.eye(3)
    A = np.vstack([ I - R for R in R_b])
    b = np.hstack( t_b - ( R_ba.dot(t_a.T) ).T )
    t_ba = np.linalg.lstsq(A, b)[0]

    # 3.final result
    M_ba = np.eye(4)
    M_ba[:3,:3], M_ba[:3,3] = R_ba, t_ba
    return M_ba

  def MakeProblemWithModelA(self, base, dict_M_ba):
    # check
    check_unique([len(trj) for trj in self.trajectory.values()])
    for trj in self.trajectory.values():
      assert isinstance(trj, AbsTrajectory)

    # define constraint function
    num_sensors = len(self.trajectory)
    num_x = num_sensors-1
    num_l = num_sensors

    @AddJacobian(split=False)
    def GlobalConstraint(*args):
      # separate parameters and observation from one large args
      # args = x_args + l_args
      x_args, l_args = list(args[:num_x]), list(args[num_x:])
      e = []
      # base trajectory is the first
      q_wa, t_wa = ToQt(l_args.pop(0))
      # loop through trajectory-pair
      for qt_ba, qt_wb in zip(x_args, l_args):
        q_ba, t_ba = ToQt(qt_ba)   # other -> base
        q_wb, t_wb = ToQt(qt_wb)
        # R_wb * R_ba = R_wa
        err_q = ( q_wb * q_ba * q_wa.Inv()).q[1:]
        # t_wb + R_wb * t_ba = t_wa
        err_t = t_wb + q_wb.RotatePoint(t_ba) - t_wa
        e += [err_q, err_t]
      return np.hstack(e)

    # assign slot
    self.trajectory[base].slot = 0
    opp_key = self.trajectory.keys()
    opp_key.remove(base)
    for slot, key in enumerate(opp_key):
      self.trajectory[key].slot = slot+1

    problem = BatchGaussHelmertProblem(GlobalConstraint, num_x, num_l)
    for key in opp_key:
      trj = self.trajectory[key]
      # parameter, get init guess
      if key in dict_M_ba:
        M_ba = dict_M_ba[key]
      else:
        M_ba = self.SolveAXBDirect(base, key)
        print( "Init guess for %s:\n%s" %(key, M_ba))
      P_ba = QuaternionPose.FromM(M_ba)
      P_ba.AddToProblemAsParameter(problem, trj.slot-1)
      self.calibration[key][base] = P_ba     # other <- base
      # observation
      trj.AddPosesToProblemAsObservation(problem, skip_first=True)
    self.trajectory[base].AddPosesToProblemAsObservation(problem, skip_first=True)

    return problem

  def MakeProblemWithModelB(self, base, dict_M_ba={}):
    # check
    check_unique([len(trj) for trj in self.trajectory.values()])
    for trj in self.trajectory.values():
      assert isinstance(trj, AbsTrajectory)
      if not np.allclose( trj[0].M, np.eye(4) ):
        print( 'First pose is not Identity, please use trj.Rebase() to make so.')

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
      q_a, t_a = ToQt(l_args.pop(0))
      # loop through trajectory-pair
      for qt_ba, qt_b in zip(x_args, l_args):
        q_ba, t_ba = ToQt(qt_ba)   # other -> base
        q_b , t_b  = ToQt(qt_b)
        err_q = ( q_b * q_ba * q_a.Inv() * q_ba.Inv() ).q[1:]
        err_t = t_ba + q_ba.RotatePoint(t_a) - q_b.RotatePoint(t_ba) - t_b
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
      if key in dict_M_ba:
        M_ba = dict_M_ba[key]
      else:
        M_ba = self.SolveAXXBDirect(base, key)
        print( "Init guess for %s:\n%s" %(key, M_ba))
      Pob = QuaternionPose.FromM(M_ba)
      Pob.AddToProblemAsParameter(problem, trj.slot-1)
      self.calibration[key][base] = Pob     # other <- base
      # observation
      trj.AddPosesToProblemAsObservation(problem, skip_first=True)
    self.trajectory[base].AddPosesToProblemAsObservation(problem, skip_first=True)
    return problem

  def MakeProblemWithModelBB(self, base, dict_M_ba={}, fix_rot=False, fix_x=False,fix_y=False,fix_z=False):
    # check
    check_unique([len(trj) for trj in self.trajectory.values()])
    for trj in self.trajectory.values():
      assert isinstance(trj, AbsTrajectory)
#      if not np.allclose( trj[0].M, np.eye(4) ):
#        print 'First pose is not Identity, please use trj.Rebase() to make so.'

    # define constraint function
    num_sensors = len(self.trajectory)
    num_x = 2*(num_sensors-1)
    num_l = num_sensors

    @AddJacobian(split=False)
    def AbsoluteConstraint(*args):  # Yvw Awa = Bvb Xba
      # separate parameters and observation from one large args
      # args = x_args + l_args
      x_args, y_args, l_args = list(args[:num_x/2]), list(args[num_x/2:num_x]), list(args[num_x:])
      e = []
      # base trajectory is the first
      q_wa, t_wa = ToQt(l_args.pop(0))
      # loop through trajectory-pair
      for qt_ba, qt_vw, qt_vb in zip(x_args, y_args, l_args):
        q_ba, t_ba = ToQt(qt_ba)   # other -> base
        q_vw, t_vw = ToQt(qt_vw)   # other -> base
        q_vb, t_vb = ToQt(qt_vb)
        err_q = ( q_vb * q_ba * q_wa.Inv() * q_vw.Inv() ).q[1:]
        err_t = q_vb.RotatePoint(t_ba) + t_vb - q_vw.RotatePoint(t_wa) - t_vw
        e += [err_q, err_t]
      return np.hstack(e)

    # assign slot
    self.trajectory[base].slot = 0
    opp_key = self.trajectory.keys()
    opp_key.remove(base)
    for cnt, key in enumerate(opp_key):
      self.trajectory[key].slot = cnt+1

    problem = BatchGaussHelmertProblem(AbsoluteConstraint, num_x, num_l)
    for key in opp_key:
      trj = self.trajectory[key]
      # parameter, get init guess
      if key in dict_M_ba:
        M_ba, M_vw = dict_M_ba[key]
      else:
        M_ba, M_vw = self.SolveAXYBDirect(base, key)
        print( "Init guess for %s:\n%s" %(key, M_ba))
      Pba = QuaternionPose.FromM(M_ba)
      Pba.AddToProblemAsParameter(problem, trj.slot-1, fix_rot, fix_x,fix_y,fix_z)
      Pvw = QuaternionPose.FromM(M_vw)
      Pvw.AddToProblemAsParameter(problem, num_x/2+trj.slot-1, fix_rot, fix_x, fix_y, fix_z)

      self.calibration[key][base] = Pba     # other <- base
      # observation
      trj.AddPosesToProblemAsObservation(problem, skip_first=True)
    self.trajectory[base].AddPosesToProblemAsObservation(problem, skip_first=True)
    return problem

  def MakeProblemWithModelC(self, base, dict_M_ba={}):
    check_unique([len(trj) for trj in self.trajectory.values()])
    for trj in self.trajectory.values():
      assert isinstance(trj, RelTrajectory)

    # define constraint function
    num_sensors = len(self.trajectory)
    num_x = num_sensors-1
    num_l = num_sensors

    @AddJacobian(split=False)
    def RelativeConstraint(*args):
      # separate parameters and observation from one large args
      # args = x_args + l_args
      x_args, l_args = list(args[:num_x]), list(args[num_x:])
      e = []
      # base trajectory is the first
      q_a, t_a = ToQt(l_args.pop(0))
      # loop through trajectory-pair
      for qt_ba, qt_b in zip(x_args, l_args):
        q_ba, t_ba = ToQt(qt_ba)   # other -> base
        q_b , t_b  = ToQt(qt_b)
        err_q = ( q_b * q_ba * q_a.Inv() * q_ba.Inv() ).q[1:]
        err_t = t_ba + q_ba.RotatePoint(t_a) - q_b.RotatePoint(t_ba) - t_b
        e += [err_q, err_t]
      return np.hstack(e)

    # assign slot
    self.trajectory[base].slot = 0
    opp_key = self.trajectory.keys()
    opp_key.remove(base)
    for slot, key in enumerate(opp_key):
      self.trajectory[key].slot = slot+1

    problem = BatchGaussHelmertProblem(RelativeConstraint, num_x, num_l)
    for key in opp_key:
      trj = self.trajectory[key]
      # parameter, get init guess
      if key in dict_M_ba:
        M_ba = dict_M_ba[key]
      else:
        M_ba = self.SolveAXXBDirect(base, key)
        print( "Init guess for %s:\n%s" %(key, M_ba))
      P_ba = QuaternionPose.FromM(M_ba)
      P_ba.AddToProblemAsParameter(problem, trj.slot-1)
      self.calibration[key][base] = P_ba     # other <- base
      # observation
      trj.AddPosesToProblemAsObservation(problem, skip_first=False)
    self.trajectory[base].AddPosesToProblemAsObservation(problem, skip_first=False)
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
    num_sensor = 2
    num_seg = 300
    def add_n_noise(sigma):
      return lambda x: x + sigma*np.random.randn(3)
    def add_u_noise(scale):
      return lambda x: x + scale*np.random.rand(3)
    noise_on = 1.0
    np.random.seed(20)

    ConjugateM = lambda dM1, M21 : M21.dot(dM1).dot(invT(M21))
    def deep_map(function, list_of_list):
      if not isinstance(list_of_list[0], list):
        return map(function, list_of_list)
      return [ deep_map(function, l) for l in list_of_list ]

    Mba_all = [np.eye(4)]+[ MfromRT(randsp(), randsp()) for _ in xrange(num_sensor-1) ] # other <- base
    print( Mba_all)
    dM_1   = [ MfromRT( d2r(10+5*np.random.rand(1))*randsp(), 0.5*np.random.rand(1) * randsp() ) for _ in xrange(num_seg)]
    dM_all = [ map(ConjugateM, dM_1, [M21]*num_seg ) for M21 in Mba_all ]

    M_all  = [ [np.eye(4)] for _ in range(num_sensor)]
    for M_trj, dM_trj in zip(M_all, dM_all):
      for dM in dM_trj:
        M_trj.append( M_trj[-1].dot( dM ) )

    Mvw_all = [np.eye(4)]+[ MfromRT(randsp(), randsp()) for _ in xrange(num_sensor-1) ] # other <- base

    cov_q = np.diag(np.r_[0.01,0.02,0.03]**2)
    cov_t = np.diag(np.r_[0.01,0.02,0.01]**2)
    cov = block_diag(cov_q, cov_t)
    fac = []
    for i in range(100):
      if 0:
        print( "A:")
        calp_glb = BatchCalibrationProblem()
        for i in xrange(num_sensor):
          if 1:
            calp_glb[i]= AbsTrajectory.FromPoseData( M_all[i], None ).Rebase(invT(Mba_all[i])).SimulateNoise( cov )
          else:
            calp_glb[i]= RelTrajectory.FromPoseData( dM_all[i], None ).SimulateNoise( cov ).ToAbs(map(invT,Mba_all)[i])
        init_guest = {i+1:T_ba for i,T_ba in enumerate(Mba_all[1:])}#{}#
        problem_glb = calp_glb.MakeProblemWithModelA(0, init_guest)
        x, Cov_xx, sigma_0, w = problem_glb.Solve(update_cov=True)
        fac.append(sigma_0)

      if 0:
        print( "B:")
        calp_abs = BatchCalibrationProblem()
        for i in xrange(num_sensor):
          calp_abs[i]= AbsTrajectory.FromPoseData( M_all[i], cov ).SimulateNoise()
        problem_abs = calp_abs.MakeProblemWithModelB(0, init_guest)
        x, Cov_xx, sigma_0, w = problem_abs.Solve()
        fac.append(sigma_0)

      if 1:
        print( "BB:")
        calp_abs = BatchCalibrationProblem()
        for i in xrange(num_sensor):
          calp_abs[i]= AbsTrajectory.FromPoseData( M_all[i], None ).Rebase(Mvw_all[i].dot(invT(Mba_all[i]))).SimulateNoise(cov)
        problem_abs = calp_abs.MakeProblemWithModelBB(0, {i+1:(T_ba, T_vw) for i,T_ba,T_vw in zip(count(), Mba_all[1:], Mvw_all[1:] )})
        x, Cov_xx, sigma_0, w = problem_abs.Solve()
        fac.append(sigma_0)

      if 0:
        print( "C:")
        calp_rel = BatchCalibrationProblem()
        for i in xrange(num_sensor):
          if 0:
            calp_rel[i]= RelTrajectory.FromPoseData( dM_all[i], cov ).SimulateNoise()
          else:
            calp_rel[i]= AbsTrajectory.FromPoseData( M_all[i], None ).ToRel().SimulateNoise(cov)
        problem_rel = calp_rel.MakeProblemWithModelC(0, init_guest)
        x, Cov_xx, sigma_0, w = problem_rel.Solve()
        fac.append(sigma_0)
    plt.close()
    plt.hist(fac,20)
    print( np.mean(fac))