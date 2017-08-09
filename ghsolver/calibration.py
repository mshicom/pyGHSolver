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

class Pose(object):
  __slots__ = 'id','cov','param','param_id','param_blk'
  def __init__(self, id=None, cov=None):
    self.id = id
    self.cov = cov
    self.param_id, self.param_blk = None,None

  @property
  def R(self):  return self.M[:3,:3]
  @property
  def t(self):  return self.M[:3,3]

  def __repr__(self):
    return "Pose %s:\n%s" % (self.id, self.M)

  def AddToProblemAsObservation(self, problem):
    self.param_id, self.param_blk = problem.AddObservation([self.param])
    if not self.cov is None:
      problem.SetSigmaWithID(self.param_id[0], self.cov)

  def AddToProblemAsParameter(self, problem):
    self.param_id, self.param_blk = problem.AddParameter([self.param])

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
  def Add(a, b):
    """used in RelTrajectory.ToAbs"""
    raise NotImplementedError("")

  @staticmethod
  def Minus(a, b):
    """used in AbsTrajectory.ToRel"""
    raise NotImplementedError("")

class MatrixPose(Pose):
  __slots__ = '_M'
  def __init__(self, M, id=None, cov=None):
    super(MatrixPose,self).__init__(id, cov)
    self._M = np.copy(M)
    self.param = np.ndarray(shape=(12,), buffer=self._M)

  @classmethod
  def FromM(cls, M, *arg,**kwarg):
    raise cls(M, *arg,**kwarg)

  @property
  def M(self):
    return self._M

  def ApplyTransform(self, M):
    self._M = np.dot(M, self._M)

class AngleAxisPose(Pose):
  __slots__ = 'rt'
  def __init__(self, r, t, id=None, cov=None):
    super(AngleAxisPose,self).__init__(id, cov)
    self.rt = np.hstack([r,t])
    self.param = self.rt
    check_magnitude(self.r)

  @property
  def r(self):    return self.rt[:3]
  @property
  def t(self):    return self.rt[3:]

  @classmethod
  def FromM(cls, M, *arg,**kwarg):
    return cls(rFromM(M), tFromM(M), *arg,**kwarg)

  @property
  def M(self):
    return MfromRT(self.r, self.t)

  def ApplyTransform(self, M):
    M_new = np.dot(M, self.M)
    self.rt[:] = np.hstack(rtFromM(M_new))

  @staticmethod
  def Add(a, b):
    c_r = axAdd(a.r, b.r)
    c_t = a.t + ax2Rot(a.r).dot(b.t)
    return np.hstack([c_r, c_t])

  @staticmethod
  def Minus(a, b):
    c_r = axAdd(-a.r, b.r)
    c_t = ax2Rot(-a.r).dot(b.t - a.t)
    return np.hstack([c_r, c_t])

  def __repr__(self):
    return "AngleAxisPose %s:(%s,%s)" % (self.id, self.r, self.t)

  @staticmethod
  def Interpolate(a, b, time):
    ratio = float(time - a.id)/( b.id - a.id)
    M0, M1 = a.M, b.M
    dm = SE3.algebra_from_group( M1.dot(invT(M0)) )
    dM = SE3.group_from_algebra( ratio*dm )
    Mt = dM.dot(M0)
    cov_time = (1-ratio)*a.cov + ratio*b.cov

    return AngleAxisPose.FromM(Mt, id=time, cov=cov_time)

    def test_Interpolate():
      p1 = AngleAxisPose(np.zeros(3), np.zeros(3), 0, block_diag(np.eye(3), np.eye(3)) )
      p2 = AngleAxisPose(np.zeros(3), np.ones(3),  1, block_diag(np.eye(3), np.eye(3)) )
      AngleAxisPose.Interpolate(p1,p2, 0)
      AngleAxisPose.Interpolate(p1,p2, 0.5)
      AngleAxisPose.Interpolate(p1,p2, 1)

class QuaternionPose(Pose):
  __slots__ = 'qt'
  parametriaztion = ProductParameterization(QuaternionParameterization(), IdentityParameterization(3))
  def __init__(self, q, t, id=None, cov=None):
    super(QuaternionPose,self).__init__(id, cov)
    q *= np.sign(q[0])
    q /= np.linalg.norm(q)
    self.qt = np.hstack([q,t])
    self.param = self.qt
  @property
  def q(self):    return self.qt[:4]
  @property
  def t(self):    return self.qt[4:]

  @classmethod
  def FromM(cls, M, *arg,**kwarg):
    return cls(qFromM(M), tFromM(M), *arg, **kwarg)

  def ApplyTransform(self, M):
    M_new = np.dot(M, self.M)
    self.qt[:] = np.hstack([qFromM(M_new), tFromM(M_new)])

  def AddToProblemAsObservation(self, problem):
    super(QuaternionPose, self).AddToProblemAsObservation(problem)
    problem.SetParameterizationWithID(self.param_id[0], self.parametriaztion)

  def AddToProblemAsParameter(self, problem):
    super(QuaternionPose, self).AddToProblemAsParameter(problem)
    problem.SetParameterizationWithID(self.param_id[0], self.parametriaztion)

  @property
  def M(self):
    M = np.eye(4)
    M[:3,:3] = Quaternion(self.q).ToRot()
    M[:3, 3] = self.t
    return M

  def __repr__(self):
    return "QuaternionPose %s:(%s,%s)" % (self.id, self.q, self.t)

from bisect import bisect_left
class Trajectory(object):
  @classmethod
  def FromPoseData(cls, M_list, cov_list, timestamp=None, pose_class=QuaternionPose):
    num_pos = len(M_list)
    if not isinstance(cov_list, list):
      cov_list = [cov_list]*num_pos

    trj = cls()
    if timestamp is None:
      trj.poses = [pose_class.FromM(M, id, cov)                            \
                      for id,(M, cov)                          \
                      in  enumerate(zip(M_list, cov_list))]
    else:
      trj.poses = [pose_class.FromM(M, ts, cov)                          \
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

  def AddPosesToProblemAsObservation(self, problem, skip_first=False):
    if skip_first:
      for p in self.poses[1:]:
        p.AddToProblemAsObservation(problem)
    else:
      for p in self.poses:
        p.AddToProblemAsObservation(problem)

  def CollectError(self):
    err = np.vstack([p.param_blk[0].err for p in self.poses if p.param_blk ])
    return err

  def CollectCost(self):
    cost_func = lambda blk: blk.err.dot( np.linalg.inv(blk.sigma) ).dot(blk.err)

    cost = [ cost_func(p.param_blk[0]) for p in self.poses if p.param_blk ]
    return np.hstack(cost)

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

  def Interpolate(self, sorted_insertion_timestamp):
    sorted_timestep = self.id
    end = len(self.poses)
    i = 0
    new_p = []
    PoseType = type(self.poses[0])
    for other in sorted_insertion_timestamp:
      i = bisect_left(sorted_timestep, other, lo=i)  #  a[:i] < x <= a[i:]
      if i == 0:
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
    return super(RelTrajectory, cls).FromPoseData(M_list, cov_list, timestamp, AngleAxisPose)

  def ToAbs(self, M0=np.eye(4) ):
    PoseClass = type(self.poses[0])
    pose_list = [PoseClass( M = M0, id = 0, cov = np.zeros(6) )]

    for dp in self.poses:
      p_last = pose_list[-1]
      if np.array_equal(p_last.M, np.eye(4)): # a hack when r,t=0
        pose = PoseClass( M = dp.M,
                     id = len(pose_list),
                     cov= dp.cov)
      else:
        pose = PoseClass.Add(p_last, dp)
        pose.id = len(pose_list)
      pose_list.append(pose)

    return AbsTrajectory.FromPoseList(pose_list)


class AbsTrajectory(Trajectory):
  def __init__(self):
    super(AbsTrajectory, self).__init__()

  @classmethod
  def FromPoseData(cls, M_list, cov_list, timestamp=None):
    return super(AbsTrajectory, cls).FromPoseData(M_list, cov_list, timestamp, AngleAxisPose)

  @property
  def r(self):
    return [p.r for p in self.poses]

  @staticmethod
  @AddJacobian
  def RelT(r1,t1,r2,t2):
    r12 = axAdd(-r1,r2)
    t12 = ax2Rot(-r1).dot(t2-t1)
    return np.hstack([r12, t12])

  def SetFirstPoseFix(self, problem):
    problem.SetVarFixedWithID(self.poses[0].param_id[0])

  def ToRel(self, interval=1):
    pose_list = []

    for p_base, p_end in zip(self.poses[:-interval],self.poses[interval:]):
      if np.allclose(p_base.M, p_end.M):
        pass

      else:
        drdt, J = AbsTrajectory.RelT(p_base.r, p_base.t, p_end.r, p_end.t)
        J = np.hstack(J)
        Cov_abs = block_diag(p_base.cov, p_end.cov)
        cov_drdt = J.dot(Cov_abs).dot(J.T)

        pose = AngleAxisPose( drdt[:3], drdt[3:],
                              id = p_end.id,
                              cov= cov_drdt)
      pose_list.append(pose)

    return RelTrajectory.FromPoseList(pose_list)

  def Plot(self, scale=1, select=slice(None), **kwarg):
    PlotPose(self.M[select], scale, **kwarg)

  def Rebase(self, M=None):
    if M is None:
      M = invT(self.poses[0].M)
    for p in self.poses:
      p.ApplyTransform(M)


class CalibrationProblem(object):
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
    M21 = np.eye(4)
    M21[:3,:3], M21[:3,3] = R21, t21
    return M21

  def MakeProblemWithAbsModel(self, base, M0_dict={}):

    def AbsoluteConstraint(rt_sa, rt_wa, rt_vs):
      r_sa,t_sa = rt_sa[:3],rt_sa[3:]
      r_wa,t_wa = rt_wa[:3],rt_wa[3:]
      r_vs,t_vs = rt_vs[:3],rt_vs[3:]

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
      if not np.allclose( trj[0].M, np.eye(4) ):
        raise RuntimeWarning('First pose is not Identity, please use trj.Rebase() to make so.')
      trj.AddPosesToProblemAsObservation(problem, skip_first=True)

    Pbase = self.trajectory[base].poses[1:]

    """parameter"""
    opp_key = self.trajectory.keys()
    opp_key.remove(base)
    for key in opp_key:
      if key in M0_dict:
        Mob = M0_dict[key]
      else:
        Mob = self.SolveDirect(base, key)
        print "Init guess for %s:\n%s" %(key, Mob)
      Pob = AngleAxisPose.FromM(Mob)
      Pob.AddToProblemAsParameter(problem)
      self.calibration[key][base] = Pob

      Popp = self.trajectory[key].poses[1:]
      for p1, p2 in zip(Pbase, Popp):
        problem.AddConstraintWithID( AbsoluteConstraint,
                                     Pob.param_id,
                                     p1.param_id + p2.param_id )
    return problem

  def MakeProblemWithRelModel(self, base, M0_dict={}):

    def RelativeConstraint(rt_sa, drt_a, drt_s):
      r_sa, t_sa = rt_sa[:3],rt_sa[3:]
      dr_a, dt_a = drt_a[:3],drt_a[3:]
      dr_s, dt_s = drt_s[:3],drt_s[3:]
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
      if key in M0_dict:
        Mob = M0_dict[key]
      else:
        Mob = self.SolveDirect(base, key)
        print "Init guess for %s:\n%s" %(key, Mob)

      Pob = AngleAxisPose.FromM(Mob)
      Pob.AddToProblemAsParameter(problem)
      self.calibration[key][base] = Pob

      Popp = self.trajectory[key].poses
      for p1, p2 in zip(Pbase, Popp):
        problem.AddConstraintWithID( RelativeConstraint,
                                     Pob.param_id,
                                     p1.param_id + p2.param_id )
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
      r_err, t_err = trj.CollectRTError()
      for j in range(3):
        b[j][s].hist( r_err[:,j], bins, edgecolor='None',color='royalblue')
        c[j][s].hist( t_err[:,j], bins, edgecolor='None',color='royalblue')
      b[0][s].set_title(str(key),fontsize=20)
      c[0][s].set_title(str(key),fontsize=20)

    # y axis lable
    for i in range(3):
      b[i][0].set_ylabel(r"$\mathbf{r}_%d$" % i,fontsize=20)
      c[i][0].set_ylabel(r"$\mathbf{t}_%d$" % i,fontsize=20)

  def DetectOutliar(self):
    raise NotImplementedError()
    robust_sigma = lambda arr : 1.4825*np.median(np.abs(arr), axis=0)

    for s, (key,trj) in enumerate(self.trajectory.items()):
      r_err, t_err = trj.CollectRTError()
      r_sig, t_sig = map(robust_sigma, [r_err, t_err])


 #%% test
if __name__ == '__main__':

  if 1:
    num_sensor = 2
    num_seg = 200
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
    r_ob_all = map(rFromM, Mob_all)
    t_ob_all = map(tFromM, Mob_all)
    print Mob_all
    dM_1   = [ MfromRT( d2r(10+5*np.random.rand(1))*randsp(), 0.5*np.random.rand(1) * randsp() ) for _ in xrange(num_seg)]
    dM_all = [dM_1] + [ map(ConjugateM, dM_1, [M21]*num_seg ) for M21 in Mob_all ]

    M_all  = [ [np.eye(4)] for _ in range(num_sensor)]
    for M_trj, dM_trj in zip(M_all, dM_all):
      for dM in dM_trj:
        M_trj.append( M_trj[-1].dot( dM ) )
    r_all = deep_map(rFromM, M_all)
    t_all = deep_map(tFromM, M_all)

    fac_abs_list,fac_rel_list = [],[]
    for test in range(1):
      r_all_noisy = deep_map(add_n_noise(noise_on*0.002), r_all)
      t_all_noisy = deep_map(add_n_noise(noise_on*0.02), t_all)
      M_all_noisy = [ map(MfromRT, r_trj, t_trj) for r_trj, t_trj in zip(r_all_noisy, t_all_noisy) ]
      for j in range(num_sensor):
        M_all_noisy[j][0] = np.eye(4)
        r_all_noisy[j][0] = np.zeros(3)
        t_all_noisy[j][0] = np.zeros(3)

      print "Abs:"
      calp_abs = CalibrationProblem()
      for i in xrange(num_sensor):
        calp_abs[i]= AbsTrajectory.FromPoseData( M_all_noisy[i], block_diag(0.002**2*np.eye(3), 0.02**2*np.eye(3)) )
      init_guest = {}#{i+1:Tob for i,Tob in enumerate(Tob_all)}#
      problem_abs = calp_abs.MakeProblemWithAbsModel(0, init_guest)
      x_abs, le_abs, fac_abs = SolveWithGESparse(problem_abs, fac=True)
      fac_abs_list.append(fac_abs)

      print "Rel:"
      calp_rel = CalibrationProblem()
      for key, trj in calp_abs.trajectory.items():
        calp_rel[key] = trj.ToRel(1)
      problem_rel = calp_rel.MakeProblemWithRelModel(0, init_guest)
      x_rel, le_rel, fac_rel = SolveWithGESparse(problem_rel, fac=True)
      fac_rel_list.append(fac_rel)

    plt.hist(fac_abs_list)
    plt.hist(fac_rel_list)
#    calp_abs.FillCalibration()
#    calp_abs[0].Plot(0.2)
  #  problem.UpdateXL(True, False)
  #  print r_ob_all,t_ob_all
  #  print calp.param.values()
