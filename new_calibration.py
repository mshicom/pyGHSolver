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

def CovFromRT(cov_r, cov_t):
  return scipy.linalg.block_diag(cov_r, cov_t)

class Pose(object):
  __slots__ = 'id','T','rt','cov_rt','rt_id','rt_blk'
  def __init__(self, T, id=None, cov_rt=None):
    self.id = id
    self.rt = np.hstack(rtFromT(T))
    self.cov_rt = cov_rt

    self.rt_id, self.rt_blk = None,None
    check_magnitude(self.r)

  @property
  def r(self):
    return self.rt[:3]
  @property
  def t(self):
    return self.rt[3:]

  def __repr__(self):
    return "Pose %s:(%s,%s)" % (self.id, self.r, self.t)

  def AddToProblemAsObservation(self, problem):
    self.rt_id, self.rt_blk = problem.AddObservation([self.rt])
    if not self.cov_rt is None:
      problem.SetSigmaWithID(self.rt_id[0], self.cov_rt)


  def AddToProblemAsParameter(self, problem):
    self.rt_id, self.rt_blk = problem.AddParameter([self.rt])

  def ApplyTransform(self, T):
    T_new = np.dot(T, self.T)
    self.rt[:] = np.hstack(rtFromT(T_new))

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
  def FromPoseData(cls, T_list, cov_rt_list, timestamp=None):
    num_pos = len(T_list)
    if not isinstance(cov_rt_list, list):
      cov_rt_list = [cov_rt_list]*num_pos

    trj = cls()
    if timestamp is None:
      trj.poses = [Pose(T, id, cov_rt)                            \
                      for id,(T, cov_rt)                          \
                      in  enumerate(zip(T_list, cov_rt_list))]
    else:
      trj.poses = [Pose(T, ts, cov_rt)                          \
                      for ts, T, cov_rt                          \
                      in  zip(timestamp, T_list, cov_rt_list)]
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

  def CollectRTError(self):
    err = np.vstack([p.rt_blk[0].err for p in self.poses if p.rt_blk ])
    return err[:3], err[3:]

  def CollectRTCost(self):
    cost_func = lambda blk: blk.err.dot( np.linalg.inv(blk.sigma) ).dot(blk.err)

    cost = [ cost_func(p.rt_blk[0]) for p in self.poses if p.rt_blk ]
    return np.hstack(cost)

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

  def __len__(self):
    return len(self.poses)

  def __getitem__(self, key):
    return self.poses[key]

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

  @staticmethod
  @AddJacobian
  def AbsT(r1,t1,dr,dt):
    r2 = axAdd(r1,dr)
    t2 = t1 + ax2Rot(r1).dot(dt)
    return np.hstack([r2, t2])

  def ToAbs(self, T0=np.eye(4) ):
    RelTrajectory.AbsT(*[np.random.rand(3)]*4)
    pose_list = [Pose( T = T0, id = 0, cov_rt = np.zeros(6) )]

    for dp in self.poses:
      p_last = pose_list[-1]
      if np.array_equal(p_last.T, np.eye(4)): # a hack when r,t=0
        pose = Pose( T = dp.T,
                     id = len(pose_list),
                     cov_rt= dp.cov_rt)
      else:
        rt, J = RelTrajectory.AbsT(p_last.r, p_last.t, dp.r, dp.t)

        J = np.hstack(J)
        Cov_rel = scipy.linalg.block_diag(p_last.cov_rt, dp.cov_rt)
        cov_rt = J.dot(Cov_rel).dot(J.T)

        pose = Pose( T = MfromRT(rt[:3],rt[3:]),
                    id = len(pose_list),
                    cov_rt= cov_rt)
      pose_list.append(pose)

    return AbsTrajectory.FromPoseList(pose_list)


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
      if np.allclose(p_base.T, p_end.T):
        pass

      else:
        drdt, J = AbsTrajectory.RelT(p_base.r, p_base.t, p_end.r, p_end.t)
        J = np.hstack(J)
        Cov_abs = scipy.linalg.block_diag(p_base.cov_rt, p_end.cov_rt)
        cov_drdt = J.dot(Cov_abs).dot(J.T)

        pose = Pose( T = MfromRT(drdt[:3],drdt[3:]),
                    id = p_end.id,
                    cov_rt= cov_drdt)
      pose_list.append(pose)

    return RelTrajectory.FromPoseList(pose_list)

  def Plot(self, scale=1, select=slice(None), **kwarg):
    PlotPose(self.T[select], scale, **kwarg)

  def Rebase(self, T=None):
    if T is None:
      T = invT(self.poses[0].T)
    for p in self.poses:
      p.ApplyTransform(T)


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
    T21 = np.eye(4)
    T21[:3,:3], T21[:3,3] = R21, t21
    return T21

  def MakeProblemWithAbsModel(self, base, T0_dict={}):

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
      if not np.allclose( trj[0].T, np.eye(4) ):
        raise RuntimeWarning('First pose is not Identity, please use trj.Rebase() to make so.')
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
      self.calibration[key][base] = Pob

      Popp = self.trajectory[key].poses[1:]
      for p1, p2 in zip(Pbase, Popp):
        problem.AddConstraintWithID( AbsoluteConstraint,
                                     Pob.rt_id,
                                     p1.rt_id + p2.rt_id )
    return problem

  def MakeProblemWithRelModel(self, base, T0_dict={}):

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
      if key in T0_dict:
        Tob = T0_dict[key]
      else:
        Tob = self.SolveDirect(base, key)
        print "Init guess for %s:\n%s" %(key, Tob)

      Pob = Pose(Tob)
      Pob.AddToProblemAsParameter(problem)
      self.calibration[key][base] = Pob

      Popp = self.trajectory[key].poses
      for p1, p2 in zip(Pbase, Popp):
        problem.AddConstraintWithID( RelativeConstraint,
                                     Pob.rt_id,
                                     p1.rt_id + p2.rt_id )
    return problem

  def FillCalibration(self):
    from itertools import permutations
    for key1,key2 in permutations( self.trajectory.keys(), 2):
      if key1 in self.calibration and key2 in self.calibration[key1]:
        pose12 = self.calibration[key1][key2]
        self.calibration[key2][key1] = Pose( invT(pose12.T) )

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
    robust_sigma = lambda arr : 1.4825*np.median(np.abs(arr), axis=0)

    for s, (key,trj) in enumerate(self.trajectory.items()):
      r_err, t_err = trj.CollectRTError()
      r_sig, t_sig = map(robust_sigma, [r_err, t_err])

#%%
from vtk_visualizer import plotxyz, get_vtk_control
import vtk
from vtk.util import numpy_support
viz = get_vtk_control()
viz.RemoveAllActors()

def AxesPolyData():
   # Create input point data.
  lpts = vtk.vtkPoints()
  lpts.InsertPoint(0, (0,0,0))
  lpts.InsertPoint(1, (1,0,0))
  lpts.InsertPoint(2, (0,1,0))
  lpts.InsertPoint(3, (0,0,1))

  lines = vtk.vtkCellArray()
  for i in xrange(1,4):
    l = vtk.vtkLine()
    l.GetPointIds().SetId(0, 0)
    l.GetPointIds().SetId(1, i)
    lines.InsertNextCell(l)

  # Create a vtkUnsignedCharArray container and store the colors in it
  colors = vtk.vtkUnsignedCharArray()
  colors.SetNumberOfComponents(3)
  colors.SetName("Colors")
  colors.InsertNextTuple3(255,0,0)
  colors.InsertNextTuple3(0,255,0)
  colors.InsertNextTuple3(0,0,255)

  # Add the lines to the polydata container
  linesPolyData = vtk.vtkPolyData()
  linesPolyData.SetPoints(lpts)
  linesPolyData.SetLines(lines)
  linesPolyData.GetCellData().SetScalars(colors)
  return linesPolyData
_axes_pd = AxesPolyData()
cubeSource = vtk.vtkCubeSource()

def PlotPose(pose, scale=1, inv=False, base=None, hold=False, color=(255,255,255)):
  if inv:
    pose = map(invT, pose)
  if not base is None:
    pose = map(lambda p:np.dot(base,p), pose )

  R_list = [p[:3,:3] for p in pose]
  t_list = [p[:3,3]  for p in pose]

  # pose matrix -> PolyData
  points = vtk.vtkPoints()  # where t goes
  polyLine = vtk.vtkPolyLine()
  if 1:
    tensors = vtk.vtkDoubleArray() # where R goes, column major
    tensors.SetNumberOfComponents(9)
    for i,(R,t) in enumerate(zip(R_list,t_list)):
      points.InsertNextPoint(*tuple(t))
      tensors.InsertNextTupleValue( tuple(R.ravel(order='F')) )
      polyLine.GetPointIds().InsertNextId(i)
  else:
    ts = np.hstack(t_list)
    Rs_flat = np.hstack([ R.ravel(order='F') for R in R_list])
    points.SetData(numpy_support.numpy_to_vtk(ts))
    tensors = numpy_support.numpy_to_vtk( Rs_flat )

  polyData = vtk.vtkPolyData()
  polyData.SetPoints(points)
  polyData.GetPointData().SetTensors(tensors)

  # PolyData -> tensorGlyph
  tensorGlyph= vtk.vtkTensorGlyph()
  try:
    tensorGlyph.SetInput(polyData)
  except:
    tensorGlyph.SetInputData(polyData)
  tensorGlyph.SetScaleFactor(scale)
  tensorGlyph.SetSourceData( _axes_pd )
#  tensorGlyph.SetSourceConnection( cubeSource.GetOutputPort() )
  tensorGlyph.ColorGlyphsOff()
#  tensorGlyph.SetColorModeToScalars()
  tensorGlyph.ThreeGlyphsOff()
  tensorGlyph.ExtractEigenvaluesOff()
  tensorGlyph.Update()

  # tensorGlyph -> actor
  mapper = vtk.vtkPolyDataMapper()
  try:
    mapper.SetInput(tensorGlyph.GetOutput())
  except:
    mapper.SetInputData(tensorGlyph.GetOutput())
#  mapper.ScalarVisibilityOn()
#  mapper.SetScalarModeToUseCellData()

  pose_actor = vtk.vtkActor()
  pose_actor.SetMapper(mapper)
#  pose_actor.GetProperty().SetColor(255, 0, 0)

  # connect the pose a color line
  polyLine_cell = vtk.vtkCellArray()
  polyLine_cell.InsertNextCell(polyLine)
  polyLine_pd = vtk.vtkPolyData()
  polyLine_pd.SetPoints(points)
  polyLine_pd.SetLines(polyLine_cell)
  lmapper = vtk.vtkPolyDataMapper()
  try:
    lmapper.SetInput(polyLine_pd)
  except:
    lmapper.SetInputData(polyLine_pd)
  line_actor = vtk.vtkActor()
  line_actor.SetMapper(lmapper)
  line_actor.GetProperty().SetColor(*color)

  if not hold:
    viz.RemoveAllActors()
  viz.AddActor(pose_actor)
  viz.AddActor(line_actor)


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

    ConjugateT = lambda dT1, T21 : T21.dot(dT1).dot(invT(T21))
    d2r =  lambda deg: np.pi*deg/180
    def deep_map(function, list_of_list):
      if not isinstance(list_of_list[0], list):
        return map(function, list_of_list)
      return [ deep_map(function, l) for l in list_of_list ]

    Tob_all = [ MfromRT(randsp(), randsp()) for _ in xrange(num_sensor-1) ] # other <- base
    r_ob_all = map(rFromT, Tob_all)
    t_ob_all = map(tFromT, Tob_all)
    print Tob_all
    dT_1   = [ MfromRT( d2r(10+5*np.random.rand(1))*randsp(), 0.5*np.random.rand(1) * randsp() ) for _ in xrange(num_seg)]
    dT_all = [dT_1] + [ map(ConjugateT, dT_1, [T21]*num_seg ) for T21 in Tob_all ]

    T_all  = [ [np.eye(4)] for _ in range(num_sensor)]
    for T_trj, dT_trj in zip(T_all, dT_all):
      for dT in dT_trj:
        T_trj.append( T_trj[-1].dot( dT ) )
    r_all = deep_map(rFromT, T_all)
    t_all = deep_map(tFromT, T_all)

    fac_abs_list,fac_rel_list = [],[]
    for test in range(100):
      r_all_noisy = deep_map(add_n_noise(noise_on*0.002), r_all)
      t_all_noisy = deep_map(add_n_noise(noise_on*0.02), t_all)
      T_all_noisy = [ map(MfromRT, r_trj, t_trj) for r_trj, t_trj in zip(r_all_noisy, t_all_noisy) ]
      for j in range(num_sensor):
        T_all_noisy[j][0] = np.eye(4)
        r_all_noisy[j][0] = np.zeros(3)
        t_all_noisy[j][0] = np.zeros(3)

      print "Abs:"
      calp_abs = CalibrationProblem()
      for i in xrange(num_sensor):
        calp_abs[i]= AbsTrajectory.FromPoseData( T_all_noisy[i], scipy.linalg.block_diag(0.002**2*np.eye(3), 0.02**2*np.eye(3)) )
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
