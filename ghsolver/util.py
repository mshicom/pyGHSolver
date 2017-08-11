#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  9 17:39:38 2017

@author: kaihong
"""
import numpy as np
import pycppad
from numpy.testing import *

def check_equal(x,y):
  if not x==y:
    raise ValueError("value not equal")
  return x

def check_nan(x):
  if not np.all( np.isfinite(x) ):
    raise ValueError("invalid value")
  return x

def check_allzero(x):
  if np.all( x == 0 ):
    raise ValueError("all zero")
  return x

def check_magnitude(r):
  if np.linalg.norm(r) > np.pi:
    raise ValueError("rotation magnitude larger than pi, will cause problems during optimizatoin")
  return r

def MakeJacobianFunction(g, *args):
  arg_sizes= [ len(np.atleast_1d(vec)) for vec in args ]

  arg_indices= np.cumsum( arg_sizes )[:-1]
  var       = np.hstack( args )
  var_in    = pycppad.independent( var )
  var_out   = np.atleast_1d( g( *np.split(var_in, arg_indices) ) )
  var_jacobian= pycppad.adfun(var_in, var_out).jacobian
  def g_jac_auto(*vec):
    J = var_jacobian( np.hstack(vec) )
    check_nan(J)
    check_allzero(J)
    return np.split(J, arg_indices, axis=1)
  return g_jac_auto

def AddJacobian(f):
  """
  Examples
  --------
  >>>  @AddJacobian
  >>>  def foo(x,y):
  >>>    return x-y
  >>>  res, jac = foo(np.ones(3), np.zeros(3))
  """
  def g(*args):
    if not hasattr(f, 'jac'):
      f.jac = MakeJacobianFunction(f, *args)
    return f(*args), f.jac(*args)
  return g

def CheckJacobianFunction(g, g_jac=None, *args):
  g_jac_auto   = MakeJacobianFunction(g, *args)
  tmp_jac_auto = g_jac_auto( *args )

  if g_jac is None:
    return tmp_jac_auto

  tmp_jac      = list( g_jac( *args ) )
  assert len(tmp_jac)==len(tmp_jac_auto)
  for a,b in zip(tmp_jac, tmp_jac_auto):
    assert_array_almost_equal(a,b)
  return

def ErrorPropogationExplicit(cov_x, func, x):
  """ y + dy = func(x + dx)
      dy = J dx, J = df/dx,
      cov_y = J cov_x J'
  """
  J = MakeJacobianFunction(func, x)(x)[0]
  return J.dot(cov_x).dot(J.T)
  def test():
    f = np.sin
    x = np.r_[0.2]
    var = np.r_[0.1]
    var_est = ErrorPropogationExplicit(var, f, x)
    var_exp = np.cos(x)**2 * var
    assert_almost_equal(var_est, var_exp)

def NullMatrix(A):
  A = np.atleast_2d(A)
  Q,R = np.linalg.qr(A.T, 'complete')
  rank = A.shape[0]
  return Q[:, rank:]

def ErrorPropogationImplicit(cov_x, constaint_func, x, y=None):
  """ G(x)=0
      dG = J dx = 0
      dy = J NullSpace(J)*dx
  """
  J = MakeJacobianFunction(func, x)(x)[0]
  Jn = NullMatrix(J)
  return Jn.dot(cov_x).dot(Jn.T)

def Montacalo(f, x0, sigma, trial=10000):
  dim_x, dim_y = len(np.atleast_1d(x)), len(np.atleast_1d(f(x0)))

  y = np.empty((trial, dim_y))
  for i in xrange(trial):
    y[i] = f( x0 + sigma*np.random.randn(dim_x) )
  y_mean = np.mean(y, axis=0)
  y_cov = np.cov((y - y_mean).T)
  return y_cov, y_mean

#cov_r1_r2 = 0.1**2 * np.eye(6)
#f = lambda x: axAdd(-x[:3],x[3:])
#x0 = np.r_[0.1,0,0, 0.2,0.3,0.1]
#cov_est = ErrorPropogationExplicit(cov_r1_r2, f, x0)
#cov_mon = Montacalo(f, x0, 0.1)

#%%
from vtk_visualizer import get_vtk_control
import vtk
from vtk.util import numpy_support

def AxesPolyData():
   # Create input point data.
  newPts = vtk.vtkPoints()
  newLines = vtk.vtkCellArray()
  newScalars = vtk.vtkUnsignedCharArray()
  newScalars.SetNumberOfComponents(3)
  newScalars.SetName("Colors")

  ptIds = [None, None]
  for i in range(3):
    ptIds[0] = newPts.InsertNextPoint( [0,0,0] )
    ptIds[1] = newPts.InsertNextPoint( np.roll([1,0,0], i))
    newLines.InsertNextCell(2, ptIds)

    c = np.roll([255,0,0], i)
    newScalars.InsertNextTuple3(*c)
    newScalars.InsertNextTuple3(*c)

  # Add the lines to the polydata container
  output = vtk.vtkPolyData()
  output.SetPoints(newPts)
  output.GetPointData().SetScalars(newScalars)
  output.SetLines(newLines)
  return output
_axes_pd = AxesPolyData()

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
  tensorGlyph.ColorGlyphsOff()
  tensorGlyph.ThreeGlyphsOff()
  tensorGlyph.ExtractEigenvaluesOff()
  tensorGlyph.Update()

  # tensorGlyph -> actor
  mapper = vtk.vtkPolyDataMapper()
  try:
    mapper.SetInput(tensorGlyph.GetOutput())
  except:
    mapper.SetInputData(tensorGlyph.GetOutput())


  pose_actor = vtk.vtkActor()
  pose_actor.GetProperty().SetLineWidth(1.5)
  pose_actor.SetMapper(mapper)

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
  line_actor.GetProperty().SetLineStipplePattern(0xf0f0)
  line_actor.GetProperty().SetLineStippleRepeatFactor(1)
  line_actor.GetProperty().SetLineWidth(1.5)

  if not hold:
    get_vtk_control().RemoveAllActors()
  get_vtk_control().AddActor(pose_actor)
  get_vtk_control().AddActor(line_actor)
