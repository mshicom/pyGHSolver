#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  7 14:32:19 2017

@author: nubot
"""
import numpy as np
import pycppad
from numpy.testing import *


from abc import ABCMeta, abstractmethod
class LocalParameterization(object):
  __metaclass__ = ABCMeta

  def __init__(self):
    self.jacobian = None

  @abstractmethod
  def GlobalSize(self):
    """ Size of x """

  @abstractmethod
  def LocalSize(self):
    """ Size of delta """

  @abstractmethod
  def Plus(self, x, delta):
    """ Generalization of the addition operation.

    x_plus_delta = Plus(x, delta)
    with the condition that Plus(x, 0) = x.
    """

  @abstractmethod
  def ComputeJacobian(self, x):
    """ Return the GlobalSize() x LocalSize() row-major jacobian matrix.
        The jacobian of Plus(x, delta) w.r.t delta at delta = 0.
    """

  def UpdataJacobian(self, x):
    self.jacobian = self.ComputeJacobian(x).copy()

  def ToLocalJacobian(self, Jx):
    """ Return J_delta given J_x.
        Acording to the Chain Rule of differentiation,
          J_delta = f'(x) * Plus'(x) =  Jx * jacobian
    """
    return Jx.dot(self.jacobian)

class IdentityParameterization(LocalParameterization):
  def __init__(self, size):
    self.size = size
    self.jacobian = np.eye(self.size)

  def Plus(self, x, delta):
    return x + delta

  def GlobalSize(self):
    return self.size

  def LocalSize(self):
    return self.size

  def ComputeJacobian(self, x):
    return self.jacobian

  def ToLocalJacobian(self, Jx):
    return Jx

  def UpdataJacobian(self, x):
    pass

class SubsetParameterization(LocalParameterization):
  def __init__(self, active_parameters_mask):
    self.active_mask = np.array(active_parameters_mask, bool)
    self.active_indice, = np.where(self.active_mask)

    self.global_size = len(active_parameters_mask)
    self.local_size  = len(self.active_indice)

    self.jacobian = np.eye(self.global_size)[:,self.active_indice]

  def Plus(self, x, delta):
    x_ = x.copy()
    x_[self.active_indice] += delta
    return x_

  def ComputeJacobian(self, x):
    return self.jacobian

  def UpdataJacobian(self, x):
    pass

  def ToLocalJacobian(self, Jx):
    return np.compress(self.active_mask, Jx, axis=1)  # = Jx[:, self.active_indice]

  def GlobalSize(self):
    return self.global_size

  def LocalSize(self):
    return self.local_size

def test_SubsetParameterization():
  par = SubsetParameterization([1,0,0])
  assert(par.GlobalSize() == 3)
  assert(par.LocalSize()  == 1)
  assert( np.all( par.ComputeJacobian(None) == np.array([ [1],[0],[0] ]) ) )
  print "test_SubsetParameterization passed "

class AutoDiffLocalParameterization(LocalParameterization):
  def __init__(self, plus_func, x0, delta0):
    self.plus_func = plus_func
    self.global_size = len(x0)
    self.local_size  = len(delta0)

    in_var = pycppad.independent( np.hstack( [x0, np.zeros(self.local_size)] ) )
    x_part, delta_part = np.split(in_var, [self.global_size])
    out_var = plus_func( x_part, delta_part )
    jac_func = pycppad.adfun(in_var, out_var).jacobian

    def jac_delta(x):
      inp = np.hstack( [x0, np.zeros(self.local_size) ])
      J = jac_func(inp)
      return J[:, self.global_size:]
    self.jac_delta = jac_delta

  def Plus(self, x, delta):
    return self.plus_func(x, delta)

  def ComputeJacobian(self, x):
    return self.jac_delta(x)

  def GlobalSize(self):
    return self.global_size

  def LocalSize(self):
    return self.local_size

def test_AutoDiffLocalParameterization():
  M = np.random.rand(3, 3)
  plus_func = lambda x,delta : x + M.dot(delta)
  x0 = np.empty(3)
  l0 = np.empty(3)
  par = AutoDiffLocalParameterization(plus_func, x0, l0)
  assert( np.all( par.ComputeJacobian(x0) == M ) )

  def SubSetPlus(x, delta):
    x_ = x.copy()
    x_[:2] += delta
    return x_
  par = AutoDiffLocalParameterization(SubSetPlus, x0, l0[:2 ])
  assert( np.all( par.ComputeJacobian(x0) == np.eye(3)[:,:2] ) )

  print "test_AutoDiffLocalParameterization passed "

def skew(v):
    return np.array([[   0, -v[2],  v[1]],
                     [ v[2],    0, -v[0]],
                     [-v[1], v[0],    0 ]])
import scipy
class SE3Parameterization(LocalParameterization):
  """ se(3) = [t, omega]
  A tutorial on SE(3) transformation parameterizations and on-manifold optimization
  [https://pixhawk.org/_media/dev/know-how/jlblanco2010geometry3d_techrep.pdf]
  """
  @staticmethod
  def Vec12(T): # column major flat
    return T[:3,:4].ravel('F')
  @staticmethod
  def Mat34(x):
    return np.reshape(x,(3,4),order='F')
  @staticmethod
  def Mat44(x):
    return np.vstack([ np.reshape(x,(3,4),order='F'), [ 0, 0, 0, 1]])

  # shared workspace for all instance
  J = np.zeros((12,6))
  J[9:12, 0:3] = np.eye(3)

  def __init__(self):
    self.jacobian = None

  def GlobalSize(self):
    """ Size of x """
    return 12

  def LocalSize(self):
    """ Size of delta """
    return 6

  def Plus(self, x, delta):
    """ x_new = exp(delta) * R(x) """
    t,omega = np.split(delta,2)
    A = np.zeros((4,4))
    A[:3,:3] = skew(omega)
    A[:3, 3] = t
    D = scipy.linalg.expm(A)
    DX = D.dot(SE3Parameterization.Mat44(x))
    return SE3Parameterization.Vec12(DX)

  def ComputeJacobian(self, x):
    """exp(ε) * R(x)"""
    c1,c2,c3,t = np.split(x, 4)
    SE3Parameterization.J[0:3 , 3:6] = skew(-c1)
    SE3Parameterization.J[3:6 , 3:6] = skew(-c2)
    SE3Parameterization.J[6:9 , 3:6] = skew(-c3)
    SE3Parameterization.J[9:12, 3:6] = skew(-t)
    return SE3Parameterization.J

def test_SE3Parameterization():
  import geometry

  p = SE3Parameterization()
  T0 = geometry.SE3.sample_uniform()
  x0 = SE3Parameterization.Vec12(T0)

  dx  = 0.05*np.random.rand(6) # [t, omega]
  dx_ = np.r_[dx[3:],dx[:3]]  # [omega, t]

  T1_ = geometry.SE3.group_from_algebra(geometry.se3.algebra_from_vector(dx_)).dot(T0)
  T1  = SE3Parameterization.Mat44(p.Plus(x0, dx))
  assert_array_almost_equal(T1_, T1)

  dT = p.ComputeJacobian(x0).dot(dx)
  assert_array_almost_equal( SE3Parameterization.Vec12(T1-T0), dT, 2)
  print "test_SE3Parameterization passed "

if __name__ == '__main__':
  test_SubsetParameterization()
  test_SE3Parameterization()
  test_AutoDiffLocalParameterization()