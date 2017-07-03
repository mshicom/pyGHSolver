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
    super(SubsetParameterization, self).__init__()
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
    super(AutoDiffLocalParameterization, self).__init__()

    self.plus_func = plus_func
    self.global_size = len(x0)
    self.local_size  = len(delta0)

    in_var = pycppad.independent( np.hstack( [x0, np.zeros(self.local_size)] ) )
    x_part, delta_part = np.split(in_var, [self.global_size])
    out_var = plus_func( x_part, delta_part )
    jac_func = pycppad.adfun(in_var, out_var).jacobian

    def jac_delta(x):
      inp = np.hstack( [x, np.zeros(self.local_size) ])
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
    super(SE3Parameterization, self).__init__()

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

def sinc_smooth(x):
  if np.abs(x) < 1e-6:
    xx = x**2
    return 1.0 + xx * (xx/120.0 - 1.0/6.0) # 1 - (x^2)/6 + (x^4)/120 + O(x^6)
  else:
    return np.sin(x)/x


def ComputeHouseholderVector(x):
  """ Algorithm 5.1.1 from 'Matrix Computations' by Golub et al. (Johns Hopkins
      Studies in Mathematical Sciences) but using the nth element of the input
      vector as pivot instead of first. This computes the vector v with v(n) = 1
      and beta such that H = I - beta * v * v^T is orthogonal and
      H * x = ||x||_2 * e_n.
  """
  sigma = np.sum(x[:-1]**2)
  v     = x.copy()
  v[-1] = 1.0
  beta  = 0.0
  x_pivot = x[-1]
  if (sigma <= 1e-12):
    if x_pivot < 0.:
      beta = 2.0
    return v, beta

  mu = np.sqrt( x_pivot**2 + sigma )
  v_pivot = 1.0
  if x_pivot <= 0.:
    v_pivot = x_pivot - mu
  else:
    v_pivot = -sigma / (x_pivot + mu)

  beta = 2.0 * v_pivot**2 / (sigma + v_pivot**2)
  v[:-1] /= v_pivot
  return v, beta

def HouseholderMatrix(x):
  """a Householder matrix H(x) of vector x can transform x to lie
     along the coordinate axis, i.e, H(x) * x = (0, . . . , 0, 1),
     which can be seen as on sphere if ||x||=1
  """
  v, beta = ComputeHouseholderVector(x)
  H = np.eye(len(v)) - v.reshape(-1,1) * ( beta * v )
  return H

def NullSpaceForVector(x):
  if 1:
    return HouseholderMatrix(x)[:-1, :]
  else:
    Q,R = np.linalg.qr(x.reshape(-1,1), 'complete')
    return Q[:, 1:].T

class SphereParameterization(LocalParameterization):
  """ This provides a parameterization for homogeneous vectors which are commonly
   used in Structure for Motion problems.  One example where they are used is
   in representing points whose triangulation is ill-conditioned. Here
   it is advantageous to use an over-parameterization since homogeneous vectors
   can represent points at infinity.

   The plus operator is defined as
   Plus(x, delta) =
      [sin(0.5 * |delta|) * delta / |delta|, cos(0.5 * |delta|)] * x
   with * defined as an operator which applies the update orthogonal to x to
   remain on the sphere. We assume that the last element of x is the scalar
   component. The size of the homogeneous vector is required to be greater than
   1.
  """
  def __init__(self, dim):
    super(SphereParameterization, self).__init__()
    self.dim = dim
    self.jacobian = np.empty((self.dim, self.dim-1))

  def GlobalSize(self):
    return self.dim

  def LocalSize(self):
    return self.dim-1

  def Plus(self, x, delta):
    y = self.ToHomoSphere(delta)

    v, beta = ComputeHouseholderVector(x)
    x_plus_delta = np.linalg.norm(x) * (y -  v * ( beta * v.dot(y) ) )
    return x_plus_delta

  @staticmethod
  def Minus(x, y):
    return NullSpaceForVector(x).dot(y)

  def ComputeJacobian(self, x):
    """ The Jacobian is equal to J = 0.5 * H.leftCols(size_ - 1) where H is the
        Householder matrix (H = I - beta * v * v').
    """
    J = self.jacobian
    v, beta = ComputeHouseholderVector(x)
    for i in range(self.dim-1):
      J[:,i]  = -0.5 * beta * v[i] * v
      J[i,i] += 0.5
    J *= np.linalg.norm(x)
    return J

  @staticmethod
  def ToHomoSphere(v):
    """ extend v to homogeneous vector y and spherecal normalize it, i.e, |y|=1
          y = [sinc(|v|/2) * v/2 , cos(|v|/2)]
    """
    norm_v_half = 0.5*np.linalg.norm(v)
    y = np.hstack([ sinc_smooth(norm_v_half) * 0.5 * v, np.cos(norm_v_half) ])
    return y

  @staticmethod
  def ToEuclidean(y, half=True):
    norm_half = np.arccos(y[-1])
    div_sin_norm_half  = 1.0/np.sqrt(1 - y[-1]**2)
    return  2 * norm_half * div_sin_norm_half * y[:-1]

def test_SphereParameterization():
  x     = np.r_[0,0,0,1.]
  param = SphereParameterization(4)

  # plus 0
  assert_array_equal(x, param.Plus(x, np.r_[0,0,0.]))
  # always on sphere
  for i in range(10):
    assert_almost_equal(1, np.linalg.norm( param.Plus(x, np.random.rand(3) ) ) )

  # close approximate
  x     = np.r_[1, 0.5, 0.2, 0]
  delta = np.r_[0.001, 0.005, 0.002]
  assert_array_almost_equal(x + param.ComputeJacobian(x).dot(delta),
                            param.Plus(x, delta),
                            4)

  print "test_SphereParameterization passed "



class HomogeneousParameterization(LocalParameterization):
  """ This provides a parameterization for homogeneous vectors which are commonly
   used in Structure for Motion problems.  One example where they are used is
   in representing points whose triangulation is ill-conditioned. Here
   it is advantageous to use an over-parameterization since homogeneous vectors
   can represent points at infinity.

   The plus operator is defined as
   Plus(x, delta) =
      [...]
   with * defined as an operator which applies the update orthogonal to x to
   remain on the sphere. We assume that the last element of x is the scalar
   component. The size of the homogeneous vector is required to be greater than
   1.
  """
  def __init__(self, dim):
    super(HomogeneousParameterization, self).__init__()
    self.dim = dim
    self.jacobian = np.empty((self.dim, self.dim-1))

  def GlobalSize(self):
    return self.dim

  def LocalSize(self):
    return self.dim-1

  def Plus(self, x, delta):
    y = self.ToHomoSphere(delta)
    v, beta = ComputeHouseholderVector(x)
    y_rotated = y -  v * ( beta * v.dot(y) )
    x_plus_delta = y_rotated/np.linalg.norm(y_rotated)
    return x_plus_delta


  def ComputeJacobian(self, x):
    J = self.jacobian
    v, beta = ComputeHouseholderVector(x)
    for i in range(self.dim-1):
      J[:,i]  = -beta * v[i] * v
      J[i,i] += 1
    J /= np.linalg.norm(x)
    return J

  @staticmethod
  def ToHomoSphere(v, positive=True):
    """ extend v to homogeneous vector y and spherecal normalize it, i.e, |y|=1
          y = normalized( [v, 1] )
        Notice it only cover the half sphere since the last element is always positive.
    """
    y = np.hstack([v, 1.0]) if positive else np.hstack([v, -1.0])
    y /= np.linalg.norm(y)
    return y

  @staticmethod
  def ToEuclidean(y):
    return y[:-1]/y[-1]

  @staticmethod
  def Minus(x, y):
    return NullSpaceForVector(x).dot(y)

def test_HomogeneousParameterization():
  x0 = np.r_[0, 0, 1.]
  x  = HomogeneousParameterization.ToHomoSphere(x0)
  param = HomogeneousParameterization( len(x) )

  # plus 0
  assert_array_almost_equal( x,
                             param.Plus(x, np.zeros(3) ) )

  delta = np.r_[0,0,5.]
#  assert_array_equal( x0 + delta,
#                     param.ToEuclidean( param.Plus(x, delta) ) )

  # always on sphere
  for i in range(10):
    assert_almost_equal(1,
                        np.linalg.norm( param.Plus(x, np.random.rand(3) ) ) )

  print "test_HomogeneousParameterization passed "


if __name__ == '__main__':
  test_SubsetParameterization()
  test_SE3Parameterization()
  test_AutoDiffLocalParameterization()
  test_SphereParameterization()
  test_HomogeneousParameterization()