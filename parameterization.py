#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  7 14:32:19 2017

@author: nubot
"""
import numpy as np
import pycppad
from numpy.testing import *

import solver2

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

  def ToGlobalCovariance(self, x, sigma):
    assert sigma.shape == (self.GlobalSize(),)*2
    J = self.ComputeJacobian(x)
    return J.T.dot(sigma).dot(J)

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

#%% AutoDiffLocalParameterization
def MakeParameterization(plus_func, x0, delta0):

  global_size = len(x0)
  local_size  = len(delta0)

  in_var = pycppad.independent( np.hstack( [x0, np.zeros(local_size)] ) )
  x_part, delta_part = np.split(in_var, [global_size])
  out_var = plus_func( x_part, delta_part )
  jac_func = pycppad.adfun(in_var, out_var).jacobian
  def jac_delta(x):
    inp = np.hstack( [x, np.zeros(local_size) ])
    J = solver2.check_nan(jac_func(inp))
    return J[:, global_size:]

  class AutoDiffLocalParameterization(LocalParameterization):
    def __init__(self):
      super(AutoDiffLocalParameterization, self).__init__()

    def Plus(self, x, delta):
      return plus_func(x, delta)

    def ComputeJacobian(self, x):
      return jac_delta(x)

    def GlobalSize(self):
      return global_size

    def LocalSize(self):
      return local_size

  return AutoDiffLocalParameterization

def test_AutoDiffLocalParameterization():
  M = np.random.rand(3, 3)
  plus_func = lambda x,delta : x + M.dot(delta)
  x0 = np.empty(3)
  l0 = np.empty(3)
  par = MakeParameterization(plus_func, x0, l0)()
  assert_array_almost_equal(par.ComputeJacobian(x0), M)

  def SubSetPlus(x, delta):
    x_ = x.copy()
    x_[:2] += delta
    return x_
  par = MakeParameterization(SubSetPlus, x0, l0[:2 ])()
  assert_array_almost_equal(par.ComputeJacobian(x0), np.eye(3)[:,:2])

  print "test_AutoDiffLocalParameterization passed "
#%% SE3Parameterization

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

def test_SE3Parameterization_solver():
  Vec = SE3Parameterization.Vec12
  Mat = SE3Parameterization.Mat44
  def InvR(x, l):
    return Vec( Mat(x) - Mat(l)  )
  import geometry
  T0 = geometry.SE3.group_from_algebra(geometry.se3.algebra_from_vector(0.1*np.random.rand(6)))
  l0 = Vec(T0)
  x0 = Vec(np.eye(4))
  problem = solver2.GaussHelmertProblem()
  problem.AddConstraintWithArray(InvR, [x0], [l0])
  problem.SetParameterization(x0, SE3Parameterization())
  problem.SetParameterization(l0, SE3Parameterization())

  try:
    x,le,fac = solver2.SolveWithGESparse(problem,fac=True)
  except solver2.CholmodNotPositiveDefiniteError:
    return

  print fac
  assert_array_almost_equal(Mat(x), T0)
  print "test_SE3Parameterization passed"

#%% SphereParameterization

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

def HomoVectorCollinearError(x, y):
  """  err = NullSpaceForVector(x).dot(y) """
  v, beta = ComputeHouseholderVector(x)
  err = y -  v * ( beta * v.dot(y) )
  return err[:-1]

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
#    if 2*norm_v_half > np.pi:
#      raise ValueError("SphereParameterization meant to be worked with vector length < np.pi ")
    y = np.hstack([ sinc_smooth(norm_v_half) * 0.5 * v, np.cos(norm_v_half) ])
    return y

  @staticmethod
  def ToEuclidean(y):
    norm_half = np.arccos(y[-1])
    div_sin_norm_half  = 1.0/np.sqrt(1 - y[-1]**2)
    return  2 * norm_half * div_sin_norm_half * y[:-1]

def test_SphereParameterization():
  param = SphereParameterization(4)
  x     = param.ToHomoSphere( np.r_[0, 0, 0.1] )

  # plus 0
  assert_array_almost_equal(x, param.Plus(x, np.r_[0,0,0.]))

  auto_jac = solver2.MakeJacobianFunction(param.Plus, x, 1e-7*np.ones(3))
  for i in range(10):
    # always on sphere
    assert_almost_equal(1, np.linalg.norm( param.Plus(x, np.random.rand(3) ) ) )
    # Jacobian
    x_ = param.ToHomoSphere( 0.5*np.random.rand(3) )
    assert_almost_equal( auto_jac(x_, np.zeros(3))[-1],
                         param.ComputeJacobian(x_) )

  print "test_SphereParameterization passed "

def test_SphereParameterization_solve():
  pa = SphereParameterization(2)
  def iden(x, y):
    return NullSpaceForVector(y).dot( x )

  x = pa.ToHomoSphere( 1.2 )

  sigma = np.atleast_2d( 0.2**2 )
  facs = np.empty(100)
  xs   = np.empty(facs.shape+(2,))
  for it in range(len(facs)):
    y = [ pa.ToHomoSphere( 1.2 + 0.2*np.random.randn(1) ) for _ in range(200) ]
    problem = solver2.GaussHelmertProblem()

    for i in range(len(y)):
      problem.AddConstraintWithArray(iden, [x], [y[i]])
      problem.SetParameterization(y[i], pa )
      problem.SetSigma(y[i], sigma)
    problem.SetParameterization(x, pa )

    xs[it],le,facs[it] = solver2.SolveWithGESparse(problem, maxit=20, fac=True)

  assert_almost_equal( np.mean(facs), 1.0, decimal=1)
  assert_almost_equal( np.mean( [pa.ToEuclidean(x_) for x_ in xs]), 1.2, decimal=1)
  print "test_SphereParameterization_solve passed "



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
  def ToSphere(y):
    """ spherecal normalize it, i.e, |y|=1"""
    return y/np.linalg.norm(y)

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

  auto_jac = solver2.MakeJacobianFunction(param.Plus, x, 1e-7*np.ones(3))
  for i in range(10):
    v = 100*np.random.rand(3)
    # always on sphere
    assert_almost_equal(1,
                        np.linalg.norm( param.Plus(x, v ) ) )
    # forward backward
    assert_almost_equal(v,
                        param.ToEuclidean( param.ToHomoSphere(v) ) )
    # Jacobian
    x_ = param.ToHomoSphere( 100*np.random.rand(3) )
    assert_almost_equal( auto_jac(x_, np.zeros(3))[-1],
                         param.ComputeJacobian(x_) )

  print "test_HomogeneousParameterization passed "

def test_HomogeneousParameterization_solve():
  pa = HomogeneousParameterization(3)
  def iden(x, y):
    return NullSpaceForVector( pa.ToHomoSphere(y) ).dot( x )

  x = pa.ToHomoSphere( [120., 120.] )

  sigma = 2.**2 * np.eye(2)
  facs = np.empty(100)
  xs   = np.empty(facs.shape+(3,))
  for it in range(len(facs)):
    y = [ 120 + 2.*np.random.randn(2) for _ in range(200) ]
    problem = solver2.GaussHelmertProblem()

    for i in range(len(y)):
      problem.AddConstraintWithArray(iden, [x], [y[i]])
      problem.SetSigma(y[i], sigma)
    problem.SetParameterization(x, pa )

    xs[it],le,facs[it] = solver2.SolveWithGESparse(problem, maxit=20, fac=True)

  assert_almost_equal( np.mean(facs), 1.0, decimal=1)
  assert_array_almost_equal( np.mean( [pa.ToEuclidean(x_) for x_ in xs], axis=0 ), np.r_[120.0,120], decimal=1)
  print "test_HomogeneousParameterization_solve passed "

def test_HomogeneousParameterization_solve2():
  pa = HomogeneousParameterization(3)
  def iden(x, y):
    return pa.ToEuclidean(x) - y

  x = pa.ToHomoSphere( [120., 120.] )

  sigma = 2.**2 * np.eye(2)
  facs = np.empty(100)
  xs   = np.empty(facs.shape+(3,))
  for it in range(len(facs)):
    y = [ 120 + 2.*np.random.randn(2) for _ in range(200) ]
    problem = solver2.GaussHelmertProblem()

    for i in range(len(y)):
      problem.AddConstraintWithArray(iden, [x], [y[i]])
      problem.SetSigma(y[i], sigma)
    problem.SetParameterization(x, pa )

    xs[it],le,facs[it] = solver2.SolveWithGESparse(problem, maxit=20, fac=True)

  assert_almost_equal( np.mean(facs), 1.0, decimal=1)
  assert_array_almost_equal( np.mean( [pa.ToEuclidean(x_) for x_ in xs], axis=0 ), np.r_[120.0,120], decimal=1)
  print "test_HomogeneousParameterization_solve passed "

#%% AngleAxisParameterization

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

def axAdd(r1, r2):
#  m1 = toRodriguez(r1)
#  m2 = toRodriguez(r2)
#  m12 = ( 4.0*(m1+m2) + 2.0*skew(m1).dot(m2) ) / (4.0 - m1.dot(m2))
#  return  fromRodriguez(m12)
  u1 = axToCayley(r1)
  u2 = axToCayley(r2)
  u12 = ( u1+u2 + skew(u1).dot(u2) ) / (1 - u1.dot(u2))
  return axFromCayley(u12)

def randsp(n=3):
    v = np.random.uniform(-1, 1, size=n)
    return v/np.linalg.norm(v)

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

AngleAxisParameterization = MakeParameterization(
                              lambda x,delta: axAdd(delta,x),
                              np.r_[0.1,0.1,0.1],
                              np.r_[0.1,0.1,0.1])

def test_AngleAxisParameterization_solve():
  pa = AngleAxisParameterization()
  def iden(x, y):
    return x - y

  x = np.random.rand(3)
  facs = np.empty(100)
  cov = 0.02**2 * np.eye(3)
  for it in range(len(facs)):
    y_list = [ axAdd(0.02*np.random.randn(3), x) for i in range(100)]
    problem = solver2.GaussHelmertProblem()
    for y in y_list:
      problem.AddConstraintWithArray(iden, [x], [y])
      problem.SetParameterization(y, pa)
      problem.SetSigma(y, cov)
    problem.SetParameterization(x, pa)

    xs,le,facs[it] = solver2.SolveWithGESparse(problem, maxit=30, fac=True)

  assert_almost_equal( np.mean(facs), 1.0, decimal=1)
  print "test_AngleAxisParameterization_solve passed "
#%%
def QuaternionProduct(z,w):
  return np.array(
         [z[0] * w[0] - z[1] * w[1] - z[2] * w[2] - z[3] * w[3],
          z[0] * w[1] + z[1] * w[0] + z[2] * w[3] - z[3] * w[2],
          z[0] * w[2] - z[1] * w[3] + z[2] * w[0] + z[3] * w[1],
          z[0] * w[3] + z[1] * w[2] - z[2] * w[1] + z[3] * w[0]] )

class QuaternionParameterization(LocalParameterization):
  def __init__(self):
    super(QuaternionParameterization, self).__init__()

  def Plus(self, x, delta):
    norm_delta = np.linalg.norm(delta)
    if norm_delta>0:
      sin_delta_by_delta = np.sin(norm_delta) / norm_delta
      q_delta = np.array([np.cos(norm_delta), sin_delta_by_delta*delta ])
      return QuaternionProduct(q_delta, x)
    else:
      return x

  def GlobalSize(self):
    return 4

  def LocalSize(self):
    return 3

  def ComputeJacobian(self, x):
    return np.array([ [-x[1], -x[2], -x[3] ],
                      [ x[0],  x[3], -x[2] ],
                      [-x[3],  x[0],  x[1] ],
                      [ x[2], -x[1],  x[0] ] ])

if __name__ == '__main__':
  test_SubsetParameterization()
  test_SE3Parameterization()
  test_SE3Parameterization_solver()

  test_AutoDiffLocalParameterization()
  test_SphereParameterization()
  test_HomogeneousParameterization()
  if 0:
    test_SphereParameterization_solve()
    test_HomogeneousParameterization_solve()
    test_AngleAxisParameterization_solve()