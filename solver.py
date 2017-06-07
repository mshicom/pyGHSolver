#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 28 10:08:11 2017

@author: kaihong
"""
import numpy as np
import scipy
import scipy.sparse
import matplotlib.pyplot as plt
from collections import namedtuple
from batch_optimize import *

import pycppad
#from cvxopt import matrix, spmatrix

def GenerateJacobianFunction(g, x_list, l_list):
  x_sizes = [x.size for x in x_list]
  l_sizes = [l.size for l in l_list]
  xl_indices = np.cumsum(x_sizes + l_sizes)[:-1]
  x_indices = np.cumsum([0]+x_sizes)
  l_indices = np.cumsum([0]+l_sizes)

  var       = np.hstack(x_list+l_list )
  var_ad    = pycppad.independent( var )
  var_jacobian= pycppad.adfun(var_ad, g( *np.split(var_ad, xl_indices) ) ).jacobian

  x_slc = [slice(start, stop) for start, stop in zip(x_indices[:-1], x_indices[1:])]
  l_slc = [slice(start, stop) for start, stop in zip(l_indices[:-1], l_indices[1:])]
  def g_jacobian(x,l):
    var = np.hstack([x[slc] for slc in x_slc ] + [l[slc] for slc in l_slc ] )
    J = var_jacobian(var)
    return np.split(J, xl_indices, axis=1)
  return g_jacobian

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
    self.jacobian = self.ComputeJacobian(x)

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



ConstraintBlock = namedtuple('ConstraintBlock', ['g_flat', 'g_jacobian',
                                                 'r_slc', 'x_var', 'l_var'])
class VariableBlock():
  __slots__ = 'addr','data','place','place_local','param'
  def __init__(self, array):
    if np.isscalar(array):
      raise RuntimeError("Input cannot be scalar, use x=np.empty(1) ")
    self.data = array
    self.addr = array.__array_interface__['data'][0]
    self.param = IdentityParameterization(array.shape[0])

  def SetPlace(self, offset):
    size = self.data.shape[0] #self.param.LocalSize()
    self.place = slice(offset, offset + size)
    return size

  def SetLocalPlace(self, offset):
    size = self.param.LocalSize()
    self.place_local = slice(offset, offset + size)
    return size

  def __hash__(self):
    return hash(self.addr)

  def __eq__(self, other):
    return self.data.shape == other.data.shape and self.addr == other.addr

class ObservationBlock(VariableBlock):
  __slots__ = '_sigma','weight'
  def __init__(self, array):
    VariableBlock.__init__(self, array)
    self._sigma = self.weight = np.ones(array.shape[0])

  @property
  def sigma(self):
    return self._sigma

  @sigma.setter
  def SetSigma(self, sigma):
    self._sigma = sigma
    self.weight = 1./sigma

  def ComputeCTWCInvAndE(self, le):
    C = self.param.ComputeJacobian(le)
    CTW = C.T * self.weight       #  = C.T.dot( np.diag(self.weight) )
    CTWC_inv = np.linalg.inv(CTW.dot(C))
    e = CTWC_inv.dot( CTW.dot(le) )
    return CTWC_inv, e

  def ComputeCTWCAndCTWE(self, le):
    C = self.param.ComputeJacobian(le)
    CTW = C.T * self.weight       #  = C.T.dot( np.diag(self.weight) )
    CTWC = CTW.dot(C)
    CTWe = CTW.dot(le)
    return CTWC, CTWe

def test_VariableBlock():
  a = np.empty((10,10))
  b = np.arange(3)
  assert len( { VariableBlock(a[0  ]), VariableBlock(b) } ) == 2             # different variable
  assert len( { VariableBlock(a[0  ]), VariableBlock(a[   1]) } ) == 2       # same length, different row
  assert len( { VariableBlock(a[0, :5]), VariableBlock(a[0,  :4]) } ) == 2   # diff length, same head
  assert len( { VariableBlock(a[0, :5]), VariableBlock(a[0, 1:5]) } ) == 2   # diff length, same end
  s = np.empty(1)
  assert len( { VariableBlock(s), VariableBlock(s) } ) == 1       # scalar

class GaussHelmertProblem:
  def __init__(self):
    # map from x (nparray.data) to x_id (int)
    self.parameter_offset = 0
    self.parameter_local_offset = -1

    self.parameter_dict = {}

    # map from l (nparray.base) to l_id (int)
    self.observation_offset = 0
    self.observation_local_offset = -1

    self.observation_dict = {}
    self.observation_weight = {}

    self.constraint_offset = 0
    self.constraint_blocks = []    # list of ConstraintBlock
    self.variance_factor = -1.0

  def AddParameter(self, parameter_list):
    """ Add x (nparray.base) to parameter_dict and return a list of x_id
    """
    x_var = []
    for x in parameter_list:
      var = VariableBlock(x)
      if var not in self.parameter_dict:
        self.parameter_offset += var.SetPlace(self.parameter_offset)
        self.parameter_dict[var] = var
        x_var.append( var )
      else:
        x_var.append(self.parameter_dict[var])
    return x_var

  def AddObservation(self, observation_list):
    """ Add l (nparray.base) to observation_dict and return a list of l_id  """
    l_var = []
    for l in observation_list:
      var = ObservationBlock(l)
      if var not in self.observation_dict:
        self.observation_offset += var.SetPlace(self.observation_offset)
        self.observation_dict[var] = var
        l_var.append( var )
      else:
        l_var.append(self.observation_dict[var])
    return l_var

  def UpdateLocalOffset(self):
    self.parameter_local_offset  = 0
    for var in self.parameter_dict.itervalues():
      self.parameter_local_offset += var.SetLocalPlace(self.parameter_local_offset)

    self.observation_local_offset = 0
    for var in self.observation_dict.itervalues():
      self.observation_local_offset += var.SetLocalPlace(self.observation_local_offset)

  def SetParameterization(self, array, parameterization):
    var = VariableBlock(array)
    if var in self.parameter_dict:
      self.parameter_dict[var].param = parameterization
    elif var in self.observation_dict:
      self.observation_dict[var].param = parameterization
    else:
      raise RuntimeWarning("Input variable not in the lists")

  def SetSigma(self, array, sigma):
    var = ObservationBlock(array)
    if var in self.observation_dict:
      self.observation_dict[var].sigma = sigma
    else:
      raise RuntimeWarning("Input variable not in the lists")

  def NumParameters(self):
    return self.parameter_offset

  def NumReducedParameters(self):
    return self.parameter_local_offset

  def NumObservations(self):
    return self.observation_offset

  def NumReducedObservations(self):
    return self.observation_local_offset

  def NumResiduals(self):
    return self.constraint_offset

  def CollectParameters(self):
    x = np.zeros(self.parameter_offset)
    for var in self.parameter_dict.itervalues():
      x[ var.place ] = var.data[:]
    return x

  def CollectObservations(self):
    l = np.zeros( self.observation_offset )
    for var in self.observation_dict.itervalues():
      l[ var.place ] = var.data[:]
    return l

  def CollectSigma(self):
    s = np.ones( self.observation_offset )
    for var in self.observation_dict.itervalues():
      if not var.sigma is None:
        s[ var.place ] = var.sigma
    return s

  def Plus_x(self, x, delta):
    for var in self.parameter_dict.itervalues():
      x[var.place] = var.param.Plus( x[var.place], delta[ var.place_local ] )
    return x

  def Plus_l(self, l, delta):
    for var in self.observation_dict.itervalues():
      l[var.place] = var.param.Plus( l[var.place], delta[ var.place_local ] )
    return l

  def SetObservationFromVector(self, new_l):
    for var in self.observation_dict.itervalues():
      var.data[:] = new_l[ var.place ]

  def SetParameterFromVector(self, new_x):
    for var in self.parameter_dict.itervalues():
      var.data[:] = new_x[ var.place ]

  def AddConstraintUsingAD(self, g, x_list, l_list):
    """
      1.add x (and l) to parameter_dict (and observation_dict)
      2.add record tuple(g, x_ids, l_ids) in constraint_blocks
    """
    x_sizes = [x.size for x in x_list]
    l_sizes = [l.size for l in l_list]
    xl_indices = np.cumsum(x_sizes + l_sizes)[:-1]
    var       = np.hstack(x_list+l_list )
    var_ad    = pycppad.independent( var )
    var_jacobian= pycppad.adfun(var_ad, g( *np.split(var_ad, xl_indices) ) ).jacobian

    res = g( *(x_list + l_list) )
    jac = var_jacobian(var)
    if not ( np.isfinite(res).all() and  np.isfinite(jac).all() ):
      RuntimeWarning("AutoDiff Not valid")
      return

    x_var = self.AddParameter(x_list)
    l_var = self.AddObservation(l_list)
    x_slc = [v.place for v in x_var]
    l_slc = [v.place for v in l_var]
    def g_flat(x,l):
      return g( *([x[slc] for slc in x_slc ] + [l[slc] for slc in l_slc ] ) )

    def g_jacobian(x,l):
      var = np.hstack([x[slc] for slc in x_slc ] + [l[slc] for slc in l_slc ] )
      J = var_jacobian(var)
      return np.split(J, xl_indices, axis=1)

    r_slc = slice(self.constraint_offset, self.constraint_offset+len(res) )
    self.constraint_offset = r_slc.stop

    self.constraint_blocks.append(
      ConstraintBlock( g_flat, g_jacobian, r_slc, x_var, l_var ))

  def AddConstraintUsingMD(self, g, g_jac, x_list, l_list):
    """
      1.add x (and l) to parameter_dict (and observation_dict)
      2.add record tuple(g, x_ids, l_ids) in constraint_blocks
    """
    x_sizes = [x.size for x in x_list]
    l_sizes = [l.size for l in l_list]

    res = g( *(x_list + l_list) )
    jac = g_jac( *(x_list + l_list) )

    inequal_size = [j.shape != (len(res), size) for j, size in zip(jac, x_sizes+l_sizes)]
    if len(jac) != len(x_list + l_list) or np.any( inequal_size ):
      RuntimeError("Jacobian Size Not fit")
      return
    if not ( np.isfinite(res).all() and  np.isfinite(jac).all() ):
      RuntimeWarning("Function Not valid")
      return

    x_var = self.AddParameter(x_list)
    l_var = self.AddObservation(l_list)
    x_slc = [v.place for v in x_var]
    l_slc = [v.place for v in l_var]
    def g_flat(x,l):
      return g( *([x[slc] for slc in x_slc ] + [l[slc] for slc in l_slc ] ) )

    def g_jacobian(x,l):
      return g_jac( *([x[slc] for slc in x_slc ] + [l[slc] for slc in l_slc ] ) )

    r_slc = slice(self.constraint_offset, self.constraint_offset+len(res) )
    self.constraint_offset = r_slc.stop

    self.constraint_blocks.append(
      ConstraintBlock( g_flat, g_jacobian, r_slc, x_var, l_var ))

  def EvaluateConstraintJacobian(self, x=None, l=None):
    if x is None:
      x = self.CollectParameters()
    if l is None:
      l = self.CollectObservations()

    A = np.zeros( ( self.NumResiduals(), self.NumParameters() ) )
    B = np.zeros( ( self.NumResiduals(), self.NumObservations() ) )
    for cb in self.constraint_blocks:
      r_slc = cb.r_slc
      js = cb.g_jacobian(x, l)
      for var in cb.x_var:
        A[r_slc, var.place] = js.pop(0)
      for var in cb.l_var:
        B[r_slc, var.place] = js.pop(0)
    return A,B

  def EvaluateConstraintJacobianWithParam(self, x=None, l=None):
    if x is None:
      x = self.CollectParameters()
    if l is None:
      l = self.CollectObservations()

    A = np.zeros( ( self.NumResiduals(), self.NumReducedParameters() ) )
    B = np.zeros( ( self.NumResiduals(), self.NumReducedObservations() ) )
    for cb in self.constraint_blocks:
      r_slc = cb.r_slc
      js = cb.g_jacobian(x, l)
      for var in cb.x_var:
        A[r_slc, var.place_local] = var.param.ToLocalJacobian( js.pop(0) )
      for var in cb.l_var:
        B[r_slc, var.place_local] = var.param.ToLocalJacobian( js.pop(0) )
    return A,B

  def EvaluateConstraintJacobianSparse(self, x=None, l=None):
    if x is None:
      x = self.CollectParameters()
    if l is None:
      l = self.CollectObservations()

    A = scipy.sparse.lil_matrix( ( self.NumResiduals(), self.NumParameters(), ) )
    B = scipy.sparse.lil_matrix( ( self.NumResiduals(), self.NumObservations(), ) )
    for cb in self.constraint_blocks:
      r_slc = cb.r_slc
      js = cb.g_jacobian(x, l)
      for x_slc in cb.x_slc:
        A[r_slc, x_slc] = js.pop(0)
      for l_slc in cb.l_slc:
        B[r_slc, l_slc] = js.pop(0)
    return A.tocsr(), B.tocsr()
#    return A, B

  def EvaluateConstraintResidual(self, x=None, l=None):
    if x is None:
      x = self.CollectParameters()
    if l is None:
      l = self.CollectObservations()
    res = np.zeros(self.NumResiduals())
    for cb in self.constraint_blocks:
      res[cb.r_slc] = cb.g_flat(x, l)
    return res

  def EvaluateCorrectionJacobian(self, le):
    CTWC_inv = np.zeros( (self.NumReducedObservations(), self.NumReducedObservations()) )
    CTWC_inv_CTWe = np.empty(self.NumReducedObservations())

    for var in self.observation_dict.itervalues():
      place, place_local = var.place, var.place_local
      CTWC_inv[place_local, place_local],  CTWC_inv_CTWe[place_local] = var.ComputeCTWCInvAndE(le[ place ])
    return CTWC_inv, CTWC_inv_CTWe

  def EvaluateCorrectionJacobianRawSparse(self, le):
    CTWC = scipy.sparse.lil_matrix( (self.NumReducedObservations(), self.NumReducedObservations()) )
    CTWe = np.empty(self.NumReducedObservations())

    for var in self.observation_dict.itervalues():
      place, place_local = var.place, var.place_local
      CTWC[place_local, place_local],  CTWe[place_local] = var.ComputeCTWCAndCTWE(le[ place ])
    return CTWC.todia(), CTWe

  def EvaluateObjective(self, correction, weight=None):
    if weight is None:
      weight = 1.0 / self.CollectSigma()
    return np.sum( weight * correction**2  )

  def SolveGaussEliminateDense(self, x_init=None, l_init=None, max_it=30, Tx=1e-5):
    s = self.CollectSigma()
    s_inv = 1.0 / s
    Sigma =  np.diag(s) #scipy.sparse.diags( s, offsets=0, format="coo" )

    l0  = self.CollectObservations()
    x   = self.CollectParameters()  if x_init is None else x_init.copy()
    l   = l0.copy()                 if l_init is None else l_init.copy()
    for it in range(max_it):
        le    = l - l0
        self.variance_factor = self.EvaluateObjective(le, s_inv)
        print it, self.variance_factor

        A, B = self.EvaluateConstraintJacobian(x, l)
        err  = self.EvaluateConstraintResidual(x, l)
        W    = np.linalg.inv( B.dot(Sigma).dot(B.T) )
        Cg   = B.dot(le) - err
        ATW  = A.T.dot(W)
        ATWA = ATW.dot(A)
        dx   = np.linalg.solve(ATWA,  ATW.dot(Cg))
        lag  = W.dot( A.dot(dx) - Cg )
        dl   = -Sigma.dot( B.T.dot(lag) ) - le
        x    += dx
        l    += dl
        if np.abs( dx*np.sqrt( np.diag(ATWA) ) ).max() < Tx:
          break
    return x, l - l0

  def SolveGaussEliminateDenseWithParam(self, x_init=None, l_init=None, max_it=30, Tx=1e-5):
    self.UpdateLocalOffset()
    s = self.CollectSigma()
    s_inv = 1.0 / s
    l0  = self.CollectObservations()
    x   = self.CollectParameters()  if x_init is None else x_init.copy()
    l   = l0.copy()                 if l_init is None else l_init.copy()
    for it in range(max_it):
        le    = l - l0
        self.variance_factor = self.EvaluateObjective(le, s_inv)
        print it, self.variance_factor

        A, B = self.EvaluateConstraintJacobianWithParam(x, l)
        CTWC_inv, CTWC_inv_CTWe = self.EvaluateCorrectionJacobian(le)
        g  = self.EvaluateConstraintResidual(x, l)
        Cg   =  B.dot(CTWC_inv_CTWe) - g

        W    = np.linalg.inv( B.dot(CTWC_inv).dot(B.T) )
        ATW  = A.T.dot(W)
        ATWA = ATW.dot(A)
        dx   = np.linalg.solve(ATWA,  ATW.dot(Cg))
        lag  = W.dot( A.dot(dx) - Cg )
        dl   = -( CTWC_inv.dot( B.T.dot(lag) ) + CTWC_inv_CTWe )
        x    = self.Plus_x(x, dx)
        l    = self.Plus_l(l, dl)
        if np.abs( dx*np.sqrt( np.diag(ATWA) ) ).max() < Tx:
          break
    return x, l - l0

  def SolveFullMatrixDense(self, x=None, l=None, maxit=10):
    dim_x = self.NumParameters()
    dim_l = self.NumObservations()
    dim_r = self.NumResiduals()
    offset =  np.cumsum([0, dim_x, dim_l, dim_r])
    segment_x = slice(offset[0], offset[1])
    segment_l = slice(offset[1], offset[2])
    segment_r = slice(offset[2], offset[3])
    r_offset = offset[2]
    l_offset = offset[1]
    dim_total = offset[-1]

    weight = 1.0 / self.CollectSigma()
    W = scipy.sparse.diags(weight, 0)

    l0 = self.CollectObservations()
    if x is None:
      x  = self.CollectParameters()
    if l is None:
      l   = l0.copy()
    e = l - l0
    self.variance_factor=0

    b = np.zeros( (dim_total,) )
    A = np.zeros( (dim_total, dim_total,) )
    for it in range(maxit):
      A[segment_l, segment_l] = W.A
      for cb in self.constraint_blocks:
        r_slc = slice(cb.r_slc.start + r_offset, cb.r_slc.stop + r_offset )
        js = cb.g_jacobian(x, l)
        for var in cb.x_var:
          M = js.pop(0)
          A[r_slc, var.place] = M
          A[var.place, r_slc] = M.T
        for var in cb.l_var:
          l_slc = slice(var.place.start + l_offset, var.place.stop + l_offset )
          M = js.pop(0)
          A[r_slc, l_slc] = M
          A[l_slc, r_slc] = M.T

      g = self.EvaluateConstraintResidual(x, l)
      b[segment_l] = -W.dot(e)
      b[segment_r] = -g

      s = scipy.linalg.solve(A, b)
      x += s[segment_x]
      l += s[segment_l]
      e = l - l0
      cost = self.EvaluateObjective(e, weight)
      if np.abs(cost - self.variance_factor)<1e-6:
        break
      print it, cost
      self.variance_factor = cost
    return x, e

  def SolveFullMatrixDenseWithParam(self, x=None, l=None, maxit=10):
    self.UpdateLocalOffset()
    dim_x = self.NumReducedParameters()
    dim_l = self.NumReducedObservations()
    dim_r = self.NumResiduals()
    offset =  np.cumsum([0, dim_x, dim_l, dim_r])
    segment_x = slice(offset[0], offset[1])
    segment_l = slice(offset[1], offset[2])
    segment_r = slice(offset[2], offset[3])
    r_offset = offset[2]
    l_offset = offset[1]
    dim_total = offset[-1]

    weight = 1.0 / self.CollectSigma()

    l0 = self.CollectObservations()
    if x is None:
      x  = self.CollectParameters()
    if l is None:
      l   = l0.copy()
    e = l - l0

    b = np.zeros( (dim_total,) )
    A = np.zeros( (dim_total, dim_total,) )
    for it in range(maxit):
      CTWC, CTWe = self.EvaluateCorrectionJacobianRawSparse(e)
      A[segment_l, segment_l] = CTWC.A
      for cb in self.constraint_blocks:
        r_slc = slice(cb.r_slc.start + r_offset, cb.r_slc.stop + r_offset )
        js = cb.g_jacobian(x, l)
        for var in cb.x_var:
          M = var.param.ToLocalJacobian( js.pop(0) )
          A[r_slc, var.place_local] = M
          A[var.place_local, r_slc] = M.T
        for var in cb.l_var:
          l_slc = slice(var.place_local.start + l_offset, var.place_local.stop + l_offset )
          M = var.param.ToLocalJacobian( js.pop(0) )
          A[r_slc, l_slc] = M
          A[l_slc, r_slc] = M.T

      g = self.EvaluateConstraintResidual(x, l)
      b[segment_l] = -CTWe
      b[segment_r] = -g

      s = scipy.linalg.solve(A, b)
      x = self.Plus_x(x, s[segment_x])
      l = self.Plus_l(l, s[segment_l])
      e = l - l0
      cost = self.EvaluateObjective(e, weight)
      if np.abs(cost - self.variance_factor)<1e-6:
        break
      print it, cost
      self.variance_factor = cost
    return x, e





#%% ConjugateGradientTrustRegion

def BoundariesIntersections(z, d, trust_radius):
  """
  Solve the scalar quadratic equation ||z + t d|| == trust_radius.
  This is like a line-sphere intersection.
  Return the two values of t, sorted from low to high.
  """
  a = np.dot(d, d)
  b = 2 * np.dot(z, d)
  c = np.dot(z, z) - trust_radius**2
  sqrt_discriminant = np.sqrt(b*b - 4*a*c)
  ta = (-b - sqrt_discriminant) / (2*a)
  tb = (-b + sqrt_discriminant) / (2*a)
  return ta, tb

def ConjugateGradientTrustRegion(A, b, x0, trust_radius):
  r = b - A.dot(x0)
  d = r.copy()
  rTr = r.T.dot(r)
  thres = 1e-12*rTr
  delta_x = np.zeros_like(b)
  hits_boundary = False
  for it in range(100):
    Ad = A.dot(d)
    dAd = d.dot(Ad)
    # see if non-positive curvature reached
    if dAd <= 0:
      # if so, stop, look at the two boundary points.
      # Find both values of t to get the boundary points such that
      # ||z + t d|| == trust_radius
      # and then choose the one with the predicted min value.
      ta, tb = BoundariesIntersections(delta_x, d, trust_radius)
      delta_x = delta_x + tb * d
      hits_boundary = True
      print "non-positive curvature"
      break

    alpha = rTr / dAd
    delta_next = delta_x + alpha * d
    # see if trust region boundary reached
    if np.linalg.norm(delta_next) >= trust_radius:
      ta, tb = BoundariesIntersections(delta_x, d, trust_radius)
      delta_x = delta_x + tb * d
      hits_boundary = True
      print "boundary reached"
      break

    # still within trust region boundary
    delta_x = delta_next
    r = r - alpha*Ad if it%10 else b - A.dot(x0 + delta_x)
    rTr_old = rTr.copy()
    rTr = r.T.dot(r)
    beta = rTr/rTr_old
    d = r + beta*d

    print np.linalg.norm(r), np.linalg.norm(delta_x)
    if rTr < thres:
      break
  return delta_x, hits_boundary
#%%
def test_AddParameter():
  problem = GaussHelmertProblem()

  a = np.array([0, 0, 0],'d')
  problem.AddParameter([a])
  assert( len(problem.parameter_dict)    ==1 )
  assert( problem.parameter_dict.keys()[0].data is a )
  """a change in outside array should be reflected in the one of parameter_dict"""
  a[0] = 2
  assert( problem.parameter_dict.keys()[0].data[0] == 2 )

  """a view of other should not be added"""
  a_view = a.view()
  problem.AddParameter([a_view])
  assert( len(problem.parameter_dict)    ==1 )

  b = np.array([1, 1, 1])
  problem.AddParameter([b])
  assert( len(problem.parameter_dict)    ==2 )

  assert( problem.NumObservations() == 0)
  assert( problem.NumParameters() == 6 )

def test_SetParameterFromVector():
  problem = GaussHelmertProblem()
  a = np.array([0, 0, 0],'d')
  b = np.array([0, 0, 0],'d')
  problem.AddParameter([a])
  problem.AddParameter([b])

  x_new = np.arange(6)
  problem.SetParameterFromVector( x_new )
  assert( np.allclose(a, x_new[:3]) )
  assert( np.allclose(b, x_new[3:]) )

def EqualityConstraint(a,b):
  return a-b
def EqualityConstraintJac(a,b):
  return np.eye(len(a)), -np.eye(len(b))

def test_Collection():
  a = np.array([0, 0, 0],'d')
  b = [np.array([i, i, i],'d') for i in range(1, 10) ]
  w = [np.diag(b_) for b_ in b]
  problem = GaussHelmertProblem()
  for i in range(len(b)):
    problem.AddConstraintUsingAD(EqualityConstraint,
                                 [a],
                                 [ b[i] ],
                                 [ w[i] ] )
  wc  = problem.CollectWeight()
  wcs = problem.CollectWeightSparse()
  bc  = problem.CollectObservations()
  ac  = problem.CollectParameters()
  assert( np.allclose(a, ac) )
  assert( sum([3 * i**3 for i in range(1,10)]) == problem.EvaluateObjective(bc) )

def test_AddConstraintUsingAD():
  problem = GaussHelmertProblem()
  a = np.array([0, 0, 0],'d')
  b = np.array([1, 1, 1],'d')

  problem.AddConstraintUsingAD(EqualityConstraint, [a], [b])

  assert ( np.allclose( problem.constraint_blocks[0].g_flat(a, b),
                       -b) )
  assert ( np.allclose( np.hstack(problem.constraint_blocks[0].g_jacobian(a,b) ),
                       np.c_[np.eye(3), -np.eye(3)] ) )
  assert( len(problem.parameter_dict) ==1 )
  assert( len(problem.observation_dict) ==1 )

def test_AddConstraintUsingMD():
  problem = GaussHelmertProblem()
  a = np.array([0, 0, 0],'d')
  b = np.array([1, 1, 1],'d')

  problem.AddConstraintUsingMD(EqualityConstraint, EqualityConstraintJac, [a], [b])
  assert ( np.allclose( problem.constraint_blocks[0].g_flat(a, b),
                       -b) )
  assert ( np.allclose( np.hstack(problem.constraint_blocks[0].g_jacobian(a,b) ),
                       np.c_[np.eye(3), -np.eye(3)] ) )
  assert( len(problem.parameter_dict) ==1 )
  assert( len(problem.observation_dict) ==1 )

def test_Solve():
  np.random.seed(0)
  dim_x = 3
  a = np.ones(dim_x)

  num_l = 100
  sigma = 0.02
  s = np.full(dim_x, sigma**2)

  for trail in range(1):
    bs = [ a + sigma * np.random.randn(3) for i in range(num_l)]
    problem = GaussHelmertProblem()
    for i in range(num_l):
      problem.AddConstraintUsingAD(EqualityConstraint,
                                   [ a ],
                                   [ bs[i] ])
      problem.SetSigma(bs[i], s)

    x, le = problem.SolveGaussEliminateDense()
    x, le = problem.SolveFullMatrixDense()

    le.reshape((-1,3))

def test_SolveWithParam():
  np.random.seed(0)
  dim_x = 3
  a = np.ones(dim_x)

  num_l = 100
  sigma = 0.02
  s = np.full(dim_x, sigma**2)

  for trail in range(1):
    bs = [ a + sigma * np.random.randn(3) for i in range(num_l)]
    problem = GaussHelmertProblem()
    for i in range(num_l):
      problem.AddConstraintUsingAD(EqualityConstraint,
                                   [ a ],
                                   [ bs[i] ])
      problem.SetSigma(bs[i], s)
      problem.SetParameterization(a, SubsetParameterization([1,0,1]))

    x1, le1 = problem.SolveGaussEliminateDenseWithParam()
    assert x1[1]==1

    problem.SetParameterization(bs[0], SubsetParameterization([1,1,0]))
    x2, le2 = problem.SolveFullMatrixDenseWithParam()
    assert x2[1]==1
    assert le2[2]==0

#%%

inv = np.linalg.inv
def skew(v):
    return np.array([[   0, -v[2],  v[1]],
                     [ v[2],    0, -v[0]],
                     [-v[1], v[0],    0 ]])
def vee(s):
    return np.array([s[2,1], s[0,2], s[1,0]])

def invT(T):
    R, t = T[:3, :3], T[:3, 3]
    Ti = np.eye(4, dtype='d')
    Ti[:3, :3] = R.T
    Ti[:3, 3]  = -R.T.dot(t)
    return Ti

def ax2Rot(r):
    p = np.linalg.norm(r)
    if np.abs(p) < 1e-12:
        return np.eye(3)
    else:
        S = skew(r/p)
        return np.eye(3) + np.sin(p)*S + (1.0-np.cos(p))*S.dot(S)

def Rot2ax(R):
    tr = np.trace(R)
    a  = np.array( [R[2,1]-R[1,2], R[0,2]-R[2,0], R[1,0]-R[0,1]] )
    an = np.linalg.norm(a)
    phi= np.arctan2(an, tr-1)
    if np.abs(phi) < 1e-12:
        return np.zeros(3,'d')
    else:
        return phi/an*a

def rotateX(roll):
    """rotate around x axis"""
    return np.array([[1, 0, 0, 0],
                     [0, np.cos(roll), -np.sin(roll), 0],
                     [0, np.sin(roll), np.cos(roll), 0],
                     [0, 0, 0, 1]],'d')
def rotateY(pitch):
    """rotate around y axis"""
    return np.array([[np.cos(pitch), 0, np.sin(pitch),  0],
                     [0, 1, 0, 0],
                     [-np.sin(pitch), 0, np.cos(pitch), 0],
                     [0, 0, 0, 1]],'d')

def rotateZ(yaw):
    """rotate around z axis"""
    return np.array([[np.cos(yaw), -np.sin(yaw), 0, 0],
                     [np.sin(yaw), np.cos(yaw), 0, 0],
                     [0, 0, 1, 0],
                     [0, 0, 0, 1]],'d')

def translate(x,y,z):
    return np.array([[1, 0, 0, x],
                     [0, 1, 0, y],
                     [0, 0, 1, z],
                     [0, 0, 0, 1]],'d')
d2r =  lambda deg: np.pi*deg/180


if __name__ == "__main__":
  test_AddParameter()
  test_SetParameterFromVector()
  test_AddConstraintUsingAD()
  test_AddConstraintUsingMD()

  test_Solve()
  test_SolveWithParam()

  print "all test succeed."

  if 0:
 #%%
    Hm = [ rotateZ(d2r(10)).dot(translate(1,0,0)),
           rotateX(d2r(30)).dot(translate(2,0,0))]
    Hm_inv = [invT(h) for h in Hm]

    xi_true, eta_true = [],[]
    for h in Hm:
      xi_true.append( h[:3,3].copy() )
      eta_true.append( Rot2ax(h[:3,:3]) )

    S = len(Hm) + 1

    def CalibrationConstraint(xi, eta, r0, t0, r1, t1):
      R10 = ax2Rot(eta)
      R1  = ax2Rot(r1)
      e_t = xi - R1.dot(xi) + R10.dot(t0) - t1
      e_r = R10.dot(r0) - r1
      return np.r_[e_t, e_r]

    ''' generate ground truth trajectories '''
  #  np.random.seed(2)
    T = 10
    dM = []
    for t in xrange(T):
      dm = [rotateX(d2r(60+20*np.random.rand(1))).dot(
            rotateY(d2r(60+20*np.random.rand(1))).dot(
            rotateZ(d2r(60+20*np.random.rand(1))).dot(
            translate(1,1,1))))]
      for h, h_inv in zip(Hm, Hm_inv):
        dm.append( h.dot(dm[0]).dot(h_inv) )
      dM.append(dm)

    Sigmas = [(1e-2*np.ones(3), 1e-2*np.ones(3)),
              (1e-2*np.ones(3), 1e-2*np.ones(3)),
              (1e-2*np.ones(3), 1e-2*np.ones(3))]
    Weight = [ ( np.diag(1.0/sigma_pair[0]**2),
                 np.diag(1.0/sigma_pair[1]**2) ) for sigma_pair in Sigmas ]
    noise_on = 1
    rs,ts = [], []
    for s in range(S):
      r,t = [],[]
      for i in xrange(T):
        r.append( np.copy( noise_on*Sigmas[s][0]*np.random.randn(3) + Rot2ax(dM[i][s][:3,:3]) ) )
        t.append( np.copy( noise_on*Sigmas[s][1]*np.random.randn(3) + dM[i][s][:3,3] ) )
      rs.append(r)
      ts.append(t)

    xi,eta = [],[]
    xi[:] = xi_true[:]
    eta[:] = eta_true[:]
  #%%
    problem = GaussHelmertProblem()
    for i in range(T):
      for s in range(1, S):
        problem.AddConstraintUsingAD(CalibrationConstraint,
                                     [ xi[s-1], eta[s-1] ],
                                     [ rs[0][i], ts[0][i], rs[s][i], ts[s][i] ])
#    problem.SetParameterization(xi[0], SubsetParameterization([1,1,0]))
        problem.SetParameterization(rs[0][i], SubsetParameterization([1,1,0]))

    dim_x = problem.NumParameters()
    dim_l = problem.NumObservations()
    dim_r = problem.NumResiduals()
    slc_x = slice(0, dim_x)
    slc_l = slice(dim_x, dim_x+dim_l)
    slc_r = slice(dim_x+dim_l, dim_x + dim_l + dim_r)

    x0 = problem.CollectParameters()
    l0 = problem.CollectObservations()
    x,l = x0.copy(),l0.copy()
    e = l - l0

  #%% cg
    JA,JB = problem.EvaluateConstraintJacobian(x, l)
    J = np.c_[JA, JB]
    lambd = np.linalg.lstsq(J.T, b[:dim_x+dim_l])[0]
    x,l,lambd = problem.SolveNewtonCG(x, l, e, trust_region=1e3)
    e = l - l0
    g = problem.EvaluateConstraintResidual(x,l)
    print np.linalg.norm(g)
  #%% trust region
    def DoglegStepsForLeastSquares(A, b, nabla):
      ''' solve p* = argmin ||Ap + b||^2, subject to ||p*||2 < nabla'''
      g = A.T.dot(b) # gradient
      hessian  = A.T.dot(A)
      # shortest step: Steepest descend direction, negative gradient
      p_1 = -g.dot(g)/g.dot(hessian.dot(g))*g
      norm_p1 = np.linalg.norm(p_1)
      if norm_p1 >= nabla:
        return nabla*p_1/norm_p1

      # longest step: Newton Step
      p_2 = np.linalg.pinv(A).dot(-b) # A'A p = -A'b
      norm_p2 = np.linalg.norm(p_2)
      if norm_p2 <= nabla:
        return p_2

      # combined step
      # solve || (1-a)*p1 + a * p2 || = nabla, s.t 1>a>0
      norm_p1_squared = norm_p1**2
      p1_dot_p2 = p_1.dot(p_2)
      A = norm_p1_squared + norm_p2**2 - 2*p1_dot_p2
      B = 2*(p1_dot_p2 - norm_p1_squared)
      C = norm_p1_squared - nabla**2
      a = np.roots([A,B,C]) # Ax^2+Bx+C = 0
      a = a[a>0]
      return (1-a)*p_1 + a*p_2

    nabla = 1

    JA,JB = problem.EvaluateConstraintJacobian(x, l)
    J = np.c_[JA,JB]
    g = problem.EvaluateConstraintResidual(x, l)
    p = DoglegStepsForLeastSquares(J, g, 0.8*nabla)

    Qj,Rj,Pj = scipy.linalg.qr(J.T, pivoting=True, mode='full')
    Z = Qj[:, -(dim_x+dim_l-dim_r):]

    g_new = problem.EvaluateConstraintResidual(x + p[:dim_x], l + p[dim_x:])
    diff = g.dot(g)- g_new.dot(g_new)
    Jp = J.dot(p)
    pre = -Jp.T.dot(g+0.5*Jp)
    rho = diff/pre
    if rho < 0.25:
      nabla /= 4
    else:
      if rho > 0.75 and np.linalg.norm(p) == nabla:
        nabla = np.minimum(2*nabla, 2)
    if rho > 0.1:
      x += p[:dim_x]
      l += p[dim_x:]
    print rho, np.linalg.norm(g)
    problem.EvaluateObjective(l-l0)
    xf, le = problem.SolveGaussEliminateDense(x, l)
    problem.EvaluateObjective(le)

    xf, le = problem.SolveGaussEliminateDense(x0, l0)
    problem.EvaluateObjective(le)


  #%% cvxopt

    A,B = problem.EvaluateConstraintJacobianSparse(x0, l0)
    Qa,Ra,Pa = scipy.linalg.qr(A.A, pivoting=True, mode='full')
    Z = Qa[:, dim_x:]
    WBZ = W_s.dot(B.T).dot(Z)
    ZBWBZ = WBZ.T.dot(WBZ)




  #%%
    alpha = 1

    mu = 0.5
    JA,JB = problem.EvaluateConstraintJacobian(x, l)
    J = np.c_[JA,JB]

    g = problem.EvaluateConstraintResidual(x, l)
    print np.linalg.norm(g)
    e = l - l0

    BSBT = JB.dot(Sigma).dot(JB.T)
    Qa,Ra,Pa = scipy.linalg.qr(JA, pivoting=True, mode='full')
    Z = Qa[:, 8:]
    np.linalg.eig(Z.T.dot(BSBT).dot(Z))


    W_mu = np.diag( 1 / ( W.diagonal() + mu ) ) #
    WSE = W_mu.dot(W.dot(e))

    M = JA.dot(JA.T)/mu + JB.dot(W_mu).dot(JB.T)
    c = g - JB.dot(WSE)
    lambd = scipy.linalg.cho_solve(scipy.linalg.cho_factor(M), c)
    dl  = -WSE - W_mu.dot(JB.T.dot(lambd))
    dx  = -JA.T.dot(lambd)/mu

    g_new_predict = g + JA.dot(dx) + JB.dot(dl)
    g_new_actual  = problem.EvaluateConstraintResidual(x + dx, l + dl)
    print np.linalg.norm(g_new_actual), np.linalg.norm(g_new_predict)

    grad = W.dot(e).dot(dl)

    alpha = np.maximum(alpha, 0.5*grad/np.linalg.norm(g))
    phi_new = problem.EvaluateMeritFunction(x + dx, l + dl, alpha)

    x += dx
    l += dl


    '''  '''
    JA,JB = problem.EvaluateConstraintJacobian(x, l)
    J = np.c_[JA,JB]

    b_lsq = np.zeros( ( dim_x + dim_l) )
    b_lsq[slc_l] = -W.dot(e)
    lambd_lsq = scipy.linalg.lstsq(J.T, b_lsq)[0] # J'*lambda = -grad f
    res_lsq = b_lsq + J.T.dot(lambd_lsq)

    A = np.zeros( (dim_l + dim_r, dim_x + dim_l) )
    A[:dim_l, slc_l] = W
    A[dim_l:, :] = J

    b = np.zeros( ( dim_l + dim_r ) )
    b[:dim_l] = res_lsq[dim_x:]
    b[dim_l:] = problem.EvaluateConstraintResidual(x, l)

    p = scipy.linalg.lstsq(A, -b)[0]
    g_new = problem.EvaluateConstraintResidual(x + p[:dim_x], l + p[dim_x:])
    diff = g.dot(g)- g_new.dot(g_new)
    print diff
    x += p[:dim_x]
    l += p[dim_x:]
    e = l - l0
    print problem.EvaluateObjective(e)
    subproblem = CGSteihaugSubproblem(p, lambda p: 0, lambda p:b, lambda p:A)
    p, hits_boundary = subproblem.solve(10)


    Qj,Rj,Pj = scipy.linalg.qr(J.T, pivoting=True, mode='full')
    Z = Qj[:, -(dim_x+dim_l-dim_r):]
    G = scipy.linalg.block_diag(np.zeros([dim_x]*2), W)
    H = Z.T.dot(G).dot(Z)
    jac = np.zeros( ( dim_x + dim_l + dim_r ) )
    jac[slc_l] = W.dot(e)
    jac[slc_r] = problem.EvaluateConstraintResidual(x, l)

    hess = np.zeros( jac.shape*2 )
    hess[slc_x, slc_r] = JA.T
    hess[slc_l, slc_r] = JB.T
    hess[slc_l, slc_l] = W
    hess[slc_r, slc_x] = JA
    hess[slc_r, slc_l] = JB

    p = np.zeros_like(jac)

    subproblem = CGSteihaugSubproblem(p, lambda p: 0, lambda p:jac, lambda p:hess)
    p, hits_boundary = subproblem.solve(10)

    """"""
    x = x0.copy()
    l = l0.copy()


  #  result = problem.Solve()


    scipy.sparse.linalg.cg





