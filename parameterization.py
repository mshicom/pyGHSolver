#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  7 14:32:19 2017

@author: nubot
"""
import numpy as np
import pycppad


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