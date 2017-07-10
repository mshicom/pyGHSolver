#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  7 14:34:54 2017

@author: nubot
"""
import numpy as np
import matplotlib.pyplot as plt
from solver2 import *
from sfm import invT, ax2Rot, Rot2ax, axAdd, MfromRT
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

tFromT = lambda T: T[:3,3].copy()
rFromT = lambda T: Rot2ax(T[:3,:3])

def deep_map(function, list_of_list):
  """ ret = [ map(func, list) for list in list_of_list ]
  Example
  -------
  >>> deep_map(lambda a:a, [[[1,0],[2,3]]])
  """
  if not isinstance(list_of_list[0], list):
    return map(function, list_of_list)
  return [ deep_map(function, l) for l in list_of_list ]

''' generate ground truth relative pos between sensors '''
# s <- a , other <- base
T_sa_group = [ rotateZ(d2r(10)).dot(translate(1,0,0)),
              rotateX(d2r(30)).dot(translate(2,0,0))]
T_as_group = map(invT, T_sa_group)
num_sensor = len(T_sa_group) + 1

t_sa_group = map(tFromT, T_sa_group)
r_sa_group = map(rFromT, T_sa_group)

''' generate ground truth relative motion '''
#np.random.seed(2)
num_pos = 200
dT_group_list = [] # T: t <- t+1
for _ in xrange(num_pos-1):
  # base
  dT_a = rotateX(d2r(10+20*np.random.rand(1))).dot(
         rotateY(d2r(10+20*np.random.rand(1))).dot(
         rotateZ(d2r(10+20*np.random.rand(1))).dot(
         translate(1,1,1))))
  # others
  dT_other = [T_sa.dot(dT_a).dot(T_as) for T_sa,T_as in zip(T_sa_group, T_as_group)]

  dT_group_list.append([dT_a] + dT_other )

dr_group_list = deep_map(rFromT, dT_group_list)
dt_group_list = deep_map(tFromT, dT_group_list)


''' generate ground truth absolute pose, world <- sensor'''
T_ws_group0 = [ np.eye(4) ]+T_as_group
T_ws_group_list = [ T_ws_group0 ]  #  M[t+1][s] = M[t][s] * dT[t][s], s = 0...S
for dT_group in dT_group_list:
  T_last_group = T_ws_group_list[-1]
  T_ws_group_list.append( [ T_s.dot(dT_s) for T_s, dT_s in zip(T_last_group, dT_group) ]  )

r_ws_group_list = deep_map(rFromT, T_ws_group_list)
t_ws_group_list = deep_map(tFromT, T_ws_group_list)

''' generate noisy absolute pose measurement, world <- sensor'''
sigma_r_abs  = 0.002
sigma_t_abs  = 0.03
Cov_r_abs = sigma_r_abs**2 * np.eye(3)
Cov_t_abs = sigma_t_abs**2 * np.eye(3)

def add_noise(sigma):
  return lambda x: x+sigma*np.random.randn(3)
r_ws_group_list_noisy = deep_map(add_noise(1.0*sigma_r_abs), r_ws_group_list)
t_ws_group_list_noisy = deep_map(add_noise(1.0*sigma_t_abs), t_ws_group_list)
T_ws_group_list_noisy = [[MfromRT(r,t) for r,t in zip(r_group,t_group)]
                            for r_group,t_group in zip(r_ws_group_list_noisy, t_ws_group_list_noisy)]

''' generate (noisy) relative pose measurement from noisy absolute pose, t <- t+1'''
dT_group_list_noisy = []
for i in xrange(1, num_pos):
  dT_group = [ invT(T_last).dot(T) for T_last,T in zip(T_ws_group_list_noisy[i-1], T_ws_group_list_noisy[i]) ]
  dT_group_list_noisy.append(dT_group)
dr_group_list_noisy = deep_map(rFromT, dT_group_list_noisy)
dt_group_list_noisy = deep_map(tFromT, dT_group_list_noisy)

''' covariance for relative pose measurement '''
if 1:
  cov_dr_group_list = []
  cov_dt_group_list = []
  for i in xrange(1, num_pos):
    cov_r_group,cov_t_group = [],[]
    for T_last,T in zip(T_ws_group_list_noisy[i-1], T_ws_group_list_noisy[i]):
      R1 = T_last[:3, :3]
      t1,t2 = T_last[:3, 3], T[:3, 3]
      t12 = R1.T.dot(t2-t1)
      cov_r_12 = Cov_r_abs + R1.T.dot(Cov_r_abs).dot(R1)
      Jr = skew(-t12)
      cov_t_12 = Jr.dot(Cov_r_abs).dot(Jr.T) + R1.T.dot(2 * Cov_t_abs).dot(R1)         # t12 = R(t2-t1)
      cov_r_group.append(cov_r_12)
      cov_t_group.append(cov_t_12)
    cov_dr_group_list.append(cov_r_group)
    cov_dt_group_list.append(cov_t_group)
else:
  @AddJacobian
  def RelT(r1,t1,r2,t2):
    r12 = axAdd(-r1,r2)
    t12 = ax2Rot(-r1).dot(t2-t1)
    return np.hstack([r12, t12])
  Cov_abs = scipy.linalg.block_diag(Cov_r_abs,Cov_t_abs,Cov_r_abs,Cov_t_abs)

  dr_group_list_noisy,dt_group_list_noisy = [],[]
  cov_dr_group_list,cov_dt_group_list = [],[]
  for i in xrange(1, num_pos):
    dr_group, dt_group,cov_dr_group,cov_dt_group = [], [],[],[]
    for r_last,t_last,r,t in zip(r_ws_group_list_noisy[i-1], t_ws_group_list_noisy[i-1],r_ws_group_list_noisy[i], t_ws_group_list_noisy[i]):
      drt,Js = RelT(r_last,t_last,r,t)
      Js = np.hstack(Js)
      cov_dr = Js[:3].dot(Cov_abs).dot(Js[:3].T)
      cov_dt = Js[3:].dot(Cov_abs).dot(Js[3:].T)

      dr_group.append(drt[:3])
      dt_group.append(drt[3:])
      cov_dr_group.append(cov_dr)
      cov_dt_group.append(cov_dt)
    dr_group_list_noisy.append(dr_group)
    dt_group_list_noisy.append(dt_group)
    cov_dr_group_list.append(cov_dr_group)
    cov_dt_group_list.append(cov_dt_group)

#%% AbsoluteConstraint
def AbsoluteConstraint(r_sa, t_sa, r_wa, t_wa, r_ws, t_ws):
  check_magnitude(r_sa)
  check_magnitude(r_wa)
  check_magnitude(r_ws)

  R_ws = ax2Rot(r_ws)
  r_wa_est = axAdd( r_ws, r_sa ) #  Rot2ax( ax2Rot(r_ws).dot( ax2Rot(r_sa) ) )
  t_wa_est = t_ws + R_ws.dot(t_sa)
  return np.hstack([t_wa - t_wa_est, r_wa - r_wa_est])

#  R_sw = ax2Rot(-r_ws)
#  r_sa_est = axAdd( -r_ws, r_wa )
#  t_sa_est = R_sw.dot( -t_ws + t_wa )
#  return np.hstack([r_sa - r_sa_est, t_sa - t_sa_est])
  def test():
    r_wa, t_wa, r_ws, t_ws = np.random.rand(4,3)

    T_as = invT(MfromRT(r_wa, t_wa)).dot( MfromRT(r_ws, t_ws) )
    r_as, t_as = Rot2ax(T_as[:3,:3]), T_as[:3,3]

    R_aw = ax2Rot(r_wa).T
    r_as_est = axAdd( -r_wa, r_ws )
    t_as_est = R_aw.dot(t_ws - t_wa)
    assert_array_almost_equal( r_as, r_as_est )
    assert_array_almost_equal( t_as, t_as_est )
if 1:
  problem = GaussHelmertProblem()
  for i in range(num_pos):
    for s in range(1, num_sensor):
      problem.AddConstraintWithArray(AbsoluteConstraint,
                                   [ r_sa_group[s-1], t_sa_group[s-1] ],
                                   [ r_ws_group_list_noisy[i][0],
                                     t_ws_group_list_noisy[i][0],
                                     r_ws_group_list_noisy[i][s],
                                     t_ws_group_list_noisy[i][s]])
      problem.SetSigma(r_ws_group_list_noisy[i][s], Cov_r_abs)
      problem.SetSigma(t_ws_group_list_noisy[i][s], Cov_t_abs)
  for i in range(num_pos):
    problem.SetSigma(r_ws_group_list_noisy[i][0], Cov_r_abs)
    problem.SetSigma(t_ws_group_list_noisy[i][0], Cov_t_abs)
  problem.SetVarFixed(r_ws_group_list_noisy[0][0])
  problem.SetVarFixed(t_ws_group_list_noisy[0][0])

  #x_est, e  = SolveWithCVX(problem)
  x_abs, e, fac,cov = SolveWithGESparse(problem, fac=True, cov=True)

  print x_abs.reshape(-1,6)
  print r_sa_group[0],t_sa_group[0]
  print r_sa_group[1],t_sa_group[1]
  print cov.diagonal()
#%% RelativeConstraint
def RelativeConstraint(r_sa, t_sa, r_a, t_a, r_s, t_s):
  check_magnitude(r_sa)
  check_magnitude(r_a)
  check_magnitude(r_s)

  R_sa = ax2Rot(r_sa)
  R_s  = ax2Rot(r_s)
  e_t = t_s + R_s.dot(t_sa) - R_sa.dot(t_a) - t_sa
  e_r = R_sa.dot(r_a) - r_s
  return np.r_[e_t, e_r]

if 1:
  problem = GaussHelmertProblem()
  for i in range(num_pos-1):
    for s in range(1, num_sensor):
      problem.AddConstraintWithArray(RelativeConstraint,
                                   [ r_sa_group[s-1], t_sa_group[s-1] ],
                                   [ dr_group_list_noisy[i][0],
                                     dt_group_list_noisy[i][0],
                                     dr_group_list_noisy[i][s],
                                     dt_group_list_noisy[i][s]])
      problem.SetSigma(dr_group_list_noisy[i][s], cov_dr_group_list[i][s])
      problem.SetSigma(dt_group_list_noisy[i][s], cov_dt_group_list[i][s])
  for i in range(num_pos-1):
    problem.SetSigma(dr_group_list_noisy[i][0], cov_dr_group_list[i][0])
    problem.SetSigma(dt_group_list_noisy[i][0], cov_dt_group_list[i][0])

  #x_est, e  = SolveWithCVX(problem)
  x_rel, e, fac,cov = SolveWithGESparse(problem, fac=True, cov=True)
  print x_rel.reshape(-1,6)
  print r_sa_group[0],t_sa_group[0]
  print r_sa_group[1],t_sa_group[1]
  print cov.diagonal()