#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  7 14:34:54 2017

@author: nubot
"""
import numpy as np
import matplotlib.pyplot as plt
from solver2 import *
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


def add_n_noise(sigma):
  return lambda x: x + sigma*np.random.randn(3)
def add_u_noise(scale):
  return lambda x: x + scale*np.random.rand(3)
noise_on = 1.0
#np.random.seed(10)

''' generate ground truth relative pos between sensors '''
# s <- a , other <- base
T_sa_group = [ rotateZ(d2r(10)).dot(translate(1,0,0)),
              rotateX(d2r(30)).dot(translate(2,0,0))][:1]
T_as_group = map(invT, T_sa_group)
num_sensor = len(T_sa_group) + 1

t_sa_group = map(tFromT, T_sa_group)
r_sa_group = map(rFromT, T_sa_group)

t_sa_group_noisy = map(add_u_noise(noise_on*0.05), t_sa_group)
r_sa_group_noisy = map(add_u_noise(noise_on*0.1), r_sa_group)

print r_sa_group[0],t_sa_group[0]
#print r_sa_group[1],t_sa_group[1]

''' generate ground truth relative motion '''
num_pos = 50
def SimMotion(num_pos, base=10, mag=5):
  dT_group_list = [] # T: t <- t+1
  for _ in xrange(num_pos-1):
    # base
    dT_a = MfromRT( d2r(base + mag*np.random.rand(1)) * randsp(), 5*np.random.rand(1) * randsp() )
    # others
    dT_other = [T_sa.dot(dT_a).dot(T_as) for T_sa,T_as in zip(T_sa_group, T_as_group)]

    dT_group_list.append([dT_a] + dT_other )

  dr_group_list = deep_map(rFromT, dT_group_list)
  dt_group_list = deep_map(tFromT, dT_group_list)

  ''' generate ground truth absolute pose, world <- sensor'''
  #T_ws_group0 = [ np.eye(4) ]+T_as_group
  T_ws_group0 = [ np.eye(4)  ]*num_sensor
  T_ws_group_list = [ T_ws_group0 ]  #  M[t+1][s] = M[t][s] * dT[t][s], s = 0...S
  for dT_group in dT_group_list:
    T_last_group = T_ws_group_list[-1]
    T_ws_group_list.append( [ T_s.dot(dT_s) for T_s, dT_s in zip(T_last_group, dT_group) ]  )

  r_ws_group_list = deep_map(rFromT, T_ws_group_list)
  t_ws_group_list = deep_map(tFromT, T_ws_group_list)
  return r_ws_group_list, t_ws_group_list, dr_group_list, dt_group_list
r_ws_group_list, t_ws_group_list, dr_group_list, dt_group_list =  SimMotion(num_pos)

sigma_r_abs  = 0.002
sigma_t_abs  = 0.03
sigma_r_rel   = 0.003  # = np.sqrt(cov_dr_group_list[0][0].diagonal())
sigma_t_rel   = 0.043  # = np.sqrt(cov_dt_group_list[0][0].diagonal())
def SimNoiseAbs(sigma_r_abs, sigma_t_abs, \
                r_ws_group_list, t_ws_group_list, dr_group_list, dt_group_list):

  num_pos    = len(r_ws_group_list)
  num_sensor = len(r_ws_group_list[0])

  ''' generate noisy absolute pose measurement, world <- sensor'''
  Cov_r = sigma_r_abs**2 * np.eye(3)
  Cov_t = sigma_t_abs**2 * np.eye(3)
  r_ws_group_list_noisy = deep_map(add_n_noise(noise_on*sigma_r_abs), r_ws_group_list)
  t_ws_group_list_noisy = deep_map(add_n_noise(noise_on*sigma_t_abs), t_ws_group_list)
  T_ws_group_list_noisy = [[MfromRT(r,t) for r,t in zip(r_group,t_group)]
                              for r_group,t_group in zip(r_ws_group_list_noisy, t_ws_group_list_noisy)]
  cov_r_group_list = [ [Cov_r]*num_sensor ]*num_pos
  cov_t_group_list = [ [Cov_t]*num_sensor ]*num_pos

  ''' generate (noisy) relative pose measurement from noisy absolute pose, t <- t+1'''
  dT_group_list_noisy = []
  for i in xrange(1, num_pos):
    dT_group = [ invT(T_last).dot(T) for T_last,T in zip(T_ws_group_list_noisy[i-1], T_ws_group_list_noisy[i]) ]
    dT_group_list_noisy.append(dT_group)
  dr_group_list_noisy = deep_map(rFromT, dT_group_list_noisy)
  dt_group_list_noisy = deep_map(tFromT, dT_group_list_noisy)

  ''' covariance for relative pose measurement '''
  @AddJacobian
  def RelT(r1,t1,r2,t2):
    r12 = axAdd(-r1,r2)
    t12 = ax2Rot(-r1).dot(t2-t1)
    return np.hstack([r12, t12])
  Cov_abs = scipy.linalg.block_diag(Cov_r,Cov_t,Cov_r,Cov_t)

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

  return r_ws_group_list_noisy, t_ws_group_list_noisy, \
         cov_r_group_list,      cov_t_group_list,      \
         dr_group_list_noisy,   dt_group_list_noisy,   \
         cov_dr_group_list,     cov_dt_group_list

r_ws_group_list_noisy, t_ws_group_list_noisy, \
cov_r_group_list,      cov_t_group_list,      \
dr_group_list_noisy, dt_group_list_noisy,     \
cov_dr_group_list, cov_dt_group_list = SimNoiseAbs(sigma_r_abs, sigma_t_abs, r_ws_group_list, t_ws_group_list, dr_group_list, dt_group_list)

def SimNoiseRel(sigma_r_rel, sigma_t_rel, \
             r_ws_group_list, t_ws_group_list, dr_group_list, dt_group_list):

  num_seg    = len(dr_group_list)
  num_sensor = len(dr_group_list[0])

  ''' generate noisy relative pose measurement for sensor a, a1 <- a2'''
  Cov_dr = sigma_r_rel**2 * np.eye(3)
  Cov_dt = sigma_t_rel**2 * np.eye(3)
  dr_group_list_noisy = deep_map(add_n_noise(noise_on*sigma_r_rel), dr_group_list)
  dt_group_list_noisy = deep_map(add_n_noise(noise_on*sigma_t_rel), dt_group_list)
  cov_dr_group_list = [ [Cov_dr]*num_sensor ]*num_seg
  cov_dt_group_list = [ [Cov_dt]*num_sensor ]*num_seg

  ''' covariance for absolute pose measurement '''
  @AddJacobian
  def AbsT(r1,t1,dr,dt):
    r2 = axAdd(r1,dr)
    t2 = t1 + ax2Rot(r1).dot(dt)
    return np.hstack([r2, t2])
  AbsT(*[np.random.rand(3)]*4)

  @AddJacobian
  def RelT(r1,t1,r2,t2):
    r12 = axAdd(-r1,r2)
    t12 = ax2Rot(-r1).dot(t2-t1)
    return np.hstack([r12, t12])

  ''' generate (noisy) absolute pose from noisy relative measurement, world <- sensor'''
#  T_ws_group_list_noisy = [ [ np.eye(4)   ]*num_sensor, map(MfromRT, dr_group_list_noisy[0], dt_group_list_noisy[0]) ]
  r_ws_group_list_noisy = [ [ np.zeros(3)   ]*num_sensor, dr_group_list_noisy[0] ]
  t_ws_group_list_noisy = [ [ np.zeros(3)   ]*num_sensor, dt_group_list_noisy[0] ]
  cov_r_group_list      = [ [np.zeros((3,3))]*num_sensor, [Cov_dr]*num_sensor ]
  cov_t_group_list      = [ [np.zeros((3,3))]*num_sensor, [Cov_dt]*num_sensor ]

  for dr_group, dt_group in zip(dr_group_list_noisy[1:], dt_group_list_noisy[1:]):
    T_ws_group, r_ws_group, t_ws_group,cov_r_group,cov_t_group = [],[],[],[],[]
    for r1,t1,dr,dt,cov_r1,cov_t1 in zip(r_ws_group_list_noisy[-1], t_ws_group_list_noisy[-1], dr_group, dt_group, cov_r_group_list[-1],cov_t_group_list[-1]):
      r2t2, Js = AbsT(r1,t1,dr,dt)
      r2, t2 = r2t2[:3], r2t2[3:]
      Js = np.hstack(Js)
      Cov_rel = scipy.linalg.block_diag(cov_r1,cov_t1,Cov_dr,Cov_dt)

      cov_rt = Js.dot(Cov_rel).dot(Js.T)
      cov_r = cov_rt[:3,:3]
      cov_t = cov_rt[3:,3:]

#      ''' test covariance'''
#      drt,Js = RelT(r1,t1,r2,t2)
#      Js = np.hstack(Js)
#      Cov_abs = scipy.linalg.block_diag(cov_r1,cov_t1,cov_rt)
#
#      cov_drt_test = Js.dot(Cov_abs).dot(Js.T)

      T_ws_group.append(MfromRT(r2, t2))
      r_ws_group.append(r2)
      t_ws_group.append(t2)

      cov_r_group.append(cov_r)
      cov_t_group.append(cov_t)
#    T_ws_group_list_noisy.append(T_ws_group)
    r_ws_group_list_noisy.append(r_ws_group)
    t_ws_group_list_noisy.append(t_ws_group)
    cov_r_group_list.append(cov_r_group)
    cov_t_group_list.append(cov_t_group)


  return r_ws_group_list_noisy, t_ws_group_list_noisy, \
         cov_r_group_list,      cov_t_group_list,      \
         dr_group_list_noisy,   dt_group_list_noisy,   \
         cov_dr_group_list,     cov_dt_group_list
#r_ws_group_list_noisy, t_ws_group_list_noisy, \
#cov_r_group_list,      cov_t_group_list,      \
#dr_group_list_noisy,   dt_group_list_noisy,   \
#cov_dr_group_list,     cov_dt_group_list = SimNoiseRel(sigma_r_rel, sigma_t_rel, r_ws_group_list, t_ws_group_list, dr_group_list, dt_group_list)



#%% AbsoluteConstraint1
def AbsoluteAdjustment(r_sa_group,            t_sa_group,
                       r_ws_group_list_noisy, t_ws_group_list_noisy,
                       cov_r_group_list,      cov_t_group_list,
                       cov=False):

#  def AbsoluteConstraint(r_sa, t_sa, r_wa, t_wa, r_ws, t_ws):
#    check_magnitude(r_sa)
#    check_magnitude(r_wa)
#    check_magnitude(r_ws)
#
#    R_ws = ax2Rot(r_ws)
#    r_wa_est = axAdd( r_ws, r_sa ) #  Rot2ax( ax2Rot(r_ws).dot( ax2Rot(r_sa) ) )
#    t_wa_est = t_ws + R_ws.dot(t_sa)
#    return np.hstack([t_wa - t_wa_est, r_wa - r_wa_est])

  def AbsoluteConstraint(r_sa, t_sa, r_wa, t_wa, r_vs, t_vs):
      check_magnitude(r_sa)
      check_magnitude(r_wa)
      check_magnitude(r_vs)
      r_vw, t_vw = r_sa, t_sa
      R_vw = ax2Rot(r_vw)
      t_vwa = t_vw + R_vw.dot(t_wa) # v -> w-> a
      # r_vwa = axAdd(r_vw, r_wa)

      t_vsa = t_vs + ax2Rot(r_vs).dot(t_sa)# v -> s-> a
      # r_vsa = axAdd(r_vs, r_sa)

      e_t = t_vwa - t_vsa
      e_r = R_vw.dot(r_wa) - r_vs
      return np.r_[e_r, e_t]


  num_sensor = len(r_ws_group_list_noisy[0])

  problem = GaussHelmertProblem()
  for r_ws_group, t_ws_group, cov_r_group, cov_t_group in zip(r_ws_group_list_noisy[1:], t_ws_group_list_noisy[1:], cov_r_group_list[1:], cov_t_group_list[1:]):
    for s in range(1, num_sensor):
      problem.AddConstraintWithArray(AbsoluteConstraint,
                                   [ r_sa_group[s-1], t_sa_group[s-1] ],
                                   [ r_ws_group[0],   t_ws_group[0],
                                     r_ws_group[s],   t_ws_group[s]])
    for s in range(num_sensor):
      problem.SetSigma(r_ws_group[s], cov_r_group[s])
      problem.SetSigma(t_ws_group[s], cov_t_group[s])

  return SolveWithGESparse(problem, fac=True, cov=cov)

if 1:
  x_abs, e_abs, fac_abs,cov_abs = \
  AbsoluteAdjustment(r_sa_group_noisy,      t_sa_group_noisy,
                     r_ws_group_list_noisy, t_ws_group_list_noisy,
                     cov_r_group_list,      cov_t_group_list,
                     True)
  print "x_abs: ",x_abs.reshape(-1,6)
  print "cov_abs: ", cov_abs.diagonal()

#%% RelativeConstraint
def RelativeAdjustment(r_sa_group,          t_sa_group,
                       dr_group_list_noisy, dt_group_list_noisy,
                       cov_dr_group_list,   cov_dt_group_list,
                       cov=False):

  def RelativeConstraint(r_sa, t_sa, dr_a, dt_a, dr_s, dt_s):
    check_magnitude(r_sa)
    check_magnitude(dr_a)
    check_magnitude(dr_s)

    R_sa = ax2Rot(r_sa)
    dR_s  = ax2Rot(dr_s)
    e_t = dt_s + dR_s.dot(t_sa) - R_sa.dot(dt_a) - t_sa
    e_r = R_sa.dot(dr_a) - dr_s
    return np.r_[e_t, e_r]

  num_sensor = len(dr_group_list_noisy[0])

  problem = GaussHelmertProblem()
  for dr_group, dt_group, cov_dr_group, cov_dt_group in zip(dr_group_list_noisy, dt_group_list_noisy, cov_dr_group_list, cov_dt_group_list):
    for s in range(1, num_sensor):
      problem.AddConstraintWithArray(RelativeConstraint,
                                   [ r_sa_group[s-1], t_sa_group[s-1] ],
                                   [ dr_group[0],     dt_group[0],
                                     dr_group[s],     dt_group[s]])
    for s in range(num_sensor):
      problem.SetSigma(dr_group[s], cov_dr_group[s])
      problem.SetSigma(dt_group[s], cov_dt_group[s])

  return SolveWithGESparse(problem, fac=True, cov=cov)

if 1:
  x_rel, e_rel, fac_rel,cov_rel = \
  RelativeAdjustment(r_sa_group_noisy,    t_sa_group_noisy,
                     dr_group_list_noisy, dt_group_list_noisy,
                     cov_dr_group_list,   cov_dt_group_list,
                     True)
  print "x_rel: ",x_rel.reshape(-1,6)
  print "cov_rel: ", cov_rel.diagonal()
#%% MixConstraint
#def MixedAdjustment(r_sa_group,          t_sa_group,
#                    dr_a_noisy_list,     dt_a_noisy_list,
#                    Cov_dr,            Cov_dt,
#                    r_ws_group_list_noisy, t_ws_group_list_noisy,
#                    Cov_r,            Cov_t,
#                    cov=False):
#
#  def MixedConstraint(r_sa, t_sa, dr_a, dt_a, r_ws1, t_ws1, r_ws2, t_ws2):
#    check_magnitude(r_sa)
#    check_magnitude(dr_a)
#    check_magnitude(r_ws1)
#    check_magnitude(r_ws2)
#
#    r_wa1 = axAdd(r_ws1, r_sa)
#    r_wa2 = axAdd(r_ws2, r_sa)
#    dr_a_est = axAdd(-r_wa1, r_wa2)
#
#    t_wa1 = t_ws1 + ax2Rot(r_ws1).dot(t_sa)
#    t_wa2 = t_ws2 + ax2Rot(r_ws2).dot(t_sa)
#    dt_a_est = ax2Rot(-r_wa1).dot(-t_wa1 + t_wa2)
#    return np.hstack([dr_a - dr_a_est, dt_a - dt_a_est])
#
#  num_sensor = len(dr_group_list_noisy[0])
#
#  problem = GaussHelmertProblem()
#  for dr_a, dt_a, r_ws1_group, t_ws1_group, r_ws2_group, t_ws2_group in zip(dr_a_noisy_list, dt_a_noisy_list, r_ws_group_list_noisy[:-1], t_ws_group_list_noisy[:-1],  r_ws_group_list_noisy[1:], t_ws_group_list_noisy[1:]):
#    for s in range(1, num_sensor):
#      problem.AddConstraintWithArray(MixedConstraint,
#                                   [ r_sa_group[s-1], t_sa_group[s-1] ],
#                                   [ dr_a,            dt_a,
#                                     r_ws1_group[s],  t_ws1_group[s],
#                                     r_ws2_group[s],  t_ws2_group[s]])
#
#  for dr_a, dt_a in zip(dr_a_noisy_list, dt_a_noisy_list):
#    problem.SetSigma(dr_a, Cov_dr)
#    problem.SetSigma(dt_a, Cov_dt)
#  for r_ws_group, t_ws_group in zip(r_ws_group_list_noisy, t_ws_group_list_noisy):
#    for s in range(1, num_sensor):
#      problem.SetSigma(r_ws_group[s], Cov_r)
#      problem.SetSigma(t_ws_group[s], Cov_t)
#  problem.SetVarFixed(r_ws_group_list_noisy[0][1])
#  problem.SetVarFixed(t_ws_group_list_noisy[0][1])
#
#  return SolveWithGESparse(problem, fac=True, cov=cov)
#
#if 0:
#  x_mix, e_mix, fac_mix,cov_mix = \
#  MixedAdjustment(r_sa_group_noisy,    t_sa_group_noisy,
#                  dr_a_noisy_list,     dt_a_noisy_list,
#                  Cov_dr,            Cov_dt,
#                  r_ws_group_list_noisy, t_ws_group_list_noisy,
#                  Cov_r,            Cov_t,
#                  cov=True)
#  print "x_mix: ",x_mix.reshape(-1,6)
#  print "cov_mix: ", cov_mix.diagonal()

#%% AbsoluteConstraint2
#def ACCAbsoluteAdjustment(r_sa_group,            t_sa_group,
#                          r_ws_group_list_noisy, t_ws_group_list_noisy,
#                          dr_group_list_noisy, dt_group_list_noisy,
#                          cov_dr_group_list,   cov_dt_group_list,
#                          cov=False):
#
#  def Acc(r1,t1,r2,t2,dr,dt):
#    r2_est = axAdd(r1,dr)
#    t2_est = t1 + ax2Rot(r1).dot(dt)
#    return np.hstack([r2-r2_est, t2-t2_est])
#
#  def AbsoluteConstraint(r_sa, t_sa, r_wa, t_wa, r_vs, t_vs):
#      check_magnitude(r_sa)
#      check_magnitude(r_wa)
#      check_magnitude(r_vs)
#      r_vw, t_vw = r_sa, t_sa
#      R_vw = ax2Rot(r_vw)
#      t_vwa = t_vw + R_vw.dot(t_wa) # v -> w-> a
#      # r_vwa = axAdd(r_vw, r_wa)
#
#      t_vsa = t_vs + ax2Rot(r_vs).dot(t_sa)# v -> s-> a
#      # r_vsa = axAdd(r_vs, r_sa)
#
#      e_t = t_vwa - t_vsa
#      e_r = R_vw.dot(r_wa) - r_vs
#      return np.r_[e_r, e_t]
#
#  num_sensor = len(r_ws_group_list_noisy[0])
#
#  problem = GaussHelmertProblem()
#  for r1_group, t1_group, r2_group, t2_group, dr_group, dt_group, cov_dr_group, cov_dt_group in zip(r_ws_group_list_noisy[:-1], t_ws_group_list_noisy[:-1], r_ws_group_list_noisy[1:], t_ws_group_list_noisy[1:], dr_group_list_noisy, dt_group_list_noisy, cov_dr_group_list,  cov_dt_group_list):
#    """Acc"""
#    for s in range(num_sensor):
#      problem.AddConstraintWithArray(Acc,
#                                   [ r1_group[s],   t1_group[s] ,
#                                     r2_group[s],   t2_group[s] ],
#                                   [ dr_group[s],   dt_group[s] ])
#      problem.SetSigma(dr_group[s], cov_dr_group[s])
#      problem.SetSigma(dt_group[s], cov_dt_group[s])
#
#    """AbsoluteConstraint"""
#    for s in range(1, num_sensor):
#      problem.AddConstraintWithArray(AbsoluteConstraint,
#                                   [ r_sa_group[s-1], t_sa_group[s-1],
#                                     r2_group[0],   t2_group[0],
#                                     r2_group[s],   t2_group[s] ], [])
#  for s in range(num_sensor):
#    problem.SetVarFixed(r_ws_group_list_noisy[0][s])
#    problem.SetVarFixed(t_ws_group_list_noisy[0][s])
#
##  return SolveWithGESparseAsGM(problem, fac=True, cov=cov)
#  return SolveWithCVX(problem, fac=True, cov=cov)
#
#if 0:
#  x_acc, e_acc, fac_acc,cov_acc = \
#  ACCAbsoluteAdjustment(r_sa_group_noisy,      t_sa_group_noisy,
#                        r_ws_group_list_noisy, t_ws_group_list_noisy,
#                        dr_group_list_noisy, dt_group_list_noisy,
#                        cov_dr_group_list,   cov_dt_group_list,
#                        True)
#  print "x_acc: ",x_acc.reshape(-1,6)
#  print "cov_acc: ", cov_acc.diagonal()


#%% rel/abs difference on rotation angle
if 0:
  """  abs/rel ratio of covariance matrix diagonal elements, at different rotation angles, num_pos=50
  sigma_r_abs  = 0.002
  sigma_t_abs  = 0.03
  seed = 7,13
  result:
   0  : array([ 0.072,  0.046,  0.06 ,  0.039,  0.02 ,  0.03 ]),
   10 : array([ 0.02 ,  0.03 ,  0.058,  0.022,  0.037,  0.097]),
   20 : array([ 0.032,  0.035,  0.04 ,  0.041,  0.048,  0.056]),
   30 : array([ 0.044,  0.036,  0.031,  0.06 ,  0.058,  0.04 ]),
   40 : array([ 0.069,  0.07 ,  0.054,  0.094,  0.103,  0.072]),
   50 : array([ 0.106,  0.093,  0.091,  0.145,  0.12 ,  0.12 ]),
   60 : array([ 0.134,  0.102,  0.128,  0.191,  0.157,  0.168]),
   70 : array([ 0.189,  0.149,  0.161,  0.258,  0.189,  0.216]),
   80 : array([ 0.229,  0.186,  0.19 ,  0.325,  0.25 ,  0.271]),
   90 : array([ 0.239,  0.21 ,  0.193,  0.337,  0.324,  0.331]),
   100: array([ 0.285,  0.237,  0.211,  0.4  ,  0.41 ,  0.414]),
   110: array([ 0.324,  0.244,  0.223,  0.443,  0.473,  0.459]),
   120: array([ 0.357,  0.256,  0.242,  0.499,  0.511,  0.492]),
   130: array([ 0.354,  0.279,  0.257,  0.552,  0.491,  0.508]),
   140: array([ 0.341,  0.32 ,  0.279,  0.619,  0.477,  0.54 ]),
   150: array([ 0.355,  0.353,  0.351,  0.726,  0.558,  0.609]),
   160: array([ 0.409,  0.352,  0.403,  0.75 ,  0.681,  0.632]),
   170: array([ 0.472,  0.44 ,  0.37 ,  0.715,  0.659,  0.621])}
  """
  abs_rel_ratio_angle = {}
  for base in range(0, 180, 10):
    np.random.seed(7)
    r_ws_group_list, t_ws_group_list,             \
    dr_group_list, dt_group_list =  SimMotion(50, base=base)

    np.random.seed(13)
    r_ws_group_list_noisy, t_ws_group_list_noisy, \
    cov_r_group_list,      cov_t_group_list,      \
    dr_group_list_noisy, dt_group_list_noisy,     \
    cov_dr_group_list, cov_dt_group_list = SimNoiseAbs(sigma_r_abs, sigma_t_abs, r_ws_group_list, t_ws_group_list, dr_group_list, dt_group_list)

    try:
      x_abs, e_abs, fac_abs,cov_abs = \
      AbsoluteAdjustment(r_sa_group_noisy,      t_sa_group_noisy,
                         r_ws_group_list_noisy, t_ws_group_list_noisy,
                         cov_r_group_list,      cov_t_group_list,
                         True)
      x_rel, e_rel, fac_rel,cov_rel = \
      RelativeAdjustment(r_sa_group_noisy,    t_sa_group_noisy,
                         dr_group_list_noisy, dt_group_list_noisy,
                         cov_dr_group_list,   cov_dt_group_list,
                         True)
    except ValueError:
      print "Failed!!!!!!!!!!!!"
      continue
    else:
      abs_rel_ratio_angle[base] = cov_abs.diagonal()/cov_rel.diagonal()
  print abs_rel_ratio_angle

if 0:
  """ abs/rel ratio of covariance matrix diagonal elements, with different number of measurements
  base_angle = 10
  sigma_r_abs  = 0.002
  sigma_t_abs  = 0.03
  seed = 7,13
  result:
   10 : array([ 0.124,  0.246,  0.451,  0.136,  0.257,  0.429]),
   50 : array([ 0.02 ,  0.03 ,  0.058,  0.022,  0.037,  0.097]),
   100: array([ 0.021,  0.028,  0.016,  0.035,  0.029,  0.045]),
   500: array([ 0.008,  0.012,  0.007,  0.02 ,  0.024,  0.032])}

  """
  abs_rel_ratio_num_poses = {}
  for num_pose in [10, 50, 100, 500]:
    np.random.seed(7)
    r_ws_group_list, t_ws_group_list,             \
    dr_group_list, dt_group_list =  SimMotion(num_pose, base=10)

    np.random.seed(13)
    r_ws_group_list_noisy, t_ws_group_list_noisy, \
    cov_r_group_list,      cov_t_group_list,      \
    dr_group_list_noisy, dt_group_list_noisy,     \
    cov_dr_group_list, cov_dt_group_list = SimNoiseAbs(sigma_r_abs, sigma_t_abs, r_ws_group_list, t_ws_group_list, dr_group_list, dt_group_list)

    try:
      x_abs, e_abs, fac_abs,cov_abs = \
      AbsoluteAdjustment(r_sa_group_noisy,      t_sa_group_noisy,
                         r_ws_group_list_noisy, t_ws_group_list_noisy,
                         cov_r_group_list,      cov_t_group_list,
                         True)
      x_rel, e_rel, fac_rel,cov_rel = \
      RelativeAdjustment(r_sa_group_noisy,    t_sa_group_noisy,
                         dr_group_list_noisy, dt_group_list_noisy,
                         cov_dr_group_list,   cov_dt_group_list,
                         True)
    except ValueError:
      print "Failed!!!!!!!!!!!!"
      continue
    else:
      abs_rel_ratio_num_poses[num_pose] = cov_abs.diagonal()/cov_rel.diagonal()
      print cov_rel.diagonal()
      print cov_abs.diagonal()

  print abs_rel_ratio_num_poses

#%% batch sim
if 0:
  x_abs_list,   x_rel_list,   x_mix_list   = [],[],[]
  fac_abs_list, fac_rel_list, fac_mix_list = [],[],[]
  for it in range(1000):
    r_ws_group_list_noisy, t_ws_group_list_noisy, \
    cov_r_group_list,      cov_t_group_list,      \
    dr_group_list_noisy, dt_group_list_noisy,     \
    cov_dr_group_list, cov_dt_group_list = SimNoiseAbs(sigma_r_abs, sigma_t_abs, r_ws_group_list, t_ws_group_list, dr_group_list, dt_group_list)

    try:
      x_abs, e_abs, fac_abs = \
      AbsoluteAdjustment(r_sa_group,            t_sa_group,
                         r_ws_group_list_noisy, t_ws_group_list_noisy,
                         cov_r_group_list,      cov_t_group_list)
      x_rel, e_rel, fac_rel = \
      RelativeAdjustment(r_sa_group,          t_sa_group,
                         dr_group_list_noisy, dt_group_list_noisy,
                         cov_dr_group_list,   cov_dt_group_list)
#      x_mix, e_mix, fac_mix = \
#      MixedAdjustment(r_sa_group,          t_sa_group,
#                      dr_a_noisy_list,     dt_a_noisy_list,
#                      Cov_dr,            Cov_dt,
#                      r_ws_group_list_noisy, t_ws_group_list_noisy,
#                      Cov_r,            Cov_t)
    except:
      print "Failed!!!!!!!!!!!!"
      continue
    x_abs_list.append(x_abs)
    fac_abs_list.append(fac_abs)

    x_rel_list.append(x_rel)
    fac_rel_list.append(fac_rel)

#    x_mix_list.append(x_mix)
#    fac_mix_list.append(fac_mix)

  x_abs_arr = np.asarray(x_abs_list)
  plt.figure(num='abs')
  plt.hist(fac_abs_list)
  fac_abs_mean = np.mean(fac_abs_list)
  x_abs_mean = np.mean(x_abs_arr, axis=0)
  x_abs_cov = np.cov(x_abs_arr - x_abs_mean, rowvar=False).diagonal()

#  x_mix_arr = np.asarray(x_mix_list)
#  plt.figure(num='mix')
#  plt.hist(fac_mix_list)
#  fac_mix_mean = np.mean(fac_mix_list)
#  x_mix_mean = np.mean(x_mix_arr, axis=0)
#  x_mix_cov = np.cov(x_mix_arr - x_mix_mean, rowvar=False).diagonal()

  x_rel_arr = np.asarray(x_rel_list)
  plt.figure(num='rel')
  plt.hist(fac_rel_list)
  fac_rel_mean = np.mean(fac_rel_list)
  x_rel_mean = np.mean(x_rel_arr, axis=0)
  x_rel_cov = np.cov(x_rel_arr - x_rel_mean, rowvar=False).diagonal()

  #np.savez("calibration_sim.npz", x_abs_arr=x_abs_arr, x_rel_arr=x_rel_arr, x_mix_arr=x_mix_arr)

