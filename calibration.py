#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  7 14:34:54 2017

@author: nubot
"""
import numpy as np
import matplotlib.pyplot as plt
from solver2 import *

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

#def test():
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
T = 1000
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
    problem.AddConstraintWithArray(CalibrationConstraint,
                                 [ xi[s-1], eta[s-1] ],
                                 [ rs[0][i], ts[0][i], rs[s][i], ts[s][i] ])
    problem.SetSigma(rs[s][i], np.diag(Sigmas[s][0]**2))
    problem.SetSigma(ts[s][i], np.diag(Sigmas[s][1]**2))
for i in range(T):
  problem.SetSigma(rs[0][i], np.diag(Sigmas[s][0]**2))
  problem.SetSigma(ts[0][i], np.diag(Sigmas[s][1]**2))

#    problem.SetParameterization(xi[0], SubsetParameterization([1,1,0]))
#      problem.SetParameterization(rs[0][i], SubsetParameterization([1,1,0]))
#SetVarFixed
#x_est, e  = SolveWithCVX(problem)
x_est, e, fac = SolveWithGESparse(problem, fac=True)

print x_est.reshape(-1,6)

