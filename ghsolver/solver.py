#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 28 10:08:11 2017

@author: kaihong
"""
import matplotlib.pyplot as plt

import numpy as np
import scipy
import scipy.sparse
import scipy.sparse.linalg

np.set_printoptions(precision=3, linewidth=90)
from numpy.testing import *
import pycppad

from parameterization import *
from matrix_tree import *
from util import *
from collections import namedtuple

#%% CompoundVector
class ArrayID(object):
  __slots__ = 'addr','data'
  def __init__(self, array):
    if np.isscalar(array):
      raise RuntimeError("Input cannot be scalar, use x=np.empty(1) ")
    self.data = array
    self.addr = array.__array_interface__['data'][0]

  def __hash__(self):
    return hash(self.addr)

  def __eq__(self, other):
    return self.data.shape == other.data.shape and self.addr == other.addr

def test_ArrayID():
  a = np.empty((10,10))
  b = np.arange(3)
  assert len( { ArrayID(a[0  ]), ArrayID(b) } ) == 2             # different variable
  assert len( { ArrayID(a[0  ]), ArrayID(a[   1]) } ) == 2       # same length, different row
  assert len( { ArrayID(a[0, :5]), ArrayID(a[0,  :4]) } ) == 2   # diff length, same head
  assert len( { ArrayID(a[0, :5]), ArrayID(a[0, 1:5]) } ) == 2   # diff length, same end
  s = np.empty(1)
  assert len( { ArrayID(s), ArrayID(s) } ) == 1       # scalar
  print("test_ArrayID passed")

class CompoundVector(object):
  def __init__(self, capacity=100000):
    self.buff = np.empty(capacity)
    self.tail = 0

  def NewSegment(self, dim):
    """ Make new segment on the tail """
    head = self.tail
    array = np.ndarray(shape  = (dim,),
                     buffer = self.buff,
                     offset = head*8 )  # 8 for double size
    self.tail += dim
    return head, array

  @property
  def flat(self):
    return np.ndarray((self.tail,), buffer=self.buff)

class CompoundVectorWithDict(CompoundVector):
  Segment = namedtuple('Segment', ['offset', 'array'])

  def __init__(self, cap = 100000):
    self.array_dict = {}
    super(CompoundVectorWithDict, self).__init__(cap)

  def AddVector(self, vector):
    """ 1. check whether it is a newcomer, use address as hash"""
    key = ArrayID(vector)
    if key not in self.array_dict:
      """ 2. Make new segment on the tail """
      dim = len(vector)
      offset, array = self.NewSegment(dim)
      array[:] = vector # copy inital value
      new_seg = CompoundVectorWithDict.Segment(offset, array)
      self.array_dict[key] = new_seg
      return new_seg
    else:
      return self.array_dict[key]

  def FindVector(self, vector):
    key = ArrayID(vector)
    return  self.array_dict.get(key, (None,None))

  def OverWriteOrigin(self):
    for dst, seg in self.array_dict.items():
      dst.data[:] = seg.array

class CompoundVectorWithMapping(CompoundVector):
  Segment = namedtuple('Segment', ['offset', 'src_array', 'dst_array', 'op'])

  def __init__(self, cap = 100000):
    self.segments = []
    super(CompoundVectorWithMapping, self).__init__(cap)

  def AddMaping(self, src_dim, dst_array, op=None):
    """ 1. Make new segment on the tail """
    offset, src_array = self.NewSegment(src_dim)
    new_seg = CompoundVectorWithMapping.Segment(offset, src_array, dst_array, op)
    self.segments.append(new_seg)
    return offset, src_dim

  def Flush(self):
    for seg in self.segments:
      if seg.op is None:
        seg.dst_array[:] = seg.src_array
      else:
        seg.dst_array[:] = seg.op(seg.src_array, seg.dst_array)


def test_CompoundVector():
  vs = np.random.rand(4,4)
  # main function
  cv = CompoundVectorWithDict()
  offset,seg = list(zip(*[cv.AddVector(v) for v in vs]))[:2]
  assert_array_equal(cv.flat, vs.ravel())
  assert_equal(offset, [0,4,8,12])

  # write, from flat to segment
  cv.flat[:] = 1
  for s in seg:
    assert_equal(s, np.ones(4))

  # write, from segment to flat
  seg[0][:] = 0
  assert_equal( cv.flat, np.hstack( [np.zeros(4), np.ones(4*3) ] ) )

  # write, from flat to origin segment
  cv.OverWriteOrigin()
  assert_equal( vs.ravel(), np.hstack( [np.zeros(4), np.ones(4*3) ]) )

  # add duplcate vector
  offset2 = list(zip(*[cv.AddVector(v) for v in vs]))[0]
  assert_equal(offset, offset2)
  offset3 = cv.AddVector(  np.empty(1)  )[0]
  assert offset3 == 16

  # CompoundVectorWithMapping
  cvm = CompoundVectorWithMapping()
  cv  = CompoundVectorWithDict()

  vs = np.random.rand(4,4)
  def op_double(src, dst):
    return 2*src
  for i,v in enumerate(vs):
    dst_offset, dst_array = cv.AddVector(v)
    src_offset, src_array = cvm.AddMaping(4, dst_array, None if i==0 else op_double)
  cvm.flat[:] = 1
  cvm.Flush()
  assert_array_equal(cv.flat, np.r_[np.ones(4), np.full(12, 2.0)])

  print("test_CompoundVector passed")


#%% GaussHelmertProblem
class VariableBlock(object):
  __slots__ = 'array', 'dim', 'isfixed', 'param', 'jac'
  def __init__(self, array):
    self.array = array
    self.dim   = len(array)
    self.isfixed = False
    self.param = None
    self.jac = []
  def __repr__(self):
    return "VariableBlock:%s, %dlinks" % (self.array, len(self.jac))

class ObservationBlock(VariableBlock):
  __slots__ = 'sigma','err'
  def __init__(self, array):
    super(ObservationBlock, self).__init__(array)
    self.sigma = None
    self.err = None
  def __repr__(self):
    return "ObservationBlock:%s, %dlinks" % (self.array, len(self.jac))

class GaussHelmertProblem(object):
  ConstraintBlock = namedtuple('ConstraintBlock', ['offset', 'g_res', 'g_jac'])

  def __init__(self):
    self.cv_x   = CompoundVectorWithDict()
    self.cv_l   = CompoundVectorWithDict()
    self.cv_res = CompoundVector()

    self.mat_sigma = None
    self.mat_jac_x = None
    self.mat_jac_l = None

    self.Jx, self.Jl, self.Sigma, self.W = None,None,None,None

    self.constraint_blocks = []    # list of ConstraintBlock
    self.variance_factor = -1.0

    self.dict_parameter_block = OrderedDict()
    self.dict_observation_block = OrderedDict()
    self.dicts = {0: self.dict_parameter_block, 1:self.dict_observation_block}

  def AddParameter(self, x_list):
    id = []
    blocks = []
    for x in x_list:
      offset, array = self.cv_x.AddVector(x)
      id.append(offset)
      obj = self.dict_parameter_block.get(offset, None)
      if obj is None:
        obj = VariableBlock(array)
        self.dict_parameter_block[offset] = obj
      blocks.append(obj)
    return id, blocks

  def AddObservation(self, l_list):
    id = []
    blocks = []
    for l in l_list:
      offset, array = self.cv_l.AddVector(l)
      id.append(offset)
      obj = self.dict_observation_block.get(offset, None)
      if obj is None:
        obj = ObservationBlock(array)
        self.dict_observation_block[offset] = obj
      blocks.append(obj)
    return id, blocks

  def SetVarFixedWithID(self, array_id, which=None):
    if not which is None:
      self.dicts[which][array_id].isfixed = True
    else:
      try:
        self.dict_parameter_block[array_id].isfixed = True
        return
      except KeyError: pass # try next
      try:
        self.dict_observation_block[array_id].isfixed = True
        return
      except KeyError:
        raise ValueError("array not found")

  def SetSigmaWithID(self, array_id, sigma):
    try:
      self.dict_observation_block[array_id].sigma = np.atleast_2d(sigma)
    except KeyError:
      raise ValueError("array not found")

  def SetParameterizationWithID(self, array_id, param, which=None):
    if not which is None:
      self.dicts[which][array_id].param = param
    else:
      try:
        self.dict_parameter_block[array_id].param = param
        return
      except KeyError: pass # try next
      try:
        self.dict_observation_block[array_id].param = param
        return
      except KeyError:
        raise ValueError("array not found")

  def LookupArrayID(self, array):
    array_id, _ = self.cv_x.FindVector(array)
    if not array_id is None:
      return array_id, 0

    array_id, _ = self.cv_l.FindVector(array)
    if not array_id is None:
      return array_id, 1
    raise ValueError("array not found")

  def SetVarFixed(self, array):
    array_id, which_var = self.LookupArrayID(array)
    return self.SetVarFixedWithID(array_id, which_var)

  def SetSigma(self, array, sigma):
    array_id, _ = self.cv_l.FindVector(array)
    self.SetSigmaWithID(array_id, sigma)

  def SetParameterization(self, array, param):
    array_id, which_var = self.LookupArrayID(array)
    return self.SetParameterizationWithID(array_id, param, which_var)


  def SetUp(self):
    """ reset everything"""
    self.mat_sigma = SparseBlock()
    self.mat_jac_x = SparseBlock()
    self.mat_jac_l = SparseBlock()
    self.cv_dx  = CompoundVectorWithMapping()
    self.cv_dl  = CompoundVectorWithMapping()
    self.cv_le  = CompoundVector()
    def EnclosePlus(obj_func):
      def callback(dv, v):
        return obj_func(v, dv)
      return callback

    def EncloseToLocalJacobian(obj_func):
      def callback(J):
        return obj_func(J)
      return callback

    for x in self.dict_parameter_block.values(): #array, dim, isfixed, param
      if x.isfixed:
        continue  # ignore fixed

      """1. dx vector, its pos"""
      param = x.param
      if param is None:
        dx_dim = x.dim
        dx_offset, dx_array = self.cv_dx.AddMaping(dx_dim, x.array, lambda dx,x_ : x_+dx)
        pre_process = None
      else:
        dx_dim = param.LocalSize()
        dx_offset, dx_array = self.cv_dx.AddMaping(dx_dim, x.array, EnclosePlus(param.Plus))
        pre_process = EncloseToLocalJacobian(param.ToLocalJacobian)

      """2. jacobian at that pos"""
      for dm in x.jac:
        dm.c = dx_offset
        dm.shape[1] = dx_dim
        dm.InitElement(dm.shape)
        dm.pre_process = pre_process
        self.mat_jac_x.AddElement(dm)

    for l in self.dict_observation_block.values(): #array, dim, isfixed, param
      if l.isfixed:
        continue  # ignore fixed
      param = l.param
      if param is None:
        dl_dim = l.dim
        dl_offset, dl_array = self.cv_dl.AddMaping(dl_dim, l.array, lambda dl,l_ : l_+dl)
        pre_process = None
      else:
        dl_dim = param.LocalSize()
        dl_offset, dl_array = self.cv_dl.AddMaping(dl_dim, l.array, EnclosePlus(param.Plus))
        pre_process = EncloseToLocalJacobian(param.ToLocalJacobian)

      for dm in l.jac:
        dm.c = dl_offset
        dm.shape[1] = dl_dim
        dm.InitElement(dm.shape)
        dm.pre_process = pre_process
        self.mat_jac_l.AddElement(dm)

      _, l.err = self.cv_le.NewSegment(dl_dim)

      """3. sigma at that pos"""
      dm = DenseMatrix(dl_offset, dl_offset, (dl_dim, dl_dim))
      dm.Write( np.eye(dl_dim) if l.sigma is None else l.sigma )
      self.mat_sigma.AddElement(dm)


  def Plus(self, dx, dl=None):
    self.cv_dx.flat[:] = dx
    self.cv_dx.Flush()

    if not dl is None:
      self.cv_dl.flat[:] = dl
      self.cv_dl.Flush()


  def AddConstraintWithArray(self, g, x_list, l_list, g_jac=None):
    """ 1. Assign poses and mapped vectors for input parameter/observation arrays"""
    x_ids,x_blocks = self.AddParameter(x_list)
    l_ids,l_blocks = self.AddObservation(l_list)

    self._add_constraint(g, g_jac, x_blocks, l_blocks)


  def AddConstraintWithID(self, g, x_ids, l_ids, g_jac=None):
    x_blocks,l_blocks = [],[]
    for x in x_ids:
      block = self.dict_parameter_block.get(x, None)
      if block is None:
        raise RuntimeError("wrong id")
      x_blocks.append(block)

    for l in l_ids:
      block = self.dict_observation_block.get(l, None)
      if block is None:
        raise RuntimeError("wrong id")
      l_blocks.append(block)

    self._add_constraint(g, g_jac, x_blocks, l_blocks)

  def _add_constraint(self, g, g_jac, x_blocks, l_blocks):
    xl_vec  = [ b.array  for b in x_blocks+l_blocks ]
    xl_sizes= [ len(vec) for vec in xl_vec ]

    """ 1. Generate Jacobian function by cppad if g_jac is not supplied"""
    if g_jac is None:
      xl_indices= np.cumsum( xl_sizes )[:-1]
      var       = np.hstack( xl_vec )
      var_in    = pycppad.independent( var )
      var_out   = np.atleast_1d( g( *np.split(var_in, xl_indices) ) )
      var_jacobian= pycppad.adfun(var_in, var_out).jacobian
      def g_jac(*vec):
        J = var_jacobian( np.hstack(vec) )
        return np.split(J, xl_indices, axis=1)

    """ 2. Sanity check of size and validation"""
    tmp_res = np.atleast_1d( g( *xl_vec ) )
    tmp_jac = list( g_jac( *xl_vec ) )

    inequal_size = [j.shape != (len(tmp_res), size) for j, size in zip(tmp_jac, xl_sizes)]
    if len(tmp_jac) != len(xl_sizes) or np.any( inequal_size ):
      raise RuntimeError("Jacobian Size Not fit")
    valid_value = [ np.isfinite(m).all() and not np.all(m==0) for m in [tmp_res] + tmp_jac ]
    if not np.all( valid_value ):
      raise RuntimeError("return value of function Not valid")
    dim_res = len(tmp_res)

    """ 3. Make and append compound vector for constraint residual """
    res_off, res_vec = self.cv_res.NewSegment(dim_res)

    """ 4. Generate functor that use the mapped vectors to calcuate residual and jacobians"""
    def g_residual():
      res_vec[:] = g(*xl_vec)

    def g_jacobians():
      jac = list( g_jac( *xl_vec ) )
      jac.reverse() # reversed, to pop(-1) instead of pop(0)
      for dm in dms:
        dm.Write( check_allzero( jac.pop() ) )

    """ 5. Make new DenseMatrix that will hold the jacobians """
    dms = []
    for b in x_blocks + l_blocks:  #'array', 'dim', 'isfixed', 'param', 'jac'
      new_dm = DenseMatrix(res_off, 0)
      new_dm.shape = [dim_res, 0]
      b.jac.append( new_dm )
      dms.append( new_dm )

    """ 6. new record in the system"""
    self.constraint_blocks.append( GaussHelmertProblem.ConstraintBlock(res_off, g_residual, g_jacobians) )


  def CompoundParameters(self):
    return self.cv_x.flat

  def CompoundObservation(self):
    return self.cv_l.flat

  def CompoundResidual(self):
    return self.cv_res.flat

  @property
  def dim_x(self):    return self.cv_x.tail
  @property
  def dim_l(self):    return self.cv_l.tail
  @property
  def dim_dx(self):   return self.cv_dx.tail
  @property
  def dim_dl(self):   return self.cv_dl.tail
  @property
  def dim_res(self):  return self.cv_res.tail
  @property
  def dims(self):     return [self.dim_x, self.dim_l, self.dim_res ]

  def MakeJacobians(self,**kwarg):
    if self.mat_jac_l is None:
      self.SetUp()
    self.Jx = self.mat_jac_x.BuildSparseMatrix(**kwarg)
    self.Jl = self.mat_jac_l.BuildSparseMatrix(**kwarg)

    return self.Jx, self.Jl

  def MakeReducedKKTMatrix(self, *args, **kwarg):
    if self.mat_jac_l is None:
      self.SetUp()
    dim_x, dim_res = self.dim_x,  self.dim_res
    dim_total = dim_x + dim_res

    self.mat_jac_l.OverwriteRC(0, 0)
    self.mat_sigma.OverwriteRC(0, 0)
    self.mat_jac_x.OverwriteRC(0, dim_res)
    mat_BSBT = MakeAWAT(self.mat_jac_l, self.mat_sigma, make_full=False)
    mat_kkt   = CompoundMatrix([mat_BSBT, self.mat_jac_x])

    return MakeSymmetric(mat_kkt).BuildSparseMatrix((dim_total, dim_total),**kwarg)

  def MakeSigmaAndWeightMatrix(self, **kwarg):
    if self.mat_sigma is None:
      self.SetUp()

    self.mat_w = MakeBlockInv(self.mat_sigma)
    self.Sigma = self.mat_sigma.BuildSparseMatrix(**kwarg)

    self.W     = self.mat_w.BuildSparseMatrix(**kwarg)
    return self.Sigma, self.W

  def MakeLargeKKTMatrix(self, **kwarg):
    dim_x, dim_l, dim_res = self.dim_x, self.dim_l, self.dim_res
    dim_total = dim_x + dim_l + dim_res

    self.mat_sigma.OverwriteRC(dim_x, dim_x)
    self.mat_jac_x.OverwriteRC(dim_x+dim_l, 0)
    self.mat_jac_l.OverwriteRC(dim_x+dim_l, dim_x)

    mat_kkt   = CompoundMatrix()
    mat_kkt.AddElement(self.mat_sigma)
    mat_kkt.AddElement(self.mat_jac_x)
    mat_kkt.AddElement(self.mat_jac_l)

    kkt = mat_kkt.BuildSparseMatrix((dim_total,dim_total), **kwarg)
    return kkt

  def MakeLargeKKTSegmentSlice(self):
    offset =  np.cumsum([0]+self.dims)
    segment_x = slice(offset[0], offset[1])
    segment_l = slice(offset[1], offset[2])
    segment_r = slice(offset[2], offset[3])
    return [segment_x, segment_l, segment_r ]

  def UpdateResidual(self, ouput=False):
    for cb in self.constraint_blocks:
      cb.g_res()

    if ouput:
      return self.cv_res.flat

  def UpdateJacobian(self, ouput=False):
    for x in self.dict_parameter_block.values(): #array, dim, isfixed, param
      if x.isfixed or x.param is None:
        continue  # ignore fixed
      x.param.UpdateJacobian(x.array)

    for l in self.dict_observation_block.values(): #array, dim, isfixed, param
      if l.isfixed or l.param is None:
        continue  # ignore fixed
      l.param.UpdateJacobian(l.array)

    for cb in self.constraint_blocks:
      cb.g_jac()

    if ouput:
      return self.Jx, self.Jl

  def UpdateXL(self, x=True, l=False ):
    if x:
      self.cv_x.OverWriteOrigin()
    if l:
      self.cv_l.OverWriteOrigin()

  def ViewJacobianPattern(self, withB=False, fig=None):
    if None in (self.Jx, self.Jl):
      return
    A,B = self.Jx, self.Jl
    if not scipy.sparse.isspmatrix_coo(A):
      A = A.tocoo()
    img_A = np.ones(A.shape, 'u8')
    img_A[A.row, A.col] = np.logical_not(A.data)
    plt.matshow(img_A)

    if withB:
      if not scipy.sparse.isspmatrix_coo(B):
        B = B.tocoo()
      img_B = np.ones(B.shape, 'u8')
      img_B[B.row, B.col] = np.logical_not(B.data)
      plt.matshow(img_B)
    plt.pause(0.001)

#%%
def EqualityConstraint(a,b):
  return a-b
def EqualityConstraintJac(a,b):
  return np.eye(len(a)), -np.eye(len(b))

def MakeAffineConstraint(A,B):
  def AffineConstraint(a, b):
    return A.dot(a) + B.dot(b)
  def AffineConstraintJac(a, b):
    return A, B
  return AffineConstraint, AffineConstraintJac

def DiagonalRepeat(M, repeats):
  return scipy.linalg.block_diag(* (M,)*repeats )

def VerticalRepeat(M, repeats):
  return np.tile( M, (int(repeats),1) )

def HorizontalRepeat(M, repeats):
  return np.tile( M, (1, int(repeats)) )

def test_ProblemBasic():
  dim_x, num_x = 3, 2
  dim_l, num_l = 4, 30*num_x

  dim_g = 3
  A = np.random.rand(dim_g, dim_x)
  B = np.random.rand(dim_g, dim_l)
  AffineConstraint, AffineConstraintJac = MakeAffineConstraint(A,B)

  x = np.zeros((num_x, dim_x))
  l = [ np.ones((int(num_l/num_x), dim_l)) for _ in range(num_x) ] # l[which_x] = vstack(l[which_l])
  sigma = np.full(dim_l, 0.5)
  problem = GaussHelmertProblem()
  for i in range(num_x):
    for j in range(int(num_l/num_x)):
      problem.AddConstraintWithArray(AffineConstraint,
                                     [ x[i] ],
                                     [ l[i][j] ])
      problem.SetSigma(l[i][j], np.diag(sigma))
  problem.SetUp()

  # res compound
  dim_res = dim_g * num_l
  res = problem.CompoundResidual()
  assert_equal( res.shape, [ dim_res ]  )

  # x l compound
  xc = problem.CompoundParameters()
  assert_equal( xc.shape, [ dim_x * num_x ]  )

  lc = problem.CompoundObservation()
  assert_equal( lc.shape, [ dim_l * num_l ]  )

  # write xl, read res and  UpdateResidual()
  xc[:] = 0
  lc[:] = 0
  problem.UpdateResidual()
  assert_array_equal(res, np.zeros(dim_res) )

  # OverWriteOrigin
  xc[:] = 1
  problem.UpdateXL()
  assert_array_equal(x, np.ones((num_x, dim_x)))

  # Create Jacobian
  Jx,Jl = problem.MakeJacobians()
  assert_equal( Jx.shape, (dim_res, dim_x*num_x) )
  assert_equal( Jl.shape, (dim_res, dim_l*num_l) )

  # Evaluate Jacobian
  problem.UpdateJacobian()
  assert_array_equal( Jx.todense(), DiagonalRepeat( VerticalRepeat(A, num_l/num_x), num_x) )
  assert_array_equal( Jl.todense(), DiagonalRepeat(B, num_l) )

  # weight
  Sig, W = problem.MakeSigmaAndWeightMatrix()
  assert_array_equal(Sig.todense(), DiagonalRepeat(np.diag(sigma), num_l) )
  assert_array_equal( W.todense(), DiagonalRepeat(np.diag(1./sigma), num_l) )

  # DIY Jacobian and AddConstraintWithID
  problem2 = GaussHelmertProblem()
  for i in range(num_x):
    xid, _ = problem2.AddParameter([ x[i] ])
    for j in range(int(num_l/num_x)):
      lid, _ = problem2.AddObservation([ l[i][j] ])
      problem2.AddConstraintWithID(AffineConstraint,
                                   xid,
                                   lid,
                                   AffineConstraintJac )
  problem2.SetUp()
  Jx2,Jl2 = problem2.MakeJacobians()
  problem2.UpdateJacobian()
  assert_array_equal( Jx2.A, DiagonalRepeat( VerticalRepeat(A, num_l/num_x), num_x) )
  assert_array_equal( Jl2.A, DiagonalRepeat(B, num_l) )
  print("test_ProblemBasic passed")


def test_ProblemMakeKKT():
  def DumpXLConstraint(x,l):
    return 0.5*x.dot(x) + 0.5*l.dot(l)
  problem = GaussHelmertProblem()
  x = np.ones(2)
  l = np.full(2, 2.)
  problem.AddConstraintWithArray(DumpXLConstraint, [x], [l])
  problem.SetSigma(l, 0.5*np.eye(2))
  problem.SetUp()
  res  = problem.CompoundResidual()
  xc   = problem.CompoundParameters()
  lc   = problem.CompoundObservation()

  KKT      = problem.MakeReducedKKTMatrix( coo=False )
  Sigma, W = problem.MakeSigmaAndWeightMatrix()
  B        = problem.mat_jac_l.BuildSparseMatrix( coo=False )

  lc[:] = np.arange(2)
  xc[:] = 2+np.arange(2)
  problem.UpdateJacobian()
  problem.UpdateResidual()
  assert_array_equal( B.A, [np.arange(2)] )
  assert_array_equal(KKT.A,
                    [[ 0.5,  2. ,  3. ],
                     [ 2. ,  0. ,  0. ],
                     [ 3. ,  0. ,  0. ]])
  assert_equal(7, res)

  lc[:] += 1
  xc[:] += 1
  problem.UpdateJacobian()
  problem.UpdateResidual()
  assert_array_equal( B.A, [np.arange(2)+1] )
  assert_array_equal(KKT.A,
                    [[ 2.5,  3. ,  4. ],
                     [ 3. ,  0. ,  0. ],
                     [ 4. ,  0. ,  0. ]])
  assert_equal(15, res)
  print("test_ProblemMakeKKT passed")
def test_FixAndParameterization():

  x = np.ones((2, 3))
  l = [np.random.rand(100,3) for _ in range(2)]
  def IdentityConstraint(x,l):
    return x-l
  def IdentityConstraintJac(x,l):
    I = np.eye(len(x))
    return I, -I
  x2_true = [np.average(l[1][:,0]), 1, l[1][-1,2]]


  problem = GaussHelmertProblem()
  for i in range(2):
    for j in range(l[0].shape[0]):
      problem.AddConstraintWithArray(IdentityConstraint, [x[i]], [l[i][j]])
  problem.SetVarFixed(x[0])
  problem.SetParameterization(x[1], SubsetParameterization([1,0,1]))
  problem.SetParameterization(l[1][-1], SubsetParameterization([1,1,0]))

  x_est,le = SolveWithCVX(problem)
#  x_est,le = SolveWithGEDense(problem)
#  x_est,le = SolveWithKKT(problem)

  problem.cv_l.OverWriteOrigin()
  # fix
  assert_array_equal(x_est[0:3], np.ones(3))
  assert_array_equal(l[0], np.ones((100,3)))

  # SubsetParameterization
  assert_array_almost_equal(x_est[3:6], x2_true)
  assert_array_almost_equal(l[1], VerticalRepeat(x2_true,100))
  print("test_FixAndParameterization passed")

from cvxopt import matrix,spmatrix
from cvxopt import solvers
def SolveWithCVX(problem, maxit=10, fac=False, cov=False, dx_thres=1e-6):
  problem.SetUp()
  res  = problem.CompoundResidual()
  lc   = problem.CompoundObservation()
  xc   = problem.CompoundParameters()
  le    = np.zeros(problem.dim_dl)

  BSBT = MakeAWAT(problem.mat_jac_l, problem.mat_sigma, make_full=True ).BuildSparseMatrix(coo=True)
  A,B = problem.MakeJacobians(coo=True)
  Sigma, W = problem.MakeSigmaAndWeightMatrix(coo=True)
  b   = matrix(np.zeros((problem.dim_dx, 1), 'd'))
  for it in range(maxit):
    problem.UpdateJacobian()
    problem.UpdateResidual()

    print( np.linalg.norm(res),np.linalg.norm(le))

    P   = spmatrix(BSBT.data, BSBT.row, BSBT.col)
    q   = matrix(res - B*le)
    At  = spmatrix(A.data, A.col, A.row)     # transpose
    try:
      sol = solvers.qp(P,q, A=At, b=b )
      lag_neg= np.array(sol['x']).ravel()
      dx  = np.array(sol['y']).ravel()
      dl  = Sigma * (lag_neg * B) - le  # B.T * lag
      problem.Plus(dx, dl)
      le   += dl
    except:
      print("Singular problem")
      break
    if np.abs(dx).max() < dx_thres:
      break
  problem.cv_le.flat[:] = le

  ret = [xc, le]
  if fac:
    factor = (le * W).dot(le) / (problem.dim_res - problem.dim_dx)
    print( 'variance factor:%f' % factor)
    ret.append(factor)
  if cov:
    Wgg   = np.linalg.inv(BSBT.A)
    ATWA= (Wgg * A).T * A
    covariance = np.linalg.inv(ATWA)
    ret.append( covariance )
  return ret

def SolveWithGEDense(problem, fac=False, cov=False):
  problem.SetUp()
  res  = problem.CompoundResidual()
  lc   = problem.CompoundObservation()
  xc   = problem.CompoundParameters()
  le   = np.zeros(problem.dim_dl)

  BSBT = MakeAWAT(problem.mat_jac_l, problem.mat_sigma, make_full=True ).BuildSparseMatrix(coo=True)
  A,B = problem.MakeJacobians(coo=True)
  Sigma, W = problem.MakeSigmaAndWeightMatrix(coo=True)
  for it in range(10):
    problem.UpdateJacobian()
    problem.UpdateResidual()

    print( np.linalg.norm(res),np.linalg.norm(le))
    Wgg = np.linalg.inv(BSBT.A)
    Cg  = B * le - res
    ATW = (Wgg * A).T
    ATWA= ATW * A
    dx  = np.linalg.solve(ATWA,  ATW.dot(Cg))
    if np.abs(dx).max() < 1e-6:
      break
    lag = Wgg.dot( A * dx - Cg )
    dl  = -Sigma * (lag * B) - le  # B.T * lag
    problem.Plus(dx, dl)
    le  += dl
  problem.cv_le.flat[:] = le

  ret = [xc, le]
  factor = (le * W).dot(le) / (problem.dim_res - problem.dim_dx)
  print( 'variance factor:%f' % factor)
  if fac:
    ret.append(factor)

  if cov:
    covariance = np.linalg.inv(ATWA)
    if np.isfinite(factor) and factor!=0 :
      covariance *= factor
    ret.append( covariance )
  return ret

from sksparse.cholmod import cholesky,CholmodNotPositiveDefiniteError
def SolveWithGESparse(problem, maxit=10, fac=False, cov=False, dx_thres=1e-6):
  problem.SetUp()
  res  = problem.CompoundResidual()
  lc   = problem.CompoundObservation()
  xc   = problem.CompoundParameters()
  le   = np.zeros(problem.dim_dl)

  BSBT = MakeAWAT(problem.mat_jac_l, problem.mat_sigma, make_full=True ).BuildSparseMatrix(coo=False)
  A,B = problem.MakeJacobians(coo=False)
  Sigma, W = problem.MakeSigmaAndWeightMatrix(coo=False)

  Sgg_factor  = None
  Sxx_factor  = None

  for it in range(maxit):
    problem.UpdateJacobian()
    problem.UpdateResidual()
    print( np.linalg.norm(res),np.linalg.norm(le))
    Cg  = B * le - res

    if Sgg_factor is None:
      Sgg_factor = cholesky(BSBT)
    else:
      Sgg_factor.cholesky_inplace(BSBT)

    F = Sgg_factor.solve_A(A)   # W * A = F -->> A = BSBT * F
                                # ATW = (BSBT * F).T / BSBT = F.T
    ATWA= (A.T * F).T           # to maintain in csc
    if Sxx_factor is None:
      Sxx_factor = cholesky(ATWA)
    else:
      Sxx_factor.cholesky_inplace(ATWA)
    dx  = Sxx_factor.solve_A( Cg * F )  #np.linalg.solve(ATWA,  ATW.dot(Cg))

    if np.abs(dx).max() < dx_thres:
      break
    lag = Sgg_factor.solve_A( A * dx - Cg ) # BSBT*lambda = A*dx - Cg
    dl  = -Sigma * (lag * B) - le  # B.T * lag
    problem.Plus(dx, dl)
    le  += dl

  problem.cv_le.flat[:] = le
  ret = [xc, le]
  factor = (le * W).dot(le) / (problem.dim_res - problem.dim_dx)
  print( 'variance factor:%f' % factor)

  if fac:
    ret.append(factor)
  if cov:
    covariance = Sxx_factor.inv()
    if np.isfinite(factor) and factor!=0 :
      covariance *= factor
    ret.append( covariance )
  return ret


def SolveWithGESparseAsGM(problem, maxit=10, fac=False, cov=False, dx_thres=1e-6):
  problem.SetUp()
  res  = problem.CompoundResidual()
  lc   = problem.CompoundObservation()
  xc   = problem.CompoundParameters()

  BSBT = MakeAWAT(problem.mat_jac_l, problem.mat_sigma, make_full=True ).BuildSparseMatrix(coo=False)
  A,B = problem.MakeJacobians(coo=False)
  Sigma, W = problem.MakeSigmaAndWeightMatrix(coo=False)
  problem.UpdateJacobian()

  Sgg_factor  = cholesky(BSBT)
  Sxx_factor  = None

  for it in range(maxit):
    problem.UpdateJacobian()
    problem.UpdateResidual()
    print( np.linalg.norm(res))
    Cg  = - res

    F = Sgg_factor.solve_A(A)   # W * A = F -->> A = BSBT * F
                                # ATW = (BSBT * F).T / BSBT = F.T
    ATWA= (A.T * F).T           # to maintain in csc
    if Sxx_factor is None:
      Sxx_factor = cholesky(ATWA)
    else:
      Sxx_factor.cholesky_inplace(ATWA)
    dx  = Sxx_factor.solve_A( Cg * F )  #np.linalg.solve(ATWA,  ATW.dot(Cg))
    if np.abs(dx).max() < dx_thres:
      break
    problem.Plus(dx)

  ret = [xc, res]
  factor = res.dot( Sgg_factor.solve_A(res) ) / (problem.dim_res - problem.dim_dx)
  print( 'variance factor:%f' % factor)
  if fac:
    ret.append(factor)
  if cov:
    covariance = Sxx_factor.inv()
    if np.isfinite(factor) and factor!=0 :
      covariance *= factor
    ret.append( covariance )
  return ret

def test_SolveWithGESparseAsGM():
  dim_x = 3
  a = np.ones(dim_x)

  num_l = 100
  sigma = 0.02
  s = sigma**2*np.eye(dim_x)
  x = np.empty((1000, 3))
  fac = np.empty(1000)
  for trail in range(1000):
    bs = [ a + sigma * np.random.randn(3) for i in range(num_l)]
    problem = GaussHelmertProblem()
    for i in range(num_l):
      problem.AddConstraintWithArray(lambda x,y:x-y,
                                   [ a ],
                                   [ bs[i] ])
      problem.SetSigma(bs[i], s)
    x[trail],_, fac[trail] = SolveWithGESparseAsGM(problem,fac=True)
  plt.hist(fac)
  np.mean(fac)
  np.mean(x,axis=0)

def SolveWithGESparseLM(problem, maxit=10, fac=False, cov=False, dx_thres=1e-6):
  problem.SetUp()
  res  = problem.CompoundResidual()
  lc   = problem.CompoundObservation()
  xc   = problem.CompoundParameters()
  le   = np.zeros(problem.dim_dl)

  BSBT = MakeAWAT(problem.mat_jac_l, problem.mat_sigma, make_full=True ).BuildSparseMatrix(coo=False)
  A,B = problem.MakeJacobians(coo=False)
  Sigma, W = problem.MakeSigmaAndWeightMatrix(coo=False)
  problem.UpdateJacobian()
  problem.UpdateResidual()

  Sgg_factor  = cholesky(BSBT)
  Sxx_factor  = None

  tau = 1e-2
  nu = 2.

  x_old = xc.copy()
  l_old = lc.copy()
  r_old = res.copy()

  for it in range(maxit):

    print( np.linalg.norm(res),np.linalg.norm(le))
    Cg  = B * le - res

    # solve
    Sgg_factor.cholesky_inplace(BSBT)
    F = Sgg_factor.solve_A(A)   # W * A = F -->> A = BSBT * F
                                # ATW = (BSBT * F).T / BSBT = F.T
    ATWA= (A.T * F).T           # to maintain in csc
    if Sxx_factor is None:
      mu = tau * ATWA.diagonal().max()
      Sxx_factor = cholesky(ATWA, beta=mu)
    else:
      Sxx_factor.cholesky_inplace(ATWA, beta=mu)
    dx  = Sxx_factor.solve_A( Cg * F )  #np.linalg.solve(ATWA,  ATW.dot(Cg))
    lag = Sgg_factor.solve_A( A * dx - Cg ) # BSBT*lambda = A*dx - Cg
    dl  = -Sigma * (lag * B) - le  # B.T * lag
    problem.Plus(dx, dl)

    problem.UpdateResidual()
    pre = A * dx + B * dl
    rho = (np.linalg.norm(r_old) - np.linalg.norm(res))/np.linalg.norm(pre)

    if rho > 0:
      l_old[:] = lc
      x_old[:] = xc
      r_old[:] = res
      le  += dl
      problem.UpdateJacobian()

      mu = mu * np.max([1.0/3, 1.0 - (2*rho - 1)**3])
      nu = 2.0

      if np.abs(dx).max() < dx_thres:
        break
    else:
      lc[:] = l_old
      xc[:] = x_old
      res[:] = r_old

      mu = mu * nu
      nu = 2*nu
      print( mu)

  ret = [xc, le]
  factor = (le * W).dot(le) / (problem.dim_res - problem.dim_dx)
  print( 'variance factor:%f' % factor)
  if fac:
    ret.append(factor)
  if cov:
    covariance = Sxx_factor.inv()
    if np.isfinite(factor) and factor!=0 :
      covariance *= factor
    ret.append( covariance )
  return ret


#from pykrylov.symmlq import Symmlq
#from symmlq import symmlq
#from cvxopt import blas
#def SolveWithKKT(problem, maxit=10, fac=False):
#  problem.SetUp()
#  res  = problem.CompoundResidual()
#  xc   = problem.CompoundParameters()
#  lc   = problem.CompoundObservation()
#
#  KKT      = problem.MakeReducedKKTMatrix( coo=True )
#  Sigma, W = problem.MakeSigmaAndWeightMatrix()
#  B        = problem.mat_jac_l.BuildSparseMatrix( coo=False )
#
#  le       = np.zeros(problem.dim_dl)
#  b        = np.zeros( len(res) + problem.dim_dx )
#  seg_lambd= slice(0, len(res))
#  seg_dx   = slice(len(res), None)
#
##  op_A   = CoordLinearOperator(KKT.data, KKT.row, KKT.col, KKT.shape[1], KKT.shape[0] )
##  solver = Symmlq(KKT)
#
#  for it in range(maxit):
#    problem.UpdateJacobian()
#    problem.UpdateResidual()
#    print np.linalg.norm(res), np.linalg.norm(le)
#    b[seg_lambd]  = B * le - res
#
#    G   = spmatrix(KKT.data, KKT.row, KKT.col)
#    q   = matrix(b)
#    sol = np.array(symmlq( q, G, show=1 )[0]).ravel()
#
##    solver.solve(b)
##    sol = solver.x
#
#    dx      = sol[seg_dx]
#    lag_neg = sol[seg_lambd]
#    dl  = Sigma * (lag_neg * B) - le
#
#    problem.Plus(dx, dl)
#    le   += dl
#
#    if np.abs(dx).max() < 1e-6:
#      break
#
#    problem.UpdateJacobian()
#    problem.UpdateResidual()
#
#  ret = [xc, le]
#  if fac:
#    factor = (le * W).dot(le) / (problem.dim_res - problem.dim_dx)
#    ret.append(factor)
#
#  return ret

#%%
_robust_func = { 'none'        : lambda y : np.ones_like(y) ,
                 'huber'       : lambda y : np.fmin(1.0, 1.0/np.abs(y)),
                 'huber_smooth': lambda y : 1.0/np.sqrt(1.0 + y**2),
                 'exp_2'       : lambda y : np.exp(-0.5*y**2/4),    }# np.exp(-0.5*y**2/c**2), c=2}

from collections import  namedtuple
from itertools import count, compress
class BatchGaussHelmertProblem(object):
  class Observation(object):
    __slots__ = "array","param","cov","err"
    def __init__(self, array, param=None, cov=None):
      # check
      dim = len(array)
      param = replace_none_or_check_param(param, dim)
      dim_delta = param.LocalSize()
      if cov is None:
        cov = np.eye(dim_delta)
      else:
        cov_shape = np.shape(cov)
        assert len(set(cov_shape))==1 and len(cov_shape)==2 # diagonal
        assert cov_shape[0] == dim_delta                    # dim
      # set
      self.array = array
      self.param = param
      self.cov = cov
      self.err = np.zeros(dim_delta)

    def UpdatePrmJacobian(self):
      self.param.UpdateJacobian(self.array)

    def Plus(self, delta):
      self.array[:] = self.param.Plus(self.array, delta)
      self.err += delta

    def Cost(self):
      W = np.linalg.pinv( self.cov )
      return self.err.dot( W ).dot(self.err)

    def ClearErr(self):
      self.err[:] = 0

    @property
    def dim(self): return len(self.array)
    @property
    def dim_delta(self): return self.param.LocalSize()

  class ObservationGroup(object):
    def __init__(self):
      self.elements = []
      self.notfixed = True

    def Add(self, obs):
      # check
      assert isinstance(obs, BatchGaussHelmertProblem.Observation)
      if self.size > 0:
        assert obs.dim == self.dim
        assert type(obs.param) == type(self[0].param)
      # set
      obs_id = self.size
      self.elements.append(obs)
      return obs_id

    def UpdatePrmJacobian(self):
      for e in self.elements:
        e.UpdatePrmJacobian()

    def ClearErr(self):
      for e in self.elements:
        e.ClearErr()

    def Cost(self):
      return sum( e.Cost() for e in self.elements  )

    def arrays(self):
      return [e.array for e in self.elements]
    def params(self):
      return [e.param for e in self.elements]
    def covs(self):
      return [e.cov for e in self.elements]
    def errs(self):
      return [e.err for e in self.elements]

    def __getitem__(self, key):
      return self.elements[key]
    @property
    def size(self):  return len(self.elements)
    @property
    def dim(self):   return self[0].dim if self.size > 0 else -1
    @property
    def dim_delta(self):   return self[0].dim_delta if self.size > 0 else -1
    def __repr__(self):
      return  str(self.arrays())


  #------------------------BatchGaussHelmertProblem----------------------------
  def __init__(self, g, num_x_arg, num_l_arg):
    self.g = g
    # prelocated space
    self.x        = [None]*num_x_arg
    self.x_params = [None]*num_x_arg
    self.l_groups = [ BatchGaussHelmertProblem.ObservationGroup() for _ in range(num_l_arg)]

    self.num_l_arg = num_l_arg
    self.num_x_arg = num_x_arg

  def x_arrays(self):
    return tuple(self.x)
  def l_arrays(self):
    return zip(* (group.arrays() for group in self.l_groups) )
  def l_params(self):
    return zip(* (group.params() for group in self.l_groups) )
  def l_covs(self):
    return zip(* (group.covs() for group in self.l_groups) )
  def l_errs(self):
    return zip(* (group.errs() for group in self.l_groups) )

  def SetParameter(self, slot, array, param=None):
    # check
    dim_arg = len(array)
    param = replace_none_or_check_param(param, dim_arg)
    # set
    self.x[slot] = array
    self.x_params[slot] = param

  def ChangeParameterParameterization(self, slot, param=None):
    assert not self.x[slot] is None
    self.SetParameter(slot, self.x[slot], param)

  def AddObservation(self, slot, array, cov=None, param=None):
    new_observation = BatchGaussHelmertProblem.Observation( array, param, cov)
    obs_id = self.l_groups[slot].Add( new_observation )
    return obs_id, new_observation.err

  def SetObservationGroupFixiation(self, slot, fix):
    self.l_groups[slot].notfixed = not bool(fix)

  def Setup(self):
    for arr in self.x:
      if arr is None:
        raise RuntimeError("some parameter not set, use SetParameter\n %s" % self.parameters)

    group_sizes = [group.size for group in self.l_groups ]
    if len(set(group_sizes)) != 1 or group_sizes[0] == 0:
      raise RuntimeError("number of observationsnot equal \n %s" % group_sizes)
    self.num_obs = group_sizes[0]

    # parameterization x
    size_list_x  = [len(arr) for arr in self.x]
    self.dim_x   = sum(size_list_x)
    self.slice_x = MakeSequenceSlice(size_list_x)

    size_list_dx = [prm.LocalSize() for prm in self.x_params]
    self.dim_dx  = sum(size_list_dx)
    self.slice_dx= MakeSequenceSlice(size_list_dx)

    # parameterization l
    size_list_l  = [grp.dim for grp in self.l_groups]
    self.slice_l = MakeSequenceSlice(size_list_l, self.dim_x)
    self.dim_l   = sum(size_list_l)

    size_list_dl = [ grp.dim_delta if grp.notfixed else 0 for grp in self.l_groups ] # not include fixed observations
    self.slice_dl= MakeSequenceSlice(size_list_dl)
    self.dim_dl  = sum(size_list_dl)

    self.l_notfixeds = [ grp.notfixed for grp in self.l_groups ]

    # dim_err
    err, J = self.g( * self.x_arrays() + next(self.l_arrays()) )
    self.dim_err = len(err)
    check_equal(J.shape, (self.dim_err, self.dim_x+self.dim_l))

    # clear observation corrections
    if 0:
      for grp in self.l_groups:
        grp.ClearErr()

  def _UpdateAllPrmJacobian(self):
    for x, param in zip(self.x, self.x_params):
      param.UpdateJacobian(x)
    for grp in self.l_groups:
      if grp.notfixed:
        grp.UpdatePrmJacobian()

  def _Plus_dx(self, dx):
    for x, param, seg_dx in zip(self.x, self.x_params, self.slice_dx):
        x[:] = param.Plus(x, dx[seg_dx])

  def _Plus_dl(self, obs_id, dl):
    for grp, seg_dl in zip(self.l_groups, self.slice_dl):
      if grp.notfixed:
        grp[obs_id].Plus(dl[seg_dl])

  def Solve(self, asGM=False, maxiter=100, Tx=1e-6, weight='none', update_cov=False):
    # 1. setup
    self.Setup()
    num_obs = self.num_obs
    f_w = _robust_func[weight]

    # 2.allocate space
    Am   = np.empty( (num_obs,) + (self.dim_err, self.dim_dx) )
    Bm   = np.empty( (num_obs,) + (self.dim_err, self.dim_dl) )
    W_gg = np.empty( (num_obs,) + (self.dim_err, self.dim_err) )
    Nm   = np.empty( (self.dim_dx, self.dim_dx)    )
    nv   = np.empty( self.dim_dx )
    Cg   = np.empty( (num_obs, self.dim_err)  )
    X_gg = np.empty( num_obs )

    # 3.fill the big covarince matrix
    Cov_ll=np.zeros( (num_obs,) + (self.dim_dl, self.dim_dl) )
    for obs_id, obs_covs in enumerate(self.l_covs()):
      for grp_id, seg, cov in compress( zip(count(), self.slice_dl, obs_covs), self.l_notfixeds):
          Cov_ll[obs_id, seg, seg ] = cov

    is_converged = False
    # 4. solve
    for it in range(maxiter):
      Nm[:,:]= 0
      nv[:]  = 0
      res    = 0
      self._UpdateAllPrmJacobian()
      x_arrays = self.x_arrays()
      for obs_id, l_arrays, l_params, l_errs, dl_covs in zip(count(), self.l_arrays(), self.l_params(), self.l_errs(), self.l_covs()):
        # update covariance matrix
        if update_cov and is_converged:
          for seg, cov, param, dl in compress( zip(self.slice_dl, dl_covs, l_params, l_errs), self.l_notfixeds):
              Cov_ll[obs_id, seg, seg] = param.PlusCov(cov, dl)
          update_cov = False
          print("trying with updated cov")

        # update Jacobian and residual
        err, J = self.g(* (x_arrays + l_arrays) )
        vv = np.hstack( compress(l_errs, self.l_notfixeds) )
        res += np.linalg.norm(err)

        # apply parameterization Jacobian
        Am[obs_id] = A = np.hstack( param.ToLocalJacobian(J[:,seg])
                                      for param, seg in zip(self.x_params, self.slice_x) )
        Bm[obs_id] = B = np.hstack( param.ToLocalJacobian(J[:,seg])
                                      for param, seg in compress(zip(l_params, self.slice_l), self.l_notfixeds))
        # residual
        Cg[obs_id] = cg = B.dot(vv) - err

        # weights of constraints
        W_gg[obs_id] = np.linalg.pinv( B.dot(Cov_ll[obs_id]).dot(B.T) ) #
        # test statistic
        X_gg[obs_id] = np.sqrt( cg.T.dot(W_gg[obs_id]).dot(cg) )

      # robust weight function
      sigma_gg = np.median(X_gg)/0.6745
      w = f_w( X_gg/sigma_gg )

      # normal equation with modified weight
      for obs_id in range(num_obs):
        ATBWB = Am[obs_id].T.dot(w[obs_id]*W_gg[obs_id])
        Nm  += ATBWB.dot(Am[obs_id])
        nv  += ATBWB.dot(Cg[obs_id])

      # solve dx and update
      dx = np.linalg.solve(Nm, nv)
      self._Plus_dx(dx)

      # solve dl and update
      if not asGM:
        for obs_id, l_errs in enumerate(self.l_errs()):
          vv    = np.hstack( compress(l_errs, self.l_notfixeds) )
          lamba = W_gg[obs_id].dot( Am[obs_id].dot(dx) - Cg[obs_id] )
          dl = -Cov_ll[obs_id].dot( Bm[obs_id].T.dot(lamba) ) - vv
          self._Plus_dl(obs_id, dl)

      print( it, res)

      is_converged = np.abs(dx).max() < Tx
      if is_converged and not update_cov:
          break

    sigma_0 = sum([grp.Cost() for grp in self.l_groups]) / (num_obs * self.dim_err - self.dim_x)
    print("sigma_0: %f" % sigma_0)
    Cov_xx  = np.linalg.inv(Nm)
    return self.x, Cov_xx, sigma_0, w

def test_BatchGaussHelmertProblem():
  def g(x0, x1, l0, l1, l2):
    e0 =     x0 - l0
    e1 = 0.5*x1 - l1
    e2 = 0.5*x1 - l2
    Jx = np.vstack( [ np.c_[ np.eye(3),       np.zeros((3,3))],
                      np.c_[ np.zeros((3,3)), 0.5*np.eye(3)],
                      np.c_[ np.zeros((3,3)), 0.2*np.eye(3)],])
    Jl = -np.eye(9)
    return np.hstack([e0,e1,e2]), np.c_[Jx, Jl]

  num = 100
  problem = BatchGaussHelmertProblem(g,2,3)
  x0 = np.ones(3)
  cov0 = np.diag([0.2,0.1,0.2])
  l0 = np.random.multivariate_normal(x0, cov0, num)

  x1 = 2*np.ones(3)
  cov1 = np.diag([0.1,0.1,0.1])
  l1 = np.random.multivariate_normal(0.5*x1, cov1, num)
  cov2 = np.diag([0.2,0.1,0.3])
  l2 = np.random.multivariate_normal(0.2*x1, cov2, num)

  problem.SetParameter(0, x0, ConstantParameterization(3))#
  problem.SetParameter(1, x1)
  for i in range(num):
    problem.AddObservation(0, l0[i], cov0)
    problem.AddObservation(1, l1[i], cov1)
    problem.AddObservation(2, l2[i], cov2)
  problem.SetObservationGroupFixiation(2, True)
  x, Cov_xx, sigma_0, w = problem.Solve()

  assert_array_equal( [x0, x1], x )
  assert_array_almost_equal(l0[0], l0[1])
  assert_array_almost_equal(l1[0], l1[1])

  assert_array_equal( x0, np.ones(3) )
  assert_array_almost_equal(x0, l0[0])
  assert_array_almost_equal(0.5*x1, l1[0])

#%%
if __name__ == '__main__':
  test_ArrayID()
  test_CompoundVector()
  test_ProblemBasic()
  test_ProblemMakeKKT()
  test_FixAndParameterization()
  test_BatchGaussHelmertProblem()