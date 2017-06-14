#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 28 10:08:11 2017

@author: kaihong
"""
import numpy as np
import scipy
import scipy.sparse

from collections import namedtuple
np.set_printoptions(precision=4, linewidth=80)
from numpy.testing import *
from itertools import chain

from parameterization import *
import pycppad
from pykrylov.linop import *
from pykrylov.minres import Minres

#%% ColumnMajorSparseMatrix
class LexicalSortable(object):
  def key(self):
    raise NotImplementedError()

  def __lt__(self, other):
    return self.key() < other.key()

class MatrixTreeNode(LexicalSortable):
  def __init__(self, parent, r, c):
    self.parent = parent
    self.r = r
    self.c = c
    self.elements = []
    self.lexical_key = [(),()]
    self.absolute_pos = None

  def AddElement(self, e):
    e.parent = self
    id = len(self.elements)
    self.elements.append( e )
    return id

  def key(self):
    """ Provide key for sorting algorithm,
        in lexical order: (c_root,..,c_leaves), (r_root,..,r_leaves)
        this is exactly the order of column-major matrix
    """
    return self.lexical_key[1] + self.lexical_key[0]

  def __call__(self, type):
    """ Collect specific type node of the tree
        type : SparseBlock / DenseMatrix / DenseMatrixSegment
    """
    if isinstance(self, type):  # ignore all the other nodes except the specific type
        yield self
    for child in self.elements:
      if isinstance(child, type):  # ignore all the other nodes except the specific type
        yield child
      else:
        for c in child.__call__(type):
          yield c

  def PropogateAbsolutePos(self, pos=np.zeros(2,'i')):
    """ The abs_pos of root is always (0,0). The parent node is responsible for
        calculating their children's pos and the children simply accept it
    """
    self.absolute_pos = pos
    for e in self.elements:
      child_pos = self.absolute_pos + [e.r, e.c]
      e.PropogateAbsolutePos(child_pos)

  def PropogateKeyPrefix(self, prefix=[(),()] ): # empty prefix for root
    self.lexical_key[0] = prefix[0]
    self.lexical_key[1] = prefix[1]
    for e in self.elements:
      child_key = (self.lexical_key[0] + (e.r,), self.lexical_key[1] + (e.c,))
      e.PropogateKeyPrefix(child_key)

  def __repr__(self):
    return str(type(self)) + "at (%d, %d)" % self.absolute_pos

  def BuildSparseMatrix(self, shape=None, coo=False):
    self.PropogateKeyPrefix()

    """ 1. lexical sort of all the DenseMatrixSegment,(i.e leaves of the tree),
    the resulting order is exactly the linear data term is CSC matrix """
    vectors = sorted(self(DenseMatrixSegment))

    """ 2. assign segments to the compound data vector   """
    len_list = [vec.length for vec in vectors]
    offset = np.cumsum( [0] + len_list ).tolist()
    nnz = offset[-1]
    data = np.empty(nnz)
    for start, vec in zip(offset, vectors):
      vec.Mapto(data, start)  # vec.data = data[start : start + vec.length]
    # cache intermedia result
    self.data = data
    self.nnz  = nnz

    """ 3. Write cached data """
    for m in self(CachedMatrix):
      m.Flush()

    """ 4. Make row and col indices for COO matrix """
    self.PropogateAbsolutePos()
    row = np.hstack(vec.row_indice() for vec in vectors)
    col = np.hstack(vec.col_indice() for vec in vectors)
    self.row, self.col = row, col

    """ 5. convert to Compressed Sparse Column (CSC)
    * compressed(col) -> indices, where only the heads of each col are recorded,
    if there are all-zero columns in right (ie. shape[1] > col[-1]), we should
    repeat
    """
    if shape and shape[1]>col[-1]:
      rep = shape[1] - col[-1]
    else:
      rep = 1

    if 1:  # fast version
      indptr, = np.where( np.diff(col) )  # np.diff : out[n] = a[n+1]-a[n]
      indptr = np.r_[0, indptr+1, [nnz]*rep]
    else:  # slow version
      indptr = np.array([0] + [ i for i in xrange(1, nnz) if col[i] != col[i-1] ] + [nnz]*rep)

    self.indptr = indptr
    if coo:
      return scipy.sparse.coo_matrix( (data, (row, col) ), shape=shape)#, shape=self.shape
    else:
      return scipy.sparse.csc_matrix( (data, row, indptr), shape=shape)#, shape=self.shape

class CompoundMatrix(MatrixTreeNode):
  def __init__(self):
    super(CompoundMatrix, self).__init__(None, 0, 0)
    self.shape = None

  def NewSparseBlock(self, r, c):
    sb = SparseBlock(r, c)
    return self.AddElement(sb)

class SparseBlock(MatrixTreeNode):
  def __init__(self, r=0, c=0):
    super(SparseBlock, self).__init__(None, r, c)
    self.shape = None

  def NewDenseMatrix(self, r, c, shape):
    dm = DenseMatrix(r, c, shape)
    return self.AddElement(dm)

  def NewDiagonalMatrix(self, r, c, dim):
    dm = DiagonalMatrix(r, c, dim)
    return self.AddElement(dm)

  def PutDenseMatrix(self, id, src):
    self.elements[id].Write( src )

  def PutDenseMatrixInCache(self, id, src):
    self.elements[id].WriteCache( src )

  def OverwriteRC(self, r, c):
    self.r = r
    self.c = c

class CachedMatrix(object):
  def __init__(self):
    self.cache = None

  def Write(array):
    raise NotImplementedError()

  def WriteCache(self, array):
    self.cache = array.copy()

  def Flush(self):
    if self.cache is None:
      return

    self.Write(self.cache)

class DenseMatrix(MatrixTreeNode, CachedMatrix):
  def __init__(self, r, c, shape):
    super(DenseMatrix, self).__init__(None, r, c)
    CachedMatrix.__init__(self)

    self.dim_vec, self.num_vec = shape[:2]
    for seq in xrange(self.num_vec):
      self.AddElement(DenseMatrixSegment(self, 0, seq, self.dim_vec))

    self.buf = np.empty((self.dim_vec, self.num_vec), order='F').T

  def Write(self, array):
    if self.elements[0].data is None:
      raise RuntimeError("slice not assigned yet")
    for dst, src in zip(self.elements, array.T):
      dst.data[:] = src

  @property
  def data(self):
    if self.elements[0].data is None:
      return None
    for src, dst in zip(self.elements, self.buf):
      dst[:] = src.data[:]
    return self.buf.T

  def __repr__(self):
      return "DenseMatrix at (%d, %d): \n" % (self.r, self.c)  + str(self.data)

class DiagonalMatrix(MatrixTreeNode, CachedMatrix):
  def __init__(self, r, c, dim):
    super(DiagonalMatrix, self).__init__(None, r, c)
    CachedMatrix.__init__(self)

    self.dim = dim
    for seq in xrange(dim):
      self.AddElement(DenseMatrixSegment(self, seq, seq, 1))

    self.buf = np.empty(dim)

  def Write(self, array):
    if self.elements[0].data is None:
      raise RuntimeError("slice not assigned yet")
    for dst, src in zip(self.elements, array.tolist()):
      dst.data[:] = src

  @property
  def data(self):
    if self.elements[0].data is None:
      return None

    for i, src in enumerate(self.elements):
      self.buf[i] = src.data[:]
    return self.buf

  def __repr__(self):
      return "DiagonalMatrix at (%d, %d): \n" % (self.r, self.c)  + str(self.data)


class DenseMatrixSegment(MatrixTreeNode):
  def __init__(self, parent, r, seq, length):
    super(DenseMatrixSegment, self).__init__(parent, r, seq)
    self.length = length
    self.data = None

  def Mapto(self, vector, start):
#    if not self.data is None:
#      raise UserWarning("Segments being remapped")
    self.data = np.ndarray(shape  = (self.length,),
                           buffer = vector,
                           offset = start*vector.itemsize)

  def row_indice(self):
    r0 = self.absolute_pos[0]
    r1 = r0 + self.length
    return np.arange(r0, r1, dtype='i')

  def col_indice(self):
    return np.full(self.length, self.absolute_pos[1], dtype='i') # all the data share the same column

  def __repr__(self):
    return str(self.key()) + " len:" + str(self.length)

  def BuildSparseMatrix(self):
    raise RuntimeError("BuildSparseMatrix is not supported in DenseMatrixSegment level")

  def ComputeShape(self):
    return (self.length,)

def test_MatrixTreeNode():

  sb = SparseBlock()
  id1 = sb.NewDenseMatrix( 0, 0, (4,3) )
  id2 = sb.NewDenseMatrix( 0, 3, (2,3) )

  # iterator
  assert len(list(sb(SparseBlock))) == 1
  assert len(list(sb(DenseMatrix))) == 2
  assert len(list(sb(DenseMatrixSegment))) == 6

  # PropogateKeyPrefix
  sb.PropogateKeyPrefix()
  assert_equal([s.key() for s in sb(DenseMatrixSegment)],
               [(0, 0, 0, 0),
                (0, 1, 0, 0),
                (0, 2, 0, 0),
                (3, 0, 0, 0),
                (3, 1, 0, 0),
                (3, 2, 0, 0)] )

  # PropogateAbsolutePos
  sb.PropogateAbsolutePos()
  assert_equal(
    [s.absolute_pos for s in sb(DenseMatrixSegment)],
    [[0, 0], [0, 1], [0, 2], [0, 3], [0, 4], [0, 5]])

  sb.PropogateAbsolutePos(np.array([1,2]))
  assert_equal(
    [s.absolute_pos for s in sb(DenseMatrixSegment)],
    [[1, 2], [1, 3], [1, 4], [1, 5], [1, 6], [1, 7]])

  # BuildSparseMatrix
  sbm = sb.BuildSparseMatrix()
  sb.data[:] = np.arange(sb.nnz)
  assert_array_equal(sbm.A,
                    [[  0.,   4.,   8.,  12.,  14.,  16.],
                     [  1.,   5.,   9.,  13.,  15.,  17.],
                     [  2.,   6.,  10.,   0.,   0.,   0.],
                     [  3.,   7.,  11.,   0.,   0.,   0.]])

  # write
  dm = np.arange(6).reshape(2,3)
  sb.PutDenseMatrix(id2, dm)
  assert_array_equal(sbm.A[0:2,3:6], dm)

  # read
  dml = list(sb(DenseMatrix))
  assert_array_equal(dml[1].data, dm)

  # CompoundMatrix
  cm = CompoundMatrix()
  sb = SparseBlock()
  id1 = sb.NewDenseMatrix( 0, 0, (4,3) )
  id2 = sb.NewDenseMatrix( 0, 3, (2,3) )
  sb2 = SparseBlock(4, 6)
  id = sb2.NewDenseMatrix( 0, 0, (3,3) )

  cm.AddElement(sb)
  cm.AddElement(sb2)
  assert_equal( len(list(cm(SparseBlock))), 2)
  assert_equal( len(list(cm(DenseMatrix))), 3)
  assert_equal( len(list(cm(DenseMatrixSegment))), 9)

  cbm = cm.BuildSparseMatrix()
  cm.data[:] = np.arange(cm.nnz)
  assert_array_equal(cbm.A,
                    [[  0.,   4.,   8.,  12.,  14.,  16.,   0.,   0.,   0.],
                     [  1.,   5.,   9.,  13.,  15.,  17.,   0.,   0.,   0.],
                     [  2.,   6.,  10.,   0.,   0.,   0.,   0.,   0.,   0.],
                     [  3.,   7.,  11.,   0.,   0.,   0.,   0.,   0.,   0.],
                     [  0.,   0.,   0.,   0.,   0.,   0.,  18.,  21.,  24.],
                     [  0.,   0.,   0.,   0.,   0.,   0.,  19.,  22.,  25.],
                     [  0.,   0.,   0.,   0.,   0.,   0.,  20.,  23.,  26.]])

  # DiagonalMatrix
  sb = SparseBlock()
  id = sb.NewDiagonalMatrix(0,0,4)
  sbm = sb.BuildSparseMatrix()
  sb.PutDenseMatrix(id, np.arange(4))
  assert_array_equal(sbm.A, np.diag(np.arange(4)))

  # Cache
  sb = SparseBlock()
  id1 = sb.NewDenseMatrix(0,0,(2,2))
  id2 = sb.NewDiagonalMatrix(2,2,4)
  sb.PutDenseMatrixInCache(id1, np.full((2,2), -1))
  sb.PutDenseMatrixInCache(id2, np.full(4, 2))

  sbm = sb.BuildSparseMatrix()
  assert_array_equal(sbm.A,
                    [[-1., -1.,  0.,  0.,  0.,  0.],
                     [-1., -1.,  0.,  0.,  0.,  0.],
                     [ 0.,  0.,  2.,  0.,  0.,  0.],
                     [ 0.,  0.,  0.,  2.,  0.,  0.],
                     [ 0.,  0.,  0.,  0.,  2.,  0.],
                     [ 0.,  0.,  0.,  0.,  0.,  2.]])
  print "test_MatrixTreeNode passed"

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
  print "test_ArrayID passed"

class CompoundVector(object):
  def __init__(self, capacity=10000):
    self.buff = np.empty(capacity)
    self.tail = 0

  def MakeVector(self, dim):
    """ Make new segment on the tail """
    head = self.tail
    seg = np.ndarray(shape  = (dim,),
                     buffer = self.buff,
                     offset = head*8 )  # 8 for double size
    self.tail += dim
    return head, seg

  @property
  def flat(self):
    return np.ndarray((self.tail,), buffer=self.buff)

class CompoundVectorWithDict(CompoundVector):
  def __init__(self, cap = 10000):
    self.vec_dict = {}
    super(CompoundVectorWithDict, self).__init__(cap)

  def AddVector(self, vector):
    """ 1. check whether it is a newcomer, use address as hash"""
    key = ArrayID(vector)
    if key not in self.vec_dict:
      """ 2. Make new segment on the tail """
      new_item = self.MakeVector(len(vector))
      new_item[1][:] = vector      # copy inital value
      self.vec_dict[key] = new_item
      return new_item
    else:
      return self.vec_dict[key]

  def FindVector(self, vector):
    key = ArrayID(vector)
    return  self.vec_dict.get(key, (None,None))

  def OverWriteOrigin(self):
    for dst, (_, src) in self.vec_dict.iteritems():
      dst.data[:] = src

  @property
  def flat(self):
    return np.ndarray((self.tail,), buffer=self.buff)

def test_CompoundVector():
  vs = np.random.rand(4,4)
  # main function
  cv = CompoundVectorWithDict()
  offset, seg = zip(*[cv.AddVector(v) for v in vs])
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
  offset2, _ = zip(*[cv.AddVector(v) for v in vs])
  assert_equal(offset, offset2)
  offset3, _ = cv.AddVector(  np.empty(1)  )
  assert offset3 == 16
  print "test_CompoundVector passed"


#%% GaussHelmertProblem

class GaussHelmertProblem(object):
  ConstraintBlock = namedtuple('ConstraintBlock', ['offset', 'g_res', 'g_jac'])
  VariableBlock   = namedtuple('VariableBlock'   , ['seg', 'seg_param', 'param'])
  ObservationBlock= namedtuple('ObservationBlock', ['seg', 'seg_param', 'param', 'mat_w_id'])

  def __init__(self):
    self.cv_x   = CompoundVectorWithDict()
    self.cv_l   = CompoundVectorWithDict()
    self.cv_res = CompoundVector()

    self.mat_kkt   = CompoundMatrix()
    self.mat_w     = SparseBlock(0,1)
    self.mat_jac_x = SparseBlock(1,0)
    self.mat_jac_l = SparseBlock(1,1)

    self.mat_kkt.AddElement(self.mat_w)
    self.mat_kkt.AddElement(self.mat_jac_x)
    self.mat_kkt.AddElement(self.mat_jac_l)

    self.Jx, self.Jl, self.W = None,None, None

    self.constraint_blocks = []    # list of ConstraintBlock
    self.variance_factor = -1.0

    self.dict_variable_block = {}
    self.dict_observation_block = {}

  def AddParameter(self, x_list):
    x_off, x_vec = [],[]
    for x in x_list:
      offset, seg = self.cv_x.AddVector(x)
      x_off.append(offset)
      x_vec.append(seg)

      if not offset in self.dict_variable_block:
        item = GaussHelmertProblem.VariableBlock(seg, None, None )
        self.dict_variable_block[offset] = item
    return x_off, x_vec

  def AddObservation(self, l_list):
    l_off, l_vec = [],[]
    for l in l_list:
      offset, seg = self.cv_l.AddVector(l)
      l_off.append(offset)
      l_vec.append(seg)

      if not offset in self.dict_observation_block:
        dim_l = len(l)
        mat_w_id = self.mat_w.NewDiagonalMatrix(offset, offset, dim_l)
        self.mat_w.PutDenseMatrixInCache(mat_w_id, np.ones(dim_l))

        item = GaussHelmertProblem.ObservationBlock(seg, None, None, mat_w_id )
        self.dict_observation_block[offset] = item
    return l_off, l_vec

  def AddConstraintUsingAD(self, g, x_list, l_list):
    x_sizes = [x.size for x in x_list]
    l_sizes = [l.size for l in l_list]
    xl_indices = np.cumsum(x_sizes + l_sizes)[:-1]

    """ 1. Generate Jacobian function by cppad """
    var       = np.hstack(x_list+l_list )
    var_ad    = pycppad.independent( var )
    var_jacobian= pycppad.adfun(var_ad, g( *np.split(var_ad, xl_indices) ) ).jacobian

    res = g( *(x_list + l_list) )
    jac = var_jacobian(var)
    if not ( np.isfinite(res).all() and  np.isfinite(jac).all() ):
      RuntimeWarning("AutoDiff Not valid")
      return
    dim_res = len(res)

    """ 2. Assign poses and mapped vectors for input parameter/observation arrays"""
    x_off, x_vec = self.AddParameter(x_list)
    l_off, l_vec = self.AddObservation(l_list)
    xl_vec = x_vec + l_vec

    """ 3. Compound vector for constraint residual """
    res_off, res_vec = self.cv_res.MakeVector(dim_res)

    jac_submat_id_x = [ self.mat_jac_x.NewDenseMatrix(res_off, c, (dim_res, dim_x) ) for c, dim_x in zip(x_off, x_sizes) ]
    jac_submat_id_l = [ self.mat_jac_l.NewDenseMatrix(res_off, c, (dim_res, dim_l) ) for c, dim_l in zip(l_off, l_sizes) ]

    """ 4. Generate constraint functor that use the mapped vectors """
    def g_res():
      res_vec[:] = g(*xl_vec)

    def g_jac():
      J = var_jacobian( np.hstack(xl_vec) )
      jac = np.split(J, xl_indices, axis=1)
      jac.reverse() # reversed, to pop(-1) instead of pop(0)
      for submat_id in jac_submat_id_x:
        self.mat_jac_x.PutDenseMatrix( submat_id, jac.pop() )

      for submat_id in jac_submat_id_l:
        self.mat_jac_l.PutDenseMatrix( submat_id, jac.pop() )

    self.constraint_blocks.append( GaussHelmertProblem.ConstraintBlock(res_off, g_res, g_jac) )

  def AddConstraintWithKnownBlocks(self, g, x_off, l_off):
    x_vec,l_vec,x_sizes,l_sizes = [],[],[],[]
    for x in x_off:
      block = self.dict_variable_block.get(x, None)
      if block is None:
        raise RuntimeError("wrong id")
      x_vec.append(block.seg)
      x_sizes.append(len(block.seg))

    for l in l_off:
      block = self.dict_observation_block.get(l, None)
      if block is None:
        raise RuntimeError("wrong id")
      l_vec.append(block.seg)
      l_sizes.append(len(block.seg))

    xl_vec = x_vec + l_vec

    xl_indices = np.cumsum(x_sizes + l_sizes)[:-1]
    """ 1. Generate Jacobian function by cppad """
    var       = np.hstack( xl_vec )
    var_ad    = pycppad.independent( var )
    var_jacobian= pycppad.adfun(var_ad, g( *np.split(var_ad, xl_indices) ) ).jacobian

    res = g( *xl_vec )
    jac = var_jacobian(var)
    if not ( np.isfinite(res).all() and  np.isfinite(jac).all() ):
      RuntimeWarning("AutoDiff Not valid")
      return
    dim_res = len(res)

    """ 3. Compound vector for constraint residual """
    res_off, res_vec = self.cv_res.MakeVector(dim_res)

    jac_submat_id_x = [ self.mat_jac_x.NewDenseMatrix(res_off, c, (dim_res, dim_x) ) for c, dim_x in zip(x_off, x_sizes) ]
    jac_submat_id_l = [ self.mat_jac_l.NewDenseMatrix(res_off, c, (dim_res, dim_l) ) for c, dim_l in zip(l_off, l_sizes) ]

    """ 4. Generate constraint functor that use the mapped vectors """
    def g_res():
      res_vec[:] = g(*xl_vec)

    def g_jac():
      J = var_jacobian( np.hstack(xl_vec) )
      jac = np.split(J, xl_indices, axis=1)
      jac.reverse() # reversed, to pop(-1) instead of pop(0)
      for submat_id in jac_submat_id_x:
        self.mat_jac_x.PutDenseMatrix( submat_id, jac.pop() )

      for submat_id in jac_submat_id_l:
        self.mat_jac_l.PutDenseMatrix( submat_id, jac.pop() )

    self.constraint_blocks.append( GaussHelmertProblem.ConstraintBlock(res_off, g_res, g_jac) )

  def SetSigma(self, array, sigma, isInv=False):
    if not isinstance(array, list):
      array = [array]
    if not isinstance(sigma, list):
      sigma = [sigma]*len(array)
    elif len(sigma) != array:
      raise ValueError("number don't match")

    for ar, si in zip(array, sigma):
      if np.isscalar(ar):
        l_off = ar
      else:
        l_off, _ = self.cv_l.FindVector(ar)
      if not l_off is None:
        w = si if isInv else 1./si
        self.mat_w.PutDenseMatrixInCache( self.dict_observation_block[l_off].mat_w_id, w)

  def SetParameterization(self, array, parameterization):
    var = VariableBlock(array)
    if var in self.parameter_dict:
      self.parameter_dict[var].param = parameterization
    elif var in self.observation_dict:
      self.observation_dict[var].param = parameterization
    else:
      raise RuntimeWarning("Input variable not in the lists")

  def CompoundParameters(self):
    return self.cv_x.flat

  def CompoundObservation(self):
    return self.cv_l.flat

  def CompoundResidual(self):
    return self.cv_res.flat

  @property
  def dim_x(self):
    return self.cv_x.tail

  @property
  def dim_l(self):
    return self.cv_l.tail

  @property
  def dim_res(self):
    return self.cv_res.tail

  @property
  def dims(self):
    return [self.dim_x, self.dim_l, self.dim_res ]

  def MakeJacobians(self):
    self.Jx = self.mat_jac_x.BuildSparseMatrix()
    self.Jl = self.mat_jac_l.BuildSparseMatrix()

    return self.Jx, self.Jl

  def MakeWeightMatrix(self):
    self.W = self.mat_w.BuildSparseMatrix()
    return self.W

  def MakeKKTMatrix(self):
    dim_x, dim_l, dim_res = self.dim_x, self.dim_l, self.dim_res
    dim_total = dim_x + dim_l + dim_res

    self.mat_w.OverwriteRC(dim_x, dim_x)
    self.mat_jac_x.OverwriteRC(dim_x+dim_l, 0)
    self.mat_jac_l.OverwriteRC(dim_x+dim_l, dim_x)

    kkt = self.mat_kkt.BuildSparseMatrix((dim_total,dim_total))
    return kkt

  def MakeKKTSegmentSlice(self):
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
    for cb in self.constraint_blocks:
      cb.g_jac()

    if ouput:
      return self.Jx, self.Jl

#%%
def EqualityConstraint(a,b):
  return a-b
def EqualityConstraintJac(a,b):
  return np.eye(len(a)), -np.eye(len(b))

def MakeAffineConstraint(A,B):
  def AffineConstraint(a, b):
    return A.dot(a) + B.dot(b)
  return AffineConstraint

def DiagonalRepeat(M, repeats):
  return scipy.linalg.block_diag(* (M,)*repeats )

def VerticalRepeat(M, repeats):
  return np.tile( M, (repeats,1) )

def HorizontalRepeat(M, repeats):
  return np.tile( M, (1, repeats) )

def test_ProblemJacobian():
  dim_x, num_x = 3, 2
  dim_l, num_l = 4, 10*num_x

  dim_g = 3
  A = np.random.rand(dim_g, dim_x)
  B = np.random.rand(dim_g, dim_l)
  AffineConstraint = MakeAffineConstraint(A,B)

  x = np.empty((num_x, dim_x))
  l = [ np.empty((num_l/num_x, dim_l)) for _ in range(num_x) ] # l[which_x] = vstack(l[which_l])
  sigma = np.full(dim_l, 0.5)
  problem = GaussHelmertProblem()
  for i in range(num_x):
    for j in range(num_l/num_x):
      problem.AddConstraintUsingAD(AffineConstraint,
                                   [ x[i] ],
                                   [ l[i][j] ])
      problem.SetSigma(l[i][j], sigma)

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
  problem.cv_x.OverWriteOrigin()
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
  W = problem.MakeWeightMatrix()
  assert_array_equal( W.todense(), DiagonalRepeat(np.diag(1./sigma), num_l) )

  # 2 method
  problem = GaussHelmertProblem()
  for i in range(num_x):
    for j in range(num_l/num_x):
      x_off, _ = problem.AddParameter( [ x[i] ] )
      l_off, _ = problem.AddObservation( [ l[i][j] ] )

      problem.AddConstraintWithKnownBlocks(AffineConstraint, x_off, l_off)
      problem.SetSigma(l_off, sigma)

  print "test_ProblemJacobian passed"


#%%
if __name__ == '__main__':

  test_MatrixTreeNode()
  test_ArrayID()
  test_CompoundVector()
  test_ProblemJacobian()


  dim_x, num_x = 3, 1
  dim_l, num_l = 4, 3*num_x

  dim_g = 3
  A = np.random.rand(dim_g, dim_x)
  B = np.random.rand(dim_g, dim_l)
  AffineConstraint = MakeAffineConstraint(A,B)

  x = np.ones((num_x, dim_x))
  l = [ np.zeros((num_l/num_x, dim_l)) for _ in range(num_x) ] # l[which_x] = vstack(l[which_l])

  problem = GaussHelmertProblem()
  for i in range(num_x):
    for j in range(num_l/num_x):
      problem.AddConstraintUsingAD(AffineConstraint,
                                   [ x[i] ],
                                   [ l[i][j] ])
  dims = problem.dims
  total_dim = sum(dims)
  res  = problem.CompoundResidual()
  xc   = problem.CompoundParameters()
  lc   = problem.CompoundObservation()
  l0   = lc.copy()
  W    = problem.MakeWeightMatrix()
  kkt  = problem.MakeKKTMatrix()
  segment_x, segment_l, segment_r = problem.MakeKKTSegmentSlice()
#  a = kkt.A
  problem.UpdateJacobian()
#  op = CholeskyOperator(kkt)
  op = CoordLinearOperator(problem.mat_kkt.data,
                           problem.mat_kkt.row,
                           problem.mat_kkt.col,
                           total_dim, total_dim,
                           symmetric=True)
  b = np.zeros(sum(dims))
  solver = Minres(op)
  for it in range(1):
    problem.UpdateJacobian()
    problem.UpdateResidual()
#    op.UpdataFactor(kkt)

    e = lc - l0
    b[segment_l] = -e
    b[segment_r] = -res
    solver.solve(b)
    s = solver.bestSolution

    xc += s[segment_x]
    lc += s[segment_l]
    print np.linalg.norm(res), np.linalg.norm(e)


