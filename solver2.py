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
np.set_printoptions(precision=4, linewidth=120)
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
    self.absolute_pos = np.r_[r,c]

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
    return str(type(self)) + "at (%d, %d)" % (self.absolute_pos[0],self.absolute_pos[1])

  def BuildSparseMatrix(self, shape=None, coo=True):
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
    for m in self(DenseMatrix):
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

  def OverwriteRC(self, r, c):
    self.r = r
    self.c = c


class DenseMatrix(MatrixTreeNode):
  def __init__(self, r, c, shape):
    super(DenseMatrix, self).__init__(None, r, c)

    self.cache = None
    self.post_callback = []
    self.shape = shape
    self._init_element()

  def _init_element(self):
    for seq in xrange(self.shape[1]):
      self.AddElement(DenseMatrixSegment(self, 0, seq, self.shape[0]))

    self.buf = np.empty(self.shape, order='F').T

  def Write(self, array):
    if self.elements[0].data is None:
      self.cache = array.copy()
    else:
      for dst, src in zip(self.elements, array.T):
        dst.data[:] = src
    for cb in self.post_callback:
      cb(self, array)

  def Flush(self):
    if self.cache is None:
      return
    self.Write(self.cache)

  @property
  def data(self):
    if self.elements[0].data is None:
      return None
    for src, dst in zip(self.elements, self.buf):
      dst[:] = src.data[:]
    return self.buf.T

  def ComputeShape(self):
    return self.shape

  def __repr__(self):
      return "DenseMatrix(%d, %d)" % (self.r, self.c)

class DiagonalMatrix(DenseMatrix):  # inherit from DenseMatrix, so that the gathering mechansim will see it the same as normal DenseMatrix
  def __init__(self, r, c, dim):
    super(DiagonalMatrix, self).__init__(r, c, (dim, dim))

  def _init_element(self):
    for seq in xrange(self.shape[0]):
      self.AddElement(DenseMatrixSegment(self, seq, seq, 1))
    self.buf = np.empty((self.shape[0],1))

  def Write(self, array):
    super(DiagonalMatrix, self).Write(array.reshape(1, -1))

  def __repr__(self):
      return "DiagonalMatrix(%d, %d)" % (self.r, self.c)



class DenseMatrixSegment(MatrixTreeNode):
  def __init__(self, parent, r, seq, length):
    super(DenseMatrixSegment, self).__init__(parent, r, seq)
    self.length = length
    self.data = None

  def Mapto(self, vector, start):
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
  sb.PutDenseMatrix(id1, np.full((2,2), -1))
  sb.PutDenseMatrix(id2, np.full(4, 2))

  sbm = sb.BuildSparseMatrix()
  assert_array_equal(sbm.A,
                    [[-1., -1.,  0.,  0.,  0.,  0.],
                     [-1., -1.,  0.,  0.,  0.,  0.],
                     [ 0.,  0.,  2.,  0.,  0.,  0.],
                     [ 0.,  0.,  0.,  2.,  0.,  0.],
                     [ 0.,  0.,  0.,  0.,  2.,  0.],
                     [ 0.,  0.,  0.,  0.,  0.,  2.]])
  print "test_MatrixTreeNode passed"

#%%
def _make_tranpose_callback(dst_block):
  def WriteTranspose(obj, array):
    dst_block.Write(array.T)
  return WriteTranspose

def MakeSymmetric(other, r=0, c=0):
  """ 1. collect dense matrix"""
  other.PropogateAbsolutePos()
  dmat = list(other(DenseMatrix))

  for dm in dmat:
    r,c = dm.absolute_pos[0], dm.absolute_pos[1]
    if r!=c: # only mirror off diagonal matrix
      dm_t = DenseMatrix(c, r, tuple(reversed(dm.shape)))
      dm.post_callback.append( _make_tranpose_callback(dm_t) )
      other.AddElement(dm_t)
    else:
      assert dm.shape[0] == dm.shape[1]
  return other

def test_MakeSymmetric():
  # main function
  sb = SparseBlock()
  id1 = sb.NewDenseMatrix( 0, 0, (2,2) )
  id2 = sb.NewDenseMatrix( 2, 0, (1,2) )

  MakeSymmetric(sb)
  sb.PutDenseMatrix(id1, np.full((2,2), 1.) )
  sb.PutDenseMatrix(id2, np.full((1,2), 2.) )
  sp = sb.BuildSparseMatrix()
  assert_array_equal(sp.A,
                    [[ 1.,  1.,  2.],
                     [ 1.,  1.,  2.],
                     [ 2.,  2.,  0.]])
  # auto update
  sb.PutDenseMatrix(id2, np.full((1,2), 0.) )
  assert_array_equal(sp.A,
                    [[ 1.,  1.,  0.],
                     [ 1.,  1.,  0.],
                     [ 0.,  0.,  0.]])
#%%
from collections import defaultdict,OrderedDict

def _make_AB_callback(dst_block, dst_block_shape, src_dm_a, src_dm_b):
  op_a, op_b = OrderedDict(), OrderedDict()
  for dm_a_, dm_b_ in zip(src_dm_a, src_dm_b):
    op_a[dm_a_] = None
    op_b[dm_b_] = None

  def CalculateSumOfAB(obj, array):
    # 1. record the incoming data
    if obj in op_a:
      op_a[obj] = array
    elif obj in op_b:
      op_b[obj] = array
    else:
      raise RuntimeError("called by wrong object")
    # 2. do calculation once all data are ready
    if not np.any( [op is None for op in op_a.values()+op_b.values()] ):
      new_data = np.zeros(dst_block_shape)
      for mat_a, mat_b in zip(op_a.values(), op_b.values()):
        new_data += mat_a.dot(mat_b)
      dst_block.Write( new_data )
      # 3. reset the dict to all None
#      for op in [op_a, op_b]:
#        for key in op.keys():
#          op[key] = None
  return CalculateSumOfAB

def MakeAB(A, B, r=0, c=0):
  AB = SparseBlock(r, c)

  """ 1. collect dense matrix"""
  row_mat = defaultdict(list)
  A.PropogateAbsolutePos()
  for dm in A(DenseMatrix):
    r = dm.absolute_pos[0]
    row_mat[r].append(dm)
  keys_r = sorted(row_mat.keys())

  col_mat = defaultdict(list)
  B.PropogateAbsolutePos()
  for dm in B(DenseMatrix):
    c = dm.absolute_pos[1]
    col_mat[c].append(dm)
  keys_c = sorted(col_mat.keys())

  # make a dict for each, to easily reference each matrices by their columns number
  row_dict = [ { dm.absolute_pos[1] : dm for dm in row_mat[r] } for r in keys_r ]
  col_dict = [ { dm.absolute_pos[0] : dm for dm in col_mat[c] } for c in keys_c ]

  # for each row, find the common columns that they share with each other
  for a, mat_ra in enumerate(row_dict):
    ra = keys_r[a]
    set_mat_ra = set(mat_ra.keys())

    for b, mat_cb in enumerate(col_dict):
      cb = keys_c[b]

      comm_indices = set_mat_ra & set(mat_cb.keys())
      if len(comm_indices) == 0: # empty
        continue

      dm_a = [mat_ra[i] for i in comm_indices]
      dm_b = [mat_cb[i] for i in comm_indices]

      # then there will a new block at (ra,cb) in the result A*B matrix
      new_dm_shape = (dm_a[0].shape[0], dm_b[0].shape[1])
      new_dm = DenseMatrix(ra, cb, new_dm_shape)
      AB.AddElement(new_dm)

      callback = _make_AB_callback(new_dm, new_dm_shape, dm_a, dm_b)

      for dm in dm_a+dm_b:
        dm.post_callback.append(callback)
  return AB

def test_MakeAB():
  A = SparseBlock()
  id1 = A.NewDenseMatrix( 0, 0, (3,1) )
  id2 = A.NewDenseMatrix( 0, 1, (3,2) )
  id3 = A.NewDenseMatrix( 0, 3, (3,3) )
  sp_a = A.BuildSparseMatrix()

  W = SparseBlock()
  id1w = W.NewDenseMatrix( 0, 0, (1,1) )
  id2w = W.NewDenseMatrix( 1, 1, (2,2) )
  id3w = W.NewDenseMatrix( 3, 3, (3,3) )
  sp_w = W.BuildSparseMatrix()
  sp_w.A

  AW = MakeAB(A,W)
  A.PutDenseMatrix(id1, np.full((3,1), 1.) )
  A.PutDenseMatrix(id2, np.full((3,2), 2.) )
  A.PutDenseMatrix(id3, np.full((3,3), 3.) )
  W.PutDenseMatrix(id1w, np.full((1,1), 3.) )
  W.PutDenseMatrix(id2w, np.full((2,2), 2.) )
  W.PutDenseMatrix(id3w, np.full((3,3), 1.) )

  sp_aw = AW.BuildSparseMatrix()
  sp_aw.A
  # = (sp_a*sp_w).A,
  assert_array_equal(sp_aw.A,
                    [[ 3.,  8.,  8.,  9.,  9.,  9.],
                     [ 3.,  8.,  8.,  9.,  9.,  9.],
                     [ 3.,  8.,  8.,  9.,  9.,  9.]])

  # auto update
  A.PutDenseMatrix(id2, np.full((3,2), 4.) )
  assert_array_equal(sp_aw.A,
                    [[  3.,  16.,  16.,   9.,   9.,   9.],
                     [  3.,  16.,  16.,   9.,   9.,   9.],
                     [  3.,  16.,  16.,   9.,   9.,   9.]])
  print "test_MakeAB passed "

def _make_AWAT_callback(dst_block, dst_block_shape, src_dm_a, src_dm_w):
  op_a, op_w = OrderedDict(), OrderedDict()
  for dm_a_, dm_w_ in zip(src_dm_a, src_dm_w):
    op_a[dm_a_] = None
    op_w[dm_w_] = None

  def CalculateSumOfAWAT(obj, array):
    # 1. record the incoming data
    if obj in op_a:
      op_a[obj] = array.copy()
    elif obj in op_w:
      op_w[obj] = array.copy()
    else:
      raise RuntimeError("called by wrong object")
    # 2. do calculation once all data are ready
    if not np.any( [op is None for op in chain(op_a.values(),op_w.values())] ):
      print "All element collected"
      new_data = np.zeros(dst_block_shape)
      for mat_a, mat_w in zip(op_a.values(), op_w.values()):
        new_data += mat_a.dot(mat_w).dot(mat_a.T)
      dst_block.Write( new_data )
      # 3. reset the dict to all None, except op_w which is usually constant
      for key in op_a.keys():
        op_a[key] = None
  return CalculateSumOfAWAT

def _make_AWBT_callback(dst_block, dst_block_shape, src_dm_a, src_dm_b, src_dm_w):
  op_a, op_b, op_w = OrderedDict(), OrderedDict(), OrderedDict()
  for dm_a_, dm_b_, dm_w_ in zip(src_dm_a, src_dm_b, src_dm_w):
    op_a[dm_a_] = None
    op_b[dm_b_] = None
    op_w[dm_w_] = None

  def CalculateSumOfAWBT(obj, array):
    # 1. record the incoming data
    if obj in op_a:
      op_a[obj] = array.copy()
    elif obj in op_b:
      op_b[obj] = array.copy()
    elif obj in op_w:
      op_w[obj] = array.copy()
    else:
      raise RuntimeError("called by wrong object")
    # 2. do calculation once all data are ready
    if not np.any( [op is None for op in chain(op_a.values(),op_b.values(),op_w.values())] ):
      print "All element collected"
      new_data = np.zeros(dst_block_shape)
      for mat_a, mat_b, mat_w in zip(op_a.values(), op_b.values(),op_w.values()):
        new_data += mat_a.dot(mat_w).dot(mat_b.T)
      dst_block.Write( new_data )
      # 3. reset the dict to all None
      for op in [op_a, op_b]:
        for key in op.keys():
          op[key] = None
  return CalculateSumOfAWBT

def MakeAWAT(A, W, make_full=True, r=0, c=0):
  ret = SparseBlock(r, c)

  """ 1. collect dense matrix and sperate mat in different row"""
  A.PropogateAbsolutePos()
  row_mat = defaultdict(list)
  for dm in list(A(DenseMatrix)):
    r = dm.absolute_pos[0]
    row_mat[r].append(dm)
  keys_r = sorted(row_mat.keys())

  W.PropogateAbsolutePos()
  w_mat = {}
  for dm in list(W(DenseMatrix)):
    r,c = dm.absolute_pos
    if r==c:
      w_mat[r] = dm
  keys_w = sorted(w_mat.keys())

  # make a dict for each, to easily reference each matrices by their columns number
  row_dict = [ { dm.absolute_pos[1] : dm for dm in row_mat[r] } for r in keys_r  ]
  # for each row, find the common columns that they share with each other
  for a, mat_ra in enumerate(row_dict):
    ra = keys_r[a]
    set_mat_ra = set(mat_ra.keys())
    set_mat_w  = set(keys_w)
    comm_c_indices = set_mat_ra & set_mat_w
    if len(comm_c_indices) != 0: # empty
      dm_a = [mat_ra[c] for c in comm_c_indices]
      dm_w = [w_mat[c]  for c in comm_c_indices]
      # then there will a new block at (ra,rb) in the result MM' matrix
      new_dm_shape = (dm_a[0].shape[0],)*2
      new_dm = DenseMatrix(ra, ra, new_dm_shape)
      ret.AddElement(new_dm)

      callback = _make_AWAT_callback(new_dm, new_dm_shape, dm_a, dm_w)
      for dm in chain(dm_a,dm_w):
        dm.post_callback.append(callback)

    # for the row below
    for b, mat_rb in enumerate(row_dict[a+1:]):
      rb = keys_r[a+1 + b]

      comm_c_indices = set_mat_ra & set(mat_rb.keys()) & set_mat_w
      if len(comm_c_indices) == 0: # empty
        continue

      dm_a = [mat_ra[c] for c in comm_c_indices]
      dm_b = [mat_rb[c] for c in comm_c_indices]
      dm_w = [w_mat[c]  for c in comm_c_indices]
      # then there will a new block at (ra,rb) in the result MM' matrix
      new_dm_shape = (dm_a[0].shape[0], dm_b[0].shape[0])
      new_dm = DenseMatrix(ra, rb, new_dm_shape)
      ret.AddElement(new_dm)

      callback = _make_AWBT_callback(new_dm, new_dm_shape, dm_a, dm_b, dm_w)

      for dm in chain(dm_a,dm_b,dm_w):
        dm.post_callback.append(callback)

  return MakeSymmetric(ret) if make_full else ret

def test_MakeAWAT():
  A = SparseBlock()
  id1 = A.NewDenseMatrix( 0, 0, (3,1) )
  id2 = A.NewDenseMatrix( 0, 1, (3,2) )
  id3 = A.NewDenseMatrix( 0, 3, (3,3) )
  sp_a = A.BuildSparseMatrix()
  sp_a.A
  W = SparseBlock()
  id1w = W.NewDenseMatrix( 0, 0, (1,1) )
  id2w = W.NewDenseMatrix( 1, 1, (2,2) )
  id3w = W.NewDenseMatrix( 3, 3, (3,3) )
  sp_w = W.BuildSparseMatrix()
  sp_w.A

  aat = MakeAWAT(A,W)
  sp_aat = aat.BuildSparseMatrix()
  assert sp_aat.shape == (3,3)

  A.PutDenseMatrix(id1, np.full((3,1), 1.) )
  A.PutDenseMatrix(id2, np.full((3,2), 2.) )
  A.PutDenseMatrix(id3, np.full((3,3), 3.) )
  W.PutDenseMatrix(id1w, np.full((1,1), 1.) )
  W.PutDenseMatrix(id2w, np.full((2,2), 1.) )
  W.PutDenseMatrix(id3w, np.full((3,3), 1.) )

  # =(sp_a*sp_w*sp_a.T).A,
  assert_array_equal(sp_aat.A,
                    [[ 98.,  98.,  98.],
                     [ 98.,  98.,  98.],
                     [ 98.,  98.,  98.]])
  # auto update
  A.PutDenseMatrix(id1, np.full((3,1), 1.) )
  A.PutDenseMatrix(id2, np.full((3,2), 4.) )
  A.PutDenseMatrix(id3, np.full((3,3), 3.) )
  assert_array_equal(sp_aat.A,
                    [[ 146.,  146.,  146.],
                     [ 146.,  146.,  146.],
                     [ 146.,  146.,  146.]])
  print "test_MakeAWAT passed"


def _make_inv_callback(dst_block):
  def WriteInv(obj, array):
    dst_block.Write(np.linalg.inv(array))
  return WriteInv

def MakeBlockInv(other):
  ret  = SparseBlock()
  other.PropogateAbsolutePos()
  for dm in other(DenseMatrix):
    r,c = dm.absolute_pos[0], dm.absolute_pos[1]

    dm_inv = DenseMatrix(r, c, dm.shape)
    dm.post_callback.append( _make_inv_callback(dm_inv) )
    ret.AddElement(dm_inv)
  return ret

def test_MakeBlockInv():
  d = SparseBlock()
  id1 = d.NewDenseMatrix(0,0,(3,3))
  id2 = d.NewDenseMatrix(3,3,(3,3))
  sp_d = d.BuildSparseMatrix()

  d_inv = MakeBlockInv(d)
  d.PutDenseMatrix(id1, 0.5*np.eye(3))
  d.PutDenseMatrix(id2, 2*np.eye(3))
  sp_dinv = d_inv.BuildSparseMatrix()

  assert_array_equal(sp_dinv.A,
                    [[ 2. ,  0. ,  0. ,  0. ,  0. ,  0. ],
                     [ 0. ,  2. ,  0. ,  0. ,  0. ,  0. ],
                     [ 0. ,  0. ,  2. ,  0. ,  0. ,  0. ],
                     [ 0. ,  0. ,  0. ,  0.5,  0. ,  0. ],
                     [ 0. ,  0. ,  0. ,  0. ,  0.5,  0. ],
                     [ 0. ,  0. ,  0. ,  0. ,  0. ,  0.5]])




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
  def __init__(self, capacity=100000):
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
  def __init__(self, cap = 100000):
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
  ObservationBlock= namedtuple('ObservationBlock', ['seg', 'seg_param', 'param', 'mat_sig_id'])

  def __init__(self):
    self.cv_x   = CompoundVectorWithDict()
    self.cv_l   = CompoundVectorWithDict()
    self.cv_res = CompoundVector()

    self.mat_sigma     = SparseBlock()
    self.mat_jac_x = SparseBlock()
    self.mat_jac_l = SparseBlock()

    self.Jx, self.Jl, self.Sigma, self.W = None,None,None,None

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
        mat_sig_id = self.mat_sigma.NewDenseMatrix(offset, offset, (dim_l,)*2)
        self.mat_sigma.PutDenseMatrix(mat_sig_id, np.eye(dim_l))

        item = GaussHelmertProblem.ObservationBlock(seg, None, None, mat_sig_id )
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
      raise RuntimeWarning("AutoDiff Not valid")
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
      raise RuntimeWarning("AutoDiff Not valid")
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

  def SetSigma(self, array, sigma):
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
        self.mat_sigma.PutDenseMatrix( self.dict_observation_block[l_off].mat_sig_id, np.diag(si))


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

  def MakeReducedKKTMatrix(self):
    dim_x, dim_res = self.dim_x,  self.dim_res
    dim_total = dim_x + dim_res

    self.mat_jac_l.OverwriteRC(0, 0)
    self.mat_sigma.OverwriteRC(0, 0)
    self.mat_jac_x.OverwriteRC(0, dim_res)
    mat_BSBT = MakeAWAT(self.mat_jac_l, self.mat_sigma, make_full=False)
    mat_kkt   = CompoundMatrix()
    mat_kkt.AddElement(mat_BSBT)
    mat_kkt.AddElement(self.mat_jac_x)
#
    return MakeSymmetric(mat_kkt).BuildSparseMatrix((dim_total,dim_total))

  def MakeSigmaAndWeightMatrix(self):
    self.mat_w = MakeBlockInv(self.mat_sigma)
    self.Sigma = self.mat_sigma.BuildSparseMatrix()

    self.W     = self.mat_w.BuildSparseMatrix()
    return self.Sigma, self.W

  def MakeLargeKKTMatrix(self):
    dim_x, dim_l, dim_res = self.dim_x, self.dim_l, self.dim_res
    dim_total = dim_x + dim_l + dim_res

    self.mat_sigma.OverwriteRC(dim_x, dim_x)
    self.mat_jac_x.OverwriteRC(dim_x+dim_l, 0)
    self.mat_jac_l.OverwriteRC(dim_x+dim_l, dim_x)

    mat_kkt   = CompoundMatrix()
    mat_kkt.AddElement(self.mat_sigma)
    mat_kkt.AddElement(self.mat_jac_x)
    mat_kkt.AddElement(self.mat_jac_l)

    kkt = mat_kkt.BuildSparseMatrix((dim_total,dim_total))
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
    for cb in self.constraint_blocks:
      cb.g_jac()

    if ouput:
      return self.Jx, self.Jl

def SolverByGaussElimination(problem, maxit=10):
  res  = problem.CompoundResidual()
  xc   = problem.CompoundParameters()
  lc   = problem.CompoundObservation()
  l0   = lc.copy()

  Sigma= problem.MakeWeightMatrix()
  Sigma.data[:] = 1./Sigma.data
  W    = problem.MakeWeightMatrix()
  Ja,Jb= problem.MakeJacobians()

  for it in range(maxit):
    problem.UpdateJacobian()
    problem.UpdateResidual()
    le  = lc - l0
    print np.linalg.norm(res), np.linalg.norm(le)

    cg  = le - res
    ATW = Ja.T * W
    ATWA= ATW * Ja
    dx  = scipy.sparse.linalg.spsolve(ATWA, ATW * cg)
    lag = W * (Ja * dx - cg)
    dl  = Sigma * lag  + le
    lc -= dl
    xc += dx
  return xc, lc - l0
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

  x = np.zeros((num_x, dim_x))
  l = [ np.ones((num_l/num_x, dim_l)) for _ in range(num_x) ] # l[which_x] = vstack(l[which_l])
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
  Sig, W = problem.MakeSigmaAndWeightMatrix()
  assert_array_equal(Sig.todense(), DiagonalRepeat(np.diag(sigma), num_l) )
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
  test_MakeSymmetric()
  test_MakeAB()
  test_MakeAWAT()
  test_MakeBlockInv()

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
      problem.SetSigma(l[i][j], 0.5*np.ones(dim_l))
  dims = problem.dims
  total_dim = sum(dims)
  res  = problem.CompoundResidual()
  xc   = problem.CompoundParameters()
  lc   = problem.CompoundObservation()
  l0   = lc.copy()
  KKT  = problem.MakeReducedKKTMatrix()
  Sig,W = problem.MakeSigmaAndWeightMatrix()
  problem.UpdateJacobian()
  KKT.A

#  kkt  = problem.MakeKKTMatrix()
#  segment_x, segment_l, segment_r = problem.MakeKKTSegmentSlice()
##  a = kkt.A
#  problem.UpdateJacobian()
##  op = CholeskyOperator(kkt)
#  op = CoordLinearOperator(problem.mat_kkt.data,
#                           problem.mat_kkt.row,
#                           problem.mat_kkt.col,
#                           total_dim, total_dim,
#                           symmetric=True)
#  b = np.zeros(sum(dims))
#  for it in range(1):
#    problem.UpdateJacobian()
#    problem.UpdateResidual()
##    op.UpdataFactor(kkt)
#
#    e = lc - l0
#    b[segment_l] = -e
#    b[segment_r] = -res
#
##    xc += s[segment_x]
##    lc += s[segment_l]
#    print np.linalg.norm(res), np.linalg.norm(e)


