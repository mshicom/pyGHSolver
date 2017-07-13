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

from collections import namedtuple
np.set_printoptions(precision=3, linewidth=90)
from numpy.testing import *
from itertools import chain

from parameterization import *
import pycppad
from copy import copy
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
    * indices[0] = 0
    * indices[i] = indices[i-1] + (nnz of column i-1)
    * indices[-1] = total nnz
    """
    num_col = col.max()+1 if shape is None else shape[1]
    col_cnt = np.zeros(num_col,'i')
    cid,cnnz = np.unique(col, return_counts=True)
    col_cnt[cid] = cnnz
    indptr = np.cumsum(np.r_[0, col_cnt])
    self.indptr = indptr

    if coo:
      return scipy.sparse.coo_matrix( (data, (row, col) ), shape=shape)#, shape=self.shape
    else:
      return scipy.sparse.csc_matrix( (data, row, indptr), shape=shape)#, shape=self.shape

class CompoundMatrix(MatrixTreeNode):
  def __init__(self, sb_list=[]):
    super(CompoundMatrix, self).__init__(None, 0, 0)
    self.shape = None
    for sb in sb_list:
      self.AddElement(sb)

  def NewSparseBlock(self, r, c):
    sb = SparseBlock(r, c)
    return self.AddElement(sb)

class SparseBlock(MatrixTreeNode):
  def __init__(self, r=0, c=0, shape=None):
    super(SparseBlock, self).__init__(None, r, c)
    self.shape = shape

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
  def __init__(self, r, c, shape=None):
    super(DenseMatrix, self).__init__(None, r, c)

    self.cache = None
    self.post_callback = []
    self.pre_process = None

    if shape:
      self.InitElement(shape)

  def InitElement(self, shape):
    self.shape = shape
    if self.elements:
      del self.elements[:]
    for seq in xrange(shape[1]):
      self.AddElement(DenseMatrixSegment(self, 0, seq, shape[0]))

    self.buf = np.empty(self.shape, order='F').T

  def Write(self, array):
    if self.pre_process:
      array = self.pre_process(array)
    if len(self.elements) == 0 or self.elements[0].data is None:
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

  def InitElement(self, shape):
    self.shape = shape
    for seq in xrange(shape[0]):
      self.AddElement(DenseMatrixSegment(self, seq, seq, 1))
    self.buf = np.empty((shape[0],1))

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
  id2 = sb.NewDenseMatrix( 0, 4, (2,3) )

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
                (4, 0, 0, 0),
                (4, 1, 0, 0),
                (4, 2, 0, 0)] )

  # PropogateAbsolutePos
  sb.PropogateAbsolutePos()
  assert_equal(
    [s.absolute_pos for s in sb(DenseMatrixSegment)],
    [[0, 0], [0, 1], [0, 2], [0, 4], [0, 5], [0, 6]])

  sb.PropogateAbsolutePos(np.array([1,2]))
  assert_equal(
    [s.absolute_pos for s in sb(DenseMatrixSegment)],
    [[1, 2], [1, 3], [1, 4], [1, 6], [1, 7], [1, 8]])

  # BuildSparseMatrix coo
  sb_coo = sb.BuildSparseMatrix(coo=True)
  sb.data[:] = np.arange(sb.nnz)
  assert_array_equal(sb_coo.A,
                    [[  0.,   4.,   8.,   0.,  12.,  14.,  16.],
                     [  1.,   5.,   9.,   0.,  13.,  15.,  17.],
                     [  2.,   6.,  10.,   0.,   0.,   0.,   0.],
                     [  3.,   7.,  11.,   0.,   0.,   0.,   0.]])

  sb_csc = sb.BuildSparseMatrix(coo=False)
  sb.data[:] = np.arange(sb.nnz)
  assert_array_equal(sb_csc.A,
                    [[  0.,   4.,   8.,   0.,  12.,  14.,  16.],
                     [  1.,   5.,   9.,   0.,  13.,  15.,  17.],
                     [  2.,   6.,  10.,   0.,   0.,   0.,   0.],
                     [  3.,   7.,  11.,   0.,   0.,   0.,   0.]])
  assert_array_equal(sb_csc.indptr, sb_coo.tocsc().indptr )

  # write
  dm = np.arange(6).reshape(2,3)
  sb.PutDenseMatrix(id2, dm)
  assert_array_equal(sb_csc.A[0:2,4:7], dm)

  # read
  dml = list(sb(DenseMatrix))
  assert_array_equal(dml[1].data, dm)

  # CompoundMatrix
  cm = CompoundMatrix()
  sb = SparseBlock()
  id1 = sb.NewDenseMatrix( 0, 0, (4,3) )
  id2 = sb.NewDenseMatrix( 0, 3, (2,3) )
  sb2 = DenseMatrix( 4, 6, (3,3) )

  cm.AddElement(sb)
  cm.AddElement(sb2)
  assert_equal( len(list(cm(SparseBlock))), 1)
  assert_equal( len(list(cm(DenseMatrix))), 3)
  assert_equal( len(list(cm(DenseMatrixSegment))), 9)

  cm_coo = cm.BuildSparseMatrix()
  cm.data[:] = np.arange(cm.nnz)
  assert_array_equal(cm_coo.A,
                    [[  0.,   4.,   8.,  12.,  14.,  16.,   0.,   0.,   0.],
                     [  1.,   5.,   9.,  13.,  15.,  17.,   0.,   0.,   0.],
                     [  2.,   6.,  10.,   0.,   0.,   0.,   0.,   0.,   0.],
                     [  3.,   7.,  11.,   0.,   0.,   0.,   0.,   0.,   0.],
                     [  0.,   0.,   0.,   0.,   0.,   0.,  18.,  21.,  24.],
                     [  0.,   0.,   0.,   0.,   0.,   0.,  19.,  22.,  25.],
                     [  0.,   0.,   0.,   0.,   0.,   0.,  20.,  23.,  26.]])

  cm_csc = cm.BuildSparseMatrix(coo=False)
  cm_csc.data[:] = np.arange(cm.nnz)
  assert_array_equal(cm_csc.A,
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
  sb.PutDenseMatrix(id1, np.full((2,2), -1.0))
  sb.PutDenseMatrix(id2, np.full(4, 2.0))

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


def MakeSymmetric(obj):

  def _make_tranpose_callback(dst_block):
    def WriteTranspose(obj, array):
      dst_block.Write(array.T)
    return WriteTranspose

  def _recursive_routine(src, parent):
    """ The parent do nothing but mirror their off-diagonal(in absolute pos) children """
    for e in copy(src.elements):  # use copy to ignore new attached element
      r, c = e.absolute_pos[0], e.absolute_pos[1]

      """ For DenseMatrix, make a callback and new DenseMatrix. Recursion end here."""
      if isinstance(e, DenseMatrix):
        if r != c:
          e_t = DenseMatrix(e.c, e.r, (e.shape[1], e.shape[0]))
          e.post_callback.append( _make_tranpose_callback(e_t) )
          parent.AddElement(e_t)

      else:
        """ For abstract block, recursion is needed, and there are 2 cases: """
        if r != c:
          """ case 1: Off-diagonal, make a mirrored sibling. And subsequce
          new node will be attaced to it.
          """
          e_t = type(e)(c, r)
          parent.AddElement(e_t)
          next_parent = e_t
        else:
          """ case 2: diagonal, do nothing, but its off-diagonal children will
          be attaced to the block itself.
          """
          next_parent = src
        """ recursion magic """
        _recursive_routine(e, next_parent)
  """ func body """
  obj.PropogateAbsolutePos()
  _recursive_routine(obj, obj)
  return obj


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
  sb.PutDenseMatrix(id2, np.full((1,2), 3.) )
  assert_array_equal(sp.A,
                    [[ 1.,  1.,  3.],
                     [ 1.,  1.,  3.],
                     [ 3.,  3.,  0.]])
  # CompoundMatrix

  sb1 = SparseBlock()
  sb1.NewDenseMatrix(0,0,(4,4))
  sb1.PutDenseMatrix(0, np.ones((4,4)))
  sb2 = SparseBlock()
  sb2.NewDenseMatrix(0,0,(4,1))
  sb2.PutDenseMatrix(0, 3*np.ones((4,1)))
  sb2.OverwriteRC(0, 5)

  cm = CompoundMatrix([sb1,sb2])
  MakeSymmetric(cm)
  sp = cm.BuildSparseMatrix(coo=False)
  assert_array_equal(sp.A,
                    [[ 1.,  1.,  1.,  1.,  0.,  3.],
                     [ 1.,  1.,  1.,  1.,  0.,  3.],
                     [ 1.,  1.,  1.,  1.,  0.,  3.],
                     [ 1.,  1.,  1.,  1.,  0.,  3.],
                     [ 0.,  0.,  0.,  0.,  0.,  0.],
                     [ 3.,  3.,  3.,  3.,  0.,  0.]])
  print "test_MakeSymmetric passed"
#%%
from collections import defaultdict,OrderedDict
def MakeAB(A, B, r=0, c=0):

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
  """ func body """
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


def MakeAWAT(A, W, make_full=True, r=0, c=0):
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
  #      print "All element collected"
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
  #      print "All element collected"
        new_data = np.zeros(dst_block_shape)
        for mat_a, mat_b, mat_w in zip(op_a.values(), op_b.values(),op_w.values()):
          new_data += mat_a.dot(mat_w).dot(mat_b.T)
        dst_block.Write( new_data )
        # 3. reset the dict to all None
        for op in [op_a, op_b]:
          for key in op.keys():
            op[key] = None
    return CalculateSumOfAWBT

  """ func body """

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




def MakeBlockInv(other):
  def _make_inv_callback(dst_block):
    def WriteInv(obj, array):
      dst_block.Write(np.linalg.inv(array))
    return WriteInv
  """ func body """
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
    for dst, seg in self.array_dict.iteritems():
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
  offset,seg = zip(*[cv.AddVector(v) for v in vs])[:2]
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
  offset2 = zip(*[cv.AddVector(v) for v in vs])[0]
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

  print "test_CompoundVector passed"


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
  __slots__ = 'sigma'
  def __init__(self, array):
    super(ObservationBlock, self).__init__(array)
    self.sigma = None
  def __repr__(self):
    return "ObservationBlock:%s, %dlinks" % (self.array, len(self.jac))

def check_nan(x):
  if not np.all( np.isfinite(x) ):
    raise ValueError("invalid value")
  return x

def check_allzero(x):
  if np.all( x == 0 ):
    raise ValueError("all zero")
  return x

def check_magnitude(r):
  if np.linalg.norm(r) > np.pi:
    raise ValueError("rotation magnitude larger than pi, will cause problems during optimizatoin")
  return r

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
    valid_value = [ np.isfinite(m).all() for m in [tmp_res] + tmp_jac ]
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
      x.param.UpdataJacobian(x.array)

    for l in self.dict_observation_block.values(): #array, dim, isfixed, param
      if l.isfixed or l.param is None:
        continue  # ignore fixed
      l.param.UpdataJacobian(l.array)

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
  return np.tile( M, (repeats,1) )

def HorizontalRepeat(M, repeats):
  return np.tile( M, (1, repeats) )

def test_ProblemBasic():
  dim_x, num_x = 3, 2
  dim_l, num_l = 4, 30*num_x

  dim_g = 3
  A = np.random.rand(dim_g, dim_x)
  B = np.random.rand(dim_g, dim_l)
  AffineConstraint, AffineConstraintJac = MakeAffineConstraint(A,B)

  x = np.zeros((num_x, dim_x))
  l = [ np.ones((num_l/num_x, dim_l)) for _ in range(num_x) ] # l[which_x] = vstack(l[which_l])
  sigma = np.full(dim_l, 0.5)
  problem = GaussHelmertProblem()
  for i in range(num_x):
    for j in range(num_l/num_x):
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
    for j in range(num_l/num_x):
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
  print "test_ProblemBasic passed"


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
  print "test_ProblemMakeKKT passed"
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
  print "test_FixAndParameterization passed"

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

    print np.linalg.norm(res),np.linalg.norm(le)

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
      print "Singular problem"
      break
    if np.abs(dx).max() < dx_thres:
      break

  ret = [xc, le]
  if fac:
    factor = (le * W).dot(le) / (problem.dim_res - problem.dim_dx)
    print 'variance factor:%f' % factor
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

    print np.linalg.norm(res),np.linalg.norm(le)
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

  ret = [xc, le]
  factor = (le * W).dot(le) / (problem.dim_res - problem.dim_dx)
  print 'variance factor:%f' % factor
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
    print np.linalg.norm(res),np.linalg.norm(le)
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

  ret = [xc, le]
  factor = (le * W).dot(le) / (problem.dim_res - problem.dim_dx)
  print 'variance factor:%f' % factor

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
    print np.linalg.norm(res)
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
  print 'variance factor:%f' % factor
  if fac:
    ret.append(factor)
  if cov:
    covariance = Sxx_factor.inv()
    if np.isfinite(factor) and factor!=0 :
      covariance *= factor
    ret.append( covariance )
  return ret

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

    print np.linalg.norm(res),np.linalg.norm(le)
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
      print mu

  ret = [xc, le]
  factor = (le * W).dot(le) / (problem.dim_res - problem.dim_dx)
  print 'variance factor:%f' % factor
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
def MakeJacobianFunction(g, *args):
  arg_sizes= [ len(np.atleast_1d(vec)) for vec in args ]

  arg_indices= np.cumsum( arg_sizes )[:-1]
  var       = np.hstack( args )
  var_in    = pycppad.independent( var )
  var_out   = np.atleast_1d( g( *np.split(var_in, arg_indices) ) )
  var_jacobian= pycppad.adfun(var_in, var_out).jacobian
  def g_jac_auto(*vec):
    J = var_jacobian( np.hstack(vec) )
    check_nan(J)
    check_allzero(J)
    return np.split(J, arg_indices, axis=1)
  return g_jac_auto

def AddJacobian(f):
  """
  Examples
  --------
  >>>  @AddJacobian
  >>>  def foo(x,y):
  >>>    return x-y
  >>>  res, jac = foo(np.ones(3), np.zeros(3))
  """
  def g(*args):
    if not hasattr(f, 'jac'):
      f.jac = MakeJacobianFunction(f, *args)
    return f(*args), f.jac(*args)
  return g

def CheckJacobianFunction(g, g_jac=None, *args):
  g_jac_auto   = MakeJacobianFunction(g, *args)
  tmp_jac_auto = g_jac_auto( *args )

  if g_jac is None:
    return tmp_jac_auto

  tmp_jac      = list( g_jac( *args ) )
  assert len(tmp_jac)==len(tmp_jac_auto)
  for a,b in zip(tmp_jac, tmp_jac_auto):
    assert_array_almost_equal(a,b)
  return

def ErrorPropogationExplicit(cov_x, func, x):
  """ y + dy = func(x + dx)
      dy = J dx, J = df/dx,
      cov_y = J cov_x J'
  """
  J = MakeJacobianFunction(func, x)(x)[0]
  return J.dot(cov_x).dot(J.T)
  def test():
    f = np.sin
    x = np.r_[0.2]
    var = np.r_[0.1]
    var_est = ErrorPropogationExplicit(var, f, x)
    var_exp = np.cos(x)**2 * var
    assert_almost_equal(var_est, var_exp)

def NullMatrix(A):
  A = np.atleast_2d(A)
  Q,R = np.linalg.qr(A.T, 'complete')
  rank = A.shape[0]
  return Q[:, rank:]

def ErrorPropogationImplicit(cov_x, constaint_func, x, y=None):
  """ G(x)=0
      dG = J dx = 0
      dy = J NullSpace(J)*dx
  """
  J = MakeJacobianFunction(func, x)(x)[0]
  Jn = NullMatrix(J)
  return Jn.dot(cov_x).dot(Jn.T)

def Montacalo(f, x0, sigma, trial=10000):
  dim_x, dim_y = len(np.atleast_1d(x)), len(np.atleast_1d(f(x0)))

  y = np.empty((trial, dim_y))
  for i in xrange(trial):
    y[i] = f( x0 + sigma*np.random.randn(dim_x) )
  y_mean = np.mean(y, axis=0)
  y_cov = np.cov((y - y_mean).T)
  return y_cov, y_mean

#cov_r1_r2 = 0.1**2 * np.eye(6)
#f = lambda x: axAdd(-x[:3],x[3:])
#x0 = np.r_[0.1,0,0, 0.2,0.3,0.1]
#cov_est = ErrorPropogationExplicit(cov_r1_r2, f, x0)
#cov_mon = Montacalo(f, x0, 0.1)

#%%
if __name__ == '__main__':

  test_MatrixTreeNode()
  test_MakeSymmetric()
  test_MakeAB()
  test_MakeAWAT()
  test_MakeBlockInv()

  test_ArrayID()
  test_CompoundVector()
  test_ProblemBasic()
  test_ProblemMakeKKT()
  test_FixAndParameterization()

  def test_SolveWithGESparse():
    dim_x = 3
    a = np.ones(dim_x)

    num_l = 100
    sigma = 0.02
    s = sigma**2*np.eye(dim_x)

    for trail in range(1):
      bs = [ a + sigma * np.random.randn(3) for i in range(num_l)]
      problem = GaussHelmertProblem()
      for i in range(num_l):
        problem.AddConstraintWithArray(lambda x,y:x-y,
                                     [ a ],
                                     [ bs[i] ])
        problem.SetSigma(bs[i], s)

#    problem.SetParameterization(bs[0], SubsetParameterization([1,1,0]))
#    problem.SetSigma(bs[0], sigma**2*np.eye(2))
    SolveWithGESparseAsGM(problem,fac=True)

    x,le,fac,cov = SolveWithGESparseLM(problem, fac=True, cov=True)
    print fac
    problem.ViewJacobianPattern()


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









