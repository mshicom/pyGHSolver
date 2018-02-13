#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  9 17:35:21 2017

@author: kaihong
"""
import numpy as np
import scipy
from numpy.testing import *
from itertools import chain
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
    for seq in range(shape[1]):
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
    for seq in range(shape[0]):
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
  print( "test_MatrixTreeNode passed")

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
  print( "test_MakeSymmetric passed")
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
      if not np.any( [op is None for op in chain(op_a.values(),op_b.values())] ):
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
  print( "test_MakeAB passed ")


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
  #      print( "All element collected")
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
  #      print( "All element collected")
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
  print( "test_MakeAWAT passed")

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

if __name__ == '__main__':
  test_MatrixTreeNode()
  test_MakeSymmetric()
  test_MakeAB()
  test_MakeAWAT()
  test_MakeBlockInv()