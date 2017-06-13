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

#%% ColumnMajorSparseMatrix
class LexicalSortable(object):
  def key(self):
    raise NotImplementedError()

  def __lt__(self, other):
    return self.key() < other.key()

class ColumnMajorSparseMatrix():

  def __init__(self, shape=None):
    self.shape = shape
    self.submatrix_list = []
    self.data = None
    self.row  = None
    self.col  = None
    self.indptr = None

  def AddDenseSubMatrix(self, row0, col0, shape):
    id = len(self.submatrix_list)
    self.submatrix_list.append( self.SubMatrix(row0, col0, shape) )
    return id

  def SubMatrixWrite(self, id, src):
    self.submatrix_list[id].Write( src )

  def BuildSparseMatrix(self, coo=False):
    """ 1. sort the submatrix to a column-major flat list """
    self.submatrix_list.sort()
    num_submatrix = len(self.submatrix_list)

    """ 2. gather the submatrixs of the same column in a group, collect
        their sub vectors and sort them again into column major """
    vectors = []
    heap = []
    for i in range( num_submatrix ):
      current = self.submatrix_list[i]
      heap.extend(current.subvector_list)

      # reached the last submatrix of this column or the end
      if i == num_submatrix-1 or self.submatrix_list[i+1].c != current.c:
        # process this group
        vectors.extend( sorted(heap) )
        # reset the heap and start the new
        heap = []

    """ 3. assign vector segments to slices """
    len_list = [vec.length for vec in vectors]
    offset = np.cumsum( [0] + len_list )
    nnz = offset[-1]
    data = np.empty(nnz)
    for start, vec in zip(offset.tolist(), vectors):
      vec.Mapto(data, start)  # vec.data = data[start : start + vec.length]

    """ 4. Make row and col indices for COO matrix.
        Since the mapping is done, we could use it to collect indivial indice """
    for vec in vectors:
      vec._feed_row_indice()
    row = data.astype('i').copy()

    for vec in vectors:
      vec._feed_col_indice()
    col = data.astype('i').copy()

    """ update shape info"""
    shape = (row.max()+1, col[-1]+1)
    if self.shape is None:
      self.shape = shape
    else:
      self.shape = tuple(np.max([shape, self.shape], axis=0))

    """ 5. convert to Compressed Sparse Column (CSC)
    * compressed(col) -> indices, where only the heads of each col are recorded
    """
    if 1:  # fast version
      indptr, = np.where( np.diff(col) )  # np.diff : out[n] = a[n+1]-a[n]
      indptr = np.r_[0, indptr+1, nnz]
    else:  # slow version
      indptr = np.array([0] + [ i for i in xrange(1, nnz) if col[i] != col[i-1] ] + [nnz])

    self.data, self.col, self.row, self.indptr = data, col, row, indptr

    if coo:
      return scipy.sparse.csc_matrix( (data, row, indptr), shape=self.shape)
    else:
      return scipy.sparse.coo_matrix( (data, (row, col) ), shape=self.shape)

  def Shift(self, delta_r, delta_c):
    for submat in self.submatrix_list:
      submat.r += delta_r
      submat.c += delta_c

  def ComputeShape(self):
    bottom_mat = max(self.submatrix_list, key=lambda o : o.r + o.dim_vec)
    right_mat  = max(self.submatrix_list, key=lambda o : o.c + o.num_vec)
    r = bottom_mat.r + bottom_mat.dim_vec
    c = right_mat.c  + right_mat.num_vec
    return r, c

  def Append(self, other):
    if self.shape is None:
      offset_r, offset_c = self.ComputeShape()
    else:
      offset_r, offset_c = self.shape

    other_list = copy(other.submatrix_list)
    for submat in other_list:
#      submat.r += offset_r
      submat.c += offset_c
    self.submatrix_list.extend(other_list)


  class SubMatrix(LexicalSortable):
    """ ordering of submatrix in a column-major sparse matrix """
    def __init__(self, r, c, array_shape): #
      self.r, self.c = r,c
      self.dim_vec, self.num_vec = array_shape[:2]
      self.subvector_list = [self.SubVector(self, i, self.dim_vec) for i in range(self.num_vec)]
      self.buf = np.empty((self.dim_vec, self.num_vec), order='F').T

    def key(self):
      return (self.c, self.r)

    def Write(self, array):
      if self.subvector_list[0].data is None:
        raise RuntimeError("slice not assigned yet")
      for dst, src in zip(self.subvector_list, array.T):
        dst.data[:] = src

    @property
    def data(self):
      if self.subvector_list[0].data is None:
        return None
      for src, dst in zip(self.subvector_list, self.buf):
        dst[:] = src.data[:]
      return self.buf.T

    def __str__(self):
      return "SubMatrix at (%d, %d): \n" % (self.r, self.c)  + str(self.data)
    __repr__ = __str__

    class SubVector(LexicalSortable):
      def __init__(self, parent, seq, length):
        self.parent = parent
        self.seq = seq
        self.length = length
        self.data = None

      def key(self):
        return (self.seq, self.parent.r)

      def Mapto(self, vector, start):
        self.data = np.ndarray(shape  = (self.length,),
                               buffer = vector,
                               offset = start*vector.itemsize)

      def _feed_row_indice(self):
        r0 = self.parent.r
        r1 = r0 + self.length
        self.data[:] = np.arange(r0,r1)

      def _feed_col_indice(self):
        self.data[:] = self.parent.c + self.seq # all the data share the same column

      def __str__(self):
        return "vec at %d row,  %dth segment of array " % (self.parent.r, self.seq) + str(self.data)
      __repr__ = __str__
def test_CreateSparseMatrix():
  dense  = np.empty((4,3))

  csm = ColumnMajorSparseMatrix()
  csm.AddDenseSubMatrix( 0, 0, (4,3) )
  sparse = csm.BuildSparseMatrix()
  assert sparse.shape == dense.shape

  # write function
  dense1  = np.arange(12).reshape(4,3)

  csm1 = ColumnMajorSparseMatrix()
  csm1.AddDenseSubMatrix( 0, 0, (4,3) )
  sparse1 = csm1.BuildSparseMatrix()
  csm1.SubMatrixWrite(0, dense1)
  assert_array_equal( sparse1.A, dense1 )

  # multiple matrix
  e = [np.ones(1), 2*np.eye(2), 3*np.eye(3)]
  dense  = scipy.linalg.block_diag(*e)
  csm2 = ColumnMajorSparseMatrix()
  csm2.AddDenseSubMatrix( 0, 0, (1,1) )
  csm2.AddDenseSubMatrix( 1, 1, (2,2) )
  csm2.AddDenseSubMatrix( 3, 3, (3,3) )
  sparse2 = csm2.BuildSparseMatrix()
  for i in range(3):
    csm2.SubMatrixWrite(i, e[i])
  assert_array_equal( sparse2.A, dense  )

  # shift
  dense  = np.empty((4,3))
  csm = ColumnMajorSparseMatrix()
  csm.AddDenseSubMatrix( 0, 0, (4,3) )
  csm.Shift(5,5)
  sparse = csm.BuildSparseMatrix()
  assert sparse.shape == (5+4,5+3)

  # merge
  csm1.Append(csm2)
  sparse = csm1.BuildSparseMatrix()
  csm1.data[:] =1
  print "test_CreateSparseMatrix passed"

#%%

class MatrixTreeNode(LexicalSortable):
  def __init__(self, parent, r, c):
    self.parent = parent
    self.r = r
    self.c = c
    self.elements = []
    self.lexical_key = [(),()]
    self.absolute_pos = [r, c]

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

  def PropogateAbsolutePos(self, offset=[0,0]):
    self.absolute_pos[0] = offset[0] + self.r
    self.absolute_pos[1] = offset[1] + self.c
    for e in self.elements:
      e.PropogateAbsolutePos(self.absolute_pos)

  def PropogateKeyPrefix(self, prefix=[(),()] ): # empty prefix for root
    self.lexical_key[0] = prefix[0] + (self.r,)
    self.lexical_key[1] = prefix[1] + (self.c,)
    for e in self.elements:
      e.PropogateKeyPrefix(self.lexical_key)

  def __repr__(self):
    return str(type(self)) + "at (%d, %d)" % self.absolute_pos

#  def ComputeShape(self):
#    self.PropogateKeyPrefix()
#
#    mats = list(self(DenseMatrix))
#    right_mat  = max(mats, key=lambda m: m.lexical_key[0])
#    bottom_mat = max(mats, key=lambda m: m.lexical_key[1])
#
#    bottom_mat = max(mats, key=lambda m : m.absolute_pos[0] + m.dim_vec)
#    right_mat  = max(mats, key=lambda m : m.absolute_pos[1] + m.num_vec)
#    r = bottom_mat.r + bottom_mat.dim_vec
#    c = right_mat.c  + right_mat.num_vec
#    return r, c

  def BuildSparseMatrix(self, coo=False):
    self.PropogateKeyPrefix()

    """ 1. lexical sort of all the DenseMatrixSegment,(i.e leaves of the tree),
    the resulting order is exactly the linear data term is CSC matrix """
    vectors = sorted(self(DenseMatrixSegment))

    """ 2. assign segments to the compound data vector   """
    len_list = [vec.length for vec in vectors]
    offset = np.cumsum( [0] + len_list )
    nnz = offset[-1]
    data = np.empty(nnz)
    for start, vec in zip(offset.tolist(), vectors):
      vec.Mapto(data, start)  # vec.data = data[start : start + vec.length]
    self.data = data
    self.nnz  = nnz

    """ 3. Make row and col indices for COO matrix.
        Since the mapping is done, we could use it to collect indivial indice """
    self.PropogateAbsolutePos()

    for vec in vectors:
      vec._feed_row_indice()
    row = data.astype('i').copy()

    for vec in vectors:
      vec._feed_col_indice()
    col = data.astype('i').copy()

#    """ update shape info"""
#    shape = (row.max()+1, col[-1]+1)
#    if self.shape is None:
#      self.shape = shape
#    else:
#      self.shape = tuple(np.max([shape, self.shape], axis=0))

    """ 5. convert to Compressed Sparse Column (CSC)
    * compressed(col) -> indices, where only the heads of each col are recorded
    """
    if 1:  # fast version
      indptr, = np.where( np.diff(col) )  # np.diff : out[n] = a[n+1]-a[n]
      indptr = np.r_[0, indptr+1, nnz]
    else:  # slow version
      indptr = np.array([0] + [ i for i in xrange(1, nnz) if col[i] != col[i-1] ] + [nnz])

    if coo:
      return scipy.sparse.csc_matrix( (data, row, indptr))#, shape=self.shape
    else:
      return scipy.sparse.coo_matrix( (data, (row, col) ))#, shape=self.shape

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

  def PutDenseMatrix(self, id, src):
    self.elements[id].Write( src )

class DenseMatrix(MatrixTreeNode):
  def __init__(self, r, c, shape):
    super(DenseMatrix, self).__init__(None, r, c)

    self.dim_vec, self.num_vec = shape[:2]
    for seq in xrange(self.num_vec):
      self.AddElement(DenseMatrixSegment(self, seq, self.dim_vec))

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

class DenseMatrixSegment(MatrixTreeNode):
  def __init__(self, parent, seq, length):
    super(DenseMatrixSegment, self).__init__(parent, 0, seq)
    self.length = length
    self.data = None

  def isLeaf(self):
    return True

  def Mapto(self, vector, start):
    self.data = np.ndarray(shape  = (self.length,),
                           buffer = vector,
                           offset = start*vector.itemsize)

  def _feed_row_indice(self):
    r0 = self.absolute_pos[0]
    r1 = r0 + self.length
    self.data[:] = np.arange(r0,r1)

  def _feed_col_indice(self):
    self.data[:] = self.absolute_pos[1] # all the data share the same column

  def __repr__(self):
    return str(self.key()) + " len:" + str(self.length)

  def BuildSparseMatrix(self):
    raise RuntimeError("BuildSparseMatrix not supported in DenseMatrixSegment level")

  def ComputeShape(self):
    return (self.length,)



def test_MatrixTreeNode():

  sb = SparseBlock()
  sb.NewDenseMatrix( 0, 0, (4,3) )
  sb.NewDenseMatrix( 0, 3, (2,3) )

  # iterator
  assert len(list(sb(SparseBlock))) == 1
  assert len(list(sb(DenseMatrix))) == 2
  assert len(list(sb(DenseMatrixSegment))) == 6

  # PropogateKeyPrefix
  sb.PropogateKeyPrefix()
  assert_equal([s.key() for s in sb(DenseMatrixSegment)],
               [(0, 0, 0, 0, 0, 0),
                (0, 0, 1, 0, 0, 0),
                (0, 0, 2, 0, 0, 0),
                (0, 3, 0, 0, 0, 0),
                (0, 3, 1, 0, 0, 0),
                (0, 3, 2, 0, 0, 0)] )

  # PropogateAbsolutePos
  sb.PropogateAbsolutePos()
  assert_equal(
    [s.absolute_pos for s in sb(DenseMatrixSegment)],
    [[0, 0], [0, 1], [0, 2], [0, 3], [0, 4], [0, 5]])

  sb.PropogateAbsolutePos([1,2])
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

  # CompoundMatrix
  cm = CompoundMatrix()
  sb2 = SparseBlock()
  sb2.NewDenseMatrix( 4, 6, (3,3) )

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

test_MatrixTreeNode()
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

  def AddVector(self, vector_list):
    ret = []
    for v in vector_list:
      """ 1. check whether it is a newcomer, use address as hash"""
      key = ArrayID(v)
      if key not in self.vec_dict:
        """ 2. Make new segment on the tail """
        new_item = self.MakeVector(len(v))
        new_item[1][:] = v      # copy inital value
        self.vec_dict[key] = new_item
        ret.append( new_item )
      else:
        ret.append( self.vec_dict[key] )
    offset, seg = zip(*ret)
    return offset, seg

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
  offset, seg = cv.AddVector(v for v in vs)
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
  offset2, _ = cv.AddVector(v for v in vs)
  assert_equal(offset, offset2)
  offset3, _ = cv.AddVector( [ np.empty(1) ] )
  assert offset3[0] == 16
  print "test_CompoundVector passed"


#%% GaussHelmertProblem
class GaussHelmertProblem(object):
  ConstraintBlock = namedtuple('ConstraintBlock', ['id', 'g_res', 'g_jac'])

  def __init__(self):
    self.cv_x   = CompoundVectorWithDict()
    self.cv_l   = CompoundVectorWithDict()
    self.cv_res = CompoundVector()

    self.csm_x = ColumnMajorSparseMatrix()
    self.csm_l = ColumnMajorSparseMatrix()
    self.Jx, self.Jl = None,None

    self.constraint_blocks = []    # list of ConstraintBlock
    self.variance_factor = -1.0

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
    x_off, x_vec = self.cv_x.AddVector(x_list)
    l_off, l_vec = self.cv_l.AddVector(l_list)
    xl_vec = x_vec + l_vec
    """ 3. Compound vector for constraint residual """
    res_off, res_vec = self.cv_res.MakeVector(dim_res)

    jac_submat_id_x = [ self.csm_x.AddDenseSubMatrix(res_off, c, (dim_res, dim_x) ) for c, dim_x in zip(x_off, x_sizes) ]
    jac_submat_id_l = [ self.csm_l.AddDenseSubMatrix(res_off, c, (dim_res, dim_l) ) for c, dim_l in zip(l_off, l_sizes) ]

    """ 4. Generate constraint functor that use the mapped vectors """
    def g_res():
      res_vec[:] = g(*xl_vec)

    def g_jac():
      J = var_jacobian( np.hstack(xl_vec) )
      jac = np.split(J, xl_indices, axis=1)
      jac.reverse() # reversed, to pop(-1) instead of pop(0)
      for submat_id in jac_submat_id_x:
        self.csm_x.SubMatrixWrite( submat_id, jac.pop() )

      for submat_id in jac_submat_id_l:
        self.csm_l.SubMatrixWrite( submat_id, jac.pop() )

    self.constraint_blocks.append( GaussHelmertProblem.ConstraintBlock(res_off, g_res, g_jac) )


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

  def MakeJacobians(self):
    self.Jx = self.csm_x.BuildSparseMatrix()
    self.Jl = self.csm_l.BuildSparseMatrix()
    return self.Jx, self.Jl

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

  problem = GaussHelmertProblem()
  for i in range(num_x):
    for j in range(num_l/num_x):
      problem.AddConstraintUsingAD(AffineConstraint,
                                   [ x[i] ],
                                   [ l[i][j] ])
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

  print "test_ProblemJacobian passed"


#%%
if __name__ == '__main__':

  test_CreateSparseMatrix()
  test_ArrayID()
  test_CompoundVector()
  test_ProblemJacobian()


  dim_x, num_x = 3, 2
  dim_l, num_l = 4, 10*num_x

  dim_g = 3
  A = np.random.rand(dim_g, dim_x)
  B = np.random.rand(dim_g, dim_l)
  AffineConstraint = MakeAffineConstraint(A,B)

  x = np.empty((num_x, dim_x))
  l = [ np.empty((num_l/num_x, dim_l)) for _ in range(num_x) ] # l[which_x] = vstack(l[which_l])

  problem = GaussHelmertProblem()
  for i in range(num_x):
    for j in range(num_l/num_x):
      problem.AddConstraintUsingAD(AffineConstraint,
                                   [ x[i] ],
                                   [ l[i][j] ])
  res = problem.CompoundResidual()
  xc = problem.CompoundParameters()
  lc = problem.CompoundObservation()

  J = ColumnMajorSparseMatrix()

  def walk(node):
    """ iterate tree in pre-order depth-first search order """
    yield node
    for child in node.elements:
        for n in walk(child):
            yield n


#  class MatrixTree(object):
#    def __init__(self):



#  sp = wm.BuildSparseMatrix()



