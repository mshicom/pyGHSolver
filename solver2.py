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

  def __init__(self):
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
    row = data.copy()

    for vec in vectors:
      vec._feed_col_indice()
    col = data.copy()

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
      return scipy.sparse.csc_matrix( (data, row, indptr ) )
    else:
      return scipy.sparse.coo_matrix( (data, (row, col) ) )

  def Shift(self, delta_r, delta_c):
    for submat in self.submatrix_list:
      submat.r += delta_r
      submat.c += delta_c

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
        self.data[:] = self.parent.c + self.seq

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
  dense  = np.arange(12).reshape(4,3)

  csm = ColumnMajorSparseMatrix()
  csm.AddDenseSubMatrix( 0, 0, (4,3) )
  sparse = csm.BuildSparseMatrix()
  csm.SubMatrixWrite(0, dense)
  assert_array_equal( sparse.A, dense  )

  # multiple matrix
  e = [np.ones(1), 2*np.eye(2), 3*np.eye(3)]
  dense  = scipy.linalg.block_diag(*e)
  csm = ColumnMajorSparseMatrix()
  csm.AddDenseSubMatrix( 0, 0, (1,1) )
  csm.AddDenseSubMatrix( 1, 1, (2,2) )
  csm.AddDenseSubMatrix( 3, 3, (3,3) )
  sparse = csm.BuildSparseMatrix()
  for i in range(3):
    csm.SubMatrixWrite(i, e[i])
  assert_array_equal( sparse.A, dense  )

  # shift
  dense  = np.empty((4,3))
  csm = ColumnMajorSparseMatrix()
  csm.AddDenseSubMatrix( 0, 0, (4,3) )
  csm.Shift(5,5)
  sparse = csm.BuildSparseMatrix()
  assert sparse.shape == (5+4,5+3)


  print "test_CreateSparseMatrix passed"
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
