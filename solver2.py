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
    for submatrix in self.submatrix_list:
      submatrix._feed_grid_matrix(0)
    row = data.copy()

    for submatrix in self.submatrix_list:
      submatrix._feed_grid_matrix(1)
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

    def _feed_grid_matrix(self, which):
      data = np.mgrid[self.r : self.r + self.dim_vec, self.c : self.c + self.num_vec][which]
      self.Write(data)

    def SetCoordinate(self, r, c):
      self.r, self.c = r,c

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
        self.data[:] = np.arange(parent) # parent

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

  print "test_CreateSparseMatrix passed"

def GenerateJacobianFunction(g, x_list, l_list):
  x_sizes = [x.size for x in x_list]
  l_sizes = [l.size for l in l_list]
  xl_indices = np.cumsum(x_sizes + l_sizes)[:-1]
  x_indices = np.cumsum([0]+x_sizes)
  l_indices = np.cumsum([0]+l_sizes)

  var       = np.hstack(x_list+l_list )
  var_ad    = pycppad.independent( var )
  var_jacobian= pycppad.adfun(var_ad, g( *np.split(var_ad, xl_indices) ) ).jacobian

  x_slc = [slice(start, stop) for start, stop in zip(x_indices[:-1], x_indices[1:])]
  l_slc = [slice(start, stop) for start, stop in zip(l_indices[:-1], l_indices[1:])]
  def g_jacobian(x,l):
    var = np.hstack([x[slc] for slc in x_slc ] + [l[slc] for slc in l_slc ] )
    J = var_jacobian(var)
    return np.split(J, xl_indices, axis=1)
  return g_jacobian



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

class VariableBlock(ArrayID):
  __slots__ = 'place','place_local','param'
  def __init__(self, array):
    super(VariableBlock, self).__init__(array)
    self.param = IdentityParameterization(array.shape[0])

  def SetPlace(self, offset):
    size = self.data.shape[0] #self.param.LocalSize()
    self.place = slice(offset, offset + size)
    return size

  def SetLocalPlace(self, offset):
    size = self.param.LocalSize()
    self.place_local = slice(offset, offset + size)
    return size

class ObservationBlock(VariableBlock):
  __slots__ = '_sigma','weight'
  def __init__(self, array):
    VariableBlock.__init__(self, array)
    self._sigma = self.weight = np.ones(array.shape[0])

  @property
  def sigma(self):
    return self._sigma

  @sigma.setter
  def sigma(self, sigma):
    self._sigma = sigma
    self.weight = 1./sigma

  def ComputeCTWCInvAndE(self, le):
    C = self.param.ComputeJacobian(le)
    CTW = C.T * self.weight       #  = C.T.dot( np.diag(self.weight) )
    CTWC_inv = np.linalg.inv(CTW.dot(C))
    e = CTWC_inv.dot( CTW.dot(le) )
    return CTWC_inv, e

  def ComputeCTWCAndCTWE(self, le):
    C = self.param.ComputeJacobian(le)
    CTW = C.T * self.weight       #  = C.T.dot( np.diag(self.weight) )
    CTWC = CTW.dot(C)
    CTWe = CTW.dot(le)
    return CTWC, CTWe

def test_VariableBlock():
  a = np.empty((10,10))
  b = np.arange(3)
  assert len( { VariableBlock(a[0  ]), VariableBlock(b) } ) == 2             # different variable
  assert len( { VariableBlock(a[0  ]), VariableBlock(a[   1]) } ) == 2       # same length, different row
  assert len( { VariableBlock(a[0, :5]), VariableBlock(a[0,  :4]) } ) == 2   # diff length, same head
  assert len( { VariableBlock(a[0, :5]), VariableBlock(a[0, 1:5]) } ) == 2   # diff length, same end
  s = np.empty(1)
  assert len( { VariableBlock(s), VariableBlock(s) } ) == 1       # scalar

ConstraintBlock = namedtuple('ConstraintBlock',
                             ['g_flat', 'g_jacobian', 'r_slc', 'x_var', 'l_var','submat_id'])

class GaussHelmertProblem:
  def __init__(self):
    # map from x (nparray.data) to x_id (int)
    self.parameter_offset = 0
    self.parameter_local_offset = -1

    self.parameter_dict = {}

    # map from l (nparray.base) to l_id (int)
    self.observation_offset = 0
    self.observation_local_offset = -1

    self.observation_dict = {}
    self.observation_weight = {}

    self.constraint_offset = 0
    self.constraint_blocks = []    # list of ConstraintBlock
    self.variance_factor = -1.0

    self.csm_x = ColumnMajorSparseMatrix()
    self.csm_l = ColumnMajorSparseMatrix()

  def AddParameter(self, parameter_list):
    """ Add x (nparray.base) to parameter_dict and return a list of x_id
    """
    x_var = []
    for x in parameter_list:
      var = VariableBlock(x)
      if var not in self.parameter_dict:
        self.parameter_offset += var.SetPlace(self.parameter_offset)
        self.parameter_dict[var] = var
        x_var.append( var )
      else:
        x_var.append(self.parameter_dict[var])
    return x_var

  def AddObservation(self, observation_list):
    """ Add l (nparray.base) to observation_dict and return a list of l_id  """
    l_var = []
    for l in observation_list:
      var = ObservationBlock(l)
      if var not in self.observation_dict:
        self.observation_offset += var.SetPlace(self.observation_offset)
        self.observation_dict[var] = var
        l_var.append( var )
      else:
        l_var.append(self.observation_dict[var])
    return l_var

  def AddConstraintUsingAD(self, g, x_list, l_list):
    """
      1.add x (and l) to parameter_dict (and observation_dict)
      2.add record tuple(g, x_ids, l_ids) in constraint_blocks
    """
    x_sizes = [x.size for x in x_list]
    l_sizes = [l.size for l in l_list]
    xl_indices = np.cumsum(x_sizes + l_sizes)[:-1]
    var       = np.hstack(x_list+l_list )
    var_ad    = pycppad.independent( var )
    var_jacobian= pycppad.adfun(var_ad, g( *np.split(var_ad, xl_indices) ) ).jacobian

    res = g( *(x_list + l_list) )
    jac = var_jacobian(var)
    if not ( np.isfinite(res).all() and  np.isfinite(jac).all() ):
      RuntimeWarning("AutoDiff Not valid")
      return

    x_var = self.AddParameter(x_list)
    l_var = self.AddObservation(l_list)
    x_slc = [v.place for v in x_var]
    l_slc = [v.place for v in l_var]
    def g_flat(x,l):
      return g( *([x[slc] for slc in x_slc ] + [l[slc] for slc in l_slc ] ) )

    def g_jacobian(x,l):
      var = np.hstack([x[slc] for slc in x_slc ] + [l[slc] for slc in l_slc ] )
      J = var_jacobian(var)
      return np.split(J, xl_indices, axis=1)
    dim_res = len(res)
    r_slc = slice(self.constraint_offset, self.constraint_offset+dim_res )
    self.constraint_offset = r_slc.stop

    submat_x_id = [ self.csm_x.AddDenseSubMatrix(r_slc.start,
                                                 var_.place.start,
                                                 (dim_res, var_.data.shape[0]))
                                                   for var_ in x_var ]
    submat_l_id = [ self.csm_l.AddDenseSubMatrix(r_slc.start,
                                                 var_.place.start,
                                                 (dim_res, var_.data.shape[0]))
                                                   for var_ in l_var ]
    self.constraint_blocks.append(
      ConstraintBlock( g_flat, g_jacobian, r_slc, x_var, l_var, [submat_x_id, submat_l_id]))

  def AddConstraintUsingMD(self, g, g_jac, x_list, l_list):
    """
      1.add x (and l) to parameter_dict (and observation_dict)
      2.add record tuple(g, x_ids, l_ids) in constraint_blocks
    """
    x_sizes = [x.size for x in x_list]
    l_sizes = [l.size for l in l_list]

    res = g( *(x_list + l_list) )
    jac = g_jac( *(x_list + l_list) )

    inequal_size = [j.shape != (len(res), size) for j, size in zip(jac, x_sizes+l_sizes)]
    if len(jac) != len(x_list + l_list) or np.any( inequal_size ):
      RuntimeError("Jacobian Size Not fit")
      return
    if not ( np.isfinite(res).all() and  np.isfinite(jac).all() ):
      RuntimeWarning("Function Not valid")
      return

    x_var = self.AddParameter(x_list)
    l_var = self.AddObservation(l_list)
    x_slc = [v.place for v in x_var]
    l_slc = [v.place for v in l_var]
    def g_flat(x,l):
      return g( *([x[slc] for slc in x_slc ] + [l[slc] for slc in l_slc ] ) )

    def g_jacobian(x,l):
      return g_jac( *([x[slc] for slc in x_slc ] + [l[slc] for slc in l_slc ] ) )

    r_slc = slice(self.constraint_offset, self.constraint_offset+len(res) )
    self.constraint_offset = r_slc.stop
    nnz = len(res)*sum(x_sizes+l_sizes)

    self.constraint_blocks.append(
      ConstraintBlock( g_flat, g_jacobian, r_slc, x_var, l_var,nnz ))

  def UpdateLocalOffset(self):
    self.parameter_local_offset  = 0
    for var in self.parameter_dict.itervalues():
      self.parameter_local_offset += var.SetLocalPlace(self.parameter_local_offset)

    self.observation_local_offset = 0
    for var in self.observation_dict.itervalues():
      self.observation_local_offset += var.SetLocalPlace(self.observation_local_offset)

  def SetParameterization(self, array, parameterization):
    var = VariableBlock(array)
    if var in self.parameter_dict:
      self.parameter_dict[var].param = parameterization
    elif var in self.observation_dict:
      self.observation_dict[var].param = parameterization
    else:
      raise RuntimeWarning("Input variable not in the lists")

  def SetSigma(self, array, sigma):
    if not isinstance(array, list):
      array = [array]
    if not isinstance(sigma, list):
      sigma = [sigma]*len(array)
    elif len(sigma) != array:
      raise ValueError("number don't match")

    for ar, si in zip(array, sigma):
      var = ObservationBlock(ar)
      if var in self.observation_dict:
        self.observation_dict[var].sigma = si
      else:
        raise RuntimeWarning("Input variable not in the lists")

  def NumParameters(self):
    return self.parameter_offset

  def NumReducedParameters(self):
    return self.parameter_local_offset

  def NumObservations(self):
    return self.observation_offset

  def NumReducedObservations(self):
    return self.observation_local_offset

  def NumResiduals(self):
    return self.constraint_offset

  def CollectParameters(self):
    x = np.zeros(self.parameter_offset)
    for var in self.parameter_dict.itervalues():
      x[ var.place ] = var.data[:]
    return x

  def CollectObservations(self):
    l = np.zeros( self.observation_offset )
    for var in self.observation_dict.itervalues():
      l[ var.place ] = var.data[:]
    return l

  def CollectSigma(self):
    s = np.ones( self.observation_offset )
    for var in self.observation_dict.itervalues():
      if not var.sigma is None:
        s[ var.place ] = var.sigma
    return s

  def Plus_x(self, x, delta):
    for var in self.parameter_dict.itervalues():
      x[var.place] = var.param.Plus( x[var.place], delta[ var.place_local ] )
    return x

  def Plus_l(self, l, delta):
    for var in self.observation_dict.itervalues():
      l[var.place] = var.param.Plus( l[var.place], delta[ var.place_local ] )
    return l

  def SetObservationFromVector(self, new_l):
    for var in self.observation_dict.itervalues():
      var.data[:] = new_l[ var.place ]

  def SetParameterFromVector(self, new_x):
    for var in self.parameter_dict.itervalues():
      var.data[:] = new_x[ var.place ]



  def EvaluateConstraintJacobian(self, x=None, l=None):
    if x is None:
      x = self.CollectParameters()
    if l is None:
      l = self.CollectObservations()

    A = np.zeros( ( self.NumResiduals(), self.NumParameters() ) )
    B = np.zeros( ( self.NumResiduals(), self.NumObservations() ) )
    for cb in self.constraint_blocks:
      r_slc = cb.r_slc
      js = cb.g_jacobian(x, l)
      for var in cb.x_var:
        A[r_slc, var.place] = js.pop(0)
      for var in cb.l_var:
        B[r_slc, var.place] = js.pop(0)
    return A,B

  def EvaluateAndUpdateConstraintJacobiantest(self, x=None, l=None):
    if x is None:
      x = self.CollectParameters()
    if l is None:
      l = self.CollectObservations()

    Jx = self.csm_x.BuildSparseMatrix()
    Jl = self.csm_l.BuildSparseMatrix()

    for cb in self.constraint_blocks:
      js = cb.g_jacobian(x, l)
      for submat_id in cb.submat_id[0]:
        self.csm_x.SubMatrixWrite( submat_id, js.pop(0) )

      for submat_id in cb.submat_id[1]:
        self.csm_l.SubMatrixWrite( submat_id, js.pop(0) )
    return Jx,Jl

  def EvaluateConstraintJacobianWithParam(self, x=None, l=None):
    if x is None:
      x = self.CollectParameters()
    if l is None:
      l = self.CollectObservations()

    A = np.zeros( ( self.NumResiduals(), self.NumReducedParameters() ) )
    B = np.zeros( ( self.NumResiduals(), self.NumReducedObservations() ) )
    for cb in self.constraint_blocks:
      r_slc = cb.r_slc
      js = cb.g_jacobian(x, l)
      for var in cb.x_var:
        A[r_slc, var.place_local] = var.param.ToLocalJacobian( js.pop(0) )
      for var in cb.l_var:
        B[r_slc, var.place_local] = var.param.ToLocalJacobian( js.pop(0) )
    return A,B

  def EvaluateConstraintJacobianSparse(self, x=None, l=None):
    if x is None:
      x = self.CollectParameters()
    if l is None:
      l = self.CollectObservations()

    A = scipy.sparse.lil_matrix( ( self.NumResiduals(), self.NumParameters(), ) )
    B = scipy.sparse.lil_matrix( ( self.NumResiduals(), self.NumObservations(), ) )
    for cb in self.constraint_blocks:
      r_slc = cb.r_slc
      js = cb.g_jacobian(x, l)
      for x_slc in cb.x_slc:
        A[r_slc, x_slc] = js.pop(0)
      for l_slc in cb.l_slc:
        B[r_slc, l_slc] = js.pop(0)
    return A.tocsr(), B.tocsr()
#    return A, B

  def EvaluateConstraintResidual(self, x=None, l=None):
    if x is None:
      x = self.CollectParameters()
    if l is None:
      l = self.CollectObservations()
    res = np.zeros(self.NumResiduals())
    for cb in self.constraint_blocks:
      res[cb.r_slc] = cb.g_flat(x, l)
    return res

  def EvaluateCorrectionJacobian(self, le):
    CTWC_inv = np.zeros( (self.NumReducedObservations(), self.NumReducedObservations()) )
    CTWC_inv_CTWe = np.empty(self.NumReducedObservations())

    for var in self.observation_dict.itervalues():
      place, place_local = var.place, var.place_local
      CTWC_inv[place_local, place_local],  CTWC_inv_CTWe[place_local] = var.ComputeCTWCInvAndE(le[ place ])
    return CTWC_inv, CTWC_inv_CTWe

  def EvaluateCorrectionJacobianRawSparse(self, le):
    CTWC = scipy.sparse.lil_matrix( (self.NumReducedObservations(), self.NumReducedObservations()) )
    CTWe = np.empty(self.NumReducedObservations())

    for var in self.observation_dict.itervalues():
      place, place_local = var.place, var.place_local
      CTWC[place_local, place_local],  CTWe[place_local] = var.ComputeCTWCAndCTWE(le[ place ])
    return CTWC.todia(), CTWe

  def EvaluateObjective(self, correction, weight=None):
    if weight is None:
      weight = 1.0 / self.CollectSigma()
    return np.sum( weight * correction**2  )

  def SolveGaussEliminateDense(self, x_init=None, l_init=None, max_it=30, Tx=1e-5):
    s = self.CollectSigma()
    s_inv = 1.0 / s
    Sigma =  np.diag(s) #scipy.sparse.diags( s, offsets=0, format="coo" )

    l0  = self.CollectObservations()
    x   = self.CollectParameters()  if x_init is None else x_init.copy()
    l   = l0.copy()                 if l_init is None else l_init.copy()
    for it in range(max_it):
        le    = l - l0
        self.variance_factor = self.EvaluateObjective(le, s_inv)
        print it, self.variance_factor

        A, B = self.EvaluateConstraintJacobian(x, l)
        err  = self.EvaluateConstraintResidual(x, l)
        W    = np.linalg.inv( B.dot(Sigma).dot(B.T) )
        Cg   = B.dot(le) - err
        ATW  = A.T.dot(W)
        ATWA = ATW.dot(A)
        dx   = np.linalg.solve(ATWA,  ATW.dot(Cg))
        lag  = W.dot( A.dot(dx) - Cg )
        dl   = -Sigma.dot( B.T.dot(lag) ) - le
        x    += dx
        l    += dl
        if np.abs( dx*np.sqrt( np.diag(ATWA) ) ).max() < Tx:
          break
    return x, l - l0

  def SolveGaussEliminateDenseWithParam(self, x_init=None, l_init=None, max_it=30, Tx=1e-5):
    self.UpdateLocalOffset()
    s = self.CollectSigma()
    s_inv = 1.0 / s
    l0  = self.CollectObservations()
    x   = self.CollectParameters()  if x_init is None else x_init.copy()
    l   = l0.copy()                 if l_init is None else l_init.copy()
    for it in range(max_it):
        le    = l - l0
        self.variance_factor = self.EvaluateObjective(le, s_inv)
        print it, self.variance_factor

        A, B = self.EvaluateConstraintJacobianWithParam(x, l)
        CTWC_inv, CTWC_inv_CTWe = self.EvaluateCorrectionJacobian(le)
        g  = self.EvaluateConstraintResidual(x, l)
        Cg   =  B.dot(CTWC_inv_CTWe) - g

        W    = np.linalg.inv( B.dot(CTWC_inv).dot(B.T) )
        ATW  = A.T.dot(W)
        ATWA = ATW.dot(A)
        dx   = np.linalg.solve(ATWA,  ATW.dot(Cg))
        lag  = W.dot( A.dot(dx) - Cg )
        dl   = -( CTWC_inv.dot( B.T.dot(lag) ) + CTWC_inv_CTWe )
        x    = self.Plus_x(x, dx)
        l    = self.Plus_l(l, dl)
        if np.abs( dx*np.sqrt( np.diag(ATWA) ) ).max() < Tx:
          break
    return x, l - l0

  def SolveFullMatrixDense(self, x=None, l=None, maxit=10):
    dim_x = self.NumParameters()
    dim_l = self.NumObservations()
    dim_r = self.NumResiduals()
    offset =  np.cumsum([0, dim_x, dim_l, dim_r])
    segment_x = slice(offset[0], offset[1])
    segment_l = slice(offset[1], offset[2])
    segment_r = slice(offset[2], offset[3])
    r_offset = offset[2]
    l_offset = offset[1]
    dim_total = offset[-1]

    weight = 1.0 / self.CollectSigma()
    W = scipy.sparse.diags(weight, 0)

    l0 = self.CollectObservations()
    if x is None:
      x  = self.CollectParameters()
    if l is None:
      l   = l0.copy()
    e = l - l0
    self.variance_factor=0

    b = np.zeros( (dim_total,) )
    A = np.zeros( (dim_total, dim_total,) )
    for it in range(maxit):
      A[segment_l, segment_l] = W.A
      for cb in self.constraint_blocks:
        r_slc = slice(cb.r_slc.start + r_offset, cb.r_slc.stop + r_offset )
        js = cb.g_jacobian(x, l)
        for var in cb.x_var:
          M = js.pop(0)
          A[r_slc, var.place] = M
          A[var.place, r_slc] = M.T
        for var in cb.l_var:
          l_slc = slice(var.place.start + l_offset, var.place.stop + l_offset )
          M = js.pop(0)
          A[r_slc, l_slc] = M
          A[l_slc, r_slc] = M.T

      g = self.EvaluateConstraintResidual(x, l)
      b[segment_l] = -W.dot(e)
      b[segment_r] = -g

      s = scipy.linalg.solve(A, b)
      x += s[segment_x]
      l += s[segment_l]
      e = l - l0
      cost = self.EvaluateObjective(e, weight)
      if np.abs(cost - self.variance_factor)<1e-6:
        break
      print it, cost
      self.variance_factor = cost
    return x, e

  def SolveFullMatrixDenseWithParam(self, x=None, l=None, maxit=10):
    self.UpdateLocalOffset()
    dim_x = self.NumReducedParameters()
    dim_l = self.NumReducedObservations()
    dim_r = self.NumResiduals()
    offset =  np.cumsum([0, dim_x, dim_l, dim_r])
    segment_x = slice(offset[0], offset[1])
    segment_l = slice(offset[1], offset[2])
    segment_r = slice(offset[2], offset[3])
    r_offset = offset[2]
    l_offset = offset[1]
    dim_total = offset[-1]

    weight = 1.0 / self.CollectSigma()

    l0 = self.CollectObservations()
    if x is None:
      x  = self.CollectParameters()
    if l is None:
      l   = l0.copy()
    e = l - l0

    b = np.zeros( (dim_total,) )
    A = np.zeros( (dim_total, dim_total,) )
    for it in range(maxit):
      CTWC, CTWe = self.EvaluateCorrectionJacobianRawSparse(e)
      A[segment_l, segment_l] = CTWC.A
      for cb in self.constraint_blocks:
        r_slc = slice(cb.r_slc.start + r_offset, cb.r_slc.stop + r_offset )
        js = cb.g_jacobian(x, l)
        for var in cb.x_var:
          M = var.param.ToLocalJacobian( js.pop(0) )
          A[r_slc, var.place_local] = M
          A[var.place_local, r_slc] = M.T
        for var in cb.l_var:
          l_slc = slice(var.place_local.start + l_offset, var.place_local.stop + l_offset )
          M = var.param.ToLocalJacobian( js.pop(0) )
          A[r_slc, l_slc] = M
          A[l_slc, r_slc] = M.T

      g = self.EvaluateConstraintResidual(x, l)
      b[segment_l] = -CTWe
      b[segment_r] = -g

      s = scipy.linalg.solve(A, b)
      x = self.Plus_x(x, s[segment_x])
      l = self.Plus_l(l, s[segment_l])
      e = l - l0
      cost = self.EvaluateObjective(e, weight)
      if np.abs(cost - self.variance_factor)<1e-6:
        break
      print it, cost
      self.variance_factor = cost
    return x, e

#%%
if __name__ == '__main__':

  test_CreateSparseMatrix()

  def EqualityConstraint(a,b):
    return a-b
  def EqualityConstraintJac(a,b):
    return np.eye(len(a)), -np.eye(len(b))
  np.random.seed(0)
  dim_x = 3
  a = np.ones(dim_x)

  num_l = 100
  sigma = 0.02
  s = np.full(dim_x, sigma**2)

  bs = [ a + sigma * np.random.randn(3) for i in range(num_l)]
  problem = GaussHelmertProblem()
  for i in range(num_l):
    problem.AddConstraintUsingAD(EqualityConstraint,
                                 [ a ],
                                 [ bs[i] ])

  Jx,Jl = problem.EvaluateAndUpdateConstraintJacobiantest()
  Jxd,Jld = problem.EvaluateConstraintJacobian()
  assert_array_equal(Jx.A, Jxd)
  assert_array_equal(Jl.A, Jld)
