# cython: nonecheck=False
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: language_level=3
# cython: infer_types=False
# cython: embedsignature=True
# cython: initializedcheck=False

import numpy as np
cimport numpy as np


cdef void pocket_real_dft(
    double* in_reals_arr, double complex* out_comps_arr, size_t n_pts) nogil


cpdef np.ndarray get_pocket_real_dft(double[:] in_reals_arr)
