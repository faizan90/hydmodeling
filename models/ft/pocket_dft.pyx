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


cdef extern from "pocket_dft.h" nogil:
    cdef:
        void _pocket_real_dft(double *in_reals_arr, size_t n_pts)


# To match the same signature as mkl_dft, n_pts_adj is computed inside.
cdef void pocket_real_dft(
        double* in_reals_arr, 
        double complex* out_comps_arr, 
        size_t n_pts) nogil:

    cdef:
        size_t i = 0
        size_t n_pts_adj = n_pts

    # Make number of steps even.
    if n_pts % 2:
        n_pts_adj += 1

    else:
        n_pts_adj += 2

    for i in range(0, n_pts - 1, 2):
        out_comps_arr[(i // 2)].imag = in_reals_arr[i]
        out_comps_arr[1 + (i//2)].real = in_reals_arr[i + 1]

    if n_pts % 2:
        i += 2
        out_comps_arr[(i//2)].imag = in_reals_arr[i]

    else:
        pass

    _pocket_real_dft(<double*> &out_comps_arr[0].imag, n_pts)

    out_comps_arr[0].real = out_comps_arr[0].imag
    out_comps_arr[0].imag = 0.0 

    # For even number of n_pts, the last imaginary value should be zero.
    if n_pts % 2:
        pass

    else:
        out_comps_arr[(n_pts_adj // 2) - 1].imag = 0.0

    return


cpdef np.ndarray get_pocket_real_dft(double[:] in_reals_arr):

    cdef:
        size_t n_pts = in_reals_arr.shape[0]
        # size_t i = 0, n_pts = in_reals_arr.shape[0]
        size_t n_pts_adj = n_pts

    # Make number of steps even.
    if n_pts % 2:
        n_pts_adj += 1
    
    else:
        n_pts_adj += 2

    cdef double complex[:] out_comps_arr = np.empty(
        n_pts_adj // 2, dtype=np.complex128)

    # cdef double complex[:] out_arr = np.empty(
    #     n_pts_adj // 2, dtype=np.complex128)
    #
    # for i in range(0, n_pts - 1, 2):
    #     out_arr[(i // 2)].imag = in_reals_arr[i]
    #     out_arr[1 + (i//2)].real = in_reals_arr[i + 1]
    #
    # if n_pts % 2:
    #     i += 2
    #     out_arr[(i//2)].imag = in_reals_arr[i]
    #
    # else:
    #     pass
    #
    # _pocket_real_dft(<double*> &out_arr[0].imag, n_pts)
    #
    # out_arr[0].real = out_arr[0].imag
    # out_arr[0].imag = 0.0 
    #
    # # For even number of n_pts, the last imaginary value should be zero.
    # if n_pts % 2:
    #     pass
    #
    # else:
    #     out_arr[out_arr.shape[0] - 1].imag = 0.0

    pocket_real_dft(&in_reals_arr[0], &out_comps_arr[0], n_pts)

    return np.asarray(out_comps_arr)
    # return np.asarray(out_arr)
