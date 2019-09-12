# cython: nonecheck=False
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: language_level=3
# cython: infer_types=False
# cython: embedsignature=True


cdef extern from "intel_dfti.h" nogil:
    cdef:
        void mkl_real_dft(
            double *in_reals_arr, double complex *out_comps_arr, long n_pts)


# cdef void do_mkl_dft(
#         double *in_reals_arr, double complex *out_comps_arr, long n_pts) nogil:
# 
#     mkl_real_dft(in_reals_arr, out_comps_arr, n_pts)
#     return