# cython: nonecheck=False
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: language_level=3
# cython: infer_types=False
# cython: embedsignature=True

# Compile this by removing the underscores from the beginning of names.
# Have dfti cimport from this.
# Take the compiled binary (.pyd for windows), paste it here and add
# underscores to the .pyx and .pyxbld again.
# All this to use the intel mkl library.


cdef void mkl_real_dft(
    double *in_reals_arr, double complex *out_comps_arr, long n_pts) nogil
