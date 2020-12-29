# cython: nonecheck=False
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: language_level=3
# cython: infer_types=False
# cython: embedsignature=True

from ..miscs.dtypes cimport DT_D, DT_UL, DT_ULL, NAN


cdef void rank_sorted_arr(
        const DT_D[::1] sorted_vals_arr, 
              DT_D[::1] ranks_arr, 
        const DT_UL rank_type) nogil except +


cdef void sort_arr(
        const DT_D[::1] vals_arr, DT_D[::1] sorted_arr) nogil except +


cdef void get_sim_probs_in_ref(
        const DT_D[::1] ref_vals_sort, 
        const DT_D[::1] ref_probs_sort, 
        const DT_D[::1] sim_vals_sort,
              DT_D[::1] sim_probs_sort) nogil except +
