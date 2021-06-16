# cython: nonecheck=False
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: language_level=3
# cython: infer_types=False
# cython: embedsignature=True

import numpy as np
cimport numpy as np

cdef extern from "../opts/data_depths.h" nogil:
    cdef:
        void quick_sort(
            double *arr, long long first_index, long long last_index)


DT_D_NP = np.float64
DT_UL_NP = np.int32


cdef void rank_sorted_arr(
        const DT_D[::1] sorted_vals_arr, 
              DT_D[::1] ranks_arr, 
        const DT_UL rank_type) nogil except +:

    """
    Find ranks of a sorted array. Ascending values.
    This function allows for repeating values in the input.

    rank_type == 1: min
    rank_type == 2: max
    rank_type == any other value: mean
    Not the most efficient.
    """
 
    cdef:
        Py_ssize_t i, j, n_vals = sorted_vals_arr.shape[0], ctr
        DT_D rank

    i = 0
    while i < n_vals:

        ctr = 0
        for j in range(i, n_vals - 1):
            if sorted_vals_arr[j] == sorted_vals_arr[j + 1]:
                ctr += 1

            else:
                break

        rank = i + 1

        if ctr:
            if rank_type == 1:  # Minimum.
                pass

            elif rank_type == 2:  # Maximum.
                rank += ctr

            else:  # Mean.
                rank += ctr * 0.5

            for j in range(ctr + 1):
#                 with gil: print(i + j, ctr, rank, sorted_vals_arr[i + j])
                ranks_arr[i + j] = rank

        else:
#             with gil: print(i, ctr, rank, sorted_vals_arr[i])
            ranks_arr[i] = rank

        i += ctr + 1

    return


cdef void sort_arr(
        const DT_D[::1] vals_arr, DT_D[::1] sorted_arr) nogil except +:

    """
    Copy entries from vals_arr in to sorted_arr and then sort the sorted_arr.
    Both should have the same length and a size greater than 1.
    No NANs of course.
    """

    cdef:
        Py_ssize_t i, n_vals = vals_arr.shape[0]

    for i in range(n_vals):
        sorted_arr[i] = vals_arr[i]

    quick_sort(&sorted_arr[0], 0, n_vals - 1)
    return


cdef void get_sim_probs_in_ref(
        const DT_D[::1] ref_vals_sort, 
        const DT_D[::1] ref_probs_sort, 
        const DT_D[::1] sim_vals_sort,
              DT_D[::1] sim_probs_sort) nogil except +:

    """
    All input array have the same shape.
    All entries in sim_probs_sort are set to NAN.
    Undefined behavior if input array size is less than 2.
    First order extrapolation if sim_vals out of bounds of ref.
    ref_vals_sort, ref_probs_sort and sim_vals_sort must sorted.
    """

    cdef:
        DT_D app_zero = 1e-15

        Py_ssize_t i, i1, i2, j, min_j
        Py_ssize_t n_vals = ref_vals_sort.shape[0]

        DT_D x, x1, x2, p1, p2

    for i in range(n_vals):
        sim_probs_sort[i] = NAN

    min_j = 0  # To avoid testing against values that have been tested already.
    for i in range(n_vals):
        x = sim_vals_sort[i]

        if x < ref_vals_sort[0]:
            i1 = 0
            i2 = 1

        elif x > ref_vals_sort[n_vals - 1]:
            i1 = n_vals - 2
            i2 = n_vals - 1

        else:
            for j in range(min_j, n_vals - 1):

                # Double equals because similar values exist in reference.
                if ref_vals_sort[j] <= x <= ref_vals_sort[j + 1]:
                    i1 = j
                    i2 = j + 1

                    min_j = max(min_j, j - 1)

                    break

        x1 = ref_vals_sort[i1]
        x2 = ref_vals_sort[i2]

        p1 = ref_probs_sort[i1]
        p2 = ref_probs_sort[i2]

        if -app_zero <= (x1 - x2) <= +app_zero:
            sim_probs_sort[i] = p1

        else:
            sim_probs_sort[i] = p1 + ((p2 - p1) * ((x - x1) / (x2 - x1)))

    return


def get_sim_probs_in_ref_cy(
        const DT_D[::1] ref_vals_sort, 
        const DT_D[::1] ref_probs_sort, 
        const DT_D[::1] sim_vals_sort):

    cdef:
        DT_D[::1] sim_probs_sort

    sim_probs_sort = np.empty(ref_vals_sort.shape[0], dtype=DT_D_NP)

    assert (ref_vals_sort.shape[0] == 
            ref_probs_sort.shape[0] == 
            sim_vals_sort.shape[0])

    get_sim_probs_in_ref(
        ref_vals_sort, ref_probs_sort, sim_vals_sort, sim_probs_sort)

    return np.asarray(sim_probs_sort)


def rank_sorted_arr_cy(
        const DT_D[::1] sorted_vals_arr, 
              DT_UL rank_type):

    cdef:
        DT_D[::1] ranks_arr

    ranks_arr = np.empty(sorted_vals_arr.shape[0], dtype=DT_D_NP)

    rank_sorted_arr(sorted_vals_arr, ranks_arr, rank_type)

    return np.asarray(ranks_arr)
