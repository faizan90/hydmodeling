# cython: nonecheck=False
# cython: boundscheck=True
# cython: wraparound=False
# cython: cdivision=True
# cython: language_level=3
# cython: infer_types=False
# cython: embedsignature=True

from ..miscs.dtypes cimport DT_D, DT_UL, DT_ULL
from .data_depths cimport depth_ftn, pre_depth, post_depth


cdef extern from "data_depths.h" nogil:
    cdef:
        void depth_ftn_c(
                const double *ref,
                const double *test,
                const double *uvecs,
                      double *dot_ref,
                      double *dot_test,
                      double *dot_test_sort,
                      long *temp_mins,
                      long *mins,
                      long *depths_arr,
                const long n_ref,
                const long n_test,
                const long n_uvecs,
                const long n_dims,
                const long n_cpus)

        void pre_depth_c(
            const double *ref,
            const double *uvecs,
                  double *dot_ref_sort,
            const long n_ref,
            const long n_uvecs,
            const long n_dims,
            const long n_cpus)

        void post_depth_c(
                const double *test,
                const double *uvecs,
                const double *dot_ref_sort,
                      double *dot_test,
                      double *dot_test_sort,
                      long *temp_mins,
                      long *mins,
                      long *depths_arr,
                const long n_ref,
                const long n_test,
                const long n_uvecs,
                const long n_dims,
                const long n_cpus)


cpdef depth_ftn_cy(
        const DT_D[:, ::1] ref,
        const DT_D[:, ::1] test,
        const DT_D[:, ::1] uvecs,
              DT_D[:, ::1] dot_ref,
              DT_D[:, ::1] dot_test,
              DT_D[:, ::1] dot_test_sort,
              DT_UL[:, ::1] temp_mins,
              DT_UL[:, ::1] mins,
              DT_UL[::1] depths_arr,
        const DT_UL n_cpus,
        const DT_UL use_c):

    cdef:
        DT_UL n_ref = ref.shape[0]
        DT_UL n_dims = ref.shape[1]
        DT_UL n_uvecs = uvecs.shape[0]
        DT_UL n_test = test.shape[0]

    if use_c:
        depth_ftn_c(
            &ref[0, 0],
            &test[0, 0],
            &uvecs[0, 0],
            &dot_ref[0, 0],
            &dot_test[0, 0],
            &dot_test_sort[0, 0],
            &temp_mins[0, 0],
            &mins[0, 0],
            &depths_arr[0],
            n_ref,
            n_test,
            n_uvecs,
            n_dims,
            n_cpus)

    else:
        depth_ftn(
            ref,
            test,
            uvecs,
            dot_ref,
            dot_test,
            dot_test_sort,
            temp_mins,
            mins,
            depths_arr,
            n_ref,
            n_ref,
            n_cpus)

    return


cpdef pre_depth_cy(
        const DT_D[:, ::1] ref,
        const DT_D[:, ::1] uvecs,
              DT_D[:, ::1] dot_ref_sort,
        const DT_UL n_cpus,
        const DT_UL use_c):

    cdef:
        DT_UL n_ref = ref.shape[0]
        DT_UL n_dims = ref.shape[1]
        DT_UL n_uvecs = uvecs.shape[0] 

    if use_c:
        pre_depth_c(
            &ref[0, 0],
            &uvecs[0, 0],
            &dot_ref_sort[0, 0],
            n_ref,
            n_uvecs,
            n_dims,
            n_cpus)

    else:
        pre_depth(ref, uvecs, dot_ref_sort, n_ref, n_cpus)

    return


cpdef void post_depth_cy(
        const DT_D[:, ::1] test,
        const DT_D[:, ::1] uvecs,
        const DT_D[:, ::1] dot_ref_sort,  # should be sorted
              DT_D[:, ::1] dot_test,
              DT_D[:, ::1] dot_test_sort,
              DT_UL[:, ::1] temp_mins,
              DT_UL[:, ::1] mins,
              DT_UL[::1] depths_arr,
        const DT_UL n_cpus,
        const DT_UL use_c):


    cdef:
        DT_UL n_ref = dot_ref_sort.shape[1]
        DT_UL n_dims = test.shape[1]
        DT_UL n_uvecs = uvecs.shape[0]
        DT_UL n_test = test.shape[0]

    if use_c:
        post_depth_c(
            &test[0, 0],
            &uvecs[0, 0],
            &dot_ref_sort[0, 0],
            &dot_test[0, 0],
            &dot_test_sort[0, 0],
            &temp_mins[0, 0],
            &mins[0, 0],
            &depths_arr[0],
            n_ref,
            n_test,
            n_uvecs,
            n_dims,
            n_cpus)

    else:
        post_depth(
            test,
            uvecs,
            dot_ref_sort,
            dot_test,
            dot_test_sort,
            temp_mins,#
            mins,
            depths_arr,
            n_ref,
            n_cpus)

    return
