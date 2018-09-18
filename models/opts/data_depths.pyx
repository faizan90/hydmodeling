# cython: nonecheck=False
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: language_level=3
# cython: infer_types=False
# cython: embedsignature=True

from cython.parallel import prange, threadid


cdef extern from "data_depths.h" nogil:
    cdef:
        void quick_sort(
                double *arr,
                long first_index,
                long last_index)

        long searchsorted(
                const double *arr,
                const double value,
                const long arr_size)


cdef void depth_ftn(
        const DT_D[:, ::1] ref,
        const DT_D[:, ::1] test,
        const DT_D[:, ::1] uvecs,
              DT_D[:, ::1] dot_ref,
              DT_D[:, ::1] dot_test,
              DT_D[:, ::1] dot_test_sort,
              DT_UL[:, ::1] temp_mins,
              DT_UL[:, ::1] mins,
              DT_UL[::1] depths_arr,
        const DT_UL n_ref,
        const DT_UL n_test,
        const DT_UL n_cpus,
        ) nogil except +:

    cdef:
        Py_ssize_t i, j, k, idx

        DT_UL tid
        DT_UL n_dims = ref.shape[1]
        DT_UL n_uvecs = uvecs.shape[0]

        int even = (n_test % 2) == 0

        DT_D dy_med, inc_mult = (1 - (1e-10))

    for i in prange(
        n_uvecs, schedule='dynamic', nogil=True, num_threads=n_cpus):

        tid = threadid()

        for j in range(n_ref):
            dot_ref[tid, j] = 0.0

            for k in range(n_dims):
                dot_ref[tid, j] = dot_ref[tid, j] + (uvecs[i, k] * ref[j, k])

        for j in range(n_test):
            dot_test[tid, j] = 0.0

            for k in range(n_dims):
                dot_test[tid, j] = (
                    dot_test[tid, j] + (uvecs[i, k] * test[j, k]))

            dot_test_sort[tid, j] = dot_test[tid, j]

        quick_sort(&dot_ref[tid, 0], 0, n_ref - 1)

        quick_sort(&dot_test_sort[tid, 0], 0, n_test - 1)

        if even:
            dy_med = 0.5 * (
                dot_test_sort[tid, n_test / 2] + 
                dot_test_sort[tid, (n_test / 2) - 1])

        else:
            dy_med = dot_test_sort[tid, n_test / 2]

        for j in range(n_test):
            dot_test[tid, j] = dy_med + (
                (dot_test[tid, j] - dy_med) * inc_mult)

        for j in range(n_test):
            temp_mins[tid, j] = searchsorted(
                &dot_ref[tid, 0], dot_test[tid, j], n_ref)
 
        for j in range(n_test):
            idx = n_ref - temp_mins[tid, j]

            if idx < temp_mins[tid, j]:
                temp_mins[tid, j] = idx

            if temp_mins[tid, j] < mins[tid, j]:
                mins[tid, j] = temp_mins[tid, j]

    for i in range(n_test):
        depths_arr[i] = mins[0, i]

        for j in range(1, n_cpus):
            if mins[j, i] >= depths_arr[i]:
                continue

            depths_arr[i] = mins[j, i]

    return


cdef void pre_depth(
        const DT_D[:, ::1] ref,
        const DT_D[:, ::1] uvecs,
              DT_D[:, ::1] dot_ref_sort,
        const DT_UL n_ref,
        const DT_UL n_cpus,
        ) nogil except +:

    '''
    pre_depth and post_depth are cpu efficient versions of depth_ftn
    but they require much more memory.
    '''

    cdef:
        Py_ssize_t i, j, k
        DT_UL n_dims = ref.shape[1]
        DT_UL n_uvecs = uvecs.shape[0]

    for i in prange(
        n_uvecs, schedule='dynamic', nogil=True, num_threads=n_cpus):

        for j in range(n_ref):
            dot_ref_sort[i, j] = 0.0

            for k in range(n_dims):
                dot_ref_sort[i, j] = (
                    dot_ref_sort[i, j] + (uvecs[i, k] * ref[j, k]))

        quick_sort(&dot_ref_sort[i, 0], 0, n_ref - 1)

    return


cdef void post_depth(
        const DT_D[:, ::1] test,
        const DT_D[:, ::1] uvecs,
        const DT_D[:, ::1] dot_ref_sort,  # should be sorted
              DT_D[:, ::1] dot_test,
              DT_D[:, ::1] dot_test_sort,
              DT_UL[:, ::1] temp_mins,
              DT_UL[:, ::1] mins,
              DT_UL[::1] depths_arr,
        const DT_UL n_ref,
        const DT_UL n_cpus,
        ) nogil except +:

    '''
    pre_depth and post_depth are cpu efficient versions of depth_ftn
    but they require much more memory.
    '''

    cdef:
        Py_ssize_t i, j, k, idx
        DT_UL tid
        DT_UL n_dims = test.shape[1]
        DT_UL n_test = test.shape[0]
        DT_UL n_uvecs = uvecs.shape[0]

        int even = (n_test % 2) == 0

        DT_D dy_med, inc_mult = (1 - (1e-10))

    for i in prange(
        n_uvecs, schedule='dynamic', nogil=True, num_threads=n_cpus):

        tid = threadid()

        for j in range(n_test):
            dot_test[tid, j] = 0.0

            for k in range(n_dims):
                dot_test[tid, j] = (
                    dot_test[tid, j] + (uvecs[i, k] * test[j, k]))

            dot_test_sort[tid, j] = dot_test[tid, j]

        quick_sort(&dot_test_sort[tid, 0], 0, n_test - 1)

        if even:
            dy_med = 0.5 * (
                dot_test_sort[tid, n_test / 2] + 
                dot_test_sort[tid, (n_test / 2) - 1])

        else:
            dy_med = dot_test_sort[tid, n_test / 2]

        for j in range(n_test):
            dot_test[tid, j] = dy_med + (
                (dot_test[tid, j] - dy_med) * inc_mult)

        for j in range(n_test):
            temp_mins[tid, j] = searchsorted(
                &dot_ref_sort[i, 0], dot_test[tid, j], n_ref)
 
        for j in range(n_test):
            idx = n_ref - temp_mins[tid, j]

            if idx < temp_mins[tid, j]:
                temp_mins[tid, j] = idx

            if temp_mins[tid, j] < mins[tid, j]:
                mins[tid, j] = temp_mins[tid, j]

    for i in range(n_test):
        depths_arr[i] = mins[0, i]

        for j in range(1, n_cpus):
            if mins[j, i] >= depths_arr[i]:
                continue

            depths_arr[i] = mins[j, i]

    return
