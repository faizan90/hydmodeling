# cython: nonecheck=False
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: language_level=3
# cython: infer_types=False
# cython: embedsignature=True

import numpy as np
cimport numpy as np

from ..miscs.dtypes cimport fc_i, pwp_i

cdef DT_D NaN = np.NaN
cdef DT_D INF = np.inf


cdef extern from "cmath":
    bint isnan(DT_D x) nogil


cdef extern from "../miscs/rand_gen_mp.h" nogil:
    cdef:
        DT_D rand_c()
        void warm_up()  # call this everytime

warm_up()


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

        void depth_ftn_c(
            const double *ref,
            const double *test,
            const double *uvecs,
                  double *dot_ref,
                  double *dot_test,
                  double *dot_test_sort,
                  long *temp_mins,
                  long *mins,
            const long n_ref,
            const long n_test,
            const long n_uvecs,
            const long n_dims,
            const long n_cpus)


cdef void get_new_chull_vecs(
          DT_UL[::1] depths_arr,
          DT_UL[:, ::1] temp_mins,
          DT_UL[:, ::1] mins,

    const DT_D[::1] pre_obj_vals,
          DT_D[::1] sort_obj_vals,

          DT_D[:, ::1] acc_vecs,
    const DT_D[:, ::1] prm_vecs,
    const DT_D[:, ::1] uvecs,
          DT_D[:, ::1] dot_ref,
          DT_D[:, ::1] dot_test,
          DT_D[:, ::1] dot_test_sort,
          DT_D[:, ::1] chull_vecs,

    const DT_UL n_cpus,
          DT_UL *chull_vecs_ctr,
    ) nogil except +:

    cdef:
        Py_ssize_t i, j, ctr
        DT_UL prm_vec_rank
        DT_UL n_prm_vecs = prm_vecs.shape[0]
        DT_UL n_prms = prm_vecs.shape[1]
        DT_UL n_uvecs = uvecs.shape[0]
        DT_UL n_acc_vecs = acc_vecs.shape[0]

    for i in range(n_prm_vecs):
        sort_obj_vals[i] = pre_obj_vals[i]
    quick_sort(&sort_obj_vals[0], 0, n_prm_vecs - 1)

    for i in range(n_prm_vecs):
        prm_vec_rank = searchsorted(
            &sort_obj_vals[0], pre_obj_vals[i], n_prm_vecs)

        if prm_vec_rank >= n_acc_vecs:
            continue

        for j in range(n_prms):
            acc_vecs[prm_vec_rank, j] = prm_vecs[i, j]

    for j in range(n_cpus):
        for i in range(n_acc_vecs):
            temp_mins[j, i] = n_acc_vecs
            mins[j, i] = n_acc_vecs

    depth_ftn_c(
        &acc_vecs[0, 0],
        &acc_vecs[0, 0],
        &uvecs[0, 0],
        &dot_ref[0, 0],
        &dot_test[0, 0],
        &dot_test_sort[0, 0],
        &temp_mins[0, 0],
        &mins[0, 0],
        n_acc_vecs,
        n_acc_vecs,
        n_uvecs,
        n_prms,
        n_cpus)

    ctr = 0
    for i in range(n_acc_vecs):
        depths_arr[i] = mins[0, i]
        for j in range(1, n_cpus):
            if mins[j, i] >= depths_arr[i]:
                continue

            depths_arr[i] = mins[j, i]

        if depths_arr[i] != 1:
            continue

        for j in range(n_prms):
            chull_vecs[ctr, j] = acc_vecs[i, j]
        ctr += 1
    chull_vecs_ctr[0] = ctr

    # just to be sure
    for i in range(chull_vecs_ctr[0], n_acc_vecs):
        for j in range(n_prms):
            chull_vecs[i, j] = NaN
    return


cdef void adjust_rope_bds(
    const DT_D[:, ::1] chull_vecs,
          DT_D[:, ::1] rope_bds_dfs,
    const DT_UL chull_vecs_ctr,
    ) nogil except +:

    cdef:
        Py_ssize_t i, j
        DT_UL n_prms = chull_vecs.shape[1]
        DT_D min_prm_val, max_prm_val

    for j in range(n_prms):
        min_prm_val = +INF
        max_prm_val = -INF
        for i in range(chull_vecs_ctr):
            if chull_vecs[i, j] < min_prm_val:
                min_prm_val = chull_vecs[i, j]

            if chull_vecs[i, j] > max_prm_val:
                max_prm_val = chull_vecs[i, j]

        rope_bds_dfs[j, 0] = min_prm_val
        rope_bds_dfs[j, 1] = max_prm_val - min_prm_val
    return


cdef void gen_vecs_in_chull(
          DT_UL[::1] depths_arr,

    const DT_UL[:, ::1] prms_flags,
    const DT_UL[:, ::1] prms_span_idxs,
          DT_UL[:, ::1] temp_mins,
          DT_UL[:, ::1] mins,

    const DT_UL[:, :, ::1] prms_idxs,

    const DT_D[:, ::1] rope_bds_dfs,
    const DT_D[:, ::1] chull_vecs,
    const DT_D[:, ::1] uvecs,

          DT_D[:, ::1] temp_rope_prm_vecs,
          DT_D[:, ::1] prm_vecs,
          DT_D[:, ::1] dot_ref,
          DT_D[:, ::1] dot_test,
          DT_D[:, ::1] dot_test_sort,

    const DT_UL n_hbv_prms,
    const DT_UL chull_vecs_ctr,
    const DT_UL n_cpus,
    ) nogil except +:

    cdef:
        Py_ssize_t i, j, k, m, ctr, cfc_i, cpwp_i
        DT_UL n_prm_vecs = prm_vecs.shape[0]
        DT_UL n_prms = prm_vecs.shape[1]
        DT_UL n_temp_rope_prm_vecs = temp_rope_prm_vecs.shape[0]
        DT_UL n_uvecs = uvecs.shape[0]

    ctr = 0
    while ctr < n_prm_vecs:
        for i in range(n_temp_rope_prm_vecs):
            for j in range(n_prms):
                temp_rope_prm_vecs[i, j] = (
                    rope_bds_dfs[j, 0] + (rand_c() * rope_bds_dfs[j, 1]))

            # check constraints
            for j in range(prms_span_idxs[fc_i, 1] - prms_span_idxs[fc_i, 0]):

                cfc_i = prms_span_idxs[fc_i, 0] + j
                cpwp_i = prms_span_idxs[pwp_i, 0] + j

                # TODO: guarantee that it will break eventually
                while (temp_rope_prm_vecs[i, cpwp_i] >
                       temp_rope_prm_vecs[i, cfc_i]):
                    temp_rope_prm_vecs[i, cpwp_i] = rope_bds_dfs[cpwp_i, 0] + (
                        rand_c() * rope_bds_dfs[cpwp_i, 1])

            for j in range(n_hbv_prms):
                for m in range(3, 6):
                    if not prms_flags[j, m]:
                        continue

                    k = prms_idxs[j, m, 0]

                    # TODO: verify this
                    # TODO: guarantee that it will break eventually
                    while (temp_rope_prm_vecs[i, k + 1] <
                           temp_rope_prm_vecs[i, k]):

                        temp_rope_prm_vecs[i, k] = rope_bds_dfs[k, 0] + (
                            rand_c() * rope_bds_dfs[k, 1])

        for j in range(n_cpus):
            for i in range(n_temp_rope_prm_vecs):
                temp_mins[j, i] = n_temp_rope_prm_vecs
                mins[j, i] = n_temp_rope_prm_vecs

        depth_ftn_c(
            &chull_vecs[0, 0],
            &temp_rope_prm_vecs[0, 0],
            &uvecs[0, 0],
            &dot_ref[0, 0],
            &dot_test[0, 0],
            &dot_test_sort[0, 0],
            &temp_mins[0, 0],
            &mins[0, 0],
            chull_vecs_ctr,
            n_temp_rope_prm_vecs,
            n_uvecs,
            n_prms,
            n_cpus)

        for i in range(n_temp_rope_prm_vecs):
            depths_arr[i] = mins[0, i]
            for j in range(1, n_cpus):
                if mins[j, i] >= depths_arr[i]:
                    continue

                depths_arr[i] = mins[j, i]

            if not depths_arr[i]:
                continue

            if ctr >= n_prm_vecs:
                break

            for j in range(n_prms):
                prm_vecs[ctr, j] = temp_rope_prm_vecs[i, j]
            ctr += 1
    return


cdef void pre_rope(
    const DT_UL[:, ::1] prms_flags,
    const DT_UL[:, ::1] prms_span_idxs,
          DT_UL[::1] depths_arr,
          DT_UL[:, ::1] temp_mins,
          DT_UL[:, ::1] mins,

    const DT_UL[:, :, ::1] prms_idxs,

    const DT_D[::1] pre_obj_vals,
          DT_D[::1] sort_obj_vals,

          DT_D[:, ::1] prm_vecs,
    const DT_D[:, ::1] uvecs,
          DT_D[:, ::1] temp_rope_prm_vecs,
          DT_D[:, ::1] acc_vecs,
          DT_D[:, ::1] rope_bds_dfs,
          DT_D[:, ::1] dot_ref,
          DT_D[:, ::1] dot_test,
          DT_D[:, ::1] dot_test_sort,
          DT_D[:, ::1] chull_vecs,

    const DT_UL n_hbv_prms,
    const DT_UL n_cpus,
          DT_UL *chull_vecs_ctr,
          DT_UL *cont_iter,
    ) nogil except +:

    get_new_chull_vecs(
        depths_arr,
        temp_mins,
        mins,
        pre_obj_vals,
        sort_obj_vals,
        acc_vecs,
        prm_vecs,
        uvecs,
        dot_ref,
        dot_test,
        dot_test_sort,
        chull_vecs,
        n_cpus,
        chull_vecs_ctr)

    with gil: assert chull_vecs_ctr[0] >= 3, chull_vecs_ctr[0]

    adjust_rope_bds(
        chull_vecs,
        rope_bds_dfs,
        chull_vecs_ctr[0])

    gen_vecs_in_chull(
        depths_arr,
        prms_flags,
        prms_span_idxs,
        temp_mins,
        mins,
        prms_idxs,
        rope_bds_dfs,
        chull_vecs,
        uvecs,
        temp_rope_prm_vecs,
        prm_vecs,
        dot_ref,
        dot_test,
        dot_test_sort,
        n_hbv_prms,
        chull_vecs_ctr[0],
        n_cpus)

    cont_iter[0] += 1
    return


cdef void post_rope(
    const DT_D[::1] pre_obj_vals,
          DT_D[::1] best_prm_vec,

    const DT_D[:, ::1] prm_vecs,

    const DT_UL max_iters,
    const DT_UL max_cont_iters,
          DT_UL *iter_curr,
          DT_UL *last_succ_i,
          DT_UL *n_succ,
          DT_UL *cont_iter,
          DT_UL *cont_opt_flag,

          DT_D *fval_pre_global,
    ) nogil except +:

    cdef:
        Py_ssize_t j, k

        DT_D fval_pre

        DT_UL n_prms = prm_vecs.shape[1]
        DT_UL n_prm_vecs = prm_vecs.shape[0]

    for j in range(n_prm_vecs):
        fval_pre = pre_obj_vals[j]

        if isnan(fval_pre):
            with gil: raise RuntimeError('fval_pre is Nan!')

        # check for global minimum and best vector
        if fval_pre >= fval_pre_global[0]:
            continue

        for k in range(n_prms):
            best_prm_vec[k] = prm_vecs[j, k]

        fval_pre_global[0] = fval_pre
        last_succ_i[0] = iter_curr[0]
        n_succ[0] += 1
        cont_iter[0] = 0

    iter_curr[0] += 1

    if cont_opt_flag[0] and (iter_curr[0] >= max_iters):
        with gil: print('***Max iterations reached!***')
        cont_opt_flag[0] = 0

    if cont_opt_flag[0] and (cont_iter[0] > max_cont_iters):
        with gil: print('***max_cont_iters reached!***')
        cont_opt_flag[0] = 0
    return