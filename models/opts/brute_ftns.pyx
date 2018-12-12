# cython: nonecheck=False
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: language_level=3
# cython: infer_types=False
# cython: embedsignature=True

from ..miscs.dtypes cimport fc_i, pwp_i

cdef extern from "cmath":
    bint isnan(DT_D x) nogil
    DT_D INFINITY


cdef void pre_brute(
              DT_UL[::1] last_idxs_vec,
              DT_UL[::1] use_prm_vec_flags,

        const DT_UL[:, ::1] prms_flags,
        const DT_UL[:, ::1] prms_span_idxs,

        const DT_UL[:, :, ::1] prms_idxs,

              DT_D[:, ::1] prm_vecs,

        const DT_ULL n_poss_combs,
        const DT_UL n_discretize,
        const DT_UL n_hbv_prms,
              DT_ULL *comb_ctr,
        ) except +:

    cdef:
        Py_ssize_t i, j, k, m, t_i

        DT_UL n_prm_vecs = prm_vecs.shape[0]
        DT_UL n_prms = prm_vecs.shape[1]

        DT_UL last_prm_idx = n_prms - 1
        DT_UL last_idx_val= n_discretize - 1

    for i in range(n_prm_vecs):
        use_prm_vec_flags[i] = 0

    for t_i in range(n_prm_vecs):
        last_idxs_vec[last_prm_idx] += 1

        for i in range(last_prm_idx, -1, -1):
            if last_idxs_vec[i] > last_idx_val:
                for j in range(i, n_prms):
                    last_idxs_vec[j] = 0

                last_idxs_vec[i - 1] += 1

        use_prm_vec_flags[t_i] = 1

        for i in range(n_prms):
            prm_vecs[t_i, i] = last_idxs_vec[i] / (<DT_D> last_idx_val)

        # check if pwp is ge than fc and adjust
        for i in range(prms_span_idxs[fc_i, 1] - prms_span_idxs[fc_i, 0]):

            if (prm_vecs[t_i, prms_span_idxs[pwp_i, 0] + i] <=
                prm_vecs[t_i, prms_span_idxs[fc_i, 0] + i]):
                continue

            use_prm_vec_flags[t_i] = 0
            break

        for i in range(n_hbv_prms):
            for m in range(3, 6):
                if not prms_flags[i, m]:
                    continue

                k = prms_idxs[i, m, 0]
                if prm_vecs[t_i, k + 1] >= prm_vecs[t_i, k]:
                    continue

                use_prm_vec_flags[t_i] = 0
                break

        comb_ctr[0] += 1

        if comb_ctr[0] >= n_poss_combs:
            break
    return


cdef void post_brute(
        const DT_UL[::1] use_prm_vec_flags,

        const DT_D[::1] curr_obj_vals,
              DT_D[::1] pre_obj_vals,
              DT_D[::1] best_prm_vec,
              DT_D[::1] iobj_vals,

              DT_D[:, ::1] prm_vecs,

        const DT_ULL n_poss_combs,
              DT_UL *iter_curr,
              DT_UL *cont_opt_flag,
              DT_ULL *comb_ctr,

              DT_D *fval_pre_global,
        ) nogil except +:

    cdef:
        Py_ssize_t j, k

        DT_UL n_prms = prm_vecs.shape[1]
        DT_UL n_prm_vecs = prm_vecs.shape[0]

        DT_D fval_pre, fval_curr, ddmv, iobj = INFINITY

    for j in range(n_prm_vecs):
        if not use_prm_vec_flags[j]:
            continue

        fval_pre = pre_obj_vals[j]
        fval_curr = curr_obj_vals[j]

        if isnan(fval_curr):
            with gil: raise RuntimeError('fval_curr is Nan!')

        if fval_curr < iobj:
            iobj = fval_curr

        if fval_curr >= fval_pre:
            continue

        pre_obj_vals[j] = fval_curr

        # check for global minimum and best vector
        if fval_curr >= fval_pre_global[0]:
            continue

        for k in range(n_prms):
            best_prm_vec[k] = prm_vecs[j, k]

        fval_pre_global[0] = fval_curr

    iobj_vals[iter_curr[0] + 1] = iobj

    iter_curr[0] += 1

    if comb_ctr[0] >= n_poss_combs:
        cont_opt_flag[0] = 0
    return