# cython: nonecheck=False
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: language_level=3
# cython: infer_types=False
# cython: embedsignature=True

from cython.parallel import prange, threadid

from ..miscs.dtypes cimport fc_i, pwp_i
from ..miscs.misc_ftns cimport del_idx


cdef extern from "cmath" nogil:
    bint isnan(DT_D x)
    DT_D INFINITY


cdef extern from "../miscs/rand_gen_mp.h" nogil:
    cdef:
        DT_D rand_c()
        void warm_up()  # call this everytime
        DT_D rand_c_mp(DT_ULL *rnd_j)

warm_up()


cdef void pre_de(
        const DT_UL[::1] idx_rng,
              DT_UL[::1] r_r,

        const DT_UL[:, ::1] prms_flags,
        const DT_UL[:, ::1] prms_span_idxs,
              DT_UL[:, ::1] del_idx_rng,
              DT_UL[:, ::1] choice_arr,

        const DT_UL[:, :, ::1] prms_idxs,

              DT_ULL[::1] seeds_arr, # a warmed up one

        const DT_D[::1] mu_sc_fac_bds,
        const DT_D[::1] cr_cnst_bds,

              DT_D[:, ::1] prm_vecs,
              DT_D[:, ::1] v_j_g,
              DT_D[:, ::1] u_j_gs,

        const DT_UL n_hbv_prms,
        const DT_UL n_cpus,
              DT_UL *cont_iter,
        ) nogil except +:

    cdef:
        Py_ssize_t tid, i, k, m, ch_r_i, ch_r_j, ch_r_l
        DT_UL t_i

        DT_UL n_prm_vecs = prm_vecs.shape[0]
        DT_UL n_prm_vecs_ol = n_prm_vecs - 1
        DT_UL n_prms = prm_vecs.shape[1]

        DT_UL n_mu_samps = 3
        DT_UL r0, r1, r2

        DT_D mu_sc_fac, cr_cnst

    # randomize the mutation and recombination factors for every
    # generation
    mu_sc_fac = (
        mu_sc_fac_bds[0] + ((mu_sc_fac_bds[1] - mu_sc_fac_bds[0]) * rand_c()))

    cr_cnst = (
        cr_cnst_bds[0] + ((cr_cnst_bds[1] - cr_cnst_bds[0]) * rand_c()))

    for t_i in prange(
        n_prm_vecs, 
        schedule='static', 
        nogil=True, 
        num_threads=n_cpus):

        tid = threadid()

        # get inidicies except t_i
        del_idx(idx_rng, del_idx_rng[tid], t_i)

        # select indicies randomly from del_idx_rng
        ch_r_l = 1
        while ch_r_l:
            ch_r_l = 0

            for ch_r_i in range(n_mu_samps):
                k = <Py_ssize_t> (rand_c_mp(&seeds_arr[tid]) * n_prm_vecs_ol)

                choice_arr[tid, ch_r_i] = del_idx_rng[tid, k]

            for ch_r_i in range(n_mu_samps):
                for ch_r_j in range(ch_r_i + 1, n_mu_samps):
                    if (choice_arr[tid, ch_r_i] != choice_arr[tid, ch_r_j]):
                        continue

                    ch_r_l = 1
                    break

        r0 = choice_arr[tid, 0]
        r1 = choice_arr[tid, 1]
        r2 = choice_arr[tid, 2]

        # mutate
        for k in range(n_prms):
            v_j_g[tid, k] = prm_vecs[r0, k] + (
                    mu_sc_fac * (prm_vecs[r1, k] - prm_vecs[r2, k]))

        # keep parameters in bounds
        for k in range(n_prms):
            if ((v_j_g[tid, k] < 0) or (v_j_g[tid, k] > 1)):

                v_j_g[tid, k] = rand_c_mp(&seeds_arr[tid])

        # get an index randomly to have atleast one parameter from
        # the mutated vector
        r_r[tid] = <DT_UL> (rand_c_mp(&seeds_arr[tid]) * n_prms)

        for k in range(n_prms):
            if ((rand_c_mp(&seeds_arr[tid]) <= cr_cnst) or (k == r_r[tid])):
                u_j_gs[t_i, k] = v_j_g[tid, k]

            else:
                u_j_gs[t_i, k] = prm_vecs[t_i, k]

        # check if pwp is ge than fc and adjust
        for i in range(prms_span_idxs[fc_i, 1] - prms_span_idxs[fc_i, 0]):
            if (u_j_gs[t_i, prms_span_idxs[pwp_i, 0] + i] <
                u_j_gs[t_i, prms_span_idxs[fc_i, 0] + i]):

                continue

            u_j_gs[t_i, prms_span_idxs[pwp_i, 0] + i] = (
                0.99 *
                rand_c_mp(&seeds_arr[tid]) * 
                u_j_gs[t_i, prms_span_idxs[fc_i, 0] + i])

        for i in range(n_hbv_prms):
            for m in range(3, 6):
                if not prms_flags[i, m]:
                    continue

                k = prms_idxs[i, m, 0]
                if u_j_gs[t_i, k + 1] >= u_j_gs[t_i, k]:
                    continue

                u_j_gs[t_i, k] = (
                    0.99 * 
                    rand_c_mp(&seeds_arr[tid]) * 
                    u_j_gs[t_i, k + 1])

    cont_iter[0] += 1
    return


cdef void post_de(
              DT_UL[::1] prm_opt_stop_arr,

        const DT_D[::1] curr_obj_vals,
              DT_D[::1] pre_obj_vals,
              DT_D[::1] best_prm_vec,
              DT_D[::1] iobj_vals,

        const DT_D[:, ::1] u_j_gs,
              DT_D[:, ::1] prm_vecs,
              DT_D[:, ::1] prms_mean_thrs_arr,

        const DT_UL max_iters,
        const DT_UL max_cont_iters,
              DT_UL *iter_curr,
              DT_UL *last_succ_i,
              DT_UL *n_succ,
              DT_UL *cont_iter,
              DT_UL *cont_opt_flag,

        const DT_D obj_ftn_tol,
              DT_D *tol_curr,
              DT_D *tol_pre,
              DT_D *fval_pre_global,
              DT_D *prm_pcnt_tol,
        ) nogil except +:

    cdef:
        Py_ssize_t j, k

        DT_UL n_prms = prm_vecs.shape[1]
        DT_UL n_prm_vecs = prm_vecs.shape[0]

        DT_D fval_pre, fval_curr, ddmv, iobj = INFINITY

    for j in range(n_prm_vecs):
        fval_pre = pre_obj_vals[j]
        fval_curr = curr_obj_vals[j]

        if isnan(fval_curr):
            with gil: raise RuntimeError('fval_curr is Nan!')

        if fval_curr < iobj:
            iobj = fval_curr

        if fval_curr >= fval_pre:
            continue

        for k in range(n_prms):
            prm_vecs[j, k] = u_j_gs[j, k]

        pre_obj_vals[j] = fval_curr

        # check for global minimum and best vector
        if fval_curr >= fval_pre_global[0]:
            continue

        for k in range(n_prms):
            best_prm_vec[k] = u_j_gs[j, k]

        tol_pre[0] = tol_curr[0]
        tol_curr[0] = (fval_pre_global[0] - fval_curr) / fval_pre_global[0]
        fval_pre_global[0] = fval_curr
        last_succ_i[0] = iter_curr[0]
        n_succ[0] += 1
        cont_iter[0] = 0

    iobj_vals[iter_curr[0] + 1] = iobj

    if (iter_curr[0] >= 300) & ((iter_curr[0] % 20) == 0):
        ddmv = 0.0

        for k in range(n_prms):
            if prm_opt_stop_arr[k]:
                continue

            prms_mean_thrs_arr[k, 0] = 0.0
            for j in range(n_prm_vecs):
                prms_mean_thrs_arr[k, 0] += prm_vecs[j, k]

            prms_mean_thrs_arr[k, 0] /= n_prm_vecs

            prms_mean_thrs_arr[k, 1] = (
                (1 - prm_pcnt_tol[0]) * prms_mean_thrs_arr[k, 0])

            prms_mean_thrs_arr[k, 2] = (
                (1 + prm_pcnt_tol[0]) * prms_mean_thrs_arr[k, 0])

            prm_opt_stop_arr[k] = 1

            for j in range(n_prm_vecs):
                if ((prm_vecs[j, k] < prms_mean_thrs_arr[k, 1]) or
                    (prm_vecs[j, k] > prms_mean_thrs_arr[k, 2])):

                    prm_opt_stop_arr[k] = 0
                    break

            if not prm_opt_stop_arr[k]:
                continue

            ddmv = 1.0

            with gil:
                print(
                    'Parameter no. %d optimized at iteration: %d!' %
                    (k, iter_curr[0]))
        if ddmv:
            ddmv = 0.0
            for k in range(n_prms):
                if prm_opt_stop_arr[k]:
                    ddmv += 1

            with gil: 
                print('%d out of %d to go!' % (n_prms - int(ddmv), n_prms))

        if ddmv == n_prms:
            with gil: print('***All parameters within prm_pcnt_tol!***')
            cont_opt_flag[0] = 0

    iter_curr[0] += 1

    if cont_opt_flag[0] and (iter_curr[0] >= max_iters):
        with gil: print('***Max iterations reached!***')

        cont_opt_flag[0] = 0

    if cont_opt_flag[0] and ((0.5 * (tol_pre[0] + tol_curr[0])) < obj_ftn_tol):
        with gil: print('***Objective tolerance reached!***')

        cont_opt_flag[0] = 0

    if cont_opt_flag[0] and (cont_iter[0] > max_cont_iters):
        with gil: print('***max_cont_iters reached!***')

        cont_opt_flag[0] = 0

    return
