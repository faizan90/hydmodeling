# cython: nonecheck=False
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: language_level=3
# cython: infer_types=False
# cython: embedsignature=True

import time
import timeit
import random
from libcpp.map cimport map as cmap
from cython.parallel import prange, threadid

import numpy as np
cimport numpy as np

from .hbv_obj_ftn cimport obj_ftn
from .misc_ftns cimport (
    get_demr, 
    get_ln_demr,
    get_mean,
    get_ln_mean,
    get_variance,
    del_idx)
from .dtypes cimport (
    DT_D,
    DT_UL,
    DT_ULL,
    fc_i, 
    pwp_i, 
    n_hbv_cols, 
    n_hbv_prms, 
    obj_longs_ct, 
    obj_doubles_ct,
    err_val)

DT_D_NP = np.float64
DT_UL_NP = np.int32


cdef extern from "rand_gen_mp.h" nogil:
    cdef:
        DT_D rand_c()
        void warm_up()  # call this everytime
        DT_D rand_c_mp(DT_ULL *rnd_j)
        void warm_up_mp(DT_ULL *seeds_arr, DT_UL n_seeds)  # call this everytime

warm_up()


cpdef dict hbv_opt_de(args):

    '''The differential evolution algorithm for distributed HBV with routing
    '''
    cdef:
        DT_UL i, j, k, m, t_i

        #======================================================================
        # Basic optimization related parameters
        DT_UL opt_flag = 1, use_obs_flow_flag
        DT_UL max_iters, max_cont_iters, off_idx, n_prms, n_hm_prms
        DT_UL iter_curr = 0, last_succ_i = 0, n_succ = 0, cont_iter = 0

        DT_D obj_ftn_tol, res, prm_pcnt_tol
        DT_D tol_curr = np.inf, tol_pre = np.inf
        DT_D fval_pre_global = np.inf, fval_pre, fval_curr

        list accept_vars
        list total_vars

        DT_UL[::1] prm_opt_stop_arr, obj_longs
        DT_UL[:, ::1] prms_flags, prms_span_idxs, f_var_infos
        DT_UL[:, :, ::1] prms_idxs

        DT_D[::1] pre_obj_vals, curr_obj_vals, best_params, params_tmp
        DT_D[::1] obj_ftn_wts, obj_doubles
        DT_D[:, ::1] curr_opt_prms, bounds, bds_dfs, prms_mean_thrs_arr
        #======================================================================

        #======================================================================
        # Differential Evolution related parameters
        DT_UL n_pop
        DT_UL r0, r1, r2
        DT_UL del_r_i, del_r_j, ch_r_i, ch_r_j, ch_r_l
        DT_UL pop_size_ol, n_mu_samps = 3

        DT_D mu_sc_fac, cr_cnst

        list idxs_shuff_pop

        DT_UL[::1] idx_rng, r_r
        DT_UL[:, ::1] del_idx_rng, choice_arr

        DT_D[::1] mu_sc_fac_bds, cr_cnst_bds
        DT_D[:, ::1] pop, pop_raw, v_j_g, u_j_gs
        
        # TODO: save populatiopn state at certain iterations to see how they evolve 
        #======================================================================

        #======================================================================
        # HBV related parameters
        DT_D rnof_q_conv

        DT_D[:, :, ::1] hbv_prms
        #======================================================================

        #======================================================================
        # Hydrological state and topology related parameters
        long curr_us_stm

        DT_UL n_cells, n_recs, cat, stm, n_route_prms
        DT_UL route_type, n_stms, cat_no, stm_idx, cat_idx

        DT_UL[::1] stms_idxs

        DT_D[::1] inflow_arr, qact_arr, f_vars, area_arr
        DT_D[:, ::1] inis_arr, temp_arr, prec_arr, petn_arr, route_prms
        DT_D[:, ::1] dem_net_arr, cats_outflow_arr, stms_inflow_arr
        DT_D[:, ::1] stms_outflow_arr, qsim_mult_arr, inflow_mult_arr
        DT_D[:, ::1] lrst_mult_arr
        DT_D[:, :, ::1] cats_outflow_mult_arr
        DT_D[:, :, ::1] stms_inflow_mult_arr, stms_outflow_mult_arr
        DT_D[:, :, :, ::1] outs_mult_arr
        cmap[long, long] cat_to_idx_map, stm_to_idx_map
        #======================================================================

        #======================================================================
        # All other variables
        DT_UL n_cpus, tid
        DT_D ddmv # a temporary double
        DT_D mean_ref, ln_mean_ref, demr, ln_demr, act_std_dev

        DT_UL[::1] n_calls
        DT_ULL[::1] seeds_arr
        #======================================================================

    bounds = args[0]

    obj_ftn_wts = args[1]

    (mu_sc_fac_bds, 
     cr_cnst_bds, 
     n_pop,
     max_iters, 
     max_cont_iters, 
     obj_ftn_tol,
     prm_pcnt_tol) = args[2]

    (inis_arr,
     temp_arr,
     prec_arr,
     petn_arr,
     qact_arr) = args[3]

    (curr_us_stm,
     stms_idxs,
     cat_to_idx_dict,
     stm_to_idx_dict) = args[4]

    (dem_net_arr,
     cats_outflow_arr,
     stms_inflow_arr,
     stms_outflow_arr) = args[5]
    
    (off_idx,
     rnof_q_conv,
     route_type,
     cat_no,
     n_cpus,
     n_stms,
     n_cells,
     use_obs_flow_flag,
     n_hm_prms) = args[6]

    (area_arr,
     prms_idxs, 
     prms_flags,
     prms_span_idxs,
     f_vars,
     f_var_infos) = args[7]

    n_prms = bounds.shape[0]
    n_recs = cats_outflow_arr.shape[0]
    pop_size_ol = n_pop - 1

    idx_rng = np.arange(0, n_pop, 1, dtype=DT_UL_NP)
    r_r = np.zeros(n_cpus, dtype=DT_UL_NP)
    n_calls = np.zeros(n_cpus, dtype=DT_UL_NP)
    seeds_arr = np.zeros(n_cpus, dtype=np.uint64)
    del_idx_rng = np.zeros((n_cpus, pop_size_ol), dtype=DT_UL_NP)
    choice_arr = np.zeros((n_cpus, 3), dtype=DT_UL_NP)

    pre_obj_vals = np.full(n_pop, np.inf, dtype=DT_D_NP)
    curr_obj_vals = np.full(n_pop, np.inf, dtype=DT_D_NP)
    best_params = np.full(n_prms, np.nan, dtype=DT_D_NP)

    params_tmp = np.zeros(n_pop, dtype=DT_D_NP)

    curr_opt_prms = np.zeros((n_cpus, n_prms), dtype=DT_D_NP)
    pop = np.zeros((n_pop, n_prms), dtype=DT_D_NP)
    pop_raw = pop.copy()
    v_j_g = np.zeros((n_cpus, n_prms), dtype=DT_D_NP)
    u_j_gs = np.zeros((n_pop, n_prms), dtype=DT_D_NP)

    hbv_prms = np.zeros((n_cpus, n_cells, n_hbv_prms), dtype=DT_D_NP)
    
    n_route_prms = n_prms - n_hm_prms
    if not n_route_prms:
        n_route_prms = 1
    route_prms = np.full((n_cpus, n_route_prms), np.nan, dtype=DT_D_NP)

    idxs_shuff_pop = list(range(0,  n_pop))
    accept_vars = []
    total_vars = []

    inflow_arr = np.zeros(n_recs, dtype=DT_D_NP)

    qsim_mult_arr = np.zeros((n_cpus, n_recs), dtype=DT_D_NP)
    inflow_mult_arr = qsim_mult_arr.copy()
    lrst_mult_arr = np.zeros((n_cpus, n_recs + 1), dtype=DT_D_NP)

    cats_outflow_mult_arr = np.zeros((n_cpus,
                                      cats_outflow_arr.shape[0],
                                      cats_outflow_arr.shape[1]), 
                                     dtype=DT_D_NP)

    stms_inflow_mult_arr = np.zeros((n_cpus,
                                     stms_inflow_arr.shape[0],
                                     stms_inflow_arr.shape[1]),
                                    dtype=DT_D_NP)
    stms_outflow_mult_arr = stms_inflow_mult_arr.copy()

    outs_mult_arr = np.zeros((n_cpus, n_cells, 1, n_hbv_cols), dtype=DT_D_NP)

    for cat in cat_to_idx_dict:
        cat_to_idx_map[cat] = cat_to_idx_dict[cat]

    for stm in stm_to_idx_dict:
        stm_to_idx_map[stm] = stm_to_idx_dict[stm]

    # prep the MP RNG
    for i in range(n_cpus):
        seeds_arr[i] = <DT_ULL> (time.time() * 10000)
        time.sleep(0.001)
    warm_up_mp(&seeds_arr[0], n_cpus)

    mean_ref = get_mean(qact_arr, &off_idx)
    ln_mean_ref = get_ln_mean(qact_arr, &off_idx)

    demr = get_demr(qact_arr, &mean_ref, &off_idx)
    ln_demr = get_ln_demr(qact_arr, &ln_mean_ref, &off_idx)

    act_std_dev = get_variance(&mean_ref, qact_arr, &off_idx)**0.5

    bds_dfs = np.zeros((n_prms, 2), dtype=DT_D_NP)

    # mean, min thresh, max thresh
    prms_mean_thrs_arr = np.zeros((n_prms, 3), dtype=DT_D_NP)

    # for param index with value 1, stop optimizing
    prm_opt_stop_arr = np.zeros(n_prms, dtype=np.int32)

    # size and other constants needed in the obj_ftn
    # so that all are in a single variable
    obj_longs = np.zeros(obj_longs_ct, dtype=DT_UL_NP)
    obj_longs[0] = off_idx
    obj_longs[1] = curr_us_stm
    obj_longs[2] = n_hm_prms
    obj_longs[3] = route_type
    obj_longs[4] = cat_no
    obj_longs[5] = n_stms
    obj_longs[6] = n_hbv_cols
    obj_longs[7] = use_obs_flow_flag
    obj_longs[8] = opt_flag
    obj_longs[9] = n_cells
    obj_longs[10] = n_hbv_prms

    obj_doubles = np.zeros(obj_doubles_ct, dtype=DT_D_NP)
    obj_doubles[0] = rnof_q_conv
    obj_doubles[1] = demr
    obj_doubles[2] = ln_demr
    obj_doubles[3] = mean_ref
    obj_doubles[4] = act_std_dev
    obj_doubles[5] = err_val

    for k in range(n_prms):
        bds_dfs[k, 0] = bounds[k, 0]
        bds_dfs[k, 1] = bounds[k, 1] - bounds[k, 0]

        if np.isclose(bds_dfs[k, 1], 0.0):
            prm_opt_stop_arr[k] = 1

            print('Parameter no. %d with a value of %0.6f is a constant!' % 
                  (k, bounds[k, 0]))

    # initiate parameter space
    for i in range(n_pop):
        for k in range(n_prms):
            pop_raw[i, k] = (<DT_D> i) / pop_size_ol

    # shuffle the parameters around, in space
    for i in range(n_prms):
        random.shuffle(idxs_shuff_pop)
        for j in range(n_pop):
            params_tmp[j] = pop_raw[<DT_UL> idxs_shuff_pop[j], i]

        for j in range(n_pop):
            if prm_opt_stop_arr[i]:
                pop[j, i] = 0.0
            else:
                pop[j, i] = params_tmp[j]

    for i in range(n_pop):
        # bring all pwps below fcs
        for j in range(prms_span_idxs[fc_i, 1] - prms_span_idxs[fc_i, 0]):
            if (pop[i, prms_span_idxs[pwp_i, 0] + j] <
                pop[i, prms_span_idxs[fc_i, 0] + j]):
                continue

            # arbitrary decrease
            pop[i, prms_span_idxs[pwp_i, 0] + j] = (
                0.99 * rand_c() * pop[i, prms_span_idxs[fc_i, 0] + j])

        for j in range(n_hbv_prms):
            for m in range(3, 6):
                if not prms_flags[j, m]:
                    continue

                k = prms_idxs[j, m, 0]
                if pop[i, k + 1] >= pop[i, k]:
                    continue 

                pop[i, k] = 0.99 * rand_c() * pop[i, k + 1]

    for i in range(n_cpus):
        for j in range(n_recs):
            for k in range(stms_inflow_arr.shape[1]):
                stms_inflow_mult_arr[i, j, k] = stms_inflow_arr[j, k]
                stms_outflow_mult_arr[i, j, k] = stms_outflow_arr[j, k]
            for k in range(cats_outflow_arr.shape[1]):
                cats_outflow_mult_arr[i, j, k] = cats_outflow_arr[j, k]

    # for the selected parameters, get obj vals
#     tid = 0
#     for i in range(n_pop):
    for i in prange(n_pop, 
                    schedule='dynamic',
                    nogil=True, 
                    num_threads=n_cpus):
        tid = threadid()

        for k in range(n_prms):
            curr_opt_prms[tid, k] = pop[i, k]

#         print(['%0.5f'  % _ for _ in curr_opt_prms[tid, :]])

        res = obj_ftn(
            &tid,
            n_calls,
            stms_idxs,
            obj_longs,
            prms_flags,
            f_var_infos,
            prms_idxs,
            obj_ftn_wts,
            curr_opt_prms[tid],
            qact_arr,
            area_arr,
            qsim_mult_arr[tid],
            lrst_mult_arr[tid],
            inflow_mult_arr[tid],
            f_vars,
            obj_doubles,
            route_prms[tid],
            inis_arr,
            temp_arr,
            prec_arr,
            petn_arr,
            cats_outflow_mult_arr[tid],
            stms_inflow_mult_arr[tid],
            stms_outflow_mult_arr[tid],
            dem_net_arr,
            hbv_prms[tid],
            bds_dfs,
            outs_mult_arr[tid],
            cat_to_idx_map,
            stm_to_idx_map)

#         raise Exception(res)
        if res == err_val:
            pre_obj_vals[i] = (2 + rand_c()) * err_val
        else:
            pre_obj_vals[i] = res

    for i in range(n_pop):
#         print('%d Ini res:' % i, pre_obj_vals[i])
        if pre_obj_vals[i] > fval_pre_global:
            continue

        if np.isnan(pre_obj_vals[i]):
            raise RuntimeError('res is Nan!')

        fval_pre_global = pre_obj_vals[i]
        for k in range(n_prms):
            best_params[k] = pop[i, k]

    print('fval_pre_global:', fval_pre_global)
    print('Catchment:', cat_no)
#     raise Exception('Stop!')

    while ((iter_curr < max_iters) and
           (0.5 * (tol_pre + tol_curr) > obj_ftn_tol) and
           (cont_iter < max_cont_iters)):

        cont_iter += 1
#         print('DE iter no:', iter_curr)

        # randomize the mutation and recombination factors for every
        # generation
        mu_sc_fac = (mu_sc_fac_bds[0] +
                     ((mu_sc_fac_bds[1] - mu_sc_fac_bds[0]) * rand_c()))
        cr_cnst = (cr_cnst_bds[0] +
                   ((cr_cnst_bds[1] - cr_cnst_bds[0]) * rand_c()))

        for t_i in prange(n_pop, 
                        schedule='static',
                        nogil=True, 
                        num_threads=n_cpus):
            tid = threadid()

            # get inidicies except t_i
            del_idx(idx_rng, del_idx_rng[tid], &t_i)

            # select indicies randomly from del_idx_rng
            ch_r_l = 1
            while ch_r_l:
                ch_r_l = 0
                for ch_r_i in range(n_mu_samps):
                    k = <DT_UL> (rand_c_mp(&seeds_arr[tid]) * (pop_size_ol))
                    choice_arr[tid, ch_r_i] = del_idx_rng[tid, k]

                for ch_r_i in range(n_mu_samps):
                    for ch_r_j in range(ch_r_i + 1, n_mu_samps):
                        if (choice_arr[tid, ch_r_i] ==
                            choice_arr[tid, ch_r_j]):
                            ch_r_l = 1

            r0 = choice_arr[tid, 0]
            r1 = choice_arr[tid, 1]
            r2 = choice_arr[tid, 2]

            # mutate
            for k in range(n_prms):
#                 if prm_opt_stop_arr[k]:
#                     v_j_g[tid, k] = pop[0, k]
#                     continue

                v_j_g[tid, k] = (
                    pop[r0, k] + (mu_sc_fac * (pop[r1, k] - pop[r2, k])))

            # keep parameters in bounds
            for k in range(n_prms):
#                 if prm_opt_stop_arr[k]:
#                     continue

                if ((v_j_g[tid, k] < 0) or (v_j_g[tid, k] > 1)):
                    v_j_g[tid, k] = rand_c_mp(&seeds_arr[tid])

            # get an index randomly to have atleast one parameter from
            # the mutated vector
            r_r[tid] = <DT_UL> (rand_c_mp(&seeds_arr[tid]) * n_prms)

            for k in range(n_prms):
#                 if prm_opt_stop_arr[k]:
#                     curr_opt_prms[tid, k] = pop[0, k]
#                     u_j_gs[t_i, k] = pop[0, k]
#                     continue

                if ((rand_c_mp(&seeds_arr[tid]) <= cr_cnst) or
                    (k == r_r[tid])):

                    u_j_gs[t_i, k] = v_j_g[tid, k]

                else:
                    u_j_gs[t_i, k] = pop[t_i, k]

                curr_opt_prms[tid, k] = u_j_gs[t_i, k]

            # check if pwp is ge than fc and adjust
            for i in range(prms_span_idxs[fc_i, 1] - prms_span_idxs[fc_i, 0]):
#                 if (prm_opt_stop_arr[prms_span_idxs[pwp_i, 0] + i] or
#                     prm_opt_stop_arr[prms_span_idxs[fc_i, 0] + i]):
#                     continue  # what if the parameter to change is locked?
 
                if (curr_opt_prms[tid, prms_span_idxs[pwp_i, 0] + i] <
                    curr_opt_prms[tid, prms_span_idxs[fc_i, 0] + i]):
                    continue
 
                curr_opt_prms[tid, prms_span_idxs[pwp_i, 0] + i] = (
                    0.99 *
                    rand_c_mp(&seeds_arr[tid]) * 
                    curr_opt_prms[tid, prms_span_idxs[fc_i, 0] + i])
 
                u_j_gs[t_i, prms_span_idxs[pwp_i, 0] + i] = (
                    curr_opt_prms[tid, prms_span_idxs[pwp_i, 0] + i])

            for i in range(n_hbv_prms):
                for m in range(3, 6):
                    if not prms_flags[i, m]:
                        continue

                    k = prms_idxs[i, m, 0]
                    if curr_opt_prms[tid, k + 1] >= curr_opt_prms[tid, k]:
                        continue

                    curr_opt_prms[tid, k] = (0.99 *
                                             rand_c_mp(&seeds_arr[tid]) *
                                             curr_opt_prms[tid, k + 1])
                    u_j_gs[i, k] = curr_opt_prms[tid, k]

            res = obj_ftn(
                &tid,
                n_calls,
                stms_idxs,
                obj_longs,
                prms_flags,
                f_var_infos,
                prms_idxs,
                obj_ftn_wts,
                curr_opt_prms[tid],
                qact_arr,
                area_arr,
                qsim_mult_arr[tid],
                lrst_mult_arr[tid],
                inflow_mult_arr[tid],
                f_vars,
                obj_doubles,
                route_prms[tid],
                inis_arr,
                temp_arr,
                prec_arr,
                petn_arr,
                cats_outflow_mult_arr[tid],
                stms_inflow_mult_arr[tid],
                stms_outflow_mult_arr[tid],
                dem_net_arr,
                hbv_prms[tid],
                bds_dfs,
                outs_mult_arr[tid],
                cat_to_idx_map,
                stm_to_idx_map)

            if res == err_val:
                curr_obj_vals[t_i] = (2 + rand_c_mp(&seeds_arr[tid])) * err_val
            else:
                curr_obj_vals[t_i] = res

        for j in range(n_pop):
            fval_pre = pre_obj_vals[j]
            fval_curr = curr_obj_vals[j]

            if np.isnan(fval_curr):
                raise RuntimeError('fval_curr is Nan!')

            if fval_curr >= fval_pre:
                continue

            for k in range(n_prms):
                pop[j, k] = u_j_gs[j, k]
                         
            pre_obj_vals[j] = fval_curr
#             accept_vars.append((mu_sc_fac, cr_cnst, fval_curr, iter_curr))
#             total_vars.append((mu_sc_fac, cr_cnst))

            # check for global minimum and best vector
            if fval_curr >= fval_pre_global:
                continue

            for k in range(n_prms):
                best_params[k] = u_j_gs[j, k]

            tol_pre = tol_curr
            tol_curr = (fval_pre_global - fval_curr) / fval_pre_global
            fval_pre_global = fval_curr
            last_succ_i = iter_curr
            n_succ += 1
            cont_iter = 0

#             print(iter_curr, 'global min: %0.8f' % fval_pre_global)

        if iter_curr > 20:
            ddmv = 0.0
            for k in range(n_prms):
                if prm_opt_stop_arr[k]:
                    ddmv += 1
   
            if ddmv == n_prms:
                print('\n***All parameters optimized!***\n', sep='')
                break
   
            for k in range(n_prms):
#             for k in prange(n_prms, 
#                             schedule='static',
#                             nogil=True, 
#                             num_threads=n_cpus):
                if prm_opt_stop_arr[k]:
                    continue
 
                prms_mean_thrs_arr[k, 0] = 0.0
                for j in range(n_pop):
                    prms_mean_thrs_arr[k, 0] += pop[j, k]
  
                prms_mean_thrs_arr[k, 0] /= n_pop
                prms_mean_thrs_arr[k, 1] = (
                    (1 - prm_pcnt_tol) * prms_mean_thrs_arr[k, 0])
                prms_mean_thrs_arr[k, 2] = (
                    (1 + prm_pcnt_tol) * prms_mean_thrs_arr[k, 0])
   
                prm_opt_stop_arr[k] = 1
                for j in range(n_pop):
                    if ((pop[j, k] < prms_mean_thrs_arr[k, 1]) or
                        (pop[j, k] > prms_mean_thrs_arr[k, 2])):
                        prm_opt_stop_arr[k] = 0
                        break
 
                if not prm_opt_stop_arr[k]:
                    continue
   
#                 with gil:
                print('\nParameter no. %d optimized at iteration: %d!' %
                      (k, iter_curr))
                ddmv += 1
                print('%d out of %d to go!\n' % (n_prms - int(ddmv), n_prms))

        iter_curr += 1

    # it is important to call the obj_ftn to makes changes one last time
    # i.e. if you want to use/validate results
    tid = 0
    for k in range(n_prms):
        curr_opt_prms[tid, k] = best_params[k]

    res = obj_ftn(
        &tid,
        n_calls,
        stms_idxs,
        obj_longs,
        prms_flags,
        f_var_infos,
        prms_idxs,
        obj_ftn_wts,
        curr_opt_prms[tid],
        qact_arr,
        area_arr,
        qsim_mult_arr[tid],
        lrst_mult_arr[tid],
        inflow_mult_arr[tid],
        f_vars,
        obj_doubles,
        route_prms[tid],
        inis_arr,
        temp_arr,
        prec_arr,
        petn_arr,
        cats_outflow_arr,  # no mult arr
        stms_inflow_arr,  # no mult arr
        stms_outflow_arr,  # no mult arr
        dem_net_arr,
        hbv_prms[tid],
        bds_dfs,
        outs_mult_arr[tid],
        cat_to_idx_map,
        stm_to_idx_map)

#     cat_idx = cat_to_idx_map[cat_no]
#     for j in range(n_recs):
#         for k in range(n_stms):
#             stm = stms_idxs[k]
#             stm_idx = stm_to_idx_map[stm]
# 
#             stms_inflow_arr[j, stm_idx] = stms_inflow_mult_arr[tid, j, stm_idx]
# 
#             stms_outflow_arr[j, stm_idx] = (
#                 stms_outflow_mult_arr[tid, j, stm_idx])
# 
#         cats_outflow_arr[j, cat_idx] = cats_outflow_mult_arr[tid, j, cat_idx]

    print('Number of iterations:', iter_curr)
    print('Objective function value:', fval_pre_global)
    print('Successful tries:', n_succ)
    print('Last successful try at:', last_succ_i)
    print('cont_iter:', cont_iter)
    print('n_calls:', np.array(n_calls))
    print('Final tolerance:', 0.5 * (tol_pre + tol_curr))
    print('Best parameters:')
    print(['%0.3f' % prm for prm in np.array(best_params).ravel()])

    return {'params': np.asarray(hbv_prms[tid]),
            'route_params': np.asarray(route_prms[tid]),
            'opt_params': np.asarray(best_params),
            'fmin': fval_pre_global,
            'n_gens': iter_curr,
            'n_succ': n_succ,
            'lst_succ_try': last_succ_i,
            'cont_iter': cont_iter,
            'pop': np.asarray(pop),
            'fin_tol': 0.5 * (tol_pre + tol_curr),
            'accept_vars': accept_vars,
            'total_vars': total_vars,
            'n_calls': np.asarray(n_calls),
            'pop_pre_obj_vals': np.asarray(pre_obj_vals),
            'pop_curr_obj_vals': np.asarray(curr_obj_vals),
            'qsim_arr': np.asarray(qsim_mult_arr[tid])}
