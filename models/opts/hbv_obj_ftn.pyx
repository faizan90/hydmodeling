# cython: nonecheck=False
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: language_level=3
# cython: infer_types=False
# cython: embedsignature=True

from ..hyds.routing cimport tfm_opt_to_route_prms
from ..hyds.opt_to_hbv_prms cimport tfm_opt_to_hbv_prms
from ..miscs.misc_ftns cimport (
    get_ns,
    get_ln_ns,
    get_kge,
    cmpt_resampled_arr,
    get_demr,
    get_mean,
    fill_diffs_arr)
from ..miscs.misc_ftns_partial cimport (
    get_ns_prt, get_ln_ns_prt, get_kge_prt, cmpt_resampled_arr_prt)
from ..miscs.sec_miscs cimport update_obj_doubles
from ..ft.dfti cimport cmpt_real_fourtrans_1d, cmpt_cumm_freq_pcorrs
from ..ft.effs cimport get_ft_eff
from..miscs.fdcs cimport sort_arr, get_sim_probs_in_ref

from ..hbvs.hbv_mult_cat_loop cimport hbv_mult_cat_loop
from ..miscs.dtypes cimport (
    curr_us_stm_i,
    n_stms_i,
    n_hm_prms_i,
    off_idx_i,
    use_step_flag_i,
    resamp_obj_ftns_flag_i,
    a_zero_i,
    use_res_cat_runoff_flag_i,
    demr_i,
    ln_demr_i,
    mean_ref_i,
    act_std_dev_i,
    min_q_thresh_i,
    route_type_i,
    NAN,
    demr_peak_i,
    ln_demr_peak_i,
    demr_sort_i,
    demr_qdiffs_i)


cdef DT_D obj_ftn(
        const long *tid,

              DT_UL[::1] n_calls,
        const DT_UL[::1] stm_idxs,
        const DT_UL[::1] obj_longs,
        const DT_UL[::1] use_step_arr,

        const DT_ULL[::1] obj_ftn_resamp_tags_arr,

        const DT_UL[:, ::1] prms_flags,
        const DT_UL[:, ::1] f_var_infos,

        const DT_UL[:, :, ::1] prms_idxs,

        const DT_D[::1] obj_ftn_wts,
              DT_D[::1] opt_prms,
        const DT_D[::1] qact_arr,
        const DT_D[::1] qact_resamp_arr,
              DT_D[::1] qsim_resamp_arr,
        const DT_D[::1] area_arr,
              DT_D[::1] qsim_arr,
              DT_D[::1] inflow_arr,
        const DT_D[::1] f_vars,
              DT_D[::1] route_prms,
              DT_D[::1] qact_qres_arr,
              DT_D[::1] qsim_qres_arr,
              DT_D[::1] obj_res_doubles,
        const DT_D[::1] qact_arr_sort,
              DT_D[::1] qsim_arr_sort,
        const DT_D[::1] qact_probs_arr_sort,
              DT_D[::1] qsim_probs_arr_sort,
        const DT_D[::1] qact_diffs_arr,
              DT_D[::1] qsim_diffs_arr,

        const DT_D[:, ::1] inis_arr,
              DT_D[:, ::1] temp_arr,
              DT_D[:, ::1] prec_arr,
              DT_D[:, ::1] petn_arr,
              DT_D[::1, :] cats_outflow_arr,
              DT_D[::1, :] stms_inflow_arr,
              DT_D[::1, :] stms_outflow_arr,
        const DT_D[:, ::1] dem_net_arr,
              DT_D[:, ::1] hbv_prms,
        const DT_D[:, ::1] bds_dfs,

              DT_D[:, :, ::1] outs_arr,

              cmap[long, long] &cat_to_idx_map,
              cmap[long, long] &stm_to_idx_map,
              ForFourTrans1DRealVec &q_ft_tfms,
        ) nogil except +:

    '''NOTE: All used obj. ftns. should have a maximum value of one.'''

    cdef:
        Py_ssize_t i

        DT_UL n_recs = qsim_arr.shape[0], stm_idx

        DT_D res, obj_ftn_wts_sum, min_q_thresh

    if obj_ftn_wts.shape[0] > 8:
        with gil: raise NotImplementedError

    tfm_opt_to_hbv_prms(
        prms_flags,
        f_var_infos,
        prms_idxs,
        f_vars,
        opt_prms,
        bds_dfs,
        hbv_prms)

    if obj_longs[route_type_i] and (obj_longs[n_stms_i] > 0):
        tfm_opt_to_route_prms(
            opt_prms[obj_longs[n_hm_prms_i]:], 
            route_prms, 
            bds_dfs[obj_longs[n_hm_prms_i]:, :])

    res = hbv_mult_cat_loop(
        stm_idxs,
        obj_longs,
        qact_arr,
        area_arr,
        qsim_arr,
        inflow_arr,
        obj_res_doubles,
        route_prms,
        inis_arr,
        temp_arr,
        prec_arr,
        petn_arr,
        cats_outflow_arr,
        stms_inflow_arr,
        stms_outflow_arr,
        dem_net_arr,
        hbv_prms,
        outs_arr,
        cat_to_idx_map,
        stm_to_idx_map)

    if res != 0.0:
        return NAN

    n_calls[tid[0]] = n_calls[tid[0]] + 1

    obj_ftn_wts_sum = 0.0

    if ((not obj_longs[use_res_cat_runoff_flag_i]) or 
        (obj_longs[curr_us_stm_i] == -2)):

        for i in range(n_recs):
            qact_qres_arr[i] = qact_arr[i]
            qsim_qres_arr[i] = qsim_arr[i]

    else:
        if (obj_longs[curr_us_stm_i] != -2):
            stm_idx = stm_to_idx_map[obj_longs[curr_us_stm_i]]
            min_q_thresh = obj_res_doubles[min_q_thresh_i]

            for i in range(n_recs):
                qact_qres_arr[i] = qact_arr[i] - stms_outflow_arr[i, stm_idx]
                qsim_qres_arr[i] = qsim_arr[i] - stms_outflow_arr[i, stm_idx]

                # The limiting is needed only for discharges to be higher than
                # zero in case of ln_NS but it is done for all obj_ftns,
                # to have them comparable. Theoretically, doing so produces
                # mass balance problems. Will also set negative values to
                # min_q_thresh. Don't uncomment.

#                 if qact_qres_arr[i] < min_q_thresh:
#                     qact_qres_arr[i] = min_q_thresh
# 
#                 if qsim_qres_arr[i] < min_q_thresh:
#                     qsim_qres_arr[i] = min_q_thresh

            if obj_longs[resamp_obj_ftns_flag_i]:
                with gil: raise NotImplementedError('Cant\'t be done!')

            update_obj_doubles(
                obj_longs, 
                use_step_arr, 
                obj_ftn_wts,
                obj_res_doubles,
                qact_qres_arr,
                qact_resamp_arr)

    if obj_longs[resamp_obj_ftns_flag_i]:
        if obj_longs[use_step_flag_i]:
            cmpt_resampled_arr_prt(
                qsim_arr, 
                qsim_resamp_arr,
                obj_ftn_resamp_tags_arr,
                use_step_arr)

        else:
            cmpt_resampled_arr(
                qsim_arr,
                qsim_resamp_arr,
                obj_ftn_resamp_tags_arr)

    if obj_ftn_wts[0]:
        if obj_longs[resamp_obj_ftns_flag_i]:
            if obj_longs[use_step_flag_i]:
                res = obj_ftn_wts[0] * (
                       get_ns_prt(
                        qact_resamp_arr,
                        qsim_resamp_arr,
                        use_step_arr,
                        obj_res_doubles[demr_i],
                        obj_longs[a_zero_i]))

            else:
                res = obj_ftn_wts[0] * get_ns(
                        qact_resamp_arr,
                        qsim_resamp_arr,
                        obj_res_doubles[demr_i],
                        obj_longs[a_zero_i])

        else:
            if obj_longs[use_step_flag_i]:
                res = obj_ftn_wts[0] * (
                       get_ns_prt(
                        qact_qres_arr,
                        qsim_qres_arr,
                        use_step_arr,
                        obj_res_doubles[demr_i],
                        obj_longs[off_idx_i]))

            else:
                res = obj_ftn_wts[0] * (
                       get_ns(
                        qact_qres_arr,
                        qsim_qres_arr,
                        obj_res_doubles[demr_i],
                        obj_longs[off_idx_i]))

        obj_ftn_wts_sum = obj_ftn_wts_sum + obj_ftn_wts[0]

    if obj_ftn_wts[1]:
        if obj_longs[resamp_obj_ftns_flag_i]:
            if obj_longs[use_step_flag_i]:
                res = res + (
                    obj_ftn_wts[1] * get_ln_ns_prt(
                        qact_resamp_arr,
                        qsim_resamp_arr,
                        use_step_arr,
                        obj_res_doubles[ln_demr_i],
                        obj_longs[a_zero_i]))

            else:
                res = res + (
                    obj_ftn_wts[1] * get_ln_ns(
                        qact_resamp_arr,
                        qsim_resamp_arr,
                        obj_res_doubles[ln_demr_i],
                        obj_longs[a_zero_i]))

        else:
            if obj_longs[use_step_flag_i]:
                res = res + (
                    obj_ftn_wts[1] * get_ln_ns_prt(
                        qact_qres_arr,
                        qsim_qres_arr,
                        use_step_arr,
                        obj_res_doubles[ln_demr_i],
                        obj_longs[off_idx_i]))

            else:
                res = res + (
                    obj_ftn_wts[1] * get_ln_ns(
                        qact_qres_arr,
                        qsim_qres_arr,
                        obj_res_doubles[ln_demr_i],
                        obj_longs[off_idx_i]))

        obj_ftn_wts_sum = obj_ftn_wts_sum + obj_ftn_wts[1]

    if obj_ftn_wts[2]: 
        if obj_longs[resamp_obj_ftns_flag_i]:
            if obj_longs[use_step_flag_i]:
                res = res + (
                    obj_ftn_wts[2] * get_kge_prt(
                        qact_resamp_arr,
                        qsim_resamp_arr,
                        use_step_arr,
                        obj_res_doubles[mean_ref_i],
                        obj_res_doubles[act_std_dev_i],
                        obj_longs[a_zero_i]))

            else:
                res = res + (
                    obj_ftn_wts[2] * get_kge(
                        qact_resamp_arr,
                        qsim_resamp_arr,
                        obj_res_doubles[mean_ref_i],
                        obj_res_doubles[act_std_dev_i],
                        obj_longs[a_zero_i]))

        else:
            if obj_longs[use_step_flag_i]:
                res = res + (
                    obj_ftn_wts[2] * get_kge_prt(
                        qact_qres_arr,
                        qsim_qres_arr,
                        use_step_arr,
                        obj_res_doubles[mean_ref_i],
                        obj_res_doubles[act_std_dev_i],
                        obj_longs[off_idx_i]))

            else:
                res = res + (
                    obj_ftn_wts[2] * get_kge(
                        qact_qres_arr,
                        qsim_qres_arr,
                        obj_res_doubles[mean_ref_i],
                        obj_res_doubles[act_std_dev_i],
                        obj_longs[off_idx_i]))

        obj_ftn_wts_sum = obj_ftn_wts_sum + obj_ftn_wts[2]

    if obj_ftn_wts[3]:
        if obj_longs[resamp_obj_ftns_flag_i]:
            with gil: raise NotImplementedError

        if obj_longs[use_res_cat_runoff_flag_i]:
            with gil: raise NotImplementedError

        for i in range(q_ft_tfms[0].n_pts):
            q_ft_tfms[tid[0] + 1].orig[i] = qsim_arr[obj_longs[off_idx_i] + i]

        res = res + (
            obj_ftn_wts[3] * get_ft_eff(
                obj_longs,
                obj_res_doubles,
                q_ft_tfms[0],
                q_ft_tfms[tid[0] + 1]))

        obj_ftn_wts_sum = obj_ftn_wts_sum + obj_ftn_wts[3]

    if obj_ftn_wts[4]:
        res = res + obj_ftn_wts[4] * (
            get_ns_prt(
                qact_qres_arr,
                qsim_qres_arr,
                use_step_arr,
                obj_res_doubles[demr_peak_i],
                obj_longs[off_idx_i]))

        obj_ftn_wts_sum = obj_ftn_wts_sum + obj_ftn_wts[4]

    if obj_ftn_wts[5]:
        res = res + obj_ftn_wts[5] * (
            get_ln_ns_prt(
                qact_qres_arr,
                qsim_qres_arr,
                use_step_arr,
                obj_res_doubles[ln_demr_peak_i],
                obj_longs[off_idx_i]))

        obj_ftn_wts_sum = obj_ftn_wts_sum + obj_ftn_wts[5]

    if obj_ftn_wts[6]:
        sort_arr(qsim_arr[obj_longs[off_idx_i]:], qsim_arr_sort)

        get_sim_probs_in_ref(
            qact_arr_sort, 
            qact_probs_arr_sort, 
            qsim_arr_sort, 
            qsim_probs_arr_sort)

        res = res + obj_ftn_wts[6] * (
               get_ns(
                qact_probs_arr_sort,
                qsim_probs_arr_sort,
                obj_res_doubles[demr_sort_i],
                0))

        obj_ftn_wts_sum = obj_ftn_wts_sum + obj_ftn_wts[6]

    if obj_ftn_wts[7]:
        if obj_longs[resamp_obj_ftns_flag_i]:
            with gil: raise NotImplementedError
 
        else:
            if obj_longs[use_step_flag_i]:
                with gil: raise NotImplementedError

            else:
                fill_diffs_arr(qsim_arr, qsim_diffs_arr)

                res = obj_ftn_wts[7] * (
                       get_ns(
                        qact_diffs_arr,
                        qsim_diffs_arr,
                        obj_res_doubles[demr_qdiffs_i],
                        obj_longs[off_idx_i]))

        obj_ftn_wts_sum = obj_ftn_wts_sum + obj_ftn_wts[7]

    return obj_ftn_wts_sum - res
