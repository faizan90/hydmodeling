# cython: nonecheck=False
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: language_level=3
# cython: infer_types=False
# cython: embedsignature=True

from ..hyds.routing cimport tfm_opt_to_route_prms
from ..hyds.opt_to_hbv_prms cimport tfm_opt_to_hbv_prms
from ..miscs.misc_ftns cimport get_ns, get_ln_ns, get_kge, cmpt_resampled_arr
from ..miscs.misc_ftns_partial cimport (
    get_ns_prt, get_ln_ns_prt, get_kge_prt, cmpt_resampled_arr_prt)

from ..hbvs.hbv_mult_cat_loop cimport hbv_mult_cat_loop
from ..miscs.dtypes cimport (
    n_stms_i,
    n_hm_prms_i,
    off_idx_i,
    use_step_flag_i,
    resamp_obj_ftns_flag_i,
    a_zero_i,
    demr_i,
    ln_demr_i,
    mean_ref_i,
    act_std_dev_i)


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
    const DT_D[::1] obj_doubles,
          DT_D[::1] route_prms,

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
    ) nogil except +:

    cdef:
        Py_ssize_t i
        DT_D res, obj_ftn_wts_sum

    tfm_opt_to_hbv_prms(
        prms_flags,
        f_var_infos,
        prms_idxs,
        f_vars,
        opt_prms,
        bds_dfs,
        hbv_prms)

    if (obj_longs[n_stms_i] > 0):
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
        obj_doubles,
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

    n_calls[tid[0]] = n_calls[tid[0]] + 1

    obj_ftn_wts_sum = 0.0
    if res != 0.0:
        return obj_ftn_wts_sum - res

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
                        &obj_doubles[demr_i],
                        &obj_longs[a_zero_i]))

            else:
                res = obj_ftn_wts[0] * get_ns(
                        qact_resamp_arr,
                        qsim_resamp_arr,
                        &obj_doubles[demr_i],
                        &obj_longs[a_zero_i])

        else:
            if obj_longs[use_step_flag_i]:
                res = obj_ftn_wts[0] * (
                       get_ns_prt(
                        qact_arr,
                        qsim_arr,
                        use_step_arr,
                        &obj_doubles[demr_i],
                        &obj_longs[off_idx_i]))
    
            else:
                res = obj_ftn_wts[0] * (
                       get_ns(
                        qact_arr,
                        qsim_arr,
                        &obj_doubles[demr_i],
                        &obj_longs[off_idx_i]))

        obj_ftn_wts_sum = obj_ftn_wts_sum + obj_ftn_wts[0]

    if obj_ftn_wts[1]:
        if obj_longs[resamp_obj_ftns_flag_i]:
            if obj_longs[use_step_flag_i]:
                res = res + (
                    obj_ftn_wts[1] * get_ln_ns_prt(
                        qact_resamp_arr,
                        qsim_resamp_arr,
                        use_step_arr,
                        &obj_doubles[ln_demr_i],
                        &obj_longs[a_zero_i]))
    
            else:
                res = res + (
                    obj_ftn_wts[1] * get_ln_ns(
                        qact_resamp_arr,
                        qsim_resamp_arr,
                        &obj_doubles[ln_demr_i],
                        &obj_longs[a_zero_i]))

        else:
            if obj_longs[use_step_flag_i]:
                res = res + (
                    obj_ftn_wts[1] * get_ln_ns_prt(
                        qact_arr,
                        qsim_arr,
                        use_step_arr,
                        &obj_doubles[ln_demr_i],
                        &obj_longs[off_idx_i]))
    
            else:
                res = res + (
                    obj_ftn_wts[1] * get_ln_ns(
                        qact_arr,
                        qsim_arr,
                        &obj_doubles[ln_demr_i],
                        &obj_longs[off_idx_i]))

        obj_ftn_wts_sum = obj_ftn_wts_sum + obj_ftn_wts[1]

    if obj_ftn_wts[2]: 
        if obj_longs[resamp_obj_ftns_flag_i]:
            if obj_longs[use_step_flag_i]:
                res = res + (
                    obj_ftn_wts[2] * get_kge_prt(
                        qact_resamp_arr,
                        qsim_resamp_arr,
                        use_step_arr,
                        &obj_doubles[mean_ref_i],
                        &obj_doubles[act_std_dev_i],
                        &obj_longs[a_zero_i]))
    
            else:
                res = res + (
                    obj_ftn_wts[2] * get_kge(
                        qact_resamp_arr,
                        qsim_resamp_arr,
                        &obj_doubles[mean_ref_i],
                        &obj_doubles[act_std_dev_i],
                        &obj_longs[a_zero_i]))

        else:
            if obj_longs[use_step_flag_i]:
                res = res + (
                    obj_ftn_wts[2] * get_kge_prt(
                        qact_arr,
                        qsim_arr,
                        use_step_arr,
                        &obj_doubles[mean_ref_i],
                        &obj_doubles[act_std_dev_i],
                        &obj_longs[off_idx_i]))
    
            else:
                res = res + (
                    obj_ftn_wts[2] * get_kge(
                        qact_arr,
                        qsim_arr,
                        &obj_doubles[mean_ref_i],
                        &obj_doubles[act_std_dev_i],
                        &obj_longs[off_idx_i]))
            
        obj_ftn_wts_sum = obj_ftn_wts_sum + obj_ftn_wts[2]

    return obj_ftn_wts_sum - res
