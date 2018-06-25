# cython: nonecheck=False
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: language_level=3
# cython: infer_types=False
# cython: embedsignature=True

from .routing cimport tfm_opt_to_route_prms
from .opt_to_hbv_prms cimport tfm_opt_to_hbv_prms
from .misc_ftns cimport get_ns, get_ln_ns, get_kge
from .hbv_mult_cat_loop cimport hbv_mult_cat_loop


cdef DT_D obj_ftn(
    const long *tid,

          DT_UL[::1] n_calls,
    const DT_UL[::1] stm_idxs,
    const DT_UL[::1] obj_longs,

    const DT_UL[:, ::1] prms_flags,
    const DT_UL[:, ::1] f_var_infos,

    const DT_UL[:, :, ::1] prms_idxs,

    const DT_D[::1] obj_ftn_wts,
          DT_D[::1] opt_prms,
    const DT_D[::1] qact_arr,
    const DT_D[::1] area_arr,
          DT_D[::1] qsim_arr,
          DT_D[::1] lrst_arr,
          DT_D[::1] inflow_arr,
    const DT_D[::1] f_vars,
    const DT_D[::1] obj_doubles,
          DT_D[::1] route_prms,

    const DT_D[:, ::1] inis_arr,
          DT_D[:, ::1] temp_arr,
          DT_D[:, ::1] prec_arr,
          DT_D[:, ::1] petn_arr,
          DT_D[:, ::1] cats_outflow_arr,
          DT_D[:, ::1] stms_inflow_arr,
          DT_D[:, ::1] stms_outflow_arr,
    const DT_D[:, ::1] dem_net_arr,
          DT_D[:, ::1] hbv_prms,    
    const DT_D[:, ::1] bds_dfs,

          DT_D[:, :, ::1] outs_arr,

          cmap[long, long] &cat_to_idx_map,
          cmap[long, long] &stm_to_idx_map,
    ) nogil except +:

    cdef:
        DT_D res, obj_ftn_wts_sum

    tfm_opt_to_hbv_prms(
        prms_flags,
        f_var_infos,
        prms_idxs,
        f_vars,
        opt_prms,
        bds_dfs,
        hbv_prms)

    if (obj_longs[5] > 0):
        tfm_opt_to_route_prms(
            opt_prms[obj_longs[2]:], 
            route_prms, 
            bds_dfs[obj_longs[2]:, :])

    res = hbv_mult_cat_loop(
        stm_idxs,
        obj_longs,
        qact_arr,
        area_arr,
        qsim_arr,
        lrst_arr,
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

    obj_ftn_wts_sum = 0.0
    if res == 0.0:
        if (obj_ftn_wts[0] != 0):
            res = (obj_ftn_wts[0] * 
                   get_ns(qact_arr, qsim_arr, &obj_doubles[1], &obj_longs[0]))
            obj_ftn_wts_sum = obj_ftn_wts_sum + obj_ftn_wts[0]

        if (obj_ftn_wts[1] != 0):
            res = res + (obj_ftn_wts[1] *
                         get_ln_ns(qact_arr, 
                                   qsim_arr, 
                                   &obj_doubles[2], 
                                   &obj_longs[0]))
            obj_ftn_wts_sum = obj_ftn_wts_sum + obj_ftn_wts[1]

        if (obj_ftn_wts[2] != 0):
            res = res + (obj_ftn_wts[2] *
                         get_kge(qact_arr,
                                 qsim_arr,
                                 &obj_doubles[3],
                                 &obj_doubles[4],
                                 &obj_longs[0]))
            obj_ftn_wts_sum = obj_ftn_wts_sum + obj_ftn_wts[2]

    n_calls[tid[0]] = n_calls[tid[0]] + 1
    return obj_ftn_wts_sum - res
