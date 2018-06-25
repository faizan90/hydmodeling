# cython: nonecheck=False
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: language_level=3
# cython: infer_types=False
# cython: embedsignature=True

from .hbv_loop cimport hbv_loop
from .routing cimport route_strms

cdef DT_UL use_c = 1


cdef extern from "hbv_c_loop.h" nogil:
    cdef:
        DT_D hbv_c_loop(
            const DT_D *temp_arr,
            const DT_D *prec_arr,
            const DT_D *petn_arr,
            const DT_D *prms_arr,
            const DT_D *inis_arr,
            const DT_D *area_arr,
                  DT_D *lrst_arr,
                  DT_D *qsim_arr,
                  DT_D *outs_arr,
            const DT_UL *n_time_steps,
            const DT_UL *n_cells,
            const DT_UL *n_prms,
            const DT_UL *n_vars_outs_arr,
            const DT_D *rnof_q_conv,
            const DT_D *err_val,
            const DT_UL *opt_flag)


cdef DT_D hbv_mult_cat_loop(
    const DT_UL[::1] stm_idxs,
    const DT_UL[::1] misc_longs,

    const DT_D[::1] qact_arr,
    const DT_D[::1] area_arr,
          DT_D[::1] qsim_arr,
          DT_D[::1] lrst_arr,
          DT_D[::1] inflow_arr,
    const DT_D[::1] misc_doubles,
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

          DT_D[:, :, ::1] outs_arr,

          cmap[long, long] &cat_to_idx_map,
          cmap[long, long] &stm_to_idx_map,
    ) nogil except +:

    cdef:
        Py_ssize_t i
        DT_UL n_recs = qsim_arr.shape[0], stm_idx, cat_idx
        DT_D res

    for i in range(n_recs):
        qsim_arr[i] = 0.0
        lrst_arr[i + 1] = 0.0

    if (misc_longs[5] > 0):
        route_strms(
            stm_idxs,
            cat_to_idx_map,
            stm_to_idx_map,
            inflow_arr,
            route_prms,
            dem_net_arr,
            cats_outflow_arr,
            stms_inflow_arr,
            stms_outflow_arr,
            &misc_longs[5],
            &misc_longs[3])

    if use_c == 1:
        res = hbv_c_loop(
            &temp_arr[0, 0],
            &prec_arr[0, 0],
            &petn_arr[0, 0],
            &hbv_prms[0, 0],
            &inis_arr[0, 0],
            &area_arr[0],
            &lrst_arr[0],
            &qsim_arr[0],
            &outs_arr[0, 0, 0],
            &n_recs,
            &misc_longs[9],
            &misc_longs[10],
            &misc_longs[6],
            &misc_doubles[0],
            &misc_doubles[5],
            &misc_longs[8])
    else:
        res = hbv_loop(
            temp_arr,
            prec_arr,
            petn_arr,
            hbv_prms,
            inis_arr,
            area_arr,
            lrst_arr,
            qsim_arr,
            outs_arr,
            &misc_doubles[0],
            &misc_doubles[5],
            &misc_longs[8])

    if res == 0.0:
        if (misc_longs[1] != -2):
            stm_idx = stm_to_idx_map[misc_longs[1]]
            for i in range(n_recs):
                qsim_arr[i] = qsim_arr[i] + (
                    stms_outflow_arr[i, stm_idx])

        cat_idx = cat_to_idx_map[misc_longs[4]]
        if misc_longs[7] == 0:
            for i in range(n_recs):
                cats_outflow_arr[i, cat_idx] = qsim_arr[i]
        elif misc_longs[7] == 1:
            for i in range(n_recs):
                cats_outflow_arr[i, cat_idx] = qact_arr[i]
        else:
            with gil: print(
                ('Incorrect use_obs_flow_flag: %d' % misc_longs[7]))
    return res
