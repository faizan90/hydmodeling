# cython: nonecheck=False
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: language_level=3
# cython: infer_types=False
# cython: embedsignature=True

import numpy as np
cimport numpy as np
from libcpp.map cimport map as cmap

from .opt_to_hbv_prms cimport tfm_opt_to_hbv_prms
from ..hbvs.hbv_loop cimport hbv_loop
from ..hbvs.hbv_mult_cat_loop cimport hbv_mult_cat_loop
from ..miscs.dtypes cimport (
    DT_D,
    DT_UL,
    DT_ULL,
    n_hbv_cols, 
    n_hbv_prms, 
    obj_longs_ct, 
    obj_doubles_ct,
    err_val,
    off_idx_i,
    curr_us_stm_i,
    n_hm_prms_i,
    route_type_i,
    cat_no_i,
    n_stms_i,
    n_hbv_cols_i,
    use_obs_flow_flag_i,
    opt_flag_i,
    n_cells_i,
    n_hbv_prms_i,
    rnof_q_conv_i,
    err_val_i,
    min_q_thresh_i,
    use_res_cat_runoff_flag_i)

DT_D_NP = np.float64
DT_UL_NP = np.int32


cdef extern from "../hbvs/hbv_c_loop.h" nogil:
    cdef:
        DT_D hbv_c_loop(
                const DT_D *temp_arr,
                const DT_D *prec_arr,
                const DT_D *petn_arr,
                const DT_D *prms_arr,
                const DT_D *inis_arr,
                const DT_D *area_arr,
                      DT_D *qsim_arr,
                      DT_D *outs_arr,
                const DT_UL *n_time_steps,
                const DT_UL *n_cells,
                const DT_UL *n_hbv_prms,
                const DT_UL *n_hbv_cols,
                const DT_D *rnof_q_conv,
                const DT_UL *opt_flag)


cpdef dict hbv_loop_py(
              DT_D[:, ::1] temp_arr,
              DT_D[:, ::1] prec_arr,
              DT_D[:, ::1] petn_arr,
              DT_D[:, ::1] prms_arr,
        const DT_D[:, ::1] inis_arr,
        const DT_D[::1] area_arr,
        const DT_D rnof_q_conv,
              DT_UL opt_flag=0):

    cdef:
        DT_UL n_time_steps = temp_arr.shape[1]
        DT_UL n_cells = temp_arr.shape[0]

        DT_D loop_ret

        DT_D[::1] qsim_arr
        DT_D[:, :, ::1] outs_arr

        dict out_dict

    assert temp_arr.shape[0] == prec_arr.shape[0] == petn_arr.shape[0]
    assert temp_arr.shape[1] == prec_arr.shape[1] == petn_arr.shape[1]
    assert temp_arr.shape[0] == prms_arr.shape[0] == area_arr.shape[0]
    assert prms_arr.shape[1] == n_hbv_prms

    if opt_flag == 0:
        outs_arr = np.empty(
            (n_cells, n_time_steps + 1, n_hbv_cols), dtype=DT_D_NP)

    else:
        outs_arr = np.empty((n_cells, 1, n_hbv_cols), dtype=DT_D_NP)

    qsim_arr = np.zeros(n_time_steps, dtype=DT_D_NP)

    loop_ret = hbv_loop(
        temp_arr,
        prec_arr,
        petn_arr,
        prms_arr,
        inis_arr,
        area_arr,
        qsim_arr,
        outs_arr,
        &rnof_q_conv,
        &opt_flag)

    out_dict = {}
    out_dict['loop_ret'] = loop_ret
    out_dict['outs_arr'] = np.asarray(outs_arr[:, 1:, :])
    out_dict['qsim_arr'] = np.asarray(qsim_arr)
    return out_dict


cpdef dict hbv_c_loop_py(
        const DT_D[:, ::1] temp_arr,
        const DT_D[:, ::1] prec_arr,
        const DT_D[:, ::1] petn_arr,
        const DT_D[:, ::1] prms_arr,
        const DT_D[:, ::1] inis_arr,
        const DT_D[::1] area_arr,
        const DT_D rnof_q_conv,
              DT_UL opt_flag=0):

    cdef:
        DT_UL n_time_steps = temp_arr.shape[1]
        DT_UL n_cells = temp_arr.shape[0]

        DT_D loop_ret

        DT_D[::1] qsim_arr
        DT_D[:, :, ::1] outs_arr

        dict out_dict

    assert temp_arr.shape[0] == prec_arr.shape[0] == petn_arr.shape[0]
    assert temp_arr.shape[1] == prec_arr.shape[1] == petn_arr.shape[1]
    assert temp_arr.shape[0] == prms_arr.shape[0] == area_arr.shape[0]
    assert prms_arr.shape[1] == n_hbv_prms

    if opt_flag == 0:
        outs_arr = np.empty(
            (n_cells, n_time_steps + 1, n_hbv_cols), dtype=DT_D_NP)

    else:
        outs_arr = np.empty((n_cells, 1, n_hbv_cols), dtype=DT_D_NP)

    qsim_arr = np.zeros(n_time_steps, dtype=DT_D_NP)

    loop_ret = hbv_c_loop(
        &temp_arr[0, 0],
        &prec_arr[0, 0],
        &petn_arr[0, 0],
        &prms_arr[0, 0],
        &inis_arr[0, 0],
        &area_arr[0],
        &qsim_arr[0],
        &outs_arr[0, 0, 0],
        &n_time_steps,
        &n_cells,
        &n_hbv_prms,
        &n_hbv_cols,
        &rnof_q_conv,
        &opt_flag)

    out_dict = {}
    out_dict['loop_ret'] = loop_ret
    out_dict['outs_arr'] = np.asarray(outs_arr[:, 1:, :])
    out_dict['qsim_arr'] = np.asarray(qsim_arr)
    return out_dict


cpdef dict hbv_mult_cat_loop_py(args, DT_UL opt_flag=0):
 
    '''Distributed HBV with multiple catchments and routing
    '''
    cdef:
        long curr_us_stm
        DT_UL route_type, cat_no, n_cpus, use_obs_flow_flag, use_step_flag
        DT_UL cat, stm, off_idx, n_cells, use_res_cat_runoff_flag
        DT_UL n_stms, n_hm_prms

        DT_D rnof_q_conv, signal, min_q_thresh

        cmap[long, long] cat_to_idx_map, stm_to_idx_map

        DT_UL[::1] stm_idxs, use_step_arr

        DT_D[::1] qact_arr, obj_ftn_wts, qsim_arr
        DT_D[::1] inflow_arr, area_arr

        DT_D[:, ::1] inis_arr, temp_arr, prec_arr, petn_arr
        DT_D[:, ::1] dem_net_arr, prms_arr
        DT_D[::1, :] stms_inflow_arr, stms_outflow_arr, cats_outflow_arr

        DT_D[:, :, ::1] outs_arr

    hbv_prms, route_prms = args[0]

    inis_arr, temp_arr, prec_arr, petn_arr, qact_arr = args[3]

    curr_us_stm, stm_idxs, cat_to_idx_dict, stm_to_idx_dict = args[4]

    dem_net_arr, cats_outflow_arr, stms_inflow_arr, stms_outflow_arr = args[5]

    (off_idx,
     rnof_q_conv,
     route_type,
     cat_no,
     n_cpus,
     n_stms,
     n_cells,
     use_obs_flow_flag,
     n_hm_prms,
     use_step_flag,
     use_step_arr,
     min_q_thresh,
     use_res_cat_runoff_flag) = args[6]

    area_arr = args[7]

    inflow_arr = np.zeros(cats_outflow_arr.shape[0], dtype=DT_D_NP)
    qsim_arr = inflow_arr.copy()

    if opt_flag == 0:
        outs_arr = np.empty(
            (n_cells, cats_outflow_arr.shape[0] + 1, n_hbv_cols),
            dtype=DT_D_NP)

    else:
        outs_arr = np.empty((n_cells, 1, n_hbv_cols), dtype=DT_D_NP)

    for cat in cat_to_idx_dict:
        cat_to_idx_map[cat] = cat_to_idx_dict[cat]

    for stm in stm_to_idx_dict:
        stm_to_idx_map[stm] = stm_to_idx_dict[stm]

    misc_longs = np.zeros(obj_longs_ct, dtype=DT_UL_NP)
    misc_longs[off_idx_i] = off_idx
    misc_longs[curr_us_stm_i] = curr_us_stm
    misc_longs[n_hm_prms_i] = n_hm_prms
    misc_longs[route_type_i] = route_type
    misc_longs[cat_no_i] = cat_no
    misc_longs[n_stms_i] = n_stms
    misc_longs[n_hbv_cols_i] = n_hbv_cols
    misc_longs[use_obs_flow_flag_i] = use_obs_flow_flag
    misc_longs[opt_flag_i] = opt_flag
    misc_longs[n_cells_i] = n_cells
    misc_longs[n_hbv_prms_i] = n_hbv_prms
    misc_longs[use_res_cat_runoff_flag_i] = use_res_cat_runoff_flag

    misc_doubles = np.zeros(obj_doubles_ct, dtype=DT_D_NP)
    misc_doubles[rnof_q_conv_i] = rnof_q_conv
    misc_doubles[err_val_i] = err_val
    misc_doubles[min_q_thresh_i] = min_q_thresh

    signal = hbv_mult_cat_loop(
        stm_idxs,
        misc_longs,
        qact_arr,
        area_arr,
        qsim_arr,
        inflow_arr,
        misc_doubles,
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

    if signal:
        print('\nWARNING: signal was %f for cat: %d!\n' % (signal, cat_no))

        qsim_arr = np.full(
            cats_outflow_arr.shape[0], np.nan, dtype=DT_D_NP)

    return {'outs_arr': np.asarray(outs_arr[:, 1:, :]),
            'qsim_arr': np.asarray(qsim_arr),
            'inflow_arr': np.asarray(inflow_arr)}


cpdef np.ndarray tfm_opt_to_hbv_prms_py(
        const DT_UL[:, ::1] prms_flags,
        const DT_UL[:, ::1] f_var_infos,

        const DT_UL[:, :, ::1] prms_idxs,

        const DT_D[::1] f_vars,
        const DT_D[::1] opt_prms,
        const DT_D[:, ::1] bds_arr,
        const DT_UL n_cells):

    cdef:
        Py_ssize_t k

        DT_D[:, ::1] hbv_prms, bds_dfs

    assert prms_flags.shape[0] == n_hbv_prms
    assert (np.all(np.asarray(prms_flags) >= 0) & 
            np.all(np.asarray(prms_flags) <= 1))
    assert (np.all(np.asarray(prms_flags).sum(axis=1) > 0) &
            np.all(np.asarray(prms_flags).sum(axis=1) < 6))

    # fc and pwp calibrated using same type of criteria
    assert (np.abs(np.asarray(prms_flags[3]) - 
                   np.asarray(prms_flags[5])).sum() == 0)

    # only one of the cols in aspect, slope or slope/aspect can be active
    assert (np.all(np.asarray(prms_flags[:, 3:]).sum(axis=1) <= 1))

    assert f_var_infos.shape[0] == 5
    assert f_var_infos.shape[1] == 2
    
    assert prms_idxs.shape[0] == n_hbv_prms
    assert prms_idxs.shape[1] == prms_flags.shape[1]
    assert prms_idxs.shape[2] == 2

    hbv_prms = np.full((n_cells, n_hbv_prms), np.nan, dtype=DT_D_NP)
    bds_dfs = np.full((opt_prms.shape[0], 2), np.nan, dtype=DT_D_NP)

    for k in range(opt_prms.shape[0]):
        bds_dfs[k, 0] = bds_arr[k, 0]
        bds_dfs[k, 1] = bds_arr[k, 1] - bds_arr[k, 0]
        
    tfm_opt_to_hbv_prms(
        prms_flags,
        f_var_infos,
        prms_idxs,
        f_vars,
        opt_prms,
        bds_dfs,
        hbv_prms)
    return np.asarray(hbv_prms)
