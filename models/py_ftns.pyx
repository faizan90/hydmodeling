# cython: nonecheck=False
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: language_level=3
# cython: infer_types=False
# cython: embedsignature=True

import timeit

import numpy as np
cimport numpy as np
from libcpp.map cimport map as cmap

from .hbv_loop cimport hbv_loop
from .hbv_mult_cat_loop cimport hbv_mult_cat_loop
from .dtypes cimport (
    DT_D,
    DT_UL,
    DT_ULL,
    n_hbv_cols, 
    n_hbv_prms, 
    obj_longs_ct, 
    obj_doubles_ct,
    err_val)

DT_D_NP = np.float64
DT_UL_NP = np.int32


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
            const DT_UL *n_hbv_prms,
            const DT_UL *n_hbv_cols,
            const DT_D *rnof_q_conv,
            const DT_D *err_val,
            const DT_UL *opt_flag)

 
cpdef dict hbv_loop_py(
          DT_D[:, ::1] temp_arr,
          DT_D[:, ::1] prec_arr,
          DT_D[:, ::1] petn_arr,
          DT_D[:, ::1] prms_arr,
    const DT_D[:, ::1] inis_arr,
    const DT_D[::1] area_arr,
    const DT_D rnof_q_conv):
 
    cdef:
        DT_UL n_time_steps = temp_arr.shape[1], opt_flag = 0
        DT_UL n_cells = temp_arr.shape[0]
 
        DT_D loop_ret#, _st, _sp
 
        DT_D[::1] lrst_arr
        DT_D[::1] qsim_arr
        DT_D[:, :, ::1] outs_arr
         
        dict out_dict
         
    assert temp_arr.shape[0] == prec_arr.shape[0] == petn_arr.shape[0]
    assert temp_arr.shape[1] == prec_arr.shape[1] == petn_arr.shape[1]
    assert temp_arr.shape[0] == prms_arr.shape[0] == area_arr.shape[0]
    assert prms_arr.shape[1] == n_hbv_prms
 
    outs_arr = np.zeros((n_cells, n_time_steps + 1, n_hbv_cols), dtype=DT_D_NP)
    qsim_arr = np.zeros(n_time_steps, dtype=DT_D_NP)
    lrst_arr = np.zeros(n_time_steps + 1, dtype=DT_D_NP)
 
#     _st = timeit.default_timer()
    loop_ret = hbv_loop(
        temp_arr,
        prec_arr,
        petn_arr,
        prms_arr,
        inis_arr,
        area_arr,
        lrst_arr,
        qsim_arr,
        outs_arr,
        &rnof_q_conv,
        &err_val,
        &opt_flag)
#     _sp = timeit.default_timer()
#     print('%0.6f secs for a loop!' % (_sp - _st))
 
    out_dict = {}
    out_dict['loop_ret'] = loop_ret
    out_dict['outs_arr'] = np.asarray(outs_arr)
    out_dict['qsim_arr'] = np.asarray(qsim_arr)
    out_dict['lrst_arr'] = np.asarray(lrst_arr)
    return out_dict


cpdef dict hbv_c_loop_py(
    const DT_D[:, ::1] temp_arr,
    const DT_D[:, ::1] prec_arr,
    const DT_D[:, ::1] petn_arr,
    const DT_D[:, ::1] prms_arr,
    const DT_D[:, ::1] inis_arr,
    const DT_D[::1] area_arr,
    const DT_D rnof_q_conv):

    cdef:
        DT_UL n_time_steps = temp_arr.shape[1], opt_flag = 0
        DT_UL n_cells = temp_arr.shape[0]

        DT_D loop_ret#, _st, _sp

        DT_D[::1] lrst_arr
        DT_D[::1] qsim_arr
        DT_D[:, :, ::1] outs_arr
        
        dict out_dict
        
    assert temp_arr.shape[0] == prec_arr.shape[0] == petn_arr.shape[0]
    assert temp_arr.shape[1] == prec_arr.shape[1] == petn_arr.shape[1]
    assert temp_arr.shape[0] == prms_arr.shape[0] == area_arr.shape[0]
    assert prms_arr.shape[1] == n_hbv_prms

    outs_arr = np.zeros((n_cells, n_time_steps + 1, n_hbv_cols), dtype=DT_D_NP)
    qsim_arr = np.zeros(n_time_steps, dtype=DT_D_NP)
    lrst_arr = np.zeros(n_time_steps + 1, dtype=DT_D_NP)

#     _st = timeit.default_timer()
    loop_ret = hbv_c_loop(
        &temp_arr[0, 0],
        &prec_arr[0, 0],
        &petn_arr[0, 0],
        &prms_arr[0, 0],
        &inis_arr[0, 0],
        &area_arr[0],
        &lrst_arr[0],
        &qsim_arr[0],
        &outs_arr[0, 0, 0],
        &n_time_steps,
        &n_cells,
        &n_hbv_prms,
        &n_hbv_cols,
        &rnof_q_conv,
        &err_val,
        &opt_flag)
#     _sp = timeit.default_timer()
#     print('%0.6f secs for a loop!' % (_sp - _st))

    out_dict = {}
    out_dict['loop_ret'] = loop_ret
    out_dict['outs_arr'] = np.asarray(outs_arr)
    out_dict['qsim_arr'] = np.asarray(qsim_arr)
    out_dict['lrst_arr'] = np.asarray(lrst_arr)
    return out_dict


cpdef dict hbv_mult_cat_loop_py(args):
 
    '''Distributed HBV with multiple catchments and routing
    '''
    cdef:
        long curr_us_stm
        DT_UL route_type, cat_no, n_cpus, use_obs_flow_flag
        DT_UL cat, stm, off_idx, n_cells, opt_flag = 0
        DT_UL n_stms, n_hm_prms

        DT_D rnof_q_conv, signal

        cmap[long, long] cat_to_idx_map, stm_to_idx_map
         
        DT_UL[::1] stm_idxs

        DT_D[::1] qact_arr, obj_ftn_wts, qsim_arr, lrst_arr
        DT_D[::1] inflow_arr, area_arr

        DT_D[:, ::1] inis_arr, temp_arr, prec_arr, petn_arr
        DT_D[:, ::1] dem_net_arr, cats_outflow_arr, prms_arr
        DT_D[:, ::1] stms_inflow_arr, stms_outflow_arr

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
     n_hm_prms) = args[6]
     
    area_arr = args[7]
 
    inflow_arr = np.zeros(cats_outflow_arr.shape[0], dtype=DT_D_NP)
    qsim_arr = inflow_arr.copy()
    lrst_arr = np.zeros(cats_outflow_arr.shape[0] + 1, dtype=DT_D_NP)
    outs_arr = np.zeros((n_cells, cats_outflow_arr.shape[0] + 1, n_hbv_cols), 
                        dtype=DT_D_NP)

    for cat in cat_to_idx_dict:
        cat_to_idx_map[cat] = cat_to_idx_dict[cat]

    for stm in stm_to_idx_dict:
        stm_to_idx_map[stm] = stm_to_idx_dict[stm]

    misc_longs = np.zeros(obj_longs_ct, dtype=DT_UL_NP)
    misc_longs[0] = off_idx
    misc_longs[1] = curr_us_stm
    misc_longs[2] = n_hm_prms
    misc_longs[3] = route_type
    misc_longs[4] = cat_no
    misc_longs[5] = n_stms
    misc_longs[6] = n_hbv_cols
    misc_longs[7] = use_obs_flow_flag
    misc_longs[8] = opt_flag
    misc_longs[9] = n_cells
    misc_longs[10] = n_hbv_prms

    misc_doubles = np.zeros(obj_doubles_ct, dtype=DT_D_NP)
    misc_doubles[0] = rnof_q_conv
    misc_doubles[5] = err_val

    print('Simulating Catchment:', cat_no)

    signal = hbv_mult_cat_loop(
        stm_idxs,
        misc_longs,
        qact_arr,
        area_arr,
        qsim_arr,
        lrst_arr,
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
        print('signal was nan!')
        
        outs_arr = np.full((n_cells, 
                            cats_outflow_arr.shape[0] + 1, 
                            n_hbv_cols), 
                           np.nan, 
                           dtype=DT_D_NP)
        qsim_arr = np.full(cats_outflow_arr.shape[0], 
                           np.nan, 
                           dtype=DT_D_NP)
        lrst_arr = np.full(cats_outflow_arr.shape[0] + 1, 
                           np.nan, 
                           dtype=DT_D_NP)

    return {'out_arr': np.array(outs_arr),
            'params': np.array(hbv_prms),
            'cats_outflow_arr': np.array(cats_outflow_arr),
            'stms_inflow_arr': np.array(stms_inflow_arr),
            'stms_outflow_arr': np.array(stms_outflow_arr),
            'lrst_arr': np.array(lrst_arr),
            'qsim_arr': np.asarray(qsim_arr)}