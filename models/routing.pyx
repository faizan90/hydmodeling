# cython: nonecheck=True
# cython: boundscheck=True
# cython: wraparound=True
# cython: cdivision=True
# cython: language_level=3
# cython: infer_types=False
# cython: embedsignature=True


cdef void tfm_opt_to_route_prms(
    const DT_D[::1] opt_prms,
          DT_D[::1] route_prms,
    const DT_D[:, ::1] bds_dfs,
    ) nogil except +:

    cdef:
        Py_ssize_t i
    
    for i in range(opt_prms.shape[0]):
        route_prms[i] = bds_dfs[i, 0] + (bds_dfs[i, 1] * opt_prms[i])
    return


cdef void musk_route(
    const DT_D[::1] inflow_arr, 
          DT_D[:, ::1] outflow_arr,

    const DT_D *lag, 
    const DT_D *wt, 
    const DT_D *del_t,

    const long *n_recs, 
    const long *stm_idx,
    const long *n_stms,
    ) nogil except +:

    cdef:
        Py_ssize_t i
        DT_D C0, C1, C2, C3

    C0 = (2 * lag[0] * (1 - wt[0])) + del_t[0]
    
    C1 = (del_t[0] - (2 * lag[0] * wt[0])) / C0

    C2 = (del_t[0] + (2 * lag[0] * wt[0])) / C0

    C3 = ((2 * lag[0] * (1 - wt[0])) - del_t[0]) / C0

    outflow_arr[0, stm_idx[0]] = inflow_arr[0]
    for i in range(1, n_recs[0]):
        outflow_arr[i, stm_idx[0]] = ((inflow_arr[i] * C1) + 
                                      (inflow_arr[i - 1] * C2) + 
                                      (outflow_arr[i - 1, stm_idx[0]] * C3))
    return


cdef void route_strms(
    const DT_UL[::1] stm_idxs,

    cmap[long, long] &cat_to_idx_map,
    cmap[long, long] &stm_to_idx_map,

          DT_D[::1] inflow_arr,
          DT_D[::1] route_prms,

    const DT_D[:, ::1] dem_net_arr,
          DT_D[:, ::1] cats_outflow_arr,
          DT_D[:, ::1] stms_inflow_arr,
          DT_D[:, ::1] stms_outflow_arr,

    const DT_UL *n_stms,
    const DT_UL *route_type,
    ) nogil except +:
    
    cdef:
        Py_ssize_t i, j, opt_k, us_cat_idx, us1_idx, us2_idx
        long stm_idx, ds_cat, us_cat, us_01, us_02, stm
        DT_UL n_recs = inflow_arr.shape[0]
        DT_UL n_dem_net_cols = dem_net_arr.shape[1]
        DT_UL n_tot_stms = stms_inflow_arr.shape[1]
        DT_D lag, wt, del_t = 1.0

    for j in range(n_recs):
        inflow_arr[j] = 0.0

    opt_k = 0
    for i in range(n_stms[0]):
        stm = stm_idxs[i]
        stm_idx = stm_to_idx_map[stm]
        ds_cat = <long> dem_net_arr[stm, 0]
        us_cat = <long> dem_net_arr[stm, 5]
        us_01 = <long> dem_net_arr[stm, 3]
        us_02 = <long> dem_net_arr[stm, 4]
        
        if (us_cat != ds_cat):
            us_cat_idx = cat_to_idx_map[us_cat]
            for j in range(n_recs):
                inflow_arr[j] = cats_outflow_arr[j, us_cat_idx]

        else:
            if (us_01 != -2):
                us1_idx = stm_to_idx_map[us_01]
                for j in range(n_recs):
                    inflow_arr[j] = stms_outflow_arr[j, us1_idx]

            if (us_02 != -2):
                us2_idx = stm_to_idx_map[us_02]
                for j in range(n_recs):
                    inflow_arr[j] = (
                        inflow_arr[j] + 
                        stms_outflow_arr[j, us2_idx])

        for j in range(n_recs):
            stms_inflow_arr[j, stm_idx] = inflow_arr[j]

        if (route_type[0] == 0):
            for j in range(n_recs):
                stms_outflow_arr[j, stm_idx] = inflow_arr[j]

        elif (route_type[0] == 1):
            lag = route_prms[opt_k]
            wt = route_prms[opt_k + 1]

            musk_route(inflow_arr,
                       stms_outflow_arr,
                       &lag,
                       &wt,
                       &del_t,
                       &n_recs,
                       &stm_idx,
                       &n_tot_stms)

        else:
            with gil: print(('Incorrect route_type: %d' % route_type[0]))

        opt_k = opt_k + 2
    return
