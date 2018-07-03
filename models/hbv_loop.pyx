# cython: nonecheck=False
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: language_level=3
# cython: infer_types=False
# cython: embedsignature=True


cdef DT_D min(const DT_D a, const DT_D b) nogil:
    if a <= b:
        return a
    else:
        return b


cdef DT_D max(const DT_D a, const DT_D b) nogil:
    if a >= b:
        return a
    else:
        return b


cdef DT_D hbv_loop(
          DT_D[:, ::1] temp_arr,
          DT_D[:, ::1] prec_arr,
          DT_D[:, ::1] petn_arr,
          DT_D[:, ::1] prms_arr,
    const DT_D[:, ::1] inis_arr,
    const DT_D[::1] area_arr,
          DT_D[::1] qsim_arr,
          DT_D[:, :, ::1] outs_arr,
    const DT_D *rnof_q_conv,
    const DT_D *err_val,
    const DT_UL *opt_flag,
    ) nogil except +:

    cdef:
        Py_ssize_t i, j, o_i

        # var idxs
        Py_ssize_t snow_i, lppt_i, somo_i, rnof_i, evtn_i
        Py_ssize_t ur_i, ur_uo_i, ur_lo_i, ur_lr_i, lr_i

        # vars
        DT_D temp, prec, petn, pre_snow, snow_melt, lppt, pre_somo
        DT_D pre_ur_sto, ur_uo_rnof, ur_lo_rnof, ur_lr_seep, lr_rnof
        DT_D rel_fc, rel_fc_beta, cell_area, pre_lr_sto, p_cm

        # prm idxs
        Py_ssize_t tt_i, cm_i, pwp_i, fc_i, beta_i, k_uu_i
        Py_ssize_t ur_thr_i, k_ul_i, k_d_i, k_ll_i, p_cm_i

        # prms
        DT_D tt, cm, pwp, fc, beta, k_uu, ur_thr, k_ul, k_d, k_ll

        # previous current cell and time arrays
        DT_D[:] temp_j_arr, prec_j_arr, petn_j_arr, prms_j_arr
        DT_D[:, :] outs_j_arr

    # indicies of variables in the outs_arr
    snow_i = 0
    lppt_i = 1
    somo_i = 2
    rnof_i = 3
    evtn_i = 4
    ur_i = 5
    ur_uo_i = 6
    ur_lo_i = 7
    ur_lr_i = 8
    lr_i = 9

    # indicies of variables in the prms_arr    
    tt_i = 0
    cm_i = 1
    p_cm_i = 2
    fc_i = 3
    beta_i = 4
    pwp_i = 5
    ur_thr_i = 6
    k_uu_i = 7
    k_ul_i = 8
    k_d_i = 9
    k_ll_i = 10

    # assign initial values
    for j in range(inis_arr.shape[0]):
        outs_arr[j, 0, snow_i] = inis_arr[j, 0]
        outs_arr[j, 0, somo_i] = inis_arr[j, 1]
        outs_arr[j, 0, ur_i] = inis_arr[j, 2]
        outs_arr[j, 0, lr_i] = inis_arr[j, 3]
        
    for j in range(temp_arr.shape[0]):
        temp_j_arr = temp_arr[j, :]
        prec_j_arr = prec_arr[j, :]
        petn_j_arr = petn_arr[j, :]
        prms_j_arr = prms_arr[j, :]

#         if opt_flag[0]:
#             outs_j_arr = outs_arr[0, :, :]
#         else:
        outs_j_arr = outs_arr[j, :, :]

        tt = prms_j_arr[tt_i]
        cm = prms_j_arr[cm_i]
        p_cm = prms_j_arr[p_cm_i]
        fc = prms_j_arr[fc_i]
        beta = prms_j_arr[beta_i]
        pwp = prms_j_arr[pwp_i]
        ur_thr = prms_j_arr[ur_thr_i]
        k_uu = prms_j_arr[k_uu_i]
        k_ul = prms_j_arr[k_ul_i]
        k_d = prms_j_arr[k_d_i]
        k_ll = prms_j_arr[k_ll_i]

        cell_area = area_arr[j]

#         with gil:
#             print(f'{tt:0.4f}, {cm:0.4f}, {p_cm:0.4f}, {fc:0.4f}, {beta:0.4f}, '
#                   f'{pwp:0.4f}, {ur_thr:0.4f}, {k_uu:0.4f}, {k_ul:0.4f}, '
#                   f'{k_d:0.4f}, {k_ll:0.4f}')

        pre_snow = outs_j_arr[0, snow_i]
        pre_somo = outs_j_arr[0, somo_i]
        pre_ur_sto = outs_j_arr[0, ur_i]
        pre_lr_sto = outs_j_arr[0, lr_i]

        for o_i in range(1, temp_arr.shape[1] + 1):
            if opt_flag[0]:
                i = 0
            else:
                i = o_i

            temp = temp_j_arr[o_i - 1]
            prec = prec_j_arr[o_i - 1]
            petn = petn_j_arr[o_i - 1]

            # snow and liquid ppt
            if temp < tt:
                outs_j_arr[i, snow_i] = pre_snow + prec
                outs_j_arr[i, lppt_i] = 0.0
            else:
                snow_melt = ((cm + (p_cm * prec)) * (temp - tt))
                outs_j_arr[i, snow_i] = max(0.0,  pre_snow - snow_melt)
                outs_j_arr[i, lppt_i] = prec + min(pre_snow, snow_melt)

            pre_snow = outs_j_arr[i, snow_i]

            lppt = outs_j_arr[i, lppt_i]

            # soil moisture and ET
            if (pre_somo < 0) or (pre_somo > fc):
                #with gil: print('%f, %f' % (pre_somo, fc))
#                 with gil:
#                     print(pre_somo, fc, outs_j_arr[i-1, lppt_i], outs_j_arr[i-1, evtn_i], rel_fc_beta, beta, pwp)
#                     print(f'{tt:0.4f}, {cm:0.4f}, {p_cm:0.4f}, {fc:0.4f}, {beta:0.4f}, '
#                           f'{pwp:0.4f}, {ur_thr:0.4f}, {k_uu:0.4f}, {k_ul:0.4f}, '
#                           f'{k_d:0.4f}, {k_ll:0.4f}')
#                     print(o_i, i, j)

                return -err_val[0]

            if pre_somo > pwp:
                outs_j_arr[i, evtn_i] = petn
            else:
                outs_j_arr[i, evtn_i] = (pre_somo / fc) * petn 

            rel_fc_beta = (pre_somo / fc)**beta

#             if rel_fc_beta > 1:
#                 with gil:
#                     print(j, i, rel_fc_beta, pre_somo, fc, pwp, beta)

            outs_j_arr[i, somo_i] = (pre_somo -
                                     outs_j_arr[i, evtn_i] +
                                     (lppt * (1 - rel_fc_beta)))
            pre_somo = outs_j_arr[i, somo_i]
            outs_j_arr[i, rnof_i] = lppt * rel_fc_beta

            # runonff, upper reservoir, upper outlet
            outs_j_arr[i, ur_uo_i] = (
                max(0.0, (pre_ur_sto - ur_thr) * k_uu))
            ur_uo_rnof = outs_j_arr[i, ur_uo_i]

            # seepage to groundwater
            outs_j_arr[i, ur_lr_i] = (
                max(0.0, (pre_ur_sto - ur_uo_rnof) * k_d))
            ur_lr_seep = outs_j_arr[i, ur_lr_i]

            # runoff, upper reservoir, lower outlet 
            outs_j_arr[i, ur_lo_i] = (
                max(0.0, (pre_ur_sto -
                          ur_uo_rnof -
                          ur_lr_seep) * k_ul))
            ur_lo_rnof = outs_j_arr[i, ur_lo_i] 

            # upper reservoir storage
            outs_j_arr[i, ur_i] = (
                max(0.0,
                    (pre_ur_sto -
                     ur_uo_rnof -
                     ur_lo_rnof -
                     ur_lr_seep +
                     outs_j_arr[i, rnof_i])))
            pre_ur_sto = outs_j_arr[i, ur_i]

            # lower reservoir runoff and storage
            lr_rnof = pre_lr_sto * k_ll
            outs_j_arr[i, lr_i] = (pre_lr_sto + ur_lr_seep - lr_rnof)
            pre_lr_sto = outs_j_arr[i, lr_i]

            # upper and lower reservoirs combined discharge
            qsim_arr[o_i - 1] += (rnof_q_conv[0] * cell_area * 
                                  (ur_uo_rnof + ur_lo_rnof + lr_rnof))

    return 0.0
