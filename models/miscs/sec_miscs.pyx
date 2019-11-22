# cython: nonecheck=False
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: language_level=3
# cython: infer_types=False
# cython: embedsignature=True

from .dtypes cimport (
    off_idx_i,
    use_step_flag_i,
    resamp_obj_ftns_flag_i,
    a_zero_i,
    demr_i,
    ln_demr_i,
    mean_ref_i,
    act_std_dev_i,
    NAN)
from .misc_ftns cimport (
    get_demr, 
    get_ln_demr,
    get_mean,
    get_ln_mean,
    get_variance,
    cmpt_resampled_arr)
from .misc_ftns_partial cimport (
    get_demr_prt, 
    get_ln_demr_prt,
    get_mean_prt,
    get_ln_mean_prt,
    get_variance_prt,
    cmpt_resampled_arr_prt)


cdef void update_obj_doubles(
        const DT_UL[::1] obj_longs,
        const DT_UL[::1] use_step_arr,

        const DT_ULL[::1] obj_ftn_resamp_tags_arr,

        const DT_D[::1] obj_ftn_wts,
              DT_D[::1] obj_doubles,
        const DT_D[::1] q_arr,
              DT_D[::1] q_resamp_arr,
        ) nogil except +:

    cdef:
        DT_D mean_ref = NAN
        DT_D ln_mean_ref = NAN
        DT_D demr = NAN
        DT_D ln_demr = NAN
        DT_D act_std_dev = NAN

    if obj_longs[resamp_obj_ftns_flag_i]:
        if obj_longs[use_step_flag_i]:
            cmpt_resampled_arr_prt(
                q_arr, 
                q_resamp_arr,
                obj_ftn_resamp_tags_arr,
                use_step_arr)

            if obj_ftn_wts[0] or obj_ftn_wts[2]:
                mean_ref = get_mean_prt(
                    q_resamp_arr,
                    use_step_arr,
                    obj_longs[a_zero_i])
    
            if obj_ftn_wts[0]:
                demr = get_demr_prt(
                    q_resamp_arr,
                    use_step_arr,
                    mean_ref,
                    obj_longs[a_zero_i])

            if obj_ftn_wts[1]:
                ln_mean_ref = get_ln_mean_prt(
                    q_resamp_arr, use_step_arr, obj_longs[a_zero_i])
    
                ln_demr = get_ln_demr_prt(
                    q_resamp_arr,
                    use_step_arr,
                    ln_mean_ref,
                    obj_longs[a_zero_i])

            if obj_ftn_wts[2]:
                act_std_dev = get_variance_prt(
                    mean_ref,
                    q_resamp_arr,
                    use_step_arr,
                    obj_longs[a_zero_i])**0.5

        else:
            cmpt_resampled_arr(
                q_arr,
                q_resamp_arr,
                obj_ftn_resamp_tags_arr)

            if obj_ftn_wts[0] or obj_ftn_wts[2]:
                mean_ref = get_mean(q_resamp_arr, obj_longs[a_zero_i])

            if obj_ftn_wts[0]:
                demr = get_demr(q_resamp_arr, mean_ref, obj_longs[a_zero_i])

            if obj_ftn_wts[1]:
                ln_mean_ref = get_ln_mean(q_resamp_arr, obj_longs[a_zero_i])

                ln_demr = get_ln_demr(
                    q_resamp_arr, ln_mean_ref, obj_longs[a_zero_i])

            if obj_ftn_wts[2]:
                act_std_dev = get_variance(
                    mean_ref, q_resamp_arr, obj_longs[a_zero_i])**0.5

    else:
        if obj_longs[use_step_flag_i]:
            if obj_ftn_wts[0] or obj_ftn_wts[2]:
                mean_ref = get_mean_prt(
                    q_arr, use_step_arr, obj_longs[off_idx_i])

            if obj_ftn_wts[0]:
                demr = get_demr_prt(
                    q_arr, use_step_arr, mean_ref, obj_longs[off_idx_i])

            if obj_ftn_wts[1]:
                ln_mean_ref = get_ln_mean_prt(
                    q_arr, use_step_arr, obj_longs[off_idx_i])
    
                ln_demr = get_ln_demr_prt(
                    q_arr, use_step_arr, ln_mean_ref, obj_longs[off_idx_i])

            if obj_ftn_wts[2]:
                act_std_dev = get_variance_prt(
                    mean_ref, q_arr, use_step_arr, obj_longs[off_idx_i])**0.5

        else:
            if obj_ftn_wts[0] or obj_ftn_wts[2]:
                mean_ref = get_mean(q_arr, obj_longs[off_idx_i])

            if obj_ftn_wts[0]:
                demr = get_demr(q_arr, mean_ref, obj_longs[off_idx_i])

            if obj_ftn_wts[1]:
                ln_mean_ref = get_ln_mean(q_arr, obj_longs[off_idx_i])
                
                ln_demr = get_ln_demr(q_arr, ln_mean_ref, obj_longs[off_idx_i])

            if obj_ftn_wts[2]:
                act_std_dev = get_variance(
                    mean_ref, q_arr, obj_longs[off_idx_i])**0.5

    obj_doubles[demr_i] = demr
    obj_doubles[ln_demr_i] = ln_demr
    obj_doubles[mean_ref_i] = mean_ref
    obj_doubles[act_std_dev_i] = act_std_dev
    return
