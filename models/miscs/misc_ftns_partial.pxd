from ..miscs.dtypes cimport DT_D, DT_UL, DT_ULL


cdef DT_D get_mean_prt(
        const DT_D[::1] in_arr,
        const DT_UL[::1] bool_arr,
        const DT_UL *off_idx,
        ) nogil


cdef DT_D get_ln_mean_prt(
        const DT_D[::1] in_arr,
        const DT_UL[::1] bool_arr,
        const DT_UL *off_idx,
        ) nogil


cdef DT_D get_demr_prt(
        const DT_D[::1] x_arr,
        const DT_UL[::1] bool_arr,
        const DT_D *mean_ref,
        const DT_UL *off_idx,
        ) nogil


cdef DT_D get_ln_demr_prt(
        const DT_D[::1] x_arr,
        const DT_UL[::1] bool_arr,
        const DT_D *ln_mean_ref,
        const DT_UL *off_idx,
        ) nogil


cdef DT_D get_ns_prt(
        const DT_D[::1] x_arr,
        const DT_D[::1] y_arr,
        const DT_UL[::1] bool_arr,
        const DT_D *demr,
        const DT_UL *off_idx,
        ) nogil


cdef DT_D get_ln_ns_prt(
        const DT_D[::1] x_arr,
        const DT_D[::1] y_arr,    
        const DT_UL[::1] bool_arr,
        const DT_D *demr,
        const DT_UL *off_idx,
        ) nogil


cdef DT_D get_variance_prt(
        const DT_D *in_mean,
        const DT_D[::1] in_arr,
        const DT_UL[::1] bool_arr,
        const DT_UL *off_idx,
        ) nogil


cdef DT_D get_covariance_prt(
        const DT_D *in_mean_1,
        const DT_D *in_mean_2,
        const DT_D[::1] in_arr_1,
        const DT_D[::1] in_arr_2,
        const DT_UL[::1] bool_arr,
        const DT_UL *off_idx,
        ) nogil


cdef DT_D _get_pcorr_prt(
        const DT_D *in_arr_1_std_dev,
        const DT_D *in_arr_2_std_dev,
        const DT_D *arrs_covar
        ) nogil


cdef DT_D get_kge_prt(
        const DT_D[::1] act_arr,
        const DT_D[::1] sim_arr,
        const DT_UL[::1] bool_arr,
        const DT_D *act_mean,
        const DT_D *act_std_dev,
        const DT_UL *off_idx,
        ) nogil


cdef DT_D get_pcorr_coeff_prt(
        const DT_D[::1] x_arr,
        const DT_D[::1] y_arr,
        const DT_UL[::1] bool_arr,
        const DT_UL *off_idx,
        ) nogil


cdef DT_D get_sum_sq_diff_prt(
        const DT_D[::1] x_arr,
        const DT_D[::1] y_arr,
        const DT_UL[::1] bool_arr,
        const DT_UL *off_idx,
        ) nogil


cdef DT_D get_ln_sum_sq_diff_prt(
        const DT_D[::1] x_arr,
        const DT_D[::1] y_arr,
        const DT_UL[::1] bool_arr,
        const DT_UL *off_idx,
        ) nogil


cdef void cmpt_resampled_arr_prt(
        const DT_D[::1] ref_arr, 
              DT_D[::1] resamp_arr, 
        const DT_ULL[::1] tags_arr,
        const DT_UL[::1] bools_arr,
        ) nogil
