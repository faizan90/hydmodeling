# cython: nonecheck=True
# cython: boundscheck=True
# cython: wraparound=True
# cython: cdivision=True
# cython: language_level=3
# cython: infer_types(None)

from ..miscs.dtypes cimport DT_D, DT_UL, DT_ULL


cpdef DT_D get_mean(
        const DT_D[::1] in_arr,
        const DT_UL off_idx,
        ) nogil


cpdef DT_D get_ln_mean(
        const DT_D[::1] in_arr,
        const DT_UL off_idx,
        ) nogil


cpdef DT_D get_demr(
        const DT_D[::1] x_arr,
        const DT_D mean_ref,
        const DT_UL off_idx,
        ) nogil


cpdef DT_D get_ln_demr(
        const DT_D[::1] x_arr,
        const DT_D ln_mean_ref,
        const DT_UL off_idx,
        ) nogil


cdef DT_D get_ns(
        const DT_D[::1] x_arr,
        const DT_D[::1] y_arr,
        const DT_D demr,
        const DT_UL off_idx,
        ) nogil


cdef DT_D get_ln_ns(
        const DT_D[::1] x_arr,
        const DT_D[::1] y_arr,    
        const DT_D demr,
        const DT_UL off_idx,
        ) nogil


cdef DT_D get_variance(
        const DT_D in_mean,
        const DT_D[::1] in_arr,
        const DT_UL off_idx,
        ) nogil


cdef DT_D get_covariance(
        const DT_D in_mean_1,
        const DT_D in_mean_2,
        const DT_D[::1] in_arr_1,
        const DT_D[::1] in_arr_2,
        const DT_UL off_idx,
        ) nogil


cdef DT_D _get_pcorr(
        const DT_D in_arr_1_std_dev,
        const DT_D in_arr_2_std_dev,
        const DT_D arrs_covar
        ) nogil


cdef DT_D get_kge(
        const DT_D[::1] act_arr,
        const DT_D[::1] sim_arr,
        const DT_D act_mean,
        const DT_D act_std_dev,
        const DT_UL off_idx,
        ) nogil


cdef DT_D get_pcorr_coeff(
        const DT_D[::1] x_arr,
        const DT_D[::1] y_arr,
        const DT_UL off_idx,
        ) nogil


cdef void del_idx(
        const DT_UL[::1] x_arr,
              DT_UL[::1] y_arr,
        const long idx,
        ) nogil


cdef void lin_regsn(
        const DT_D[::1] x_arr,
        const DT_D[::1] y_arr,
              DT_D[::1] y_arr_interp,
        const DT_UL off_idx,
              DT_D *corr,
              DT_D *slope,
              DT_D *intercept,
        ) nogil


cdef DT_D get_sum_sq_diff(
        const DT_D[::1] x_arr,
        const DT_D[::1] y_arr,
        const DT_UL off_idx,
        ) nogil


cdef DT_D get_ln_sum_sq_diff(
        const DT_D[::1] x_arr,
        const DT_D[::1] y_arr,
        const DT_UL off_idx,
        ) nogil


cdef DT_D cmpt_aspect_scale(const DT_D in_aspect) nogil


cdef void cmpt_aspect_scale_arr(
        const DT_D *in_aspect_arr,
              DT_D *out_scale_arr,
        const DT_UL n_cells) nogil


cdef DT_D cmpt_slope_scale(const DT_D in_slope) nogil


cdef void cmpt_slope_scale_arr(
        const DT_D *in_slope_arr,
              DT_D *out_scale_arr,
        const DT_UL n_cells) nogil


cdef DT_D cmpt_aspect_and_slope_scale(
        const DT_D in_aspect,
        const DT_D in_slope) nogil


cdef void cmpt_aspect_and_slope_scale_arr(
        const DT_D *in_aspect_arr,
        const DT_D *in_slope_arr,
              DT_D *out_scale_arr,
        const DT_UL n_cells) nogil


cdef DT_D cmpt_tt_from_scale(
        const DT_D in_scale,
        const DT_D min_tt,
        const DT_D max_tt,
        const DT_D exponent) nogil


cdef void cmpt_tt_from_scale_arr(
        const DT_D *in_scale_arr,
              DT_D *out_tt_arr,
        const DT_D min_tt,
        const DT_D max_tt,
        const DT_D exponent,
        const DT_UL n_cells) nogil


cdef void cmpt_resampled_arr(
        const DT_D[::1] ref_arr, 
              DT_D[::1] resamp_arr, 
        const DT_ULL[::1] tags_arr,
        ) nogil

cpdef DT_D get_hargreaves_pet(
        DT_UL d_o_y, 
        DT_D lat, 
        DT_D t_min, 
        DT_D t_max, 
        DT_D t_avg, 
        DT_UL leap) nogil
