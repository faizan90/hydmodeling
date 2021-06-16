# cython: nonecheck=False
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: language_level=3
# cython: infer_types=False
# cython: embedsignature=True

from libcpp.vector cimport vector

ctypedef double DT_D
ctypedef long DT_UL  # unfortunately, this has to stay signed
ctypedef unsigned long long DT_ULL
ctypedef double complex DT_DC

ctypedef struct ForFourTrans1DReal:
    DT_D *orig      # The input array. N should be even.
    DT_DC *ft       # Fourier transform of orig.
    DT_D *amps      # Amplitudes of ft. Starting from index 1 to N//2.
    DT_D *angs      # Angles of ft. Starting from index 1 to N//2.
    DT_UL n_pts     # Number of values in orig

ctypedef vector[ForFourTrans1DReal *] ForFourTrans1DRealVec

cdef Py_ssize_t fc_i, pwp_i

cdef Py_ssize_t off_idx_i, curr_us_stm_i, n_hm_prms_i, route_type_i, cat_no_i
cdef Py_ssize_t n_stms_i, n_hbv_cols_i, use_obs_flow_flag_i, opt_flag_i
cdef Py_ssize_t n_cells_i, n_hbv_prms_i, use_step_flag_i
cdef Py_ssize_t resamp_obj_ftns_flag_i, a_zero_i
cdef Py_ssize_t ft_beg_idx_i, ft_end_idx_i
cdef Py_ssize_t use_res_cat_runoff_flag_i

cdef Py_ssize_t rnof_q_conv_i, demr_i, ln_demr_i, mean_ref_i, act_std_dev_i
cdef Py_ssize_t err_val_i, min_q_thresh_i, ft_demr_1_i, ft_demr_2_i
cdef Py_ssize_t demr_peak_i, ln_demr_peak_i, demr_sort_i, demr_qdiffs_i

cdef DT_UL n_hbv_cols, n_hbv_prms, obj_longs_ct, obj_doubles_ct
cdef DT_D err_val

cdef DT_UL max_bds_adj_atpts

cdef DT_D NAN
