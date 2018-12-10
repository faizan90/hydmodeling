import numpy as np
cimport numpy as np

ctypedef double DT_D
ctypedef long DT_UL
ctypedef unsigned long long DT_ULL

cdef Py_ssize_t fc_i, pwp_i

cdef Py_ssize_t off_idx_i, curr_us_stm_i, n_hm_prms_i, route_type_i, cat_no_i
cdef Py_ssize_t n_stms_i, n_hbv_cols_i, use_obs_flow_flag_i, opt_flag_i
cdef Py_ssize_t n_cells_i, n_hbv_prms_i, use_step_flag_i
cdef Py_ssize_t resamp_obj_ftns_flag_i, a_zero_i

cdef Py_ssize_t rnof_q_conv_i, demr_i, ln_demr_i, mean_ref_i, act_std_dev_i
cdef Py_ssize_t err_val_i, min_q_thresh_i

cdef DT_UL n_hbv_cols, n_hbv_prms, obj_longs_ct, obj_doubles_ct
cdef DT_D err_val
