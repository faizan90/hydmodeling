# cython: nonecheck=False
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: language_level=3
# cython: infer_types=False
# cython: embedsignature=True

# Indicies of some variables in the bounds arr
cdef Py_ssize_t fc_i = 3, pwp_i = 5

# obj_longs indicies
cdef Py_ssize_t off_idx_i = 0, curr_us_stm_i = 1, n_hm_prms_i = 2
cdef Py_ssize_t route_type_i = 3, cat_no_i = 4, n_stms_i = 5, n_hbv_cols_i = 6
cdef Py_ssize_t use_obs_flow_flag_i = 7, opt_flag_i = 8, n_cells_i = 9
cdef Py_ssize_t n_hbv_prms_i = 10, use_step_flag_i = 11

# obj_doubles indicies
cdef Py_ssize_t rnof_q_conv_i = 0, demr_i = 1, ln_demr_i = 2, mean_ref_i = 3
cdef Py_ssize_t act_std_dev_i = 4, err_val_i = 5, min_q_thresh_i = 6

cdef DT_UL n_hbv_cols = 10, n_hbv_prms = 11

cdef DT_UL obj_longs_ct = 12, obj_doubles_ct = 7

cdef DT_D err_val = 1e9


def get_fc_pwp_is():
    return (<long> fc_i, <long> pwp_i)
