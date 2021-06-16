# cython: nonecheck=False
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: language_level=3
# cython: infer_types=False
# cython: embedsignature=True

import numpy as np
cimport numpy as np

# Indicies of some variables in the bounds arr
cdef fc_i = 3, pwp_i = 5

# obj_longs indicies
cdef off_idx_i = 0, curr_us_stm_i = 1, n_hm_prms_i = 2
cdef route_type_i = 3, cat_no_i = 4, n_stms_i = 5, n_hbv_cols_i = 6
cdef use_obs_flow_flag_i = 7, opt_flag_i = 8, n_cells_i = 9
cdef n_hbv_prms_i = 10, use_step_flag_i = 11
cdef resamp_obj_ftns_flag_i = 12, a_zero_i = 13
cdef ft_beg_idx_i = 14, ft_end_idx_i = 15
cdef use_res_cat_runoff_flag_i = 16

# obj_doubles indicies, update obj_doubles_ct accordingly.
cdef rnof_q_conv_i = 0, demr_i = 1, ln_demr_i = 2, mean_ref_i = 3
cdef act_std_dev_i = 4, err_val_i = 5, min_q_thresh_i = 6
cdef ft_demr_1_i = 7, ft_demr_2_i = 8
cdef demr_peak_i = 9, ln_demr_peak_i = 10
cdef demr_sort_i = 11, demr_qdiffs_i = 12

# other variables
cdef n_hbv_cols = 10, n_hbv_prms = 11

cdef obj_longs_ct = 17, obj_doubles_ct = 13

cdef err_val = 1e9

cdef max_bds_adj_atpts = 1000

cdef NAN = np.nan

def get_fc_pwp_is():
    return (<long> fc_i, <long> pwp_i)
