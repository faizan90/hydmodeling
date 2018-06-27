# cython: nonecheck=False
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: language_level=3
# cython: infer_types=False
# cython: embedsignature=True

# Indicies of some variables in the bounds arr
cdef Py_ssize_t fc_i = 3, pwp_i = 5

cdef DT_UL n_hbv_cols = 9, n_hbv_prms = 11
cdef DT_UL obj_longs_ct = 11, obj_doubles_ct = 6
cdef DT_D err_val = 1e9

def get_fc_pwp_is():
    return (<long> fc_i, <long> pwp_i)
