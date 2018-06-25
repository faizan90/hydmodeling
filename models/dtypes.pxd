import numpy as np
cimport numpy as np

ctypedef double DT_D
ctypedef long DT_UL
ctypedef unsigned long long DT_ULL

cdef Py_ssize_t fc_i, pwp_i
cdef DT_UL n_hbv_cols, n_hbv_prms, obj_longs_ct, obj_doubles_ct
cdef DT_D err_val
