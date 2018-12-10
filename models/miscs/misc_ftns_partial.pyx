# cython: nonecheck=False
# cython: boundscheck=True
# cython: wraparound=False
# cython: cdivision=True
# cython: language_level=3
# cython: infer_types=False
# cython: embedsignature=True


import numpy as np
cimport numpy as np

DT_D_NP = np.float64
DT_UL_NP = np.int32


cdef extern from "math.h" nogil:
    cdef:
        DT_D log(DT_D x)


cdef DT_D get_mean_prt(
        const DT_D[::1] in_arr,
        const DT_UL[::1] bool_arr,
        const DT_UL *off_idx,
        ) nogil:

    cdef:
        Py_ssize_t i
        DT_UL ctr = 0
        DT_D _sum = 0.0
        
    for i in range(off_idx[0], in_arr.shape[0]):
        if bool_arr[i]:
            _sum += in_arr[i]
            ctr += 1
    return _sum / ctr


cdef DT_D get_ln_mean_prt(
        const DT_D[::1] in_arr,
        const DT_UL[::1] bool_arr,
        const DT_UL *off_idx,
        ) nogil:

    cdef:
        Py_ssize_t i
        DT_UL ctr = 0
        DT_D _sum = 0.0
        
    for i in range(off_idx[0], in_arr.shape[0]):
        if bool_arr[i]:
            _sum += log(in_arr[i])
            ctr += 1
    return _sum / ctr


cdef DT_D get_demr_prt(
        const DT_D[::1] x_arr,
        const DT_UL[::1] bool_arr,
        const DT_D *mean_ref,
        const DT_UL *off_idx,
        ) nogil:

    cdef:
        Py_ssize_t i;
        DT_D demr = 0.0

    for i in range(off_idx[0], x_arr.shape[0]):
        if bool_arr[i]:
            demr += (x_arr[i] - mean_ref[0])**2
    return demr


cdef DT_D get_ln_demr_prt(
        const DT_D[::1] x_arr,
        const DT_UL[::1] bool_arr,
        const DT_D *ln_mean_ref,
        const DT_UL *off_idx,
        ) nogil:

    cdef:
        Py_ssize_t i
        DT_D demr = 0.0

    for i in range(off_idx[0], x_arr.shape[0]):
        if bool_arr[i]:
            demr += (log(x_arr[i]) - ln_mean_ref[0])**2
    return demr


cdef DT_D get_ns_prt(
        const DT_D[::1] x_arr,
        const DT_D[::1] y_arr,
        const DT_UL[::1] bool_arr,
        const DT_D *demr,
        const DT_UL *off_idx,
        ) nogil:
    
    cdef:
        Py_ssize_t i
        DT_D numr = 0.0
    
    for i in range(off_idx[0], x_arr.shape[0]):
        if bool_arr[i]:
            numr += (x_arr[i] - y_arr[i])**2
    return (1.0 - (numr / demr[0]))


cdef DT_D get_ln_ns_prt(
        const DT_D[::1] x_arr,
        const DT_D[::1] y_arr,    
        const DT_UL[::1] bool_arr,
        const DT_D *demr,
        const DT_UL *off_idx,
        ) nogil:
    
    cdef:
        Py_ssize_t i
        DT_D numr = 0.0
    
    for i in range(off_idx[0], x_arr.shape[0]):
        if bool_arr[i]:
            numr += (log(x_arr[i] / y_arr[i]))**2
    return (1.0 - (numr / demr[0]))


cdef DT_D get_variance_prt(
        const DT_D *in_mean,
        const DT_D[::1] in_arr,
        const DT_UL[::1] bool_arr,
        const DT_UL *off_idx,
        ) nogil:

    cdef:
        Py_ssize_t i
        DT_UL ctr = 0
        DT_D _sum = 0.0
        
    for i in range(off_idx[0], in_arr.shape[0]):
        if bool_arr[i]:
            _sum += (in_arr[i] - in_mean[0])**2
            ctr += 1
    return _sum / ctr
        

cdef DT_D get_covariance_prt(
        const DT_D *in_mean_1,
        const DT_D *in_mean_2,
        const DT_D[::1] in_arr_1,
        const DT_D[::1] in_arr_2,
        const DT_UL[::1] bool_arr,
        const DT_UL *off_idx,
        ) nogil: 

    cdef:
        Py_ssize_t i
        DT_UL ctr = 0
        DT_D _sum = 0.0

    for i in range(off_idx[0], in_arr_1.shape[0]):
        if bool_arr[i]:
            _sum += ((in_arr_1[i] - in_mean_1[0]) * 
                     (in_arr_2[i] - in_mean_2[0]))
            ctr += 1
    return _sum / ctr


cdef DT_D _get_pcorr_prt(
        const DT_D *in_arr_1_std_dev,
        const DT_D *in_arr_2_std_dev,
        const DT_D *arrs_covar
        ) nogil:

    return arrs_covar[0] / (in_arr_1_std_dev[0] * in_arr_2_std_dev[0])


cdef DT_D get_kge_prt(
        const DT_D[::1] act_arr,
        const DT_D[::1] sim_arr,
        const DT_UL[::1] bool_arr,
        const DT_D *act_mean,
        const DT_D *act_std_dev,
        const DT_UL *off_idx,
        ) nogil:

    cdef:
        DT_D sim_mean, sim_std_dev, covar, correl, b, g, kge

    sim_mean = get_mean_prt(sim_arr, bool_arr, off_idx);
    sim_std_dev = get_variance_prt(&sim_mean, sim_arr, bool_arr, off_idx)**0.5

    covar = get_covariance_prt(
        act_mean, &sim_mean, act_arr, sim_arr, bool_arr, off_idx)
    correl = _get_pcorr_prt(act_std_dev, &sim_std_dev, &covar)

    b = sim_mean / act_mean[0]
    g = (sim_std_dev / sim_mean) / (act_std_dev[0] / act_mean[0])

    kge = 1.0 - (((correl - 1)**2) + ((b - 1)**2) + ((g - 1)**2))**0.5
    return kge


cdef DT_D get_pcorr_coeff_prt(
        const DT_D[::1] x_arr,
        const DT_D[::1] y_arr,
        const DT_UL[::1] bool_arr,
        const DT_UL *off_idx,
        ) nogil:

    cdef:
        DT_D x_mean, y_mean, x_std_dev, y_std_dev, covar

    x_mean = get_mean_prt(x_arr, bool_arr, off_idx)
    y_mean = get_mean_prt(y_arr, bool_arr, off_idx)

    x_std_dev = get_variance_prt(&x_mean, x_arr, bool_arr, off_idx)**0.5
    y_std_dev = get_variance_prt(&y_mean, y_arr, bool_arr, off_idx)**0.5

    covar = get_covariance_prt(
        &x_mean, &y_mean, x_arr, y_arr, bool_arr, off_idx)
    return _get_pcorr_prt(&x_std_dev, &y_std_dev, &covar)


cdef DT_D get_sum_sq_diff_prt(
        const DT_D[::1] x_arr,
        const DT_D[::1] y_arr,
        const DT_UL[::1] bool_arr,
        const DT_UL *off_idx,
        ) nogil:

    cdef:
        Py_ssize_t i
        DT_D sum_sq_diff = 0.0

    for i in range(off_idx[0], x_arr.shape[0]):
        if bool_arr[i]:
            sum_sq_diff += (x_arr[i] - y_arr[i])**2
    return sum_sq_diff
    
    
cdef DT_D get_ln_sum_sq_diff_prt(
        const DT_D[::1] x_arr,
        const DT_D[::1] y_arr,
        const DT_UL[::1] bool_arr,
        const DT_UL *off_idx,
        ) nogil:

    cdef:
        Py_ssize_t i
        DT_D sum_sq_diff = 0.0

    for i in range(off_idx[0], x_arr.shape[0]):
        if bool_arr[i]:
            sum_sq_diff += (log(x_arr[i] / y_arr[i]))**2
    return sum_sq_diff


cdef void cmpt_resampled_arr_prt(
        const DT_D[::1] ref_arr, 
              DT_D[::1] resamp_arr, 
        const DT_ULL[::1] tags_arr,
        const DT_UL[::1] bools_arr,
        ) nogil:

    cdef:
        Py_ssize_t i, j
        
        DT_ULL beg_tag_idx, end_tag_idx
        
        DT_UL tag_val_ct

        DT_D tag_vals_sum

    for i in range(tags_arr.shape[0] - 1):
        beg_tag_idx = tags_arr[i]
        end_tag_idx = tags_arr[i + 1]
        
        tag_vals_sum = 0
        tag_val_ct = 0
        for j in range(beg_tag_idx, end_tag_idx):
            if bools_arr[j]:
                tag_vals_sum += ref_arr[j]
                tag_val_ct += 1

        resamp_arr[i] = tag_vals_sum / tag_val_ct
    return


def get_pcorr_prt_cy(
        const DT_D[::1] x_arr, 
        const DT_D[::1] y_arr, 
        const DT_UL[::1] bool_arr,
        const DT_UL off_idx):

    assert x_arr.shape[0] > off_idx    
    assert x_arr.shape[0] == y_arr.shape[0] == bool_arr.shape[0]
    
    return get_pcorr_coeff_prt(x_arr, y_arr, bool_arr, &off_idx)


def get_ns_prt_cy(
        const DT_D[::1] x_arr, 
        const DT_D[::1] y_arr, 
        const DT_UL[::1] bool_arr,
        const DT_UL off_idx):

    cdef:
        DT_D demr, mean

    assert x_arr.shape[0] > off_idx
    assert x_arr.shape[0] == y_arr.shape[0] == bool_arr.shape[0]

    mean = get_mean_prt(x_arr, bool_arr, &off_idx)
    demr = get_demr_prt(x_arr, bool_arr, &mean, &off_idx)
    return get_ns_prt(x_arr, y_arr, bool_arr, &demr, &off_idx)


def get_ln_ns_prt_cy(
        const DT_D[::1] x_arr, 
        const DT_D[::1] y_arr, 
        const DT_UL[::1] bool_arr,
        const DT_UL off_idx):

    cdef:
        DT_D demr, mean

    assert x_arr.shape[0] > off_idx    
    assert x_arr.shape[0] == y_arr.shape[0] == bool_arr.shape[0]

    mean = get_ln_mean_prt(x_arr, bool_arr, &off_idx)
    demr = get_ln_demr_prt(x_arr, bool_arr, &mean, &off_idx)
    return get_ln_ns_prt(x_arr, y_arr, bool_arr, &demr, &off_idx)


def get_kge_prt_cy(
        const DT_D[::1] x_arr, 
        const DT_D[::1] y_arr, 
        const DT_UL[::1] bool_arr,
        const DT_UL off_idx):

    cdef:
        DT_D mean_ref, act_std_dev

    assert x_arr.shape[0] > off_idx    
    assert x_arr.shape[0] == y_arr.shape[0] == bool_arr.shape[0]

    mean_ref = get_mean_prt(x_arr, bool_arr, &off_idx)
    act_std_dev = get_variance_prt(&mean_ref, x_arr, bool_arr, &off_idx)**0.5
    return get_kge_prt(
        x_arr, y_arr, bool_arr, &mean_ref, &act_std_dev, &off_idx)


def get_ns_var_res_prt_cy(
        const DT_D[::1] ref_arr, 
        const DT_D[::1] sim_arr,
        const DT_D[::1] cycle_arr,
        const DT_UL[::1] bool_arr,
        const DT_UL off_idx):

    assert (ref_arr.shape[0] == 
            sim_arr.shape[0] == 
            cycle_arr.shape[0] == 
            bool_arr.shape[0]), 'Inputs have unequal shapes!'
    assert off_idx < ref_arr.shape[0], 'off_idx is too big!'
    
    cdef:
        DT_D numr, demr
        
    numr = get_sum_sq_diff_prt(ref_arr, sim_arr, bool_arr, &off_idx)
    demr = get_sum_sq_diff_prt(ref_arr, cycle_arr, bool_arr, &off_idx)
    return 1 - (numr / demr)


def get_ln_ns_var_res_prt_cy(
        const DT_D[::1] ref_arr,
        const DT_D[::1] sim_arr,
        const DT_D[::1] cycle_arr,
        const DT_UL[::1] bool_arr,
        const DT_UL off_idx):

    assert (ref_arr.shape[0] == 
            sim_arr.shape[0] == 
            cycle_arr.shape[0] == 
            bool_arr.shape[0]), 'Inputs have unequal shapes!'
    assert off_idx < ref_arr.shape[0], 'off_idx is too big!'
    
    cdef:
        DT_D numr, demr

    numr = get_ln_sum_sq_diff_prt(ref_arr, sim_arr, bool_arr, &off_idx)
    demr = get_ln_sum_sq_diff_prt(ref_arr, cycle_arr, bool_arr, &off_idx)
    return 1 - (numr / demr)
