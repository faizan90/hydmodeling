# cython: nonecheck=False
# cython: boundscheck=False
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
        DT_D cos(DT_D x)
        DT_D sin(DT_D x)
        DT_D M_PI
        DT_D acos(DT_D x)
        DT_D tan(DT_D x)


cpdef DT_D get_mean(
        const DT_D[::1] in_arr,
        const DT_UL off_idx,
        ) nogil:

    cdef:
        Py_ssize_t i
        DT_D _sum = 0.0

    for i in range(off_idx, in_arr.shape[0]):
        _sum += in_arr[i]

    return _sum / (in_arr.shape[0] - off_idx)


cpdef DT_D get_ln_mean(
        const DT_D[::1] in_arr,
        const DT_UL off_idx,
        ) nogil:

    cdef:
        Py_ssize_t i
        DT_D _sum = 0.0

    for i in range(off_idx, in_arr.shape[0]):
        _sum += log(in_arr[i])

    return _sum / (in_arr.shape[0] - off_idx)


cpdef DT_D get_demr(
        const DT_D[::1] x_arr,
        const DT_D mean_ref,
        const DT_UL off_idx,
        ) nogil:

    cdef:
        Py_ssize_t i;
        DT_D demr = 0.0

    for i in range(off_idx, x_arr.shape[0]):
        demr += (x_arr[i] - mean_ref)**2

    return demr


cpdef DT_D get_ln_demr(
        const DT_D[::1] x_arr,
        const DT_D ln_mean_ref,
        const DT_UL off_idx,
        ) nogil:

    cdef:
        Py_ssize_t i
        DT_D demr = 0.0

    for i in range(off_idx, x_arr.shape[0]):
        demr += (log(x_arr[i]) - ln_mean_ref)**2

    return demr


cdef DT_D get_ns(
        const DT_D[::1] x_arr,
        const DT_D[::1] y_arr,
        const DT_D demr,
        const DT_UL off_idx,
        ) nogil:

    cdef:
        Py_ssize_t i
        DT_D numr = 0.0

    for i in range(off_idx, x_arr.shape[0]):
        numr += (x_arr[i] - y_arr[i])**2

    return (1.0 - (numr / demr))


cdef DT_D get_ln_ns(
        const DT_D[::1] x_arr,
        const DT_D[::1] y_arr,    
        const DT_D demr,
        const DT_UL off_idx,
        ) nogil:

    cdef:
        Py_ssize_t i
        DT_D numr = 0.0

    for i in range(off_idx, x_arr.shape[0]):
        numr += (log(x_arr[i] / y_arr[i]))**2

    return (1.0 - (numr / demr))


cdef DT_D get_variance(
        const DT_D in_mean,
        const DT_D[::1] in_arr,
        const DT_UL off_idx,
        ) nogil:

    cdef:
        Py_ssize_t i
        DT_D _sum = 0.0
        
    for i in range(off_idx, in_arr.shape[0]):
        _sum += (in_arr[i] - in_mean)**2

    return _sum / (in_arr.shape[0] - off_idx + 1)
        

cdef DT_D get_covariance(
        const DT_D in_mean_1,
        const DT_D in_mean_2,
        const DT_D[::1] in_arr_1,
        const DT_D[::1] in_arr_2,
        const DT_UL off_idx,
        ) nogil: 

    cdef:
        Py_ssize_t i
        DT_D _sum = 0.0
        
    for i in range(off_idx, in_arr_1.shape[0]):
        _sum += ((in_arr_1[i] - in_mean_1) * 
                 (in_arr_2[i] - in_mean_2))

    return _sum / (in_arr_1.shape[0] - off_idx)


cdef DT_D _get_pcorr(
        const DT_D in_arr_1_std_dev,
        const DT_D in_arr_2_std_dev,
        const DT_D arrs_covar
        ) nogil:

    return arrs_covar / (in_arr_1_std_dev * in_arr_2_std_dev)


cdef DT_D get_kge(
        const DT_D[::1] act_arr,
        const DT_D[::1] sim_arr,
        const DT_D act_mean,
        const DT_D act_std_dev,
        const DT_UL off_idx,
        ) nogil:

    cdef:
        DT_D sim_mean, sim_std_dev, covar, correl, b, g, kge

    sim_mean = get_mean(sim_arr, off_idx);
    sim_std_dev = get_variance(sim_mean, sim_arr, off_idx)**0.5

    covar = get_covariance(act_mean, sim_mean, act_arr, sim_arr, off_idx)
    correl = _get_pcorr(act_std_dev, sim_std_dev, covar)

    b = sim_mean / act_mean
    g = sim_std_dev / act_std_dev

    kge = 1.0 - (((correl - 1)**2) + ((b - 1)**2) + ((g - 1)**2))**0.5
    return kge


cdef DT_D get_pcorr_coeff(
        const DT_D[::1] x_arr,
        const DT_D[::1] y_arr,
        const DT_UL off_idx,
        ) nogil:

    cdef:
        DT_D x_mean, y_mean, x_std_dev, y_std_dev, covar

    x_mean = get_mean(x_arr, off_idx)
    y_mean = get_mean(y_arr, off_idx)

    x_std_dev = get_variance(x_mean, x_arr, off_idx)**0.5
    y_std_dev = get_variance(y_mean, y_arr, off_idx)**0.5

    covar = get_covariance(x_mean, y_mean, x_arr, y_arr, off_idx)
    return _get_pcorr(x_std_dev, y_std_dev, covar)


cdef void del_idx(
        const DT_UL[::1] x_arr,
              DT_UL[::1] y_arr,
        const long idx,
        ) nogil:

    cdef:
        DT_UL i = 0, j = 0

    while (i < y_arr.shape[0]):
        if (i != idx):
            y_arr[i] = x_arr[j]

        else:
            j += 1
            y_arr[i] = x_arr[j]

        i += 1
        j += 1
    return


cdef void lin_regsn(
        const DT_D[::1] x_arr,
        const DT_D[::1] y_arr,
              DT_D[::1] y_arr_interp,
        const DT_UL off_idx,
              DT_D *corr,
              DT_D *slope,
              DT_D *intercept,
        ) nogil:

    cdef:
        Py_ssize_t i
        DT_D x_m, y_m, covar, x_s, y_s

    x_m = get_mean(x_arr, off_idx)
    x_s = get_variance(x_m, x_arr, off_idx)**0.5

    y_m = get_mean(y_arr, off_idx)
    y_s = get_variance(y_m, y_arr, off_idx)**0.5

    covar = get_covariance(x_m, y_m, x_arr, y_arr, off_idx)

    corr[0] = _get_pcorr(x_s, y_s, covar)

    slope[0] = corr[0] * y_s / x_s
    intercept[0] = y_m - (slope[0] * x_m)

    for i in range(x_arr.shape[0]):
        y_arr_interp[i] = (slope[0] * x_arr[i]) + intercept[0]
    return


cdef DT_D get_sum_sq_diff(
        const DT_D[::1] x_arr,
        const DT_D[::1] y_arr,
        const DT_UL off_idx,
        ) nogil:

    cdef:
        Py_ssize_t i
        DT_D sum_sq_diff = 0.0

    for i in range(off_idx, x_arr.shape[0]):
        sum_sq_diff += (x_arr[i] - y_arr[i])**2

    return sum_sq_diff


cdef DT_D get_ln_sum_sq_diff(
        const DT_D[::1] x_arr,
        const DT_D[::1] y_arr,
        const DT_UL off_idx,
        ) nogil:

    cdef:
        Py_ssize_t i
        DT_D sum_sq_diff = 0.0

    for i in range(off_idx, x_arr.shape[0]):
        sum_sq_diff += (log(x_arr[i] / y_arr[i]))**2

    return sum_sq_diff


cdef inline DT_D cmpt_aspect_scale(const DT_D in_aspect) nogil:
    return 0.5 * (cos(in_aspect) + 1)


cdef void cmpt_aspect_scale_arr(
        const DT_D *in_aspect_arr,
              DT_D *out_scale_arr,
        const DT_UL n_cells) nogil:

    cdef:
        Py_ssize_t i

    for i in range(n_cells):
        out_scale_arr[i] = cmpt_aspect_scale(in_aspect_arr[i])
    return


cdef inline DT_D cmpt_slope_scale(const DT_D in_slope) nogil:
    return 0.5 * (sin(in_slope) + 1)


cdef void cmpt_slope_scale_arr(
        const DT_D *in_slope_arr,
              DT_D *out_scale_arr,
        const DT_UL n_cells) nogil:

    cdef:
        Py_ssize_t i

    for i in range(n_cells):
        out_scale_arr[i] = cmpt_slope_scale(in_slope_arr[i])
    return


cdef inline DT_D cmpt_aspect_and_slope_scale(
        const DT_D in_aspect,
        const DT_D in_slope) nogil:
    return 0.5 * ((cos(in_aspect) * sin(in_slope)) + 1)


cdef void cmpt_aspect_and_slope_scale_arr(
        const DT_D *in_aspect_arr,
        const DT_D *in_slope_arr,
              DT_D *out_scale_arr,
        const DT_UL n_cells) nogil:

    cdef:
        Py_ssize_t i

    for i in range(n_cells):
        out_scale_arr[i] = cmpt_aspect_and_slope_scale(
            in_aspect_arr[i],
            in_slope_arr[i])
    return


cdef inline DT_D cmpt_tt_from_scale(
        const DT_D in_scale,
        const DT_D min_tt,
        const DT_D max_tt,
        const DT_D exponent) nogil:

    return min_tt + (
        (max_tt - min_tt) * (in_scale**exponent))


cdef void cmpt_tt_from_scale_arr(
        const DT_D *in_scale_arr,
              DT_D *out_tt_arr,
        const DT_D min_tt,
        const DT_D max_tt,
        const DT_D exponent,
        const DT_UL n_cells) nogil:

    cdef:
        Py_ssize_t i

    for i in range(n_cells):
        out_tt_arr[i] = cmpt_tt_from_scale(
            in_scale_arr[i], 
            min_tt, 
            max_tt, 
            exponent)
    return


cdef void cmpt_resampled_arr(
        const DT_D[::1] ref_arr, 
              DT_D[::1] resamp_arr, 
        const DT_ULL[::1] tags_arr,
        ) nogil:

    cdef:
        Py_ssize_t i

        DT_ULL j, beg_tag_idx, end_tag_idx

        DT_D tag_vals_sum

    for i in range(tags_arr.shape[0] - 1):
        beg_tag_idx = tags_arr[i]
        end_tag_idx = tags_arr[i + 1]

        tag_vals_sum = 0
        for j in range(beg_tag_idx, end_tag_idx):
            tag_vals_sum += ref_arr[j]

        resamp_arr[i] = tag_vals_sum / (end_tag_idx - beg_tag_idx)
    return

def get_resampled_arr_cy(ref_arr, tags_arr):

    cdef:
        DT_D[::1] resamp_arr = np.full(tags_arr.shape[0] - 1, np.nan)

    cmpt_resampled_arr(ref_arr, resamp_arr, tags_arr)
    return np.asarray(resamp_arr)


def get_pcorr_cy(
        const DT_D[::1] x_arr, 
        const DT_D[::1] y_arr, 
        const DT_UL off_idx):

    assert x_arr.shape[0] > off_idx    
    assert x_arr.shape[0] == y_arr.shape[0]
    
    return get_pcorr_coeff(x_arr, y_arr, off_idx)


def get_ns_cy(
        const DT_D[::1] x_arr, 
        const DT_D[::1] y_arr, 
        const DT_UL off_idx):

    cdef:
        DT_D demr, mean

    assert x_arr.shape[0] > off_idx    
    assert x_arr.shape[0] == y_arr.shape[0]

    mean = get_mean(x_arr, off_idx)
    demr = get_demr(x_arr, mean, off_idx)
    return get_ns(x_arr, y_arr, demr, off_idx)


def get_ln_ns_cy(
        const DT_D[::1] x_arr, 
        const DT_D[::1] y_arr, 
        const DT_UL off_idx):

    cdef:
        DT_D demr, mean

    assert x_arr.shape[0] > off_idx    
    assert x_arr.shape[0] == y_arr.shape[0]

    mean = get_ln_mean(x_arr, off_idx)
    demr = get_ln_demr(x_arr, mean, off_idx)
    return get_ln_ns(x_arr, y_arr, demr, off_idx)


def get_kge_cy(
        const DT_D[::1] x_arr, 
        const DT_D[::1] y_arr, 
        const DT_UL off_idx):

    cdef:
        DT_D mean_ref, act_std_dev

    assert x_arr.shape[0] > off_idx    
    assert x_arr.shape[0] == y_arr.shape[0]

    mean_ref = get_mean(x_arr, off_idx)
    act_std_dev = get_variance(mean_ref, x_arr, off_idx)**0.5
    return get_kge(x_arr, y_arr, mean_ref, act_std_dev, off_idx)


def get_aspect_scale_arr_cy(const DT_D[::1] in_aspect_arr):

    cdef:
        DT_UL n_cells = in_aspect_arr.shape[0]
        DT_D[::1] out_scale_arr

    out_scale_arr = np.empty(n_cells, dtype=DT_D_NP)

    cmpt_aspect_scale_arr(&in_aspect_arr[0], &out_scale_arr[0], n_cells)
    return np.asarray(out_scale_arr)


def get_slope_scale_arr_cy(const DT_D[::1] in_slope_arr):

    cdef:
        DT_UL n_cells = in_slope_arr.shape[0]
        DT_D[::1] out_scale_arr

    out_scale_arr = np.empty(n_cells, dtype=DT_D_NP)

    cmpt_slope_scale_arr(&in_slope_arr[0], &out_scale_arr[0], n_cells)
    return np.asarray(out_scale_arr)


def get_aspect_and_slope_scale_arr_cy(
        const DT_D[::1] in_aspect_arr, 
        const DT_D[::1] in_slope_arr):

    cdef:
        DT_UL n_cells = in_aspect_arr.shape[0]
        DT_D[::1] out_scale_arr

    assert in_aspect_arr.shape[0] == in_slope_arr.shape[0]

    out_scale_arr = np.empty(n_cells, dtype=DT_D_NP)

    cmpt_aspect_and_slope_scale_arr(
        &in_aspect_arr[0],
        &in_slope_arr[0],
        &out_scale_arr[0],
        n_cells)
    return np.asarray(out_scale_arr)


def get_ns_var_res_cy(
        const DT_D[::1] ref_arr, 
        const DT_D[::1] sim_arr,
        const DT_D[::1] cycle_arr,
        const DT_UL off_idx):

    assert ref_arr.shape[0] == sim_arr.shape[0] == cycle_arr.shape[0], (
        'Inputs have unequal shapes!')
    assert off_idx < ref_arr.shape[0], 'off_idx is too big!'

    cdef:
        DT_D numr, demr

    numr = get_sum_sq_diff(ref_arr, sim_arr, off_idx)
    demr = get_sum_sq_diff(ref_arr, cycle_arr, off_idx)
    return 1 - (numr / demr)


def get_ln_ns_var_res_cy(
        const DT_D[::1] ref_arr,
        const DT_D[::1] sim_arr,
        const DT_D[::1] cycle_arr,
        const DT_UL off_idx):

    assert ref_arr.shape[0] == sim_arr.shape[0] == cycle_arr.shape[0], (
        'Inputs have unequal shapes!')

    assert off_idx < ref_arr.shape[0], 'off_idx is too big!'

    cdef:
        DT_D numr, demr

    numr = get_ln_sum_sq_diff(ref_arr, sim_arr, off_idx)
    demr = get_ln_sum_sq_diff(ref_arr, cycle_arr, off_idx)
    return 1 - (numr / demr)


def lin_regsn_cy(
        const DT_D[::1] x_arr,
        const DT_D[::1] y_arr,
        const DT_UL off_idx):

    cdef:
        DT_D corr, slope, intercept
        DT_D[::1] y_arr_interp

    assert x_arr.shape[0] == y_arr.shape[0], 'Inputs have unequal shapes!'
    assert off_idx < x_arr.shape[0], 'off_idx is too big!'

    y_arr_interp = np.full(x_arr.shape[0], np.nan)

    lin_regsn(x_arr, y_arr, y_arr_interp, off_idx, &corr, &slope, &intercept)

    return np.asarray(y_arr_interp), corr, slope, intercept


cpdef tuple get_asymms_sample(DT_D[:] u, DT_D[:] v):

    cdef:
        Py_ssize_t i

        DT_UL n_vals
        DT_D asymm_1, asymm_2

    n_vals = u.shape[0]

    asymm_1 = 0.0
    asymm_2 = 0.0
    for i in range(n_vals):
        asymm_1 += (u[i] + v[i] - 1)**3
        asymm_2 += (u[i] - v[i])**3

    asymm_1 = asymm_1 / n_vals
    asymm_2 = asymm_2 / n_vals

    return (asymm_1, asymm_2)


cpdef DT_D get_hargreaves_pet(
        DT_UL d_o_y, 
        DT_D lat, 
        DT_D t_min, 
        DT_D t_max, 
        DT_D t_avg, 
        DT_UL leap) nogil:

    """
    Purpose: To get the potential evapotranspiration at a given latitude
    for a given date and temperature. If t_min > t_max, return pet = 0.
    Also, if pet < 0, return pet = 0.

    Description of the arguments:
        d_o_y (int): day of the year
        lat (radians): latitude of the point
        t_min (celsius): minimum temperature on that day
        t_max (celsius): maximum temperature on that day
        t_avg (celsius): average temperature on that day
        leap (flag): non-zero means year has 365 days else 366
    """

    cdef:
        int tot_days, c1, c2, c3, c4

        DT_D ndec, nws, dfr, ra, pet, two_pi = 2 * M_PI

    c1 = t_min < t_avg < t_max

    c2 = (t_max - t_min) < 0.22

    c3 = (t_max - t_min) > 26.0

    c4 = t_max <= 0.0

    if (not c1) or c2 or c3 or c4:
        return 0.0

    if leap == 0:
        tot_days = 366

    else:
        tot_days = 365

    ndec = 0.409 * sin(((two_pi * d_o_y) / tot_days) - 1.39)

    nws = acos(-tan(lat) * tan(ndec))
 
    dfr = 1 + (0.033 * cos((two_pi * d_o_y) / tot_days))

    # fac_1 = 15.342618001389575 # ((1440 * 0.082 * 0.4082)/pi)

    ra = (
        15.342618 * 
        dfr * 
        ((nws * sin(lat) * sin(ndec)) + (cos(lat) * cos(ndec) * sin(nws))))

    # fac_2 = 0.002295 # (0.0135 * 0.17)

    pet = 0.002295 * ra * ((t_max - t_min) ** 0.5) * (t_avg + 17.8)

    if pet < 0:
        pet = 0.0

    return pet


cpdef void fill_diffs_arr(const DT_D[::1] in_arr, DT_D[::1] diffs_arr) nogil:

    cdef:
        Py_ssize_t i, n_vals = in_arr.shape[0] - 1

    for i in range(n_vals):
        diffs_arr[i] = in_arr[i + 1] - in_arr[i]

    return
