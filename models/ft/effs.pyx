# cython: nonecheck=False
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: language_level=3
# cython: infer_types=False
# cython: embedsignature=True

from libc.math cimport cos
from libc.stdlib cimport malloc, free

cdef extern from "stdlib.h" nogil:
    cdef:
        DT_D abs(DT_D)

cdef extern from "complex.h" nogil:
    cdef:
        DT_D cabs(DT_DC)


# should come from a binary compiled on intel (for use on amd systems as well)
from .intel_dft_mkl cimport mkl_real_dft

# cdef extern from "intel_dfti.h" nogil:
#     cdef:
#         void mkl_real_dft(
#                 double *in_reals_arr,
#                 DT_DC *out_comps_arr,
#                 long n_pts)


from .dfti cimport cmpt_real_fourtrans_1d
from ..miscs.misc_ftns cimport get_mean, get_variance

from ..miscs.dtypes cimport (
    off_idx_i,
    use_step_flag_i,
    resamp_obj_ftns_flag_i,
    a_zero_i,
    ft_beg_idx_i, 
    ft_end_idx_i,
    mean_ref_i,
    act_std_dev_i,
    ft_demr_1_i,
    ft_demr_2_i)


cdef DT_D get_ft_eff(
        const DT_UL[::1] obj_longs,
 
        const DT_D[::1] obj_doubles,
 
        const ForFourTrans1DReal *obs_for_four_trans_struct,
              ForFourTrans1DReal *sim_for_four_trans_struct,
        ) nogil except +:

    cdef:
        Py_ssize_t i

        DT_UL n_pts = obs_for_four_trans_struct.n_pts

        DT_D abs_ft_sq_diff, abs_ft_diff

    mkl_real_dft(
        sim_for_four_trans_struct.orig, sim_for_four_trans_struct.ft, n_pts)

    abs_ft_sq_diff = 0.0
    for i in range(obj_longs[ft_beg_idx_i], obj_longs[ft_end_idx_i]):
        abs_ft_sq_diff += (cabs(
            obs_for_four_trans_struct.ft[i] -
            sim_for_four_trans_struct.ft[i])**2)

    abs_ft_diff = 0.0
    for i in range((n_pts // 2) + 1):
        abs_ft_diff += cabs(
            obs_for_four_trans_struct.ft[i] - sim_for_four_trans_struct.ft[i])

    # weights should sum upto 1
    return (
        (0.85 * (1.0 - (abs_ft_sq_diff / obj_doubles[ft_demr_1_i]))) +
        (0.15 * (1.0 - (abs_ft_diff    / obj_doubles[ft_demr_2_i]))))


cpdef tuple cmpt_obj_val_cy_debug(
        const DT_D[::1] qobs_arr,
              DT_D[::1] qsim_arr,
        const DT_UL off_idx,
        const DT_UL ft_maxi_freq_idx) except +:
    
    cdef:
        Py_ssize_t i

        DT_UL n_recs, n_pts_ft, q_sorig, q_sft, q_sampang

        DT_D ft_sdiff

        DT_D *obs_amps
        DT_D *sim_amps
        DT_D *obs_angs
        DT_D *sim_angs

        ForFourTrans1DReal obs_for_four_trans_struct
        ForFourTrans1DReal sim_for_four_trans_struct

    n_recs = qobs_arr.shape[0]

    assert n_recs == qsim_arr.shape[0]

    n_pts_ft = n_recs - off_idx

    if n_pts_ft % 2:
        n_pts_ft -= 1

    q_sorig = n_pts_ft * sizeof(DT_D)
    q_sft = ((n_pts_ft // 2) + 1) * sizeof(DT_DC)
    q_sampang = (((n_pts_ft // 2) + 1) * sizeof(DT_D))

    obs_for_four_trans_struct.orig = <DT_D *> malloc(q_sorig)
    obs_for_four_trans_struct.ft = <DT_DC *> malloc(q_sft)
    obs_for_four_trans_struct.amps = <DT_D *> malloc(q_sampang)
    obs_for_four_trans_struct.angs = <DT_D *> malloc(q_sampang)
    obs_for_four_trans_struct.n_pts = n_pts_ft

    sim_for_four_trans_struct.orig = <DT_D *> malloc(q_sorig)
    sim_for_four_trans_struct.ft = <DT_DC *> malloc(q_sft)
    sim_for_four_trans_struct.amps = <DT_D *> malloc(q_sampang)
    sim_for_four_trans_struct.angs = <DT_D *> malloc(q_sampang)
    sim_for_four_trans_struct.n_pts = n_pts_ft

    for i in range(n_pts_ft):
        obs_for_four_trans_struct.orig[i] = qobs_arr[off_idx + i]
        sim_for_four_trans_struct.orig[i] = qsim_arr[off_idx + i]
 
    cmpt_real_fourtrans_1d(&obs_for_four_trans_struct)
    cmpt_real_fourtrans_1d(&sim_for_four_trans_struct)
 
    obs_amps = obs_for_four_trans_struct.amps
    obs_angs = obs_for_four_trans_struct.angs
 
    sim_amps = sim_for_four_trans_struct.amps
    sim_angs = sim_for_four_trans_struct.angs
 
    ft_sdiff = 0.0
    for i in range(ft_maxi_freq_idx, (n_pts_ft // 2) + 1):
        ft_sdiff += ((obs_amps[i]**2) - (
            (obs_amps[i] * sim_amps[i]) * (cos(obs_angs[i] - sim_angs[i]))))**2

    free(obs_for_four_trans_struct.orig)
    free(obs_for_four_trans_struct.ft)
    free(obs_for_four_trans_struct.amps)
    free(obs_for_four_trans_struct.angs)

    free(sim_for_four_trans_struct.orig)
    free(sim_for_four_trans_struct.ft)
    free(sim_for_four_trans_struct.amps)
    free(sim_for_four_trans_struct.angs)

    return (ft_sdiff, )
