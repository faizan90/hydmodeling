# cython: nonecheck=False
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: language_level=3
# cython: infer_types=False
# cython: embedsignature=True

from libc.math cimport cos

cdef extern from "stdlib.h" nogil:
    cdef:
        DT_D abs(DT_D)


from ..miscs.misc_ftns cimport get_mean, get_variance

from ..miscs.dtypes cimport (
    off_idx_i,
    use_step_flag_i,
    resamp_obj_ftns_flag_i,
    a_zero_i,
    ft_maxi_freq_idx_i,
    mean_ref_i,
    act_std_dev_i)


cdef DT_D get_ft_eff(
        const DT_UL[::1] obj_longs,

              DT_D[::1] qsim_arr,
        const DT_D[::1] obj_doubles,

        const ForFourTrans1DReal *obs_for_four_trans_struct,
              ForFourTrans1DReal *sim_for_four_trans_struct,
        ) nogil except +:

    cdef:
        Py_ssize_t i

        DT_UL n_pts = obs_for_four_trans_struct.n_pts

        DT_D mean_sdiff, var_sdiff, ft_sdiff
        DT_D qsim_mean, pow_spec_sum, prt_pow_spec_sum, eff

        DT_D *obs_amps
        DT_D *sim_amps
        DT_D *obs_angs
        DT_D *sim_angs

        DT_DC ft

    qsim_mean = get_mean(qsim_arr, &obj_longs[off_idx_i])

    mean_sdiff = (obj_doubles[mean_ref_i] - qsim_mean)**2
  
    var_sdiff = (
        (obj_doubles[act_std_dev_i]**2) -
        get_variance(&qsim_mean, qsim_arr, &obj_longs[off_idx_i]))**2

#     mean_sdiff = 0.0
#     var_sdiff = 0.0

    obs_amps = obs_for_four_trans_struct.amps
    obs_angs = obs_for_four_trans_struct.angs

    sim_amps = sim_for_four_trans_struct.amps
    sim_angs = sim_for_four_trans_struct.angs

    ft_sdiff = 0.0
    pow_spec_sum = 0.0
    prt_pow_spec_sum = 0.0
    for i in range(obj_longs[ft_maxi_freq_idx_i] + 1, (n_pts // 2) - 1):
        ft_sdiff += (
            (obs_amps[i] * sim_amps[i]) * (cos(obs_angs[i] - sim_angs[i])))

        prt_pow_spec_sum += (obs_amps[i] * sim_amps[i])

#     for i in range((n_pts // 2) - 1):
#         pow_spec_sum += (obs_amps[i] * sim_amps[i])

    # keep this, it works
    eff = ((1.0 / 3.0) * (
        (-mean_sdiff / obj_doubles[mean_ref_i]) +
        (-var_sdiff / (obj_doubles[act_std_dev_i]**2)) +
        ((ft_sdiff / prt_pow_spec_sum) - 1))) + 1

#     eff = (-mean_sdiff / obj_doubles[mean_ref_i])
#     eff = (-var_sdiff / (obj_doubles[act_std_dev_i]**2))
#     eff = (ft_sdiff / prt_pow_spec_sum) - 1

#     with gil:
# #         if eff > 1:
# #             print(f'WARNING: eff ge 1 ({eff})!')
#         print(eff)

    return eff
