from .dtypes cimport DT_D, DT_UL
from libcpp.map cimport map as cmap


cdef DT_D obj_ftn(
    const long *tid,

          DT_UL[::1] n_calls,
    const DT_UL[::1] stm_idxs,
    const DT_UL[::1] obj_longs,
    const DT_UL[::1] use_step_arr,

    const DT_UL[:, ::1] prms_flags,
    const DT_UL[:, ::1] f_var_infos,

    const DT_UL[:, :, ::1] prms_idxs,

    const DT_D[::1] obj_ftn_wts,
          DT_D[::1] opt_prms,
    const DT_D[::1] qact_arr,
    const DT_D[::1] area_arr,
          DT_D[::1] qsim_arr,
          DT_D[::1] lrst_arr,
          DT_D[::1] inflow_arr,
    const DT_D[::1] f_vars,
    const DT_D[::1] obj_doubles,
          DT_D[::1] route_prms,

    const DT_D[:, ::1] inis_arr,
          DT_D[:, ::1] temp_arr,
          DT_D[:, ::1] prec_arr,
          DT_D[:, ::1] petn_arr,
          DT_D[:, ::1] cats_outflow_arr,
          DT_D[:, ::1] stms_inflow_arr,
          DT_D[:, ::1] stms_outflow_arr,
    const DT_D[:, ::1] dem_net_arr,
          DT_D[:, ::1] hbv_prms,    
    const DT_D[:, ::1] bds_dfs,

          DT_D[:, :, ::1] outs_arr,

          cmap[long, long] &cat_to_idx_map,
          cmap[long, long] &stm_to_idx_map,
    ) nogil except +
