from ..miscs.dtypes cimport DT_D, DT_UL


cdef void tfm_opt_to_hbv_prms(
    const DT_UL[:, ::1] prms_flags,
    const DT_UL[:, ::1] f_var_infos,
    
    const DT_UL[:, :, ::1] prms_idxs,

    const DT_D[::1] f_vars,    
    const DT_D[::1] opt_prms,
    const DT_D[:, ::1] bds_dfs,

          DT_D[:, ::1] hbv_prms,
    ) nogil except +
