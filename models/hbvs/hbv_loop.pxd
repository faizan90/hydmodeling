from ..miscs.dtypes cimport DT_D, DT_UL

cdef DT_D hbv_loop(
          DT_D[:, ::1] temp_arr,
          DT_D[:, ::1] prec_arr,
          DT_D[:, ::1] petn_arr,
          DT_D[:, ::1] prms_arr,
    const DT_D[:, ::1] inis_arr,
    const DT_D[::1] area_arr,
          DT_D[::1] qsim_arr,
          DT_D[:, :, ::1] outs_arr,
    const DT_D *rnof_q_conv,
    const DT_UL *opt_flag,
    ) nogil except +
