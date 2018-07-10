from .dtypes cimport DT_D, DT_UL

import numpy as np
cimport numpy as np


cdef void get_new_chull_vecs(
          DT_UL[::1] depths_arr,
          DT_UL[:, ::1] temp_mins,
          DT_UL[:, ::1] mins,

    const DT_D[::1] pre_obj_vals,
          DT_D[::1] sort_obj_vals,

          DT_D[:, ::1] acc_vecs,
    const DT_D[:, ::1] prm_vecs,
    const DT_D[:, ::1] uvecs,
          DT_D[:, ::1] dot_ref,
          DT_D[:, ::1] dot_test,
          DT_D[:, ::1] dot_test_sort,
          DT_D[:, ::1] chull_vecs,

    const DT_UL n_cpus,
          DT_UL *chull_vecs_ctr,
    ) nogil except +


cdef void set_rope_bds(
    const DT_D[:, ::1] chull_vecs,
          DT_D[:, ::1] rope_bds_dfs,
    const DT_UL chull_vecs_ctr,
    ) nogil except +


cdef void gen_vecs_in_chull(
          DT_UL[::1] depths_arr,

    const DT_UL[:, ::1] prms_flags,
    const DT_UL[:, ::1] prms_span_idxs,
          DT_UL[:, ::1] temp_mins,
          DT_UL[:, ::1] mins,
    
    const DT_UL[:, :, ::1] prms_idxs,

    const DT_D[:, ::1] rope_bds_dfs,
    const DT_D[:, ::1] chull_vecs,
    const DT_D[:, ::1] uvecs,

          DT_D[:, ::1] temp_rope_prm_vecs,
          DT_D[:, ::1] prm_vecs,
          DT_D[:, ::1] dot_ref,
          DT_D[:, ::1] dot_test,
          DT_D[:, ::1] dot_test_sort,

    const DT_UL n_hbv_prms,
    const DT_UL chull_vecs_ctr,
    const DT_UL n_cpus,
    ) nogil except +
