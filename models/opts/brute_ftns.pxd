from ..miscs.dtypes cimport DT_D, DT_UL, DT_ULL


cdef void pre_brute(
              DT_UL[::1] last_idxs_vec,
              DT_UL[::1] use_prm_vec_flags,

        const DT_UL[:, ::1] prms_flags,
        const DT_UL[:, ::1] prms_span_idxs,

        const DT_UL[:, :, ::1] prms_idxs,

              DT_D[:, ::1] prm_vecs,

        const DT_ULL n_poss_combs,
        const DT_UL n_discretize,
        const DT_UL n_hbv_prms,
              DT_ULL *comb_ctr,
        ) nogil except +


cdef void post_brute(
        const DT_UL[::1] use_prm_vec_flags,

        const DT_D[::1] curr_obj_vals,
              DT_D[::1] pre_obj_vals,
              DT_D[::1] best_prm_vec,
              DT_D[::1] iobj_vals,

              DT_D[:, ::1] prm_vecs,
              DT_D[:, :, ::1] iter_acc_prm_vecs,

        const DT_ULL n_poss_combs,
              DT_UL *iter_curr,
              DT_UL *cont_opt_flag,
              DT_ULL *comb_ctr,

              DT_D *fval_pre_global,
        ) nogil except +
