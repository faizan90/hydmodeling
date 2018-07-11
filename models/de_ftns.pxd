from .dtypes cimport DT_D, DT_UL, DT_ULL


cdef void pre_de(
    const DT_UL[::1] idx_rng,
          DT_UL[::1] r_r,

    const DT_UL[:, ::1] prms_flags,
    const DT_UL[:, ::1] prms_span_idxs,
          DT_UL[:, ::1] del_idx_rng,
          DT_UL[:, ::1] choice_arr,

    const DT_UL[:, :, ::1] prms_idxs,

          DT_ULL[::1] seeds_arr, # a warmed up one

    const DT_D[::1] mu_sc_fac_bds,
    const DT_D[::1] cr_cnst_bds,

          DT_D[:, ::1] prm_vecs,
          DT_D[:, ::1] v_j_g,
          DT_D[:, ::1] u_j_gs,

    const DT_UL n_hbv_prms,
    const DT_UL n_cpus,
    ) nogil except +


cdef void post_de(
          DT_UL[::1] prm_opt_stop_arr,

    const DT_D[::1] curr_obj_vals,
          DT_D[::1] pre_obj_vals,
          DT_D[::1] best_prm_vec,

    const DT_D[:, ::1] u_j_gs,
          DT_D[:, ::1] prm_vecs,
          DT_D[:, ::1] prms_mean_thrs_arr,

    const DT_UL max_iters,
    const DT_UL max_cont_iters,
          DT_UL *iter_curr,
          DT_UL *last_succ_i,
          DT_UL *n_succ,
          DT_UL *cont_iter,
          DT_UL *cont_opt_flag,

    const DT_D obj_ftn_tol,
          DT_D *tol_curr,
          DT_D *tol_pre,
          DT_D *fval_pre_global,
          DT_D *prm_pcnt_tol,
    ) nogil except +
