# cython: nonecheck=False
# cython: boundscheck=True
# cython: wraparound=True
# cython: cdivision=True
# cython: language_level=3
# cython: infer_types=False
# cython: embedsignature=True


cdef void select_best_params(
    DT_UL[:, ::1] depths_arr,

    DT_D[::1] pre_obj_vals,

    DT_D[:, ::1] prm_vecs,
    DT_D[:, ::1] uvecs,

    const DT_UL n_acc_pars,
    const DT_UL n_cpus,
          DT_D *res_new
    ) nogil except +:

    cdef:
        DT_D min_res = np.inf
        DT_UL counter = 0
        DT_UL[::1] sort_idxs
        DT_UL[::1] depths_arr
        DT_D[:,::1] chull
        DT_D[:, ::1] acc_pars = np.zeros((n_acc_pars, prm_vecs.shape[1]), dtype=np.float64)

    sort_idxs = np.argsort(pre_obj_vals).copy(order = 'c').astype(np.int32)
    for i in range(n_acc_pars):
        for j in range(prm_vecs.shape[1]):
            acc_pars[i, j] = prm_vecs[sort_idxs[i], j]

        if pre_obj_vals[sort_idxs[i]]<min_res:
            min_res = pre_obj_vals[sort_idxs[i]]

    res_new[0] = min_res

    depths_arr = depth_ftn_mp(acc_pars, acc_pars, uvecs, n_cpus)
    for i in range(acc_pars.shape[0]):
        if depths_arr[i] == 1:
            counter += 1

    chull = np.zeros((counter, prm_vecs.shape[1]), dtype=np.float64)
    counter = 0
    for i in range(acc_pars.shape[0]):
        if depths_arr[i] == 1:
            for j in range(acc_pars.shape[1]):
                chull[counter, j] = acc_pars[i,j]
            counter += 1
    return np.asarray(chull), np.asarray(acc_pars)