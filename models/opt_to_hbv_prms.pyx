# cython: nonecheck=True
# cython: boundscheck=True
# cython: wraparound=True
# cython: cdivision=True
# cython: language_level=3
# cython: infer_types=False
# cython: embedsignature=True


cdef void tfm_opt_to_hbv_prms(
    const DT_UL[:, ::1] prms_flags,
    const DT_UL[:, ::1] f_var_infos,

    const DT_UL[:, :, ::1] prms_idxs,

    const DT_D[::1] f_vars,
    const DT_D[::1] opt_prms,
    const DT_D[:, ::1] bds_dfs,

          DT_D[:, ::1] hbv_prms,
    ) nogil except +:

    cdef:
        Py_ssize_t i, j, k, m
        DT_UL n_cells = hbv_prms.shape[0], n_hbv_prms = hbv_prms.shape[1]
        DT_D scl_fac, p1, p2, p3

    for i in range(n_cells):
        for j in range(n_hbv_prms):
            hbv_prms[i, j] = 0.0

    for i in range(n_cells):
        for j in range(n_hbv_prms):
            if prms_flags[j, 0]:  # lumped parameter value for all cells
                k = prms_idxs[j, 0, 0]
                hbv_prms[i, j] = bds_dfs[k, 0] + (bds_dfs[k, 1] * opt_prms[k])
                continue

            # each row in the f_var_infos represents a variable such as soil, 
            # landuse, aspect, slope. Using them to get an hbv parameter is
            # implemented independently
            # TODO: verify this by prints, something does not look alright
            # TODO: check if the speed gets better if the cells are calculated
            # inside the ifs
            scl_fac = 0.0
            if prms_flags[j, 1]:  # landuse
                m = f_var_infos[0, 0] + (i * f_var_infos[0, 1])
                scl_fac += 1.0

                for k in range(prms_idxs[j, 1, 0], prms_idxs[j, 1, 1]):
                    hbv_prms[i, j] += (bds_dfs[k, 0] + 
                       (bds_dfs[k, 1] * opt_prms[k])) * f_vars[m]
                    m += 1

            if prms_flags[j, 2]:  # soil
                m = f_var_infos[1, 0] + (i * f_var_infos[1, 1])
                scl_fac += 1.0

                for k in range(prms_idxs[j, 2, 0], prms_idxs[j, 2, 1]):
                    hbv_prms[i, j] += (bds_dfs[k, 0] + 
                       (bds_dfs[k, 1] * opt_prms[k])) * f_vars[m]
                    m += 1

            if prms_flags[j, 3]:  # aspect
                m = f_var_infos[2, 0] + i  # it's one per cell
                k = prms_idxs[j, 3, 0]
                scl_fac += 1.0

                p1 = bds_dfs[k, 0] + (bds_dfs[k, 1] * opt_prms[k])
                p2 = bds_dfs[k + 1, 0] + (bds_dfs[k + 1, 1] * opt_prms[k + 1])
                p3 = bds_dfs[k + 2, 0] + (bds_dfs[k + 2, 1] * opt_prms[k + 2])

                hbv_prms[i, j] += (p1 + ((p2 - p1) * f_vars[m]**p3))

            if prms_flags[j, 4]:  # slope
                m = f_var_infos[3, 0] + i  # it's one per cell
                k = prms_idxs[j, 4, 0]
                scl_fac += 1.0

                p1 = bds_dfs[k, 0] + (bds_dfs[k, 1] * opt_prms[k])
                p2 = bds_dfs[k + 1, 0] + (bds_dfs[k + 1, 1] * opt_prms[k + 1])
                p3 = bds_dfs[k + 2, 0] + (bds_dfs[k + 2, 1] * opt_prms[k + 2])
                
                hbv_prms[i, j] += (p1 + ((p2 - p1) * f_vars[m]**p3))

            if prms_flags[j, 5]:  # aspect and slope
                m = f_var_infos[4, 0] + i  # it's one per cell
                k = prms_idxs[j, 5, 0]
                scl_fac += 1.0

                p1 = bds_dfs[k, 0] + (bds_dfs[k, 1] * opt_prms[k])
                p2 = bds_dfs[k + 1, 0] + (bds_dfs[k + 1, 1] * opt_prms[k + 1])
                p3 = bds_dfs[k + 2, 0] + (bds_dfs[k + 2, 1] * opt_prms[k + 2])

                hbv_prms[i, j] += (p1 + ((p2 - p1) * f_vars[m]**p3))

            scl_fac = 1.0 / scl_fac
            hbv_prms[i, j] = scl_fac * hbv_prms[i, j]
    return
