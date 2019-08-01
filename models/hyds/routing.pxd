# cython: nonecheck=False
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: language_level=3
# cython: infer_types=False
# cython: embedsignature=True

from ..miscs.dtypes cimport DT_D, DT_UL
from libcpp.map cimport map as cmap


cdef void tfm_opt_to_route_prms(
        const DT_D[::1] opt_prms,
              DT_D[::1] route_prms,
        const DT_D[:, ::1] bds_dfs,
        ) nogil except +


cdef void musk_route(
        const DT_D[::1] inflow_arr, 
              DT_D[::1, :] outflow_arr,

        const DT_D *lag, 
        const DT_D *wt, 
        const DT_D *del_t,

        const long *n_recs, 
        const long *stm_idx,
        const long *n_stms,
        ) nogil except +


cdef void route_strms(
        const DT_UL[::1] stm_idxs,

        cmap[long, long] &cat_to_idx_map,
        cmap[long, long] &stm_to_idx_map,

              DT_D[::1] inflow_arr,
              DT_D[::1] route_prms,

        const DT_D[:, ::1] dem_net_arr,
              DT_D[::1, :] cats_outflow_arr,
              DT_D[::1, :] stms_inflow_arr,
              DT_D[::1, :] stms_outflow_arr,

        const DT_UL *n_stms,
        const DT_UL *route_type,
        ) nogil except +
