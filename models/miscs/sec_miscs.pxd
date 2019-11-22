# cython: nonecheck=True
# cython: boundscheck=True
# cython: wraparound=True
# cython: cdivision=True
# cython: language_level=3
# cython: infer_types(None)

from ..miscs.dtypes cimport DT_D, DT_UL, DT_ULL


cdef void update_obj_doubles(
        const DT_UL[::1] obj_longs,
        const DT_UL[::1] use_step_arr,

        const DT_D[::1] obj_ftn_wts,
              DT_D[::1] obj_doubles,
        const DT_D[::1] q_arr,
        const DT_D[::1] q_resamp_arr,
        ) nogil except +
