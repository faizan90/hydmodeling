# cython: nonecheck=False
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: language_level=3
# cython: infer_types=False
# cython: embedsignature=True

from ..miscs.dtypes cimport DT_D, DT_UL, DT_ULL, DT_DC
from ..miscs.dtypes cimport ForFourTrans1DReal

cdef DT_D get_ft_eff(
        const DT_UL[::1] obj_longs,
    
              DT_D[::1] qsim_arr,
        const DT_D[::1] obj_doubles,
    
        const ForFourTrans1DReal *obs_for_four_trans_struct,
              ForFourTrans1DReal *sim_for_four_trans_struct,
        ) nogil except +