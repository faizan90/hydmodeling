from ..miscs.dtypes cimport DT_D, DT_UL, DT_ULL, DT_DC
from ..miscs.dtypes cimport ForFourTrans1DReal


cdef void cmpt_real_fourtrans_1d(
    ForFourTrans1DReal *for_four_trans_struct) nogil except +


cdef void cmpt_cumm_freq_pcorrs(
        const ForFourTrans1DReal *obs_for_four_trans_struct,
        const ForFourTrans1DReal *sim_for_four_trans_struct,
              DT_D *freq_corrs) nogil except +
