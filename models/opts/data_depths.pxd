from ..miscs.dtypes cimport DT_D, DT_UL, DT_ULL


cdef void depth_ftn(
        const DT_D[:, ::1] ref,
        const DT_D[:, ::1] test,
        const DT_D[:, ::1] uvecs,
              DT_D[:, ::1] dot_ref,
              DT_D[:, ::1] dot_test,
              DT_D[:, ::1] dot_test_sort,
              DT_UL[:, ::1] temp_mins,
              DT_UL[:, ::1] mins,
              DT_UL[::1] depths_arr,
        const DT_UL n_ref,
        const DT_UL n_cpus,
        ) nogil except +


cdef void pre_depth(
        const DT_D[:, ::1] ref,
        const DT_D[:, ::1] uvecs,
              DT_D[:, ::1] dot_ref_sort,
        const DT_UL n_ref,
        const DT_UL n_cpus,
        ) nogil except +


cdef void post_depth(
        const DT_D[:, ::1] test,
        const DT_D[:, ::1] uvecs,
        const DT_D[:, ::1] dot_ref_sort,  # should be sorted
              DT_D[:, ::1] dot_test,
              DT_D[:, ::1] dot_test_sort,
              DT_UL[:, ::1] temp_mins,
              DT_UL[:, ::1] mins,
              DT_UL[::1] depths_arr,
        const DT_UL n_ref,
        const DT_UL n_cpus,
        ) nogil except +
