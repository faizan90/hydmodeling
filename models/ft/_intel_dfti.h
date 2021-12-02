#pragma once
#include <complex.h>
#include <mkl_dfti.h>


void mkl_real_dft(
		double *in_reals_arr,
		_Dcomplex *out_comps_arr,
		long n_pts) {

	DFTI_DESCRIPTOR_HANDLE desc_hdl;
	MKL_LONG status;

	status = DftiCreateDescriptor(
				&desc_hdl,
				DFTI_DOUBLE,
				DFTI_REAL,
				(MKL_LONG) 1,
				(MKL_LONG) n_pts);

	status = DftiSetValue(
				desc_hdl,
				DFTI_PLACEMENT,
				DFTI_NOT_INPLACE);

	status = DftiSetValue(
				desc_hdl,
				DFTI_CONJUGATE_EVEN_STORAGE,
				DFTI_COMPLEX_COMPLEX);

	status = DftiCommitDescriptor(desc_hdl);

	status = DftiComputeForward(
				desc_hdl,
				in_reals_arr,
				out_comps_arr);

	status = DftiFreeDescriptor(&desc_hdl);

	return;

}


void mkl_set_desc(DFTI_DESCRIPTOR_HANDLE &desc_hdl, long n_pts) {

	MKL_LONG status;

	status = DftiCreateDescriptor(
				&desc_hdl,
				DFTI_DOUBLE,
				DFTI_REAL,
				(MKL_LONG) 1,
				(MKL_LONG) n_pts);

	status = DftiSetValue(
				desc_hdl,
				DFTI_PLACEMENT,
				DFTI_NOT_INPLACE);

	status = DftiSetValue(
				desc_hdl,
				DFTI_CONJUGATE_EVEN_STORAGE,
				DFTI_COMPLEX_COMPLEX);

	status = DftiCommitDescriptor(desc_hdl);

	return;
}


void mkl_dft_real_with_desc(
		DFTI_DESCRIPTOR_HANDLE &desc_hdl,
		double *in_reals_arr,
		_Dcomplex *out_comps_arr) {

	MKL_LONG status;

	status = DftiComputeForward(
				desc_hdl,
				in_reals_arr,
				out_comps_arr);

	return;
}


void mkl_free_desc(DFTI_DESCRIPTOR_HANDLE &desc_hdl) {

	MKL_LONG status;

	status = DftiFreeDescriptor(&desc_hdl);

	return;
}

