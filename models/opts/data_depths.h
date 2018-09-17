#pragma once
#include <omp.h>
#include <stdio.h>
#include <math.h>
#include <Windows.h>
#define WIN32_LEAN_AND_MEAN
#include <stdint.h>

#include "../miscs/rand_gen_mp.h"

double log(double x);
double pow(double x, double y);
double rand_c_mp(unsigned long long &seed);


void quick_sort(double *arr, long first_index, long last_index) {

	long pivotIndex, index_a, index_b;
	double temp;

	if (first_index < last_index) {
		pivotIndex = first_index;
		index_a = first_index;
		index_b = last_index;

		while (index_a < index_b) {
			while (arr[index_a] <= arr[pivotIndex] && index_a < last_index) {
				index_a++;
			}

			while (arr[index_b] > arr[pivotIndex]) {
				index_b--;
			}

			if (index_a < index_b) {
				temp = arr[index_a];
				arr[index_a] = arr[index_b];
				arr[index_b] = temp;
			}
		}

		temp = arr[pivotIndex];
		arr[pivotIndex] = arr[index_b];
		arr[index_b] = temp;

		quick_sort(arr, first_index, index_b - 1);
		quick_sort(arr, index_b + 1, last_index);
	}
	return;
}


long searchsorted(
		const double *arr, const double value, const long arr_size) {

	// arr must be sorted
	long first = 0, last = arr_size - 1, curr_idx;

	if (value <= arr[0]) {
		return 0;
	}

	else if (value > arr[last]) {
		return arr_size;
	}

	while (first <= last) {
		curr_idx = (long) (0.5 * (first + last));

		if ((value > arr[curr_idx]) && (value <= arr[curr_idx + 1])) {
			return curr_idx + 1;
		}

		else if (value < arr[curr_idx]) {
			last = curr_idx - 1;
		}

		else if (value > arr[curr_idx]) {
			first = curr_idx + 1;
		}

		else {
			return curr_idx;
		}
	}
	return 0;
}


int gettimeofday(struct timeval *tp) {

	static const uint64_t EPOCH = ((uint64_t) 116444736000000000ULL);

    SYSTEMTIME  system_time;
    FILETIME    file_time;
    uint64_t    time;

    GetSystemTime(&system_time);

    SystemTimeToFileTime(&system_time, &file_time);

    time =  ((uint64_t)file_time.dwLowDateTime);
    time += ((uint64_t)file_time.dwHighDateTime) << 32;

    tp->tv_sec  = (long) ((time - EPOCH) / 10000000L);

    tp->tv_usec = (long) (system_time.wMilliseconds * 1000);

    return 0;
}


double usph_norm_ppf_c(double p) {

	double t, z;

	if (p <= 0.0) {
		return -INFINITY;
	}

	else if (p >= 1.0) {
		return INFINITY;
	}

	if (p > 0.5) {
		t = pow(-2.0 * log(1 - p), 0.5);
	}

	else {
		t = pow(-2.0 * log(p), 0.5);
	}

    z = -0.322232431088 + t * (-1.0 + t * (-0.342242088547 + t * (
        (-0.020423120245 + t * -0.453642210148e-4))));

    z = z / (0.0993484626060 + t * (0.588581570495 + t * (
        (0.531103462366 + t * (0.103537752850 + t * 0.3856070063e-2)))));

    z = z + t;

    if (p < 0.5) {
        z = -z;
    }

    return z;
}


void gen_usph_vecs_norm_dist_c(
		unsigned long long *seeds_arr,
		double *rn_ct_arr,
		double *ndim_usph_vecs,
		long n_vecs,
		long n_dims,
		long n_cpus) {

	size_t j, tid;
	timeval t0;
	long long i;
	unsigned long long re_seed_i = (unsigned long long) (1e9);

	for (tid = 0; tid < n_cpus; ++tid) {
		gettimeofday(&t0);

		seeds_arr[tid] = (unsigned long long) (t0.tv_usec * (long) 324234543);

		for (j = 0; j < 1000; ++j) {
			rand_c_mp(&seeds_arr[tid]);
		}

		Sleep(45);
	}

	omp_set_num_threads(n_cpus);

	#pragma omp parallel for private(j, tid)
	for (i = 0; i < n_vecs; ++i) {
		tid = omp_get_thread_num();

		double mag = 0.0;

		double *usph_vec = &ndim_usph_vecs[i * n_dims];

		for (j = 0; j < n_dims; ++j) {
			usph_vec[j] = usph_norm_ppf_c(rand_c_mp(&seeds_arr[tid]));

			mag = mag + pow(usph_vec[j], 2.0);
		}

		mag = pow(mag, 0.5);

		for (j = 0; j < n_dims; ++j) {
			usph_vec[j] = usph_vec[j]  / mag;
		}

		rn_ct_arr[tid] = rn_ct_arr[tid] + n_dims;
		if ((rn_ct_arr[tid]  / re_seed_i) > 1) {
			seeds_arr[tid] = (unsigned long long) (t0.tv_usec * (long) 937212);

			for (j = 0; j < 1000; ++j) {
				rand_c_mp(&seeds_arr[tid]);
			}

			Sleep(35);

			rn_ct_arr[tid] = 0.0;
		}
	}

	return;
}


void depth_ftn_c(
		const double *ref,
		const double *test,
		const double *uvecs,
			  double *dot_ref,
			  double *dot_test,
			  double *dot_test_sort,
			  long *temp_mins,
			  long *mins,
			  long *depths_arr,
		const long n_ref,
		const long n_test,
		const long n_uvecs,
		const long n_dims,
		const long n_cpus) {

	int even = (n_test % 2) == 0;

	long long i;

	double inc_mult = (double) (1 - (double) (1e-15));

	omp_set_num_threads(n_cpus);

	#pragma omp parallel for schedule(dynamic)
	for (i = 0; i < n_uvecs; ++i) {
		size_t tid, j, k;

		double *uvec, *sdot_ref, *sdot_test, *sdot_test_sort;

		long *stemp_mins, *smins, idx, offset;

		double stest_med;

		tid = omp_get_thread_num();

		uvec = (double *) &uvecs[i * n_dims];

		sdot_ref = &dot_ref[tid * n_ref];

		for (j = 0; j < n_ref; ++j) {
			sdot_ref[j] = 0.0;

			offset = (long) (j * n_dims);


			for (k = 0; k < n_dims; ++k) {
				sdot_ref[j] = sdot_ref[j] + (uvec[k] * ref[offset + k]);
			}

		}

		sdot_test = &dot_test[tid * n_test];
		sdot_test_sort = &dot_test_sort[tid * n_test];

		for (j = 0; j < n_test; ++j) {
			sdot_test[j] = 0.0;
			offset = (long) (j * n_dims);

			for (k = 0; k < n_dims; ++k) {
				sdot_test[j] = sdot_test[j] + (uvec[k] * test[offset + k]);
			}

			sdot_test_sort[j] = sdot_test[j];
		}

		quick_sort(&sdot_ref[0], 0, n_ref - 1);
		quick_sort(&sdot_test_sort[0], 0, n_test - 1);

		if (even) {
			stest_med = 0.5 * (
					sdot_test_sort[n_test / 2] +
					sdot_test_sort[(n_test / 2) - 1]);
		}

		else {
			stest_med = sdot_test_sort[n_test / 2];
		}

		for (j = 0; j < n_test; ++j) {
			sdot_test[j] = (
					(sdot_test[j] - stest_med) * inc_mult) + stest_med;
		}

		smins = &mins[tid * n_test];
		stemp_mins = &temp_mins[tid * n_test];

		for (j = 0; j < n_test; ++j) {
			stemp_mins[j] = searchsorted(
					&sdot_ref[0], sdot_test[j], n_ref);
		}

		for (j = 0; j < n_test; ++j) {
			idx = n_ref - stemp_mins[j];

			if (idx < stemp_mins[j]) {
				stemp_mins[j] = idx;
			}

			if (stemp_mins[j] < smins[j]) {
				smins[j] = stemp_mins[j];
			}
		}
	}

	for (i = 0; i < n_test; ++i) {
		depths_arr[i] = mins[i];

		for (size_t j = 1; j < n_cpus; ++j) {
			if (mins[(j * n_test) + i] >= depths_arr[i]) {
				continue;
			}

			depths_arr[i] = mins[(j * n_test) + i];
		}
	}

	return;
}


void pre_depth_c(
		const double *ref,
		const double *uvecs,
			  double *dot_ref_sort,
		const long n_ref,
		const long n_uvecs,
		const long n_dims,
		const long n_cpus) {

	// pre_depth_c and post_depth_c are cpu efficient versions of depth_ftn_c
	// but they require much more memory.

	omp_set_num_threads(n_cpus);

	#pragma omp parallel for schedule(dynamic)
	for (long long i = 0; i < n_uvecs; ++i) {
		size_t j, k;

		double *uvec, *sdot_ref_sort;

		long offset;

		uvec = (double *) &uvecs[i * n_dims];

		sdot_ref_sort = &dot_ref_sort[i * n_ref];

		for (j = 0; j < n_ref; ++j) {
			sdot_ref_sort[j] = 0.0;

			offset = (long) (j * n_dims);


			for (k = 0; k < n_dims; ++k) {
				sdot_ref_sort[j] = (
						sdot_ref_sort[j] + (uvec[k] * ref[offset + k]));
			}
		}

		quick_sort(&sdot_ref_sort[0], 0, n_ref - 1);
	}

	return;
}

void post_depth_c(
		const double *test,
		const double *uvecs,
		const double *dot_ref_sort, // should be sorted
			  double *dot_test,
			  double *dot_test_sort,
			  long *temp_mins,
			  long *mins,
			  long *depths_arr,
		const long n_ref,
		const long n_test,
		const long n_uvecs,
		const long n_dims,
		const long n_cpus) {

	// pre_depth_c and post_depth_c are cpu efficient versions of depth_ftn_c
	// but they require much more memory.

	int even = (n_test % 2) == 0;

	long long i;

	double inc_mult = (double) (1 - (double) (1e-15));

	omp_set_num_threads(n_cpus);

	#pragma omp parallel for schedule(dynamic)
	for (i = 0; i < n_uvecs; ++i) {
		size_t tid, j, k;

		double *uvec, *sdot_ref_sort, *sdot_test, *sdot_test_sort;

		long *stemp_mins, *smins, idx, offset;

		double stest_med;

		tid = omp_get_thread_num();

		uvec = (double *) &uvecs[i * n_dims];

		sdot_ref_sort = (double *) &dot_ref_sort[i * n_ref];

		sdot_test = &dot_test[tid * n_test];
		sdot_test_sort = &dot_test_sort[tid * n_test];

		for (j = 0; j < n_test; ++j) {
			sdot_test[j] = 0.0;
			offset = (long) (j * n_dims);

			for (k = 0; k < n_dims; ++k) {
				sdot_test[j] = sdot_test[j] + (uvec[k] * test[offset + k]);
			}

			sdot_test_sort[j] = sdot_test[j];
		}

		quick_sort(&sdot_test_sort[0], 0, n_test - 1);

		if (even) {
			stest_med = 0.5 * (
					sdot_test_sort[n_test / 2] +
					sdot_test_sort[(n_test / 2) - 1]);
		}

		else {
			stest_med = sdot_test_sort[n_test / 2];
		}

		for (j = 0; j < n_test; ++j) {
			sdot_test[j] = (
					(sdot_test[j] - stest_med) * inc_mult) + stest_med;
		}

		smins = &mins[tid * n_test];
		stemp_mins = &temp_mins[tid * n_test];

		for (j = 0; j < n_test; ++j) {
			stemp_mins[j] = searchsorted(
					&sdot_ref_sort[0], sdot_test[j], n_ref);
		}

		for (j = 0; j < n_test; ++j) {
			idx = n_ref - stemp_mins[j];

			if (idx < stemp_mins[j]) {
				stemp_mins[j] = idx;
			}

			if (stemp_mins[j] < smins[j]) {
				smins[j] = stemp_mins[j];
			}
		}
	}

	for (i = 0; i < n_test; ++i) {
		depths_arr[i] = mins[i];

		for (size_t j = 1; j < n_cpus; ++j) {
			if (mins[(j * n_test) + i] >= depths_arr[i]) {
				continue;
			}

			depths_arr[i] = mins[(j * n_test) + i];
		}
	}

	return;
}
