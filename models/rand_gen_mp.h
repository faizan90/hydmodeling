#pragma once
#include <time.h>
#include <limits.h>
//#include <stdlib.h>

typedef double DT_D;
typedef long DT_UL;
typedef unsigned long long DT_ULL;

const DT_D MOD_long_long = (DT_D) ULLONG_MAX;
const DT_ULL A = 21, B = 35, C = 4;

static DT_ULL SEED = time(NULL) * 1000;
static DT_ULL rnd_i = SEED;

DT_D rand_c() {
    rnd_i ^= rnd_i << A;
    rnd_i ^= rnd_i >> B;
    rnd_i ^= rnd_i << C;
    return rnd_i / MOD_long_long;
}


void warm_up() {
	DT_UL i;
	for (i=0; i<1000; ++i) {
	    //printf("%0.10f\n", rand_c());
	    rand_c();
	}
}


DT_D rand_c_mp(DT_ULL *rnd_j) {
	DT_ULL rnd_k = *rnd_j;

    rnd_k ^= rnd_k << A;
    rnd_k ^= rnd_k >> B;
    rnd_k ^= rnd_k << C;
    *rnd_j = rnd_k;
    return rnd_k / MOD_long_long;
}


void warm_up_mp(DT_ULL *seeds_arr, DT_UL n_seeds) {
	DT_UL i, j;
	for (j=0; j<n_seeds; ++j) {
		for (i=0; i<1000; ++i) {
		    //printf("%0.10f\n", rand_c_mp(&seeds_arr[j]));
		    rand_c_mp(&seeds_arr[j]);
		}
	}
}
