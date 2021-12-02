#include "pocketfft.h"

void _pocket_real_dft(double *in_reals_arr, long n_pts) {

    rfft_plan plan = make_rfft_plan(n_pts);

    rfft_forward(plan, in_reals_arr, 1.);

    destroy_rfft_plan(plan);

    return;
}