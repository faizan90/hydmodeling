#pragma once
#include <math.h>
#include <stdio.h>

typedef double DT_D;
typedef long DT_UL;

inline DT_D min(DT_D a, DT_D b) {
    if (a <= b) {
        return a;
    }

    else {
        return b;
    }
}

inline DT_D max(DT_D a, DT_D b) {
    if (a >= b) {
        return a;
    }

    else {
        return b;
    }
}


DT_D hbv_c_loop(
	const DT_D *temp_arr,
	const DT_D *prec_arr,
	const DT_D *petn_arr,
	const DT_D *prms_arr,
	const DT_D *inis_arr,
	const DT_D *area_arr,
          DT_D *qsim_arr,
          DT_D *outs_arr,
    const DT_UL *n_time_steps,
    const DT_UL *n_cells,
	const DT_UL *n_prms,
	const DT_UL *n_vars_outs_arr,
	const DT_D *rnof_q_conv,
	const DT_UL *opt_flag
    ) {

	/*
	 * Implementation of distributed hbv_loop in cpp
	 * outs_arr can be initialized with values
	 * all other arrays should either hold some
	 * input values or zeros in case of outputs
	 */

	size_t i, j, k, m, n, p, cur_p, out_inc_n;

	// var idxs
	size_t snow_i, lppt_i, somo_i, rnof_i, evtn_i;
	size_t ur_i, ur_uo_i, ur_lo_i, ur_lr_i, lr_i;

	// vars
	DT_D temp, prec, petn, pre_snow, snow_melt, lppt, pre_somo;
	DT_D pre_ur_sto, ur_uo_rnof, ur_lo_rnof, ur_lr_seep, lr_rnof;
	DT_D rel_fc_beta, cell_area, pre_lr_sto, p_cm, avail_somo;
	DT_D pet_scale;

	// prm idxs
	size_t tt_i, cm_i, pwp_i, fc_i, beta_i, k_uu_i;
	size_t ur_thr_i, k_ul_i, k_d_i, k_ll_i, p_cm_i;

	// prms
	DT_D tt, cm, pwp, fc, beta, k_uu, ur_thr, k_ul, k_d, k_ll;

	// Previous current cell and time arrays
	DT_D *temp_j_arr, *prec_j_arr, *petn_j_arr, *prms_j_arr;
	DT_D *outs_j_arr;

	// Indicies of variables in the outs_arr
	snow_i = 0;
	lppt_i = 1;
	somo_i = 2;
	rnof_i = 3;
	evtn_i = 4;
	ur_i = 5;
	ur_uo_i = 6;
	ur_lo_i = 7;
	ur_lr_i = 8;
	lr_i = 9;

	// Indicies of variables in the prms_arr
	tt_i = 0;
	cm_i = 1;
	p_cm_i = 2;
	fc_i = 3;
	beta_i = 4;
	pwp_i = 5;
	ur_thr_i = 6;
	k_uu_i = 7;
	k_ul_i = 8;
	k_d_i = 9;
	k_ll_i = 10;

	if (*opt_flag) {
		out_inc_n = *n_vars_outs_arr;
	}

	else {
		out_inc_n = ((*n_time_steps) + 1) * *n_vars_outs_arr;
	}

	// Assign initial values
	for (i = 0, j = 0;
		 j < (*n_cells * out_inc_n);
		 i = (i + 4), j = (j + out_inc_n)) {

		outs_arr[j + snow_i] = inis_arr[i + 0];
		outs_arr[j + somo_i] = inis_arr[i + 1];
		outs_arr[j + ur_i] = inis_arr[i + 2];
		outs_arr[j + lr_i] = inis_arr[i + 3];
	}

	for (i = 0, j = 0, k = 0, m = 0;
		 j < (*n_cells * *n_time_steps);
		 i = (i + *n_prms), j = (j + *n_time_steps), k = (k + out_inc_n), ++m) {

		temp_j_arr = (DT_D *) &temp_arr[j];
        prec_j_arr = (DT_D *) &prec_arr[j];
        petn_j_arr = (DT_D *) &petn_arr[j];
        prms_j_arr = (DT_D *) &prms_arr[i];
        outs_j_arr = &outs_arr[k];

        tt = prms_j_arr[tt_i];
        cm = prms_j_arr[cm_i];
        p_cm = prms_j_arr[p_cm_i];
        fc = prms_j_arr[fc_i];
        beta = prms_j_arr[beta_i];
        pwp = prms_j_arr[pwp_i];
        ur_thr = prms_j_arr[ur_thr_i];
        k_uu = prms_j_arr[k_uu_i];
        k_ul = prms_j_arr[k_ul_i];
        k_d = prms_j_arr[k_d_i];
    	k_ll = prms_j_arr[k_ll_i];

        cell_area = area_arr[m];

		pre_snow = outs_j_arr[snow_i];
		pre_somo = outs_j_arr[somo_i];
		pre_ur_sto = outs_j_arr[ur_i];
		pre_lr_sto = outs_j_arr[lr_i];

        for (n = 0, p = *n_vars_outs_arr;
             n < *n_time_steps;
             ++n, p = (p + *n_vars_outs_arr)) {

        	// In case of optimization overwrite one vector. This is to save
        	// memory.
        	if (*opt_flag) {
        		cur_p = 0;
        	}

        	else {
        		cur_p = p;
        	}

            temp = temp_j_arr[n];
            prec = prec_j_arr[n];
            petn = petn_j_arr[n];

            // Snow and liquid ppt
            if (temp < tt) {
                outs_j_arr[cur_p + snow_i] = pre_snow + prec;
                outs_j_arr[cur_p + lppt_i] = 0.0;
            }

            else {
                snow_melt = ((cm + (p_cm * prec)) * (temp - tt));
            	outs_j_arr[cur_p + snow_i] = max(0.0,  pre_snow - snow_melt);
                outs_j_arr[cur_p + lppt_i] = prec + min(pre_snow, snow_melt);
            }

            pre_snow = outs_j_arr[cur_p + snow_i];

            lppt = outs_j_arr[cur_p + lppt_i];

            // Soil moisture and ET
			// If rel_fc_beta goes above 1 i.e. pre_somo > fc, that is self-
			// corrected in the next step by reducing the sm and giving that
			// water to runoff.
			// Also, for pre_somo greater than fc, rel_fc_beta is more than one
			// this is corrected with a min with one. This means that somo
			// won't change for that step and all water will be runoff
            rel_fc_beta = min(1.0, pow((pre_somo / fc), beta));

            avail_somo = pre_somo + (lppt * (1 - rel_fc_beta));

    		if (pre_somo > pwp) {
    			pet_scale = 1.0;
    		}

    		else {
    			pet_scale = pre_somo / pwp;
    		}

			outs_j_arr[cur_p + evtn_i] = min(avail_somo, pet_scale * petn);

			// Sometimes somo goes slightly below 0 for certain parameters'
			// combinations, also this will allow for drought modelling
			// without causing problems.
			// Comparison with 0.0 for rounding errors in case avail_somo and
			// outs_j_arr[i, evtn_i] are almost equal
            outs_j_arr[cur_p + somo_i] =  max(
                    0.0, avail_somo - outs_j_arr[cur_p + evtn_i]);

            pre_somo = outs_j_arr[cur_p + somo_i];

            // Total runoff
            outs_j_arr[cur_p + rnof_i] = lppt * rel_fc_beta;

            // Runonff, upper reservoir, upper outlet
            outs_j_arr[cur_p + ur_uo_i] = (
            	max(0.0, (pre_ur_sto - ur_thr) * k_uu));

            ur_uo_rnof = outs_j_arr[cur_p + ur_uo_i];

            // Runoff, upper reservoir, lower outlet
            outs_j_arr[cur_p + ur_lo_i] = (
            	max(0.0, (pre_ur_sto - ur_uo_rnof) * k_ul));

            ur_lo_rnof = outs_j_arr[cur_p + ur_lo_i];

            // Seepage to groundwater
            outs_j_arr[cur_p + ur_lr_i] = (
            	max(0.0, (pre_ur_sto - ur_uo_rnof - ur_lo_rnof) * k_d));

            ur_lr_seep = outs_j_arr[cur_p + ur_lr_i];

            // Upper reservoir storage
            outs_j_arr[cur_p + ur_i] = (
            	max(0.0,
            		(pre_ur_sto -
            		 ur_uo_rnof -
					 ur_lo_rnof -
					 ur_lr_seep +
					 outs_j_arr[cur_p + rnof_i])));

            pre_ur_sto = outs_j_arr[cur_p + ur_i];

            // Lower reservoir runoff and storage
            lr_rnof = pre_lr_sto * k_ll;

            outs_j_arr[cur_p + lr_i] = (pre_lr_sto + ur_lr_seep - lr_rnof);

            pre_lr_sto = outs_j_arr[cur_p + lr_i];

            // Upper and lower reservoirs combined discharge
            qsim_arr[n] += (
            		rnof_q_conv[0] *
					cell_area *
					(ur_uo_rnof + ur_lo_rnof + lr_rnof));
        }
	}

	return 0.0;
}
