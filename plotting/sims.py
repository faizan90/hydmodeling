"""
Created on %(date)s

@author: %(Faizan Anwar, IWS Uni-Stuttgart)s

"""

import os

import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties as f_props

from ..models import (
    hbv_loop_py,
    get_ns_cy,
    get_ln_ns_cy,
    get_pcorr_cy,
    get_kge_cy,
    lin_regsn_cy)

from ..misc import mkdir_hm, traceback_wrapper

plt.ioff()


@traceback_wrapper
def plot_cat_hbv_sim(plot_args):

    '''Plot all HBV variables along with some model information for all
    kfolds.
    '''

    cat_db, (wat_bal_stps,
             full_sim_flag,
             wat_bal_flag,
             show_warm_up_steps_flag) = plot_args

    extra_us_inflow_flag = False

    with h5py.File(cat_db, 'r') as db:
        out_dir = db['data'].attrs['main']
        kfolds = db['data'].attrs['kfolds']
        cat = db.attrs['cat']

        off_idx = db['data'].attrs['off_idx']
        conv_ratio = db['data'].attrs['conv_ratio']
        prm_syms = db['data/all_prms_labs'][...]
        use_obs_flow_flag = db['data'].attrs['use_obs_flow_flag']
        area_arr = db['data/area_arr'][...]

        cv_flag = db['data'].attrs['cv_flag']

        valid_not_in_db_flag = 'valid' not in db

        if cv_flag:
            cv_kf_dict = {}
            for i in range(1, kfolds + 1):
                cd_db = db[f'valid/kf_{i:02d}']
                kf_dict = {key: cd_db[key][...] for key in cd_db}
                kf_dict['use_obs_flow_flag'] = use_obs_flow_flag

                cv_kf_dict[i] = kf_dict
                break

        all_kfs_dict = {}
        for i in range(1, kfolds + 1):
            cd_db = db[f'calib/kf_{i:02d}']
            kf_dict = {key: cd_db[key][...] for key in cd_db}
            kf_dict['use_obs_flow_flag'] = use_obs_flow_flag

            extra_us_inflow_flag = 'extra_us_inflow' in kf_dict

            all_kfs_dict[i] = kf_dict

        if kfolds > 1:
            all_tem_arr = np.concatenate([
                all_kfs_dict[i]['tem_arr'] for i in all_kfs_dict], axis=1)

            all_ppt_arr = np.concatenate([
                all_kfs_dict[i]['ppt_arr'] for i in all_kfs_dict], axis=1)

            all_pet_arr = np.concatenate([
                all_kfs_dict[i]['pet_arr'] for i in all_kfs_dict], axis=1)

            all_qact_arr = np.concatenate([
                all_kfs_dict[i]['qact_arr'] for i in all_kfs_dict], axis=0)

            if 'extra_us_inflow' in kf_dict:
                all_us_inflow_arr = np.concatenate(
                    [all_kfs_dict[i]['extra_us_inflow']
                     for i in all_kfs_dict],
                    axis=0)

    for i in range(1, kfolds + 1):
        kf_dict = all_kfs_dict[i]
        kf_i = f'{i:02d}_calib'

        sim = PlotCatHBVSimKf(
            kf_i,
            cat,
            kf_dict,
            area_arr,
            conv_ratio,
            prm_syms,
            off_idx,
            out_dir,
            wat_bal_stps,
            show_warm_up_steps_flag)

        if wat_bal_flag:
            sim.wat_bal_sim()

        if (wat_bal_flag or full_sim_flag) and extra_us_inflow_flag:
            sim.flows_cmp()

        if full_sim_flag:
            sim.full_sim()

        if ((kfolds == 1) and (not cv_flag)) or valid_not_in_db_flag:
            continue

        kf_i = f'{i:02d}_valid'

        if cv_flag:
            # side effects
            kf_dict['tem_arr'] = cv_kf_dict[i]['tem_arr']
            kf_dict['ppt_arr'] = cv_kf_dict[i]['ppt_arr']
            kf_dict['pet_arr'] = cv_kf_dict[i]['pet_arr']
            kf_dict['qact_arr'] = cv_kf_dict[i]['qact_arr']

            if 'extra_us_inflow' in kf_dict:
                kf_dict['extra_us_inflow'] = (
                    cv_kf_dict[i]['extra_us_inflow'])

        else:
            # side effects
            kf_dict['tem_arr'] = all_tem_arr
            kf_dict['ppt_arr'] = all_ppt_arr
            kf_dict['pet_arr'] = all_pet_arr
            kf_dict['qact_arr'] = all_qact_arr

            if 'extra_us_inflow' in kf_dict:
                kf_dict['extra_us_inflow'] = all_us_inflow_arr

        sim = PlotCatHBVSimKf(
            kf_i,
            cat,
            kf_dict,
            area_arr,
            conv_ratio,
            prm_syms,
            off_idx,
            out_dir,
            wat_bal_stps,
            show_warm_up_steps_flag)

        if wat_bal_flag:
            sim.wat_bal_sim()

        if (wat_bal_flag or full_sim_flag) and extra_us_inflow_flag:
            sim.flows_cmp()

        if full_sim_flag:
            sim.full_sim()

    return


class PlotCatHBVSimKf:

    def __init__(
            self,
            kf_str,
            cat,
            kf_dict,
            area_arr,
            conv_ratio,
            prm_syms,
            off_idx,
            out_dir,
            wat_bal_stps,
            show_warm_up_steps_flag):

        self.kf_str = kf_str
        self.cat = cat
        self.kf_dict = kf_dict
        self.conv_ratio = conv_ratio
        self.prm_syms = prm_syms
        self.off_idx = off_idx
        self.out_dir = out_dir
        self.wat_bal_stps = wat_bal_stps
        self.show_warm_up_steps_flag = show_warm_up_steps_flag
        self.extra_us_inflow_flag = 'extra_us_inflow' in kf_dict

        if self.show_warm_up_steps_flag:
            self.plt_strt_idx = 0

        else:
            self.plt_strt_idx = self.off_idx

        if 'use_obs_flow_flag' in self.kf_dict:
            self.use_obs_flow_flag = bool(self.kf_dict['use_obs_flow_flag'])

        else:
            self.use_obs_flow_flag = False

        rarea_arr = area_arr.reshape(-1, 1)
        rrarea_arr = area_arr.reshape(-1, 1, 1)

        temp_dist_arr = self.kf_dict['tem_arr']
        prec_dist_arr = self.kf_dict['ppt_arr']
        pet_dist_arr = self.kf_dict['pet_arr']
        prms_dist_arr = self.kf_dict['hbv_prms']

        self.prms_arr = (rarea_arr * prms_dist_arr).sum(axis=0)

        self.n_recs = temp_dist_arr.shape[1]
        self.n_cells = temp_dist_arr.shape[0]

        all_outputs_dict = hbv_loop_py(
            temp_dist_arr,
            prec_dist_arr,
            pet_dist_arr,
            prms_dist_arr,
            kf_dict['ini_arr'],
            area_arr,
            conv_ratio)

        assert all_outputs_dict['loop_ret'] == 0.0

        self.temp_arr = (rarea_arr * temp_dist_arr).sum(axis=0)
        self.prec_arr = (rarea_arr * prec_dist_arr).sum(axis=0)
        self.pet_arr = (rarea_arr * pet_dist_arr).sum(axis=0)

        del temp_dist_arr, prec_dist_arr, pet_dist_arr, prms_dist_arr

        all_output = all_outputs_dict['outs_arr']

        self.full_sims_dir = os.path.join(self.out_dir, '03_hbv_figs')

        mkdir_hm(self.full_sims_dir)

        if self.extra_us_inflow_flag:
            self.flows_cmp_dir = os.path.join(
                self.out_dir, '14_inflows_outflows')

            mkdir_hm(self.flows_cmp_dir)

#         if True:
#             print('Saving snow array...')
#             out_data_loc = os.path.join(
#                 self.full_sims_dir,
#                 f'kf_{self.kf_str}_HBV_snow_{self.cat}.npy')
#
#             np.save(out_data_loc, all_output[:, :, 0])

        self.q_sim_arr = all_outputs_dict['qsim_arr']

        all_output = (rrarea_arr * all_output).sum(axis=0)

        del all_outputs_dict

        self.snow_arr = all_output[:, 0]
        self.liqu_arr = all_output[:, 1]
        self.sm_arr = all_output[:, 2]
        self.tot_run_arr = all_output[:, 3]
        self.evap_arr = all_output[:, 4]
        self.ur_sto_arr = all_output[:, 5]
        self.ur_run_uu = all_output[:, 6]
        self.ur_run_ul = all_output[:, 7]
        self.ur_to_lr_run = all_output[:, 8]
        self.lr_sto_arr = all_output[:, 9]
        self.lr_run_arr = self.lr_sto_arr * self.prms_arr[10]
        self.comb_run_arr = self.ur_run_uu + self.ur_run_ul + self.lr_run_arr

        self.q_sim_arr = (self.q_sim_arr).copy(order='C')
        self.q_act_arr = kf_dict['qact_arr']
        self.q_act_arr_diff = self.q_act_arr.copy()
        self.q_sim_arr_diff = self.q_sim_arr.copy()

        if self.extra_us_inflow_flag:
            self.extra_us_inflow = kf_dict['extra_us_inflow']
            self.q_sim_arr = self.q_sim_arr + self.extra_us_inflow
            self.q_act_arr_diff = self.q_act_arr - self.extra_us_inflow

        self.ns = get_ns_cy(self.q_act_arr, self.q_sim_arr, off_idx)
        self.ln_ns = get_ln_ns_cy(self.q_act_arr, self.q_sim_arr, off_idx)
        self.kge = get_kge_cy(self.q_act_arr, self.q_sim_arr, off_idx)
        self.q_correl = get_pcorr_cy(
            self.q_act_arr, self.q_sim_arr, off_idx)

        (self.tt,
         self.cm,
         self.p_cm,
         self.fc,
         self.beta,
         self.pwp,
         self.ur_thresh,
         self.k_uu,
         self.k_ul,
         self.k_d,
         self.k_ll) = self.prms_arr

        cum_q = np.cumsum(self.q_act_arr_diff[off_idx:] / conv_ratio)
        cum_q_sim = np.cumsum(self.comb_run_arr[off_idx:])

        self.min_vol_diff_err = 0.5
        self.max_vol_diff_err = 1.5

        self.vol_diff_arr = cum_q_sim / cum_q

        self.vol_diff_arr[
            self.vol_diff_arr < self.min_vol_diff_err + 0.05] = (
            self.min_vol_diff_err + 0.05)

        self.vol_diff_arr[
            self.vol_diff_arr > self.max_vol_diff_err - 0.05] = (
            self.max_vol_diff_err - 0.05)

        self.vol_diff_arr = np.concatenate((np.full(
            shape=(off_idx - 1),
            fill_value=np.nan),
            self.vol_diff_arr))

        self.bal_idxs = [0]
        self.bal_idxs.extend(list(range(off_idx, self.n_recs, wat_bal_stps)))

        sim_dict = {
            'temp': self.temp_arr,
            'prec': self.prec_arr,
            'pet': self.pet_arr,
            'snow': self.snow_arr,
            'liqu': self.liqu_arr,
            'sm': self.sm_arr,
            'tot_run': self.tot_run_arr,
            'evap': self.evap_arr,
            'ur_sto': self.ur_sto_arr,
            'ur_run_uu': self.ur_run_uu,
            'ur_run_ul': self.ur_run_ul,
            'ur_to_lr_run': self.ur_to_lr_run,
            'lr_sto': self.lr_sto_arr,
            'lr_run': self.lr_run_arr,
            'comb_run': self.comb_run_arr,
            'q_sim': self.q_sim_arr
            }

        sim_df = pd.DataFrame(sim_dict, dtype=float)
        sim_df.to_csv(os.path.join(
            self.full_sims_dir,
            f'kf_{self.kf_str}_HBV_sim_{self.cat}.csv'), sep=';')

        out_labs = []
        out_labs.extend(self.prm_syms)

        if 'route_labs' in self.kf_dict:
            out_labs.extend(self.kf_dict['route_labs'])

        out_labs.extend(['ns', 'ln_ns', 'kge', 'obj_ftn', 'p_corr'])

        out_params_df = pd.DataFrame(index=out_labs, columns=['value'])
        out_params_df['value'] = np.concatenate(
            (self.prms_arr,
             [self.ns,
              self.ln_ns,
              self.kge,
              'nothing_yet',
              self.q_correl]))

        out_params_loc = os.path.join(
            self.full_sims_dir,
            f'kf_{self.kf_str}_HBV_model_params_{self.cat}.csv')

        out_params_df.to_csv(
            out_params_loc, sep=str(';'), index_label='param')
        return

    def wat_bal_sim(self):

        wat_bals_dir = os.path.join(self.out_dir, '04_wat_bal_figs')

        mkdir_hm(wat_bals_dir)

        out_fig_loc = os.path.join(
            wat_bals_dir, f'kf_{self.kf_str}_HBV_water_bal_{self.cat}.png')

        act_bal_w_et_arr = []
        sim_bal_w_et_arr = []

        act_bal_wo_et_arr = []
        sim_bal_wo_et_arr = []

        prec_sum_arr = []
        evap_sum_arr = []
        q_act_sum_arr = []
        q_sim_sum_arr = []
        et_ratio_arr = []

        min_vol_ratio_err = 0
        max_vol_ratio_err = 1.5

        for i in range(1, len(self.bal_idxs) - 1):
            curr_q_act = self.q_act_arr_diff[
                self.bal_idxs[i]:self.bal_idxs[i + 1]]

            curr_q_act_sum = np.sum(curr_q_act / self.conv_ratio)

            prec_sum = np.sum(self.prec_arr[
                self.bal_idxs[i]:self.bal_idxs[i + 1]])

            evap_sum = np.sum(self.evap_arr[
                self.bal_idxs[i]:self.bal_idxs[i + 1]])

            curr_comb_sum = np.sum(
                self.comb_run_arr[self.bal_idxs[i]:self.bal_idxs[i + 1]])

            curr_sto = (
                self.ur_sto_arr[self.bal_idxs[i + 1]] +
                self.lr_sto_arr[self.bal_idxs[i + 1]] +
                self.snow_arr[self.bal_idxs[i + 1]] +
                self.sm_arr[self.bal_idxs[i + 1]])

            pre_sto = (
                self.ur_sto_arr[self.bal_idxs[i]] +
                self.lr_sto_arr[self.bal_idxs[i]] +
                self.snow_arr[self.bal_idxs[i]] +
                self.sm_arr[self.bal_idxs[i]])

            # ET accounted for
            act_bal_w_et_arr.append(
                curr_q_act_sum / (pre_sto + prec_sum - evap_sum - curr_sto))

            sim_bal_w_et_arr.append(
                curr_comb_sum / (pre_sto + prec_sum - evap_sum - curr_sto))

            # ET not accounted for
            act_bal_wo_et_arr.append(
                curr_q_act_sum / (pre_sto + prec_sum - curr_sto))

            sim_bal_wo_et_arr.append(
                curr_comb_sum / (pre_sto + prec_sum - curr_sto))

            prec_sum_arr.append(prec_sum)
            evap_sum_arr.append(evap_sum)
            q_act_sum_arr.append(curr_q_act_sum)
            q_sim_sum_arr.append(curr_comb_sum)
            et_ratio_arr.append(evap_sum / prec_sum)

        act_bal_w_et_arr = np.asarray(act_bal_w_et_arr)
        act_bal_wo_et_arr = np.asarray(act_bal_wo_et_arr)

        sim_bal_w_et_arr = np.array(sim_bal_w_et_arr)
        sim_bal_wo_et_arr = np.array(sim_bal_wo_et_arr)
        et_ratio_arr = np.array(et_ratio_arr)

        act_bal_w_et_arr[act_bal_w_et_arr < min_vol_ratio_err] = (
            min_vol_ratio_err)

        act_bal_w_et_arr[act_bal_w_et_arr > max_vol_ratio_err] = (
            max_vol_ratio_err)

        act_bal_w_et_arr = (
            np.concatenate(([np.nan, np.nan], act_bal_w_et_arr), axis=0))

        sim_bal_w_et_arr[sim_bal_w_et_arr < min_vol_ratio_err] = (
            min_vol_ratio_err)

        sim_bal_w_et_arr[sim_bal_w_et_arr > max_vol_ratio_err] = (
            max_vol_ratio_err)

        sim_bal_w_et_arr = (
            np.concatenate(([np.nan, np.nan], sim_bal_w_et_arr), axis=0))

        act_bal_wo_et_arr[act_bal_wo_et_arr < min_vol_ratio_err] = (
            min_vol_ratio_err)

        act_bal_wo_et_arr[act_bal_wo_et_arr > max_vol_ratio_err] = (
            max_vol_ratio_err)

        act_bal_wo_et_arr = (
            np.concatenate(([np.nan, np.nan], act_bal_wo_et_arr), axis=0))

        sim_bal_wo_et_arr[sim_bal_wo_et_arr < min_vol_ratio_err] = (
            min_vol_ratio_err)

        sim_bal_wo_et_arr[sim_bal_wo_et_arr > max_vol_ratio_err] = (
            max_vol_ratio_err)

        sim_bal_wo_et_arr = (
            np.concatenate(([np.nan, np.nan], sim_bal_wo_et_arr), axis=0))

        et_ratio_arr[et_ratio_arr < min_vol_ratio_err] = (
            min_vol_ratio_err)

        et_ratio_arr[et_ratio_arr > max_vol_ratio_err] = (
            max_vol_ratio_err)

        et_ratio_arr = (
            np.concatenate(([np.nan, np.nan], et_ratio_arr), axis=0))

        prec_sum_arr = np.concatenate(
            ([np.nan, np.nan], prec_sum_arr), axis=0)

        evap_sum_arr = np.concatenate(
            ([np.nan, np.nan], evap_sum_arr), axis=0)

        q_act_sum_arr = (
            np.concatenate(([np.nan, np.nan], q_act_sum_arr), axis=0))

        q_sim_sum_arr = (
            np.concatenate(([np.nan, np.nan], q_sim_sum_arr), axis=0))

        font_size = 5
        t_rows = 7
        t_cols = 1

        fig = plt.figure(figsize=(13, 6), dpi=150)

        plt.suptitle('HBV Simulation - Water Balance')

        i = 0

        vol_err_ax = plt.subplot2grid(
            (t_rows, t_cols), (i, 0), rowspan=1, colspan=t_cols)
        i += 1

        discharge_ax = plt.subplot2grid(
            (t_rows, t_cols), (i, 0), rowspan=2, colspan=t_cols)
        i += 2

        balance_ratio_ax = plt.subplot2grid(
            (t_rows, t_cols), (i, 0), rowspan=2, colspan=t_cols)
        i += 2

        balance_sum_ax = plt.subplot2grid(
            (t_rows, t_cols), (i, 0), rowspan=2, colspan=t_cols)
        i += 2

        vol_err_ax.axhline(1, color='k', lw=1)
        vol_err_ax.plot(
            self.vol_diff_arr,
            lw=0.5,
            label='Cumm. Runoff Error',
            alpha=0.95)

        vol_err_ax.set_ylim(self.min_vol_diff_err, self.max_vol_diff_err)
        vol_err_ax.set_xlim(self.plt_strt_idx, vol_err_ax.get_xlim()[1])

        vol_txt_x_crd = self.plt_strt_idx + (
            (self.vol_diff_arr.shape[0] - self.plt_strt_idx) * 0.5)

        vol_err_ax.text(
            vol_txt_x_crd,
            1.25,
            'runoff over-estimation',
            fontsize=font_size * 0.9)

        vol_err_ax.text(
            vol_txt_x_crd,
            0.75,
            'runoff under-estimation',
            fontsize=font_size * 0.9)

        vol_err_ax.set_xticklabels([])

        discharge_ax.plot(
            self.q_act_arr,
            'r-',
            label='Actual Flow',
            lw=0.5,
            alpha=0.7)

        discharge_ax.plot(
            self.q_sim_arr,
            'b-',
            label='Simulated Flow',
            lw=0.5,
            alpha=0.5)

        discharge_ax.set_xticklabels([])

        scatt_size = 5

        balance_ratio_ax.axhline(1, color='k', lw=1)
        balance_ratio_ax.scatter(
            self.bal_idxs,
            act_bal_w_et_arr,
            marker='o',
            label='Actual Outflow (Q + ET)',
            alpha=0.6,
            s=scatt_size)

        balance_ratio_ax.scatter(
            self.bal_idxs,
            sim_bal_w_et_arr,
            marker='+',
            label='Simulated Outflow (Q + ET)',
            alpha=0.6,
            s=scatt_size)

        balance_ratio_ax.scatter(
            self.bal_idxs,
            act_bal_wo_et_arr,
            marker='o',
            label='Actual Outflow (Q)',
            alpha=0.6,
            s=scatt_size)

        balance_ratio_ax.scatter(
            self.bal_idxs,
            sim_bal_wo_et_arr,
            marker='+',
            label='Simulated Outflow (Q)',
            alpha=0.6,
            s=scatt_size)

        balance_ratio_ax.scatter(
            self.bal_idxs,
            et_ratio_arr,
            marker='+',
            label='Evapotranspiration (ET)',
            alpha=0.6,
            s=scatt_size)

        balance_ratio_ax.set_ylim(min_vol_ratio_err, max_vol_ratio_err)
        balance_ratio_ax.set_xticklabels([])

        balance_sum_ax.scatter(
            self.bal_idxs,
            prec_sum_arr,
            alpha=0.6,
            marker='o',
            label='Precipitation Sum',
            s=scatt_size)

        balance_sum_ax.scatter(
            self.bal_idxs,
            evap_sum_arr,
            alpha=0.6,
            marker='+',
            label='Evapotranspiration Sum',
            s=scatt_size)

        balance_sum_ax.scatter(
            self.bal_idxs,
            q_act_sum_arr,
            alpha=0.6,
            marker='o',
            label='Actual Runoff Sum',
            s=scatt_size)

        balance_sum_ax.scatter(
            self.bal_idxs,
            q_sim_sum_arr,
            alpha=0.6,
            marker='+',
            label='Simulated Runoff Sum',
            s=scatt_size)

        balance_sum_ax.set_ylim(0, balance_sum_ax.get_ylim()[1])

        leg_font = f_props()
        leg_font.set_size(font_size)

        plot_axes = [
            discharge_ax,
            vol_err_ax,
            balance_ratio_ax,
            balance_sum_ax,
            ]

        for ax in plot_axes:
            for tlab in ax.get_yticklabels():
                tlab.set_fontsize(font_size)

            ax.grid()
            ax.set_aspect('auto')

            ax.set_xlim(self.plt_strt_idx, discharge_ax.get_xlim()[1])

            leg = ax.legend(framealpha=0.5, prop=leg_font, ncol=5, loc=0)
            leg.get_frame().set_edgecolor('black')

        for tlab in balance_sum_ax.get_xticklabels():
            tlab.set_fontsize(font_size)

        fig.savefig(out_fig_loc, bbox_inches='tight')

        plt.close('all')
        return

    def flows_cmp(self):

        cum_q = np.cumsum(self.q_act_arr[self.off_idx:])
        cum_q_sim = np.cumsum(self.q_sim_arr[self.off_idx:])

        tot_vol_diff_arr = cum_q_sim / cum_q

        tot_vol_diff_arr[
            tot_vol_diff_arr < self.min_vol_diff_err + 0.05] = (
            self.min_vol_diff_err + 0.05)

        tot_vol_diff_arr[
            tot_vol_diff_arr > self.max_vol_diff_err - 0.05] = (
            self.max_vol_diff_err - 0.05)

        tot_vol_diff_arr = np.concatenate((np.full(
            shape=(self.off_idx - 1),
            fill_value=np.nan),
            tot_vol_diff_arr))

        t_rows = 5
        t_cols = 1
        font_size = 6

        out_fig_loc = os.path.join(
            self.flows_cmp_dir,
            f'kf_{self.kf_str}_cat_flows_cmp_{self.cat}.png')

        fig = plt.figure(figsize=(13, 6), dpi=250)

        plt.suptitle('HBV Simulation - Inflow/Outflow')

        i = 0

        discharge_ax = plt.subplot2grid(
            (t_rows, t_cols), (i, 0), rowspan=1, colspan=1)
        i += 1

        us_add_flow_ax = plt.subplot2grid(
            (t_rows, t_cols), (i, 0), rowspan=1, colspan=1)
        i += 1

        res_discharge_ax = plt.subplot2grid(
            (t_rows, t_cols), (i, 0), rowspan=1, colspan=1)
        i += 1

        vol_err_ax = plt.subplot2grid(
            (t_rows, t_cols), (i, 0), rowspan=1, colspan=1)
        i += 1

        tot_vol_err_ax = plt.subplot2grid(
            (t_rows, t_cols), (i, 0), rowspan=1, colspan=1)
        i += 1

        discharge_ax.plot(
            self.q_act_arr,
            'r-',
            label='Actual Flow',
            lw=0.5,
            alpha=0.7)

        discharge_ax.plot(
            self.q_sim_arr,
            'b-',
            label='Simulated Flow',
            lw=0.5,
            alpha=0.5)

        discharge_ax.set_xticklabels([])

        us_add_flow_ax.plot(
            self.extra_us_inflow,
            'r-',
            label='Upstream Inflow',
            lw=0.5,
            alpha=0.7)

        us_add_flow_ax.plot(
            self.q_sim_arr_diff,
            'b-',
            label='Simulated Residual Outflow',
            lw=0.5,
            alpha=0.7)

        us_add_flow_ax.set_xticklabels([])

        res_discharge_ax.plot(
            self.q_act_arr_diff,
            'r-',
            label='Actual Residual Outflow',
            lw=0.5,
            alpha=0.7)

        res_discharge_ax.plot(
            self.q_sim_arr_diff,
            'b-',
            label='Simulated Residual Outflow',
            lw=0.5,
            alpha=0.5)

        vol_err_ax.axhline(1.0, color='k', lw=1)

        vol_err_ax.plot(
            self.vol_diff_arr,
            lw=0.5,
            label='Cumm. Residual Runoff Error',
            alpha=0.95)

        vol_err_ax.set_ylim(self.min_vol_diff_err, self.max_vol_diff_err)

        vol_txt_x_crd = self.plt_strt_idx + (
            (self.vol_diff_arr.shape[0] - self.plt_strt_idx) * 0.5)

        vol_err_ax.text(
            vol_txt_x_crd,
            1.25,
            'runoff over-estimation',
            fontsize=font_size * 0.9)

        vol_err_ax.text(
            vol_txt_x_crd,
            0.75,
            'runoff under-estimation',
            fontsize=font_size * 0.9)

        tot_vol_err_ax.axhline(1.0, color='k', lw=1)

        tot_vol_err_ax.plot(
            tot_vol_diff_arr,
            lw=0.5,
            label='Cumm. Total Runoff Error',
            alpha=0.95)

        tot_vol_err_ax.set_ylim(self.min_vol_diff_err, self.max_vol_diff_err)

        vol_txt_x_crd = self.plt_strt_idx + (
            (tot_vol_diff_arr.shape[0] - self.plt_strt_idx) * 0.5)

        tot_vol_err_ax.text(
            vol_txt_x_crd,
            1.25,
            'runoff over-estimation',
            fontsize=font_size * 0.9)

        tot_vol_err_ax.text(
            vol_txt_x_crd,
            0.75,
            'runoff under-estimation',
            fontsize=font_size * 0.9)

        leg_font = f_props()
        leg_font.set_size(font_size - 1)

        plot_axes = [
            discharge_ax,
            res_discharge_ax,
            vol_err_ax,
            us_add_flow_ax,
            tot_vol_err_ax,
            ]

        for ax in plot_axes:
            for tlab in ax.get_yticklabels():
                tlab.set_fontsize(font_size - 1)

            for tlab in ax.get_xticklabels():
                tlab.set_fontsize(font_size - 1)

            ax.grid()

            ax.set_xlim(self.plt_strt_idx, discharge_ax.get_xlim()[1])

            leg = ax.legend(framealpha=0.7, prop=leg_font, ncol=4, loc=0)
            leg.get_frame().set_edgecolor('black')

        fig.tight_layout(rect=[0, 0.01, 1, 0.95], h_pad=0.0)

        plt.savefig(out_fig_loc, bbox_inches='tight')

        plt.close('all')
        return

    def full_sim(self):

        full_sim_text = np.asarray([
            ('Max. actual Q = %0.4f' %
             self.q_act_arr[self.off_idx:].max()).rstrip('0'),
            ('Max. sim. Q = %0.4f' %
             self.q_sim_arr[self.off_idx:].max()).rstrip('0'),
            ('Min. actual Q = %0.4f' %
             self.q_act_arr[self.off_idx:].min()).rstrip('0'),
            ('Min. sim. Q = %0.4f' %
             self.q_sim_arr[self.off_idx:].min()).rstrip('0'),
            ('Mean actual Q = %0.4f' %
             np.mean(self.q_act_arr[self.off_idx:])).rstrip('0'),
            ('Mean sim. Q = %0.4f' %
             np.mean(self.q_sim_arr[self.off_idx:])).rstrip('0'),
            ('Max. input P = %0.4f' % self.prec_arr.max()).rstrip('0'),
            ('Max. SD = %0.2f' %
             self.snow_arr[self.off_idx:].max()).rstrip('0'),
            ('Max. liquid P = %0.4f' %
             self.liqu_arr[self.off_idx:].max()).rstrip('0'),
            ('Max. input T = %0.2f' %
             self.temp_arr[self.off_idx:].max()).rstrip('0'),
            ('Min. input T = %0.2f' %
             self.temp_arr[self.off_idx:].min()).rstrip('0'),
            ('Max. PET = %0.2f' %
             self.pet_arr[self.off_idx:].max()).rstrip('0'),
            ('Max. sim. ET = %0.2f' %
             self.evap_arr[self.off_idx:].max()).rstrip('0'),
            ('Min. PET = %0.2f' %
             self.pet_arr[self.off_idx:].min()).rstrip('0'),
            ('Min. sim. ET = %0.2f' %
             self.evap_arr[self.off_idx:].min()).rstrip('0'),
            ('Max. sim. SM = %0.2f' %
             self.sm_arr[self.off_idx:].max()).rstrip('0'),
            ('Min. sim. SM = %0.2f' %
             self.sm_arr[self.off_idx:].min()).rstrip('0'),
            ('Mean sim. SM = %0.2f' %
             np.mean(self.sm_arr[self.off_idx:])).rstrip('0'),
            'Warm up steps = %d' % self.off_idx,
            'use_obs_flow_flag = %s' % self.use_obs_flow_flag,
            ('NS = %0.4f' % self.ns).rstrip('0'),
            ('Ln-NS = %0.4f' % self.ln_ns).rstrip('0'),
            ('KG = %0.4f' % self.kge).rstrip('0'),
            ('P_Corr = %0.4f' % self.q_correl).rstrip('0'),
            ('TT = %0.4f' % self.tt).rstrip('0'),
            ('CM = %0.4f' % self.cm).rstrip('0'),
            ('P_CM = %0.4f' % self.p_cm).rstrip('0'),
            ('FC = %0.4f' % self.fc).rstrip('0'),
            (r'$\beta$ = %0.4f' % self.beta).rstrip('0'),
            ('PWP = %0.4f' % self.pwp).rstrip('0'),
            ('UR$_{thresh}$ = %0.4f' % self.ur_thresh).rstrip('0'),
            ('$K_{uu}$ = %0.4f' % self.k_uu).rstrip('0'),
            ('$K_{ul}$ = %0.4f' % self.k_ul).rstrip('0'),
            ('$K_d$ = %0.4f' % self.k_d).rstrip('0'),
            ('$K_{ll}$ = %0.4f' % self.k_ll).rstrip('0'),
            ''])

        out_fig_loc = os.path.join(
            self.full_sims_dir,
            f'kf_{self.kf_str}_HBV_model_plot_{self.cat}.png')

        act_bal_arr = []
        sim_bal_arr = []

        prec_sum_arr = []

        min_vol_err_w_et = 0.5
        max_vol_err_w_et = 1.5

        for i in range(1, len(self.bal_idxs) - 1):
            # ET accounted for
            curr_q_act = self.q_act_arr_diff[
                self.bal_idxs[i]:self.bal_idxs[i + 1]]

            curr_q_act_sum = np.sum(curr_q_act / self.conv_ratio)

            curr_prec_sum = np.sum(self.prec_arr[
                self.bal_idxs[i]:self.bal_idxs[i + 1]])

            curr_evap_sum = np.sum(self.evap_arr[
                self.bal_idxs[i]:self.bal_idxs[i + 1]])

            curr_comb_sum = np.sum(self.comb_run_arr[
                self.bal_idxs[i]:self.bal_idxs[i + 1]])

            curr_sto = (
                self.ur_sto_arr[self.bal_idxs[i + 1]] +
                self.lr_sto_arr[self.bal_idxs[i + 1]] +
                self.snow_arr[self.bal_idxs[i + 1]] +
                self.sm_arr[self.bal_idxs[i + 1]])

            pre_sto = (
                self.ur_sto_arr[self.bal_idxs[i]] +
                self.lr_sto_arr[self.bal_idxs[i]] +
                self.snow_arr[self.bal_idxs[i]] +
                self.sm_arr[self.bal_idxs[i]])

            act_bal_arr.append(
                curr_q_act_sum / (
                    pre_sto + curr_prec_sum - curr_evap_sum - curr_sto))

            sim_bal_arr.append(
                curr_comb_sum / (
                    pre_sto + curr_prec_sum - curr_evap_sum - curr_sto))

            prec_sum_arr.append(curr_prec_sum)

        act_bal_arr = np.asarray(act_bal_arr)

        act_bal_arr[act_bal_arr < min_vol_err_w_et] = min_vol_err_w_et
        act_bal_arr[act_bal_arr > max_vol_err_w_et] = max_vol_err_w_et

        act_bal_arr = np.concatenate(
            ([np.nan, np.nan], act_bal_arr), axis=0)
        sim_bal_arr = np.concatenate(
            ([np.nan, np.nan], sim_bal_arr), axis=0)

        prec_sum_arr = np.concatenate(
            ([np.nan, np.nan], prec_sum_arr), axis=0)

        steps_range = np.arange(0., self.n_recs)
        temp_trend, temp_corr, temp_slope, *__ = lin_regsn_cy(
            steps_range, self.temp_arr, 0)

        temp_stats_str = ''
        temp_stats_str += f'correlation: {temp_corr:0.5f}, '
        temp_stats_str += f'slope: {temp_slope:0.5f}'

        pet_trend, pet_corr, pet_slope, *__ = lin_regsn_cy(
            steps_range, self.pet_arr, 0)

        pet_stats_str = ''
        pet_stats_str += f'PET correlation: {pet_corr:0.5f}, '
        pet_stats_str += f'PET slope: {pet_slope:0.5f}, '

        et_trend, et_corr, et_slope, *__ = lin_regsn_cy(
            steps_range, self.evap_arr.copy(order='c'), 0)

        pet_stats_str += f'ET correlation: {et_corr:0.5f}, '
        pet_stats_str += f'ET slope: {et_slope:0.5f}'

        fig = plt.figure(figsize=(11, 27), dpi=250)

        t_rows = 14
        t_cols = 1
        font_size = 6

        plt.suptitle('HBV Simulation - Variables')

        i = 0

        params_ax = plt.subplot2grid(
            (t_rows, t_cols), (i, 0), rowspan=1, colspan=1)
        i += 1

        discharge_ax = plt.subplot2grid(
            (t_rows, t_cols), (i, 0), rowspan=1, colspan=1)
        i += 1

        prec_ax = plt.subplot2grid(
            (t_rows, t_cols), (i, 0), rowspan=1, colspan=1)
        i += 1

        liqu_ax = plt.subplot2grid(
            (t_rows, t_cols), (i, 0), rowspan=1, colspan=1)
        i += 1

        snow_ax = plt.subplot2grid(
            (t_rows, t_cols), (i, 0), rowspan=1, colspan=1)
        i += 1

        temp_ax = plt.subplot2grid(
            (t_rows, t_cols), (i, 0), rowspan=1, colspan=1)
        i += 1

        pet_ax = plt.subplot2grid(
            (t_rows, t_cols), (i, 0), rowspan=1, colspan=1)
        i += 1

        sm_ax = plt.subplot2grid(
            (t_rows, t_cols), (i, 0), rowspan=1, colspan=1)
        i += 1

        tot_run_ax = plt.subplot2grid(
            (t_rows, t_cols), (i, 0), rowspan=1, colspan=1)
        i += 1

        u_res_sto_ax = plt.subplot2grid(
            (t_rows, t_cols), (i, 0), rowspan=1, colspan=1)
        i += 1

        ur_run_uu_ax = plt.subplot2grid(
            (t_rows, t_cols), (i, 0), rowspan=1, colspan=1)
        i += 1

        ur_run_ul_ax = plt.subplot2grid(
            (t_rows, t_cols), (i, 0), rowspan=1, colspan=1)
        i += 1

        ur_to_lr_run_ax = plt.subplot2grid(
            (t_rows, t_cols), (i, 0), rowspan=1, colspan=1)
        i += 1

        l_res_sto_ax = plt.subplot2grid(
            (t_rows, t_cols), (i, 0), rowspan=1, colspan=1)

        bar_x = np.arange(0, self.n_recs, 1)

        discharge_ax.plot(self.q_act_arr, 'r-', label='Actual Flow', lw=0.5)

        discharge_ax.plot(
            self.q_sim_arr, 'b-', label='Simulated Flow', lw=0.5, alpha=0.5)

        scatt_size = 5

        prec_ax.plot(
            bar_x,
            self.prec_arr,
            label='Precipitation',
            lw=0.1)

        prec_sums_ax = prec_ax.twinx()

        prec_sums_ax.scatter(
            self.bal_idxs,
            prec_sum_arr,
            c='b',
            label='Precipitation Sum',
            alpha=0.5,
            s=scatt_size)

        temp_ax.plot(self.temp_arr, 'b-', lw=0.5, label='Temperature')
        temp_ax.plot(temp_trend, 'b-.', lw=0.9, label='Temperature Trend')

        temp_pet_x_crd = self.plt_strt_idx + (
            (temp_ax.get_xlim()[1] - self.plt_strt_idx) * 0.02)

        temp_ax.text(
            temp_pet_x_crd,
            (temp_ax.get_ylim()[0] +
             (temp_ax.get_ylim()[1] - temp_ax.get_ylim()[0]) * 0.1),
            str(temp_stats_str),
            fontsize=font_size,
            bbox={'facecolor': 'white', 'alpha': 0.5, 'pad': 2})

        pet_ax.plot(
            self.pet_arr, 'r-', lw=0.5, label='Potential Evapotranspiration')

        pet_ax.plot(
            self.evap_arr,
            'b-',
            lw=0.5,
            label='Evapotranspiration',
            alpha=0.6)

        pet_ax.plot(pet_trend, 'r-.', lw=0.9, label='PET Trend')
        pet_ax.plot(et_trend, 'b-.', lw=0.9, label='ET Trend')

        pet_ax.text(
            temp_pet_x_crd,
            (pet_ax.get_ylim()[1] -
             (pet_ax.get_ylim()[1] - pet_ax.get_ylim()[0]) * 0.1),
            str(pet_stats_str),
            fontsize=font_size,
            bbox={'facecolor': 'white', 'alpha': 0.5, 'pad': 2})

        snow_ax.plot(self.snow_arr, lw=0.1, label='Snow')

        liqu_ax.plot(
            bar_x,
            self.liqu_arr,
            label='Liquid Precipitation',
            lw=0.1)

        sm_ax.plot(self.sm_arr, lw=0.1, label='Soil Moisture')
        sm_ax.axhline(self.fc, color='r', ls='-.', lw=1, label='fc')
        sm_ax.axhline(self.pwp, color='b', ls='-.', lw=1, label='pwp')

        tot_run_ax.plot(
            bar_x,
            self.tot_run_arr,
            label='Total Runoff',
            lw=0.1)

        u_res_sto_ax.plot(
            self.ur_sto_arr, lw=0.1, label='Upper Reservoir - Storage')

        ur_run_uu_ax.plot(
            bar_x,
            self.ur_run_uu,
            label='Upper Reservoir - Quick Runoff',
            lw=0.1)

        ur_run_ul_ax.plot(
            bar_x,
            self.ur_run_ul,
            label='Upper Reservoir - Slow Runoff',
            lw=0.1)

        ur_to_lr_run_ax.plot(
            bar_x,
            self.ur_to_lr_run,
            label='Upper Reservoir  - Percolation',
            lw=0.1)

        l_res_sto_ax.plot(
            self.lr_sto_arr, lw=0.1, label='Lower Reservoir - Storage')

        prec_ax.fill_between(bar_x, 0, self.prec_arr, alpha=0.3)
        liqu_ax.fill_between(bar_x, 0, self.liqu_arr, alpha=0.3)
        tot_run_ax.fill_between(bar_x, 0, self.tot_run_arr, alpha=0.3)
        snow_ax.fill_between(bar_x, 0, self.snow_arr, alpha=0.3)
        sm_ax.fill_between(bar_x, 0, self.sm_arr, alpha=0.3)
        u_res_sto_ax.fill_between(bar_x, 0, self.ur_sto_arr, alpha=0.3)
        ur_run_uu_ax.fill_between(bar_x, 0, self.ur_run_uu, alpha=0.3)
        ur_run_ul_ax.fill_between(bar_x, 0, self.ur_run_ul, alpha=0.3)
        ur_to_lr_run_ax.fill_between(bar_x, 0, self.ur_to_lr_run, alpha=0.3)
        l_res_sto_ax.fill_between(bar_x, 0, self.lr_sto_arr, alpha=0.3)

        full_sim_text = full_sim_text.reshape(6, 6)

        table = params_ax.table(
            cellText=full_sim_text,
            loc='center',
            bbox=(0, 0, 1, 1),
            cellLoc='left')

        table.auto_set_font_size(False)
        table.set_fontsize(font_size)

        params_ax.set_axis_off()

        leg_font = f_props()
        leg_font.set_size(font_size - 1)

        plot_axes = [
            prec_sums_ax,
            discharge_ax,
            prec_ax,
            temp_ax,
            pet_ax,
            snow_ax,
            liqu_ax,
            sm_ax,
            tot_run_ax,
            u_res_sto_ax,
            ur_run_uu_ax,
            ur_run_ul_ax,
            ur_to_lr_run_ax,
            l_res_sto_ax,
            ]

        for ax in plot_axes:
            for tlab in ax.get_yticklabels():
                tlab.set_fontsize(font_size - 1)

            for tlab in ax.get_xticklabels():
                tlab.set_fontsize(font_size - 1)

            ax.grid()

            ax.set_xlim(self.plt_strt_idx, discharge_ax.get_xlim()[1])

            leg = ax.legend(framealpha=0.7, prop=leg_font, ncol=4, loc=1)
            leg.get_frame().set_edgecolor('black')

        leg = prec_sums_ax.legend(
            framealpha=0.7, prop=leg_font, ncol=4, loc=2)

        leg.get_frame().set_edgecolor('black')

        prec_sums_ax.grid(False)

        fig.tight_layout(rect=[0, 0.01, 1, 0.97], h_pad=0.0)

        plt.savefig(out_fig_loc, bbox_inches='tight')

        plt.close('all')
        return
