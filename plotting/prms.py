'''
Created on Sep 23, 2018

@author: Faizan
'''

import os
from glob import glob
from pathlib import Path

import h5py
import numpy as np
from numpngw import write_apng
import matplotlib.pyplot as plt
from adjustText import adjust_text
from matplotlib.gridspec import GridSpec

plt.ioff()


def plot_cats_best_prms_2d(dbs_dir):

    '''Plot catchment best parameters for a kfolds in 2D.'''

    cats_dbs = glob(os.path.join(dbs_dir, 'cat_*.hdf5'))

    assert cats_dbs

    kf_prms_dict = {}
    rows_dict = {}
    cols_dict = {}

    with h5py.File(cats_dbs[0], 'r') as db:
        kfolds = db['data'].attrs['kfolds']
        shape = db['data/shape'][...]

        bds_db = db['data/bds_dict']
        bds_dict = {key: bds_db[key][...] for key in bds_db}
        prms_labs = db['data/all_prms_labs'][...]
        lumped_prms_flag = db['data'].attrs['run_as_lump_flag']

        out_dir = db['data'].attrs['main']
        out_dir = os.path.join(out_dir, r'07_2d_kfold_prms')
        if not os.path.exists(out_dir):
            try:
                os.mkdir(out_dir)

            except FileExistsError:
                pass

    min_row = +np.inf
    max_row = -np.inf
    min_col = +np.inf
    max_col = -np.inf

    aux_vars_dict = {}
    aux_prms_labs = []
    for cat_db in cats_dbs:
        with h5py.File(cat_db, 'r') as db:
            cat = db.attrs['cat']
            assert kfolds == db['data'].attrs['kfolds']

            if cat not in rows_dict:
                rows_dict[cat] = db['data/rows'][...]

                if rows_dict[cat].min() < min_row:
                    min_row = rows_dict[cat].min()

                if rows_dict[cat].max() > max_row:
                    max_row = rows_dict[cat].max()

            if cat not in cols_dict:
                cols_dict[cat] = db['data/cols'][...]

                if cols_dict[cat].min() < min_col:
                    min_col = cols_dict[cat].min()

                if cols_dict[cat].max() > max_col:
                    max_col = cols_dict[cat].max()

            for i in range(1, kfolds + 1):
                if i not in kf_prms_dict:
                    kf_prms_dict[i] = {}

                kf_prms_dict[i][cat] = db[f'calib/kf_{i:02d}/hbv_prms'][...]

            aux_vars_dict[cat] = {}
            cdata_db = db['cdata']

            if 'lulc_arr' in cdata_db:
                if 'lulc_arr' not in aux_prms_labs:
                    aux_prms_labs.append('lulc_arr')

                lulc_arr = cdata_db['lulc_arr'][...]

                assert lulc_arr.ndim == 2

                for i in range(lulc_arr.shape[1]):
                    aux_vars_dict[cat][f'lulc_{i:02d}'] = lulc_arr[:, i]

            if 'soil_arr' in cdata_db:
                if 'soil_arr' not in aux_prms_labs:
                    aux_prms_labs.append('soil_arr')

                soil_arr = cdata_db['soil_arr'][...]

                assert soil_arr.ndim == 2

                for i in range(soil_arr.shape[1]):
                    aux_vars_dict[cat][f'soil_{i:02d}'] = soil_arr[:, i]

            if 'aspect_scale_arr' in cdata_db:
                aspect_scale_arr = cdata_db['aspect_scale_arr'][...]

                assert aspect_scale_arr.ndim == 1

                aux_vars_dict[cat]['aspect_scale'] = aspect_scale_arr

            if 'slope_scale_arr' in cdata_db:
                slope_scale_arr = cdata_db['slope_scale_arr'][...]

                assert slope_scale_arr.ndim == 1

                aux_vars_dict[cat]['slope_scale'] = slope_scale_arr

            if 'aspect_slope_scale_arr' in cdata_db:
                aspect_slope_scale_arr = (
                    cdata_db['aspect_slope_scale_arr'][...])

                assert aspect_slope_scale_arr.ndim == 1

                aux_vars_dict[cat]['aspect_slope_scale'] = (
                    aspect_slope_scale_arr)

    plot_min_max_lims = (min_row - 1, max_row + 1, min_col - 1, max_col + 1)

    aux_prms_labs = list(aux_vars_dict[cat].keys())
    plot_aux_vars_2d(
        aux_vars_dict,
        out_dir,
        aux_prms_labs,
        shape,
        rows_dict,
        cols_dict,
        plot_min_max_lims,
        lumped_prms_flag)

    plot_cats_best_prms_2d_kf(
        kfolds,
        kf_prms_dict,
        out_dir,
        prms_labs,
        shape,
        rows_dict,
        cols_dict,
        plot_min_max_lims,
        lumped_prms_flag,
        bds_dict)

    return


def plot_aux_vars_2d(
        aux_vars_dict,
        out_dir,
        aux_prms_labs,
        shape,
        rows_dict,
        cols_dict,
        plot_min_max_lims,
        lumped_prms_flag):

    loc_rows = 1
    loc_cols = 1

    sca_fac = 3
    loc_rows *= sca_fac
    loc_cols *= sca_fac

    legend_rows = 1
    legend_cols = loc_cols

    plot_shape = (loc_rows + legend_rows, loc_cols)

    for prm in aux_prms_labs:
        curr_row = 0
        curr_col = 0
        ax = plt.subplot2grid(
            plot_shape,
            loc=(curr_row, curr_col),
            rowspan=sca_fac,
            colspan=sca_fac)

        plot_grid = np.full(shape, np.nan)
        for cat in aux_vars_dict:
            cell_prms = aux_vars_dict[cat][prm]
            plot_grid[rows_dict[cat], cols_dict[cat]] = cell_prms

            if lumped_prms_flag:
                val = str(cell_prms[0])
                bf_pt, _af_pt = val.split('.')
                af_pt = _af_pt[:(5 - len(bf_pt))]
                val = f'{bf_pt}.{af_pt}'
                ax.text(
                    cols_dict[cat][0],
                    rows_dict[cat][0],
                    f'{cat}\n({val})'.rstrip('0'),
                    va='center',
                    ha='center', zorder=2,
                    rotation=45)
            else:
                ax.text(
                    int(cols_dict[cat].mean()),
                    int(rows_dict[cat].mean()),
                    f'{cat}',
                    va='center',
                    ha='center', zorder=2)

            ps = ax.imshow(
                plot_grid,
                origin='lower',
                cmap=plt.get_cmap('gist_rainbow'),
                zorder=1)

            ax.set_ylim(plot_min_max_lims[0], plot_min_max_lims[1])
            ax.set_xlim(plot_min_max_lims[2], plot_min_max_lims[3])

            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_axis_off()

            curr_col += sca_fac
            if curr_col >= loc_cols:
                curr_col = 0
                curr_row += sca_fac

        cb_ax = plt.subplot2grid(
            plot_shape,
            loc=(plot_shape[0] - 1, 0),
            rowspan=1,
            colspan=legend_cols)

        cb_ax.set_axis_off()
        cb = plt.colorbar(
            ps, ax=cb_ax, fraction=0.4, aspect=20, orientation='horizontal')

        cb.set_label(prm)
        cb.ax.set_xticklabels(cb.ax.get_xticklabels(), rotation=90)

        fig = plt.gcf()

        fig.set_size_inches(
            (plot_shape[1] * 1.2) + 1, (plot_shape[0] * 1.2) + 1)

        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f'{prm}.png'), bbox_inches='tight')
        plt.close()
    return


def plot_cats_best_prms_2d_kf(
        kfolds,
        kf_prms_dict,
        out_dir,
        prms_labs,
        shape,
        rows_dict,
        cols_dict,
        plot_min_max_lims,
        lumped_prms_flag,
        bds_dict):

    loc_rows = max(1, int(0.25 * kfolds))
    loc_cols = max(1, int(np.ceil(kfolds / loc_rows)))

    sca_fac = 3
    loc_rows *= sca_fac
    loc_cols *= sca_fac

    legend_rows = 1
    legend_cols = loc_cols

    plot_shape = (loc_rows + legend_rows, loc_cols)

    for p_i, prm in enumerate(prms_labs):
        curr_row = 0
        curr_col = 0
        for kf_i in kf_prms_dict:
            ax = plt.subplot2grid(
                plot_shape,
                loc=(curr_row, curr_col),
                rowspan=sca_fac,
                colspan=sca_fac)

            plot_grid = np.full(shape, np.nan)
            min_bd, max_bd = bds_dict[prm + '_bds']
            for cat in kf_prms_dict[kf_i]:
                cell_prms = kf_prms_dict[kf_i][cat][:, p_i]
                plot_grid[rows_dict[cat], cols_dict[cat]] = cell_prms

                if lumped_prms_flag:
                    val = '%0.16f' % cell_prms[0]
                    bf_pt, af_pt = val.split('.')
                    af_pt = af_pt[:(5 - len(bf_pt))]
                    val = f'{bf_pt}.{af_pt}'
                    ax.text(
                        cols_dict[cat][0],
                        rows_dict[cat][0],
                        f'{cat}\n({val})'.rstrip('0'),
                        va='center',
                        ha='center', zorder=2,
                        rotation=45)
                else:
                    ax.text(
                        int(cols_dict[cat].mean()),
                        int(rows_dict[cat].mean()),
                        f'{cat}',
                        va='center',
                        ha='center', zorder=2)

            ps = ax.imshow(
                plot_grid,
                origin='lower',
                cmap=plt.get_cmap('gist_rainbow'),
                vmin=min_bd,
                vmax=max_bd,
                zorder=1)

            ax.set_ylim(plot_min_max_lims[0], plot_min_max_lims[1])
            ax.set_xlim(plot_min_max_lims[2], plot_min_max_lims[3])

            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_title(f'kfold no: {kf_i}')
            ax.set_axis_off()

            curr_col += sca_fac
            if curr_col >= loc_cols:
                curr_col = 0
                curr_row += sca_fac

        cb_ax = plt.subplot2grid(
            plot_shape,
            loc=(plot_shape[0] - 1, 0),
            rowspan=1,
            colspan=legend_cols)

        cb_ax.set_axis_off()
        cb = plt.colorbar(
            ps, ax=cb_ax, fraction=0.4, aspect=20, orientation='horizontal')
        cb.set_label(prm)

        fig = plt.gcf()
        fig.set_size_inches(
            (plot_shape[1] * 1.2) + 1, (plot_shape[0] * 1.2) + 1)

        plt.tight_layout()

        plt.savefig(os.path.join(out_dir, f'{prm}.png'), bbox_inches='tight')
        plt.close()
    return


def plot_cat_best_prms_1d(cat_db):

    '''Plot calibrated parameters for all kfolds.'''

    with h5py.File(cat_db, 'r') as db:
        cat = db.attrs['cat']
        kfolds = db['data'].attrs['kfolds']
        bds_arr = db['cdata/bds_arr'][...]
        best_prms_labs = db['cdata/use_prms_labs'][...]

        cv_flag = db['data'].attrs['cv_flag']
        if cv_flag:
            kfolds = 1

        out_dir = db['data'].attrs['main']

        out_dir = os.path.join(out_dir, r'05_kfolds_perf')
        if not os.path.exists(out_dir):
            try:
                os.mkdir(out_dir)

            except FileExistsError:
                pass

        best_prms_list = []
        for i in range(1, kfolds + 1):
            best_prms_list.append(db[f'calib/kf_{i:02d}/opt_prms'][...])

    plt.figure(figsize=(max(20, best_prms_list[0].shape[0]), 12))

    tick_font_size = 10
    best_params_arr = np.array(best_prms_list)
    norm_pop = best_params_arr.copy()

    stats_cols = ['min', 'max', 'mean', 'stdev', 'min_bd', 'max_bd']
    n_stats_cols = len(stats_cols)

    best_params_arr = (
        (best_params_arr * (bds_arr[:, 1] - bds_arr[:, 0])) +
        bds_arr[:, 0])

    n_params = bds_arr.shape[0]
    curr_min = best_params_arr.min(axis=0)
    curr_max = best_params_arr.max(axis=0)
    curr_mean = best_params_arr.mean(axis=0)
    curr_stdev = best_params_arr.std(axis=0)
    min_opt_bounds = bds_arr.min(axis=1)
    max_opt_bounds = bds_arr.max(axis=1)

    xx, yy = np.meshgrid(
        np.arange(-0.5, n_params, 1), np.arange(-0.5, n_stats_cols, 1))

    stats_arr = np.vstack([
        curr_min,
        curr_max,
        curr_mean,
        curr_stdev,
        min_opt_bounds,
        max_opt_bounds])

    stats_ax = plt.subplot2grid((4, 1), (0, 0), rowspan=1, colspan=1)
    stats_ax.pcolormesh(
        xx,
        yy,
        stats_arr,
        cmap=plt.get_cmap('Blues'),
        vmin=0.0,
        vmax=1e30)

    stats_xx, stats_yy = np.meshgrid(
        np.arange(n_stats_cols), np.arange(n_params))

    stats_xx = stats_xx.ravel()
    stats_yy = stats_yy.ravel()

    [stats_ax.text(
        stats_yy[i],
        stats_xx[i],
        (f'{stats_arr[stats_xx[i], stats_yy[i]]:0.4f}').rstrip('0'),
        va='center',
        ha='center')
     for i in range(int(n_stats_cols * n_params))]

    stats_ax.set_xticks(list(range(0, n_params)))
    stats_ax.set_xticklabels(best_prms_labs)
    stats_ax.set_xlim(-0.5, n_params - 0.5)
    stats_ax.set_yticks(list(range(0, n_stats_cols)))
    stats_ax.set_ylim(-0.5, n_stats_cols - 0.5)
    stats_ax.set_yticklabels(stats_cols)

    stats_ax.spines['left'].set_position(('outward', 10))
    stats_ax.spines['right'].set_position(('outward', 10))
    stats_ax.spines['top'].set_position(('outward', 10))
    stats_ax.spines['bottom'].set_visible(False)

    stats_ax.set_ylabel('Statistics')

    stats_ax.tick_params(
        labelleft=True, labelbottom=False, labeltop=True, labelright=True)

    stats_ax.xaxis.set_ticks_position('top')
    stats_ax.yaxis.set_ticks_position('both')

    for tick in stats_ax.get_xticklabels():
        tick.set_rotation(60)

    params_ax = plt.subplot2grid(
        (4, 1),
        (1, 0),
        rowspan=3,
        colspan=1,
        sharex=stats_ax)

    plot_range = list(range(0, bds_arr.shape[0]))
    plt_texts = []
    for i in range(kfolds):
        params_ax.plot(
            plot_range, norm_pop[i], alpha=0.85, label=f'Fold no: {i + 1}')

        if kfolds > 6:
            continue

        for j in range(bds_arr.shape[0]):
            text = params_ax.text(
                plot_range[j],
                norm_pop[i, j],
                (f'{best_params_arr[i, j]:0.4f}').rstrip('0'),
                va='top',
                ha='left')
            plt_texts.append(text)

    if kfolds <= 6:
        adjust_text(plt_texts, only_move={'points': 'y', 'text': 'y'})

    params_ax.set_ylim(0., 1.)
    params_ax.set_xticks(list(range(best_params_arr.shape[1])))
    params_ax.set_xticklabels(best_prms_labs)
    params_ax.set_xlim(-0.5, n_params - 0.5)
    params_ax.set_ylabel('Normalized value')
    params_ax.grid()
    params_ax.legend(framealpha=0.5)

    params_ax.tick_params(
        labelleft=True,
        labelbottom=True,
        labeltop=False,
        labelright=True)
    params_ax.yaxis.set_ticks_position('both')

    for tick in params_ax.get_xticklabels():
        tick.set_rotation(60)

    title_str = 'Comparison of kfold best parameters'
    plt.suptitle(title_str, size=tick_font_size + 10)
    plt.subplots_adjust(hspace=0.15)

    out_params_fig_loc = os.path.join(
        out_dir, f'kfolds_prms_compare_{cat}.png')

    plt.savefig(out_params_fig_loc, bbox='tight_layout')

    plt.close()
    return


def plot_cat_prm_vecs(cat_db):

    '''Plot all kfold best parameter vectors for a given catchment hdf5 file.

    Along with some given statistics.
    '''

    with h5py.File(cat_db, 'r') as db:
        out_dir = db['data'].attrs['main']
        out_dir = os.path.join(out_dir, r'06_prm_vecs')

        if not os.path.exists(out_dir):
            try:
                os.mkdir(out_dir)

            except FileExistsError:
                pass

        kfolds = db['data'].attrs['kfolds']
        cat = db.attrs['cat']
        opt_schm = db['cdata/opt_schm_vars_dict'].attrs['opt_schm']

        calib_db = db['calib']

        bounds_arr = db['cdata/bds_arr'][...]
        prm_syms = db['cdata/use_prms_labs'][...]

        for i in range(1, kfolds + 1):
            kf_str = f'kf_{i:02d}'
            prm_vecs = calib_db[kf_str + '/prm_vecs'][...]
            cobj_vals = calib_db[kf_str + '/curr_obj_vals'][...]
            pobj_vals = calib_db[kf_str + '/pre_obj_vals'][...]

            plot_cat_prm_vecs_kf(
                i,
                cat,
                out_dir,
                bounds_arr,
                prm_vecs,
                cobj_vals,
                pobj_vals,
                prm_syms,
                opt_schm)
    return


def plot_cat_prm_vecs_kf(
        kf_i,
        cat,
        out_dir,
        bounds_arr,
        prm_vecs,
        cobj_vals,
        pobj_vals,
        prm_syms,
        opt_schm):

    '''Plot given kfold best parameter vectors along with some statisitics.

    This function is supposed to be called by plot_cat_prm_vecs_kf.
    '''

    plt.figure(figsize=(max(35, bounds_arr.shape[0]), 13))
    tick_font_size = 10

    stats_cols = ['min', 'max', 'mean', 'stdev', 'min_bd', 'max_bd']
    n_stats_cols = len(stats_cols)

    norm_prm_vecs = prm_vecs.copy()
    for i in range(prm_vecs.shape[0]):
        for j in range(prm_vecs.shape[1]):
            prm_vecs[i, j] = (
                (prm_vecs[i, j] * (bounds_arr[j, 1] - bounds_arr[j, 0])) +
                bounds_arr[j, 0])

            if not np.isfinite(prm_vecs[i, j]):
                prm_vecs[i, j] = bounds_arr[j, 0]

    n_params = bounds_arr.shape[0]

    curr_min = prm_vecs.min(axis=0)
    curr_max = prm_vecs.max(axis=0)
    curr_mean = prm_vecs.mean(axis=0)
    curr_stdev = prm_vecs.std(axis=0)
    min_opt_bounds = bounds_arr.min(axis=1)
    max_opt_bounds = bounds_arr.max(axis=1)

    xx_crds, yy_crds = np.meshgrid(
        np.arange(-0.5, n_params, 1), np.arange(-0.5, n_stats_cols, 1))

    stats_arr = np.vstack([
        curr_min,
        curr_max,
        curr_mean,
        curr_stdev,
        min_opt_bounds,
        max_opt_bounds
        ])

    stats_ax = plt.subplot2grid((4, 1), (0, 0), rowspan=1, colspan=1)
    stats_ax.pcolormesh(
        xx_crds,
        yy_crds,
        stats_arr,
        cmap=plt.get_cmap('Blues'),
        vmin=0.0,
        vmax=1e30)

    stats_xx, stats_yy = np.meshgrid(
        np.arange(n_stats_cols), np.arange(n_params))

    stats_xx = stats_xx.ravel()
    stats_yy = stats_yy.ravel()

    [stats_ax.text(
        stats_yy[i],
        stats_xx[i],
        (f'{stats_arr[stats_xx[i], stats_yy[i]]:0.4f}').rstrip('0'),
        va='center',
        ha='center')

     for i in range(int(n_stats_cols * n_params))]

    stats_ax.set_xticks(list(range(0, n_params)))
    stats_ax.set_xticklabels(prm_syms)
    stats_ax.set_xlim(-0.5, n_params - 0.5)
    stats_ax.set_yticks(list(range(0, n_stats_cols)))
    stats_ax.set_ylim(-0.5, n_stats_cols - 0.5)
    stats_ax.set_yticklabels(stats_cols)

    stats_ax.spines['left'].set_position(('outward', 10))
    stats_ax.spines['right'].set_position(('outward', 10))
    stats_ax.spines['top'].set_position(('outward', 10))
    stats_ax.spines['bottom'].set_visible(False)

    stats_ax.set_ylabel('Statistics')

    stats_ax.tick_params(
        labelleft=True,
        labelbottom=False,
        labeltop=True,
        labelright=True)

    stats_ax.xaxis.set_ticks_position('top')
    stats_ax.yaxis.set_ticks_position('both')

    for tick in stats_ax.get_xticklabels():
        tick.set_rotation(60)

    params_ax = plt.subplot2grid(
        (4, 1),
        (1, 0),
        rowspan=3,
        colspan=1,
        sharex=stats_ax)

    plot_range = list(range(n_params))
    for i in range(prm_vecs.shape[0]):
        params_ax.plot(plot_range, norm_prm_vecs[i], alpha=0.1, color='k')

    params_ax.set_ylim(0., 1.)
    params_ax.set_xticks(plot_range)
    params_ax.set_xticklabels(prm_syms)
    params_ax.set_xlim(-0.5, n_params - 0.5)
    params_ax.set_ylabel('Normalized value')
    params_ax.grid()

    params_ax.tick_params(
        labelleft=True, labelbottom=True, labeltop=False, labelright=True)

    params_ax.yaxis.set_ticks_position('both')

    for tick in params_ax.get_xticklabels():
        tick.set_rotation(60)

    title_str = (
        f'Distributed HBV parameters - '
        f'{opt_schm} final prm_vecs (n={prm_vecs.shape[0]})')

    plt.suptitle(title_str, size=tick_font_size + 10)
    plt.subplots_adjust(hspace=0.15)

    plt.savefig(
        str(Path(out_dir, f'hbv_prm_vecs_{cat}_kf_{kf_i:02d}.png')),
        bbox_inches='tight')

    plt.close()

    # curr_obj_vals
    __, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), sharex=True)

    n_vals = float(cobj_vals.shape[0])

    probs = 1 - (np.arange(1.0, n_vals + 1) / (n_vals + 1))

    ax1.set_title(
        f'Final objective function distribution\n'
        f'Min. obj.: {cobj_vals.min():0.4f}, '
        f'max. obj.: {cobj_vals.max():0.4f}')

    ax1.plot(np.sort(cobj_vals), probs, marker='o', alpha=0.8)
    ax1.set_ylabel('Non-exceedence Probability (-)')
    ax1.set_xlim(ax1.set_xlim()[1], ax1.set_xlim()[0])
    ax1.grid()

    ax2.hist(cobj_vals, bins=20)
    ax2.set_xlabel('Obj. ftn. values (-)')
    ax2.set_ylabel('Frequency (-)')
    plt.savefig(
        str(Path(out_dir, f'hbv_cobj_cdf_{cat}_kf_{kf_i:02d}.png')),
        bbox_inches='tight')

    # pre_obj_vals
    __, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), sharex=True)

    n_vals = float(pobj_vals.shape[0])

    probs = 1 - (np.arange(1.0, n_vals + 1) / (n_vals + 1))

    ax1.set_title(
        f'2nd last objective function distribution\n'
        f'Min. obj.: {pobj_vals.min():0.4f}, '
        f'max. obj.: {pobj_vals.max():0.4f}')

    ax1.plot(np.sort(pobj_vals), probs, marker='o', alpha=0.8)
    ax1.set_ylabel('Non-exceedence Probability (-)')
    ax1.set_xlim(ax1.set_xlim()[1], ax1.set_xlim()[0])
    ax1.grid()

    ax2.hist(pobj_vals, bins=20)
    ax2.set_xlabel('Obj. ftn. values (-)')
    ax2.set_ylabel('Frequency (-)')
    plt.savefig(
        str(Path(out_dir, f'hbv_pobj_cdf_{cat}_kf_{kf_i:02d}.png')),
        bbox_inches='tight')
    plt.close('all')
    return


def plot_cat_prm_vecs_evo(plot_args):

    (cat_db,
     save_obj_flag,
     save_png_flag,
     save_gif_flag,
     anim_secs) = plot_args

    '''Plot parameter vector evolution for all kfolds.'''

    assert save_obj_flag or save_png_flag or save_gif_flag

    with h5py.File(cat_db, 'r') as db:
        out_dir = db['data'].attrs['main']
        out_dir = os.path.join(out_dir, r'10_prm_vecs_evo')

        if not os.path.exists(out_dir):
            try:
                os.mkdir(out_dir)

            except FileExistsError:
                pass

        kfolds = db['data'].attrs['kfolds']
        cat = db.attrs['cat']

        calib_db = db['calib']

        prm_syms = db['cdata/use_prms_labs'][...]

        for kf_i in range(1, kfolds + 1):

            kf_str = f'kf_{kf_i:02d}'

            if save_obj_flag:
                gobj_vals = np.ma.masked_invalid(
                    calib_db[kf_str + '/gobj_vals'][...])

                iobj_vals = np.ma.masked_invalid(
                    calib_db[kf_str + '/iobj_vals'][...])

                plot_cat_obj_evo_kf(
                    cat,
                    kf_i,
                    out_dir,
                    gobj_vals,
                    iobj_vals)

            if save_png_flag or save_gif_flag:
                iter_prm_vecs = calib_db[kf_str + '/iter_prm_vecs'][...]  #

                plot_cat_prm_vecs_evo_kf(
                    iter_prm_vecs,
                    cat,
                    kf_i,
                    out_dir,
                    prm_syms,
                    save_png_flag,
                    save_gif_flag,
                    anim_secs)

    return


def plot_cat_obj_evo_kf(
        cat,
        kf_i,
        out_dir,
        gobj_vals,
        iobj_vals):

    '''Plot the objective function evolution for a given kfold and catchment.
    '''

    min_obj_lim = max(0, np.nanmin(gobj_vals))
    max_obj_lim = min(2, np.nanmax(gobj_vals))

    plot_lim_pad = 0.025 * (max_obj_lim - min_obj_lim)

    plt.figure(figsize=(20, 10))

    x_rng = np.arange(gobj_vals.shape[0])

    min_gobj = np.nanmin(gobj_vals)

    plt.scatter(
        x_rng, iobj_vals, marker='o', label='iter', alpha=0.7, color='r')

    plt.plot(x_rng, gobj_vals, label='global', alpha=0.8, color='k')

    plt.xlabel('Iteration number (-)')
    plt.ylabel('Objective function value(-)')

    plt.ylim(min_obj_lim - plot_lim_pad, max_obj_lim + plot_lim_pad)

    plt.grid()
    plt.legend()

    above_limit_ct = int((iobj_vals > max_obj_lim + plot_lim_pad).sum())
    tot_obj_vals = iobj_vals.shape[0]

    plt.title(
        f'Objective function value evolution for the catchment {cat} '
        f'and kfold no. {kf_i:02d}\n Min. obj. val: {min_gobj:0.5f}\n'
        f'{above_limit_ct} out of {tot_obj_vals} above the y-limit '
        f'of this figure')

    out_fig_name = f'obj_val_evo_{cat}_kf_{kf_i:02d}.png'

    plt.savefig(str(Path(out_dir, out_fig_name)), bbox_inches='tight')
    plt.close()
    return


def plot_cat_prm_vecs_evo_kf(
        iter_prm_vecs,
        cat,
        kf_i,
        out_dir,
        prm_syms,
        save_png_flag,
        save_gif_flag,
        anim_secs):

    '''Plot parameter vector evolution for a given kfolds.'''

    prm_cols = 7
    prm_rows = 7

    tot_sps_per_fig = prm_cols * prm_rows

    tot_iters = iter_prm_vecs.shape[0]

    min_lim = 0  # np.nanmin(iter_prm_vecs)
    max_lim = 1  # np.nanmax(iter_prm_vecs)

    tot_figs = int(np.ceil(tot_iters / tot_sps_per_fig))

    prm_iter_ct = 0
    stop_plotting = False

    if save_gif_flag:
        opt_vecs_fig_arrs_list = []

        blines = 10  # buffer lines on all sides for gifs

    for fig_no in range(tot_figs):
        if stop_plotting:
            break

        fig = plt.figure(figsize=(19, 9), dpi=200)
        axes = GridSpec(prm_rows, prm_cols)

        for i in range(prm_rows):
            for j in range(prm_cols):

                ax = plt.subplot(axes[i, j])

                for k in range(iter_prm_vecs.shape[1]):
                    ax.plot(
                        iter_prm_vecs[prm_iter_ct, k],
                        alpha=0.005,
                        color='k')

                ax.text(
                    0.95,
                    0.95,
                    f'({prm_iter_ct})',
                    horizontalalignment='right',
                    verticalalignment='top',
                    transform=ax.transAxes)

                ax.set_xticklabels([])
                ax.set_yticklabels([])

                ax.set_ylim(min_lim, max_lim)

                prm_iter_ct += 1

                if prm_iter_ct >= tot_iters:
                    stop_plotting = True
                    break

            if stop_plotting:
                break

        plt.suptitle(
            f'Parameter vector evolution per iteration\n'
            f'Vectors per iteration: {iter_prm_vecs.shape[1]}')

        if save_png_flag:
            out_fig_name = (
                f'hbv_prm_vecs_evo_{cat}_kf_{kf_i:02d}_fig_{fig_no:02d}.png')

            plt.savefig(str(Path(out_dir, out_fig_name)), bbox_inches='tight')

        if save_gif_flag:
            fig.canvas.draw()  # calling draw is important here

            fig_arr = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            fig_arr = fig_arr.reshape(
                fig.canvas.get_width_height()[::-1] + (3,))

            mins_arr = fig_arr[:, :, :].min(axis=2)

            not_white_idxs = np.where(mins_arr < 255)

            r_top = not_white_idxs[0].min()
            r_bot = not_white_idxs[0].max()

            c_left = not_white_idxs[1].min()
            c_rght = not_white_idxs[1].max()

            r_top -= min(blines, r_top)
            r_bot += min(blines, mins_arr.shape[0] - r_bot)

            c_left -= min(blines, c_left)
            c_rght += min(blines, mins_arr.shape[1] - c_rght)

            fig_arr = fig_arr[r_top:r_bot + 1, c_left:c_rght + 1, :]

            opt_vecs_fig_arrs_list.append(fig_arr)

        plt.close()

    if save_gif_flag:
        out_fig_name = (
            f'hbv_prm_vecs_evo_{cat}_kf_{kf_i:02d}_anim.png')

        write_apng(
            str(Path(out_dir, out_fig_name)),
            opt_vecs_fig_arrs_list,
            delay=1000,
            use_palette=True,
            num_plays=1)

        del opt_vecs_fig_arrs_list

    ##########################################################################
    if save_gif_flag:
        chull_fig_arrs_list = []

    chull_min = 0
    chull_max = 1
    n_dims = iter_prm_vecs.shape[-1]

    for opt_iter in range(iter_prm_vecs.shape[0]):

        fig = plt.figure(figsize=(15, 15), dpi=200)
        grid_axes = GridSpec(n_dims, n_dims)
        for i in range(n_dims):
            for j in range(n_dims):
                if i >= j:
                    continue

                ax = plt.subplot(grid_axes[i, j])

                ax.set_aspect('equal', 'datalim')

                ax.set_xlim(chull_min, chull_max)
                ax.set_ylim(chull_min, chull_max)

                ax.scatter(
                    iter_prm_vecs[opt_iter, :, i],
                    iter_prm_vecs[opt_iter, :, j],
                    s=2,
                    color='k',
                    alpha=0.05)

                ax.set_xticks([])
                ax.set_xticklabels([])

                ax.set_yticks([])
                ax.set_yticklabels([])

                ax.text(
                    0.95,
                    0.95,
                    f'({prm_syms[i]}, {prm_syms[j]})',
                    horizontalalignment='right',
                    verticalalignment='top',
                    transform=ax.transAxes)

        plt.suptitle(
            f'Convex hull of {n_dims}D in 2D\n'
            f'Total points: {iter_prm_vecs.shape[1]}\n'
            f'Iteration no.: {opt_iter + 1} out of {iter_prm_vecs.shape[0]}',
            x=0.4,
            y=0.5,
            va='bottom',
            ha='right')

        if save_png_flag:
            out_fig_name = (
                f'hbv_prm_vecs_chull_{cat}_kf_{kf_i:02d}_iter_'
                f'{opt_iter:02d}.png')

            plt.savefig(
                str(Path(out_dir, out_fig_name)), bbox_inches='tight')

        if save_gif_flag:
            fig.canvas.draw()  # calling draw is important here

            fig_arr = np.frombuffer(
                fig.canvas.tostring_rgb(), dtype=np.uint8)

            fig_arr = fig_arr.reshape(
                fig.canvas.get_width_height()[::-1] + (3,))

            mins_arr = fig_arr[:, :, :].min(axis=2)

            not_white_idxs = np.where(mins_arr < 255)

            r_top = not_white_idxs[0].min()
            r_bot = not_white_idxs[0].max()

            c_left = not_white_idxs[1].min()
            c_rght = not_white_idxs[1].max()

            r_top -= min(blines, r_top)
            r_bot += min(blines, mins_arr.shape[0] - r_bot)

            c_left -= min(blines, c_left)
            c_rght += min(blines, mins_arr.shape[1] - c_rght)

            fig_arr = fig_arr[r_top:r_bot + 1, c_left:c_rght + 1, :]

            chull_fig_arrs_list.append(fig_arr)

        plt.close('all')

    if save_gif_flag:
        out_fig_name = (
            f'hbv_prm_vecs_chull_{cat}_kf_{kf_i:02d}_anim.png')

        write_apng(
            str(Path(out_dir, out_fig_name)),
            chull_fig_arrs_list,
            delay=(1000 * anim_secs) / iter_prm_vecs.shape[0],
            use_palette=True,
            num_plays=1)

        del chull_fig_arrs_list
    return
