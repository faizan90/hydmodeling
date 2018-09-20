import os
from glob import glob

import h5py
import pandas as pd
from pathos.multiprocessing import ProcessPool

from .sims import plot_hbv, \
    _plot_prm_vecs, \
    _plot_hbv_kf, \
    plot_error, \
    plot_hull

from .k_folds import (
    plot_cat_kfold_effs,
    _kfold_best_prms,
    plot_kfolds_best_hbv_prms_2d,
    plot_ann_cycs_fdcs_comp,
    plot_prm_trans)


def plot_kfold_effs(dbs_dir, hgs_db_path, compare_ann_cyc_flag, n_cpus):
    '''Plot the k-fold efficiency results
    '''

    cats_dbs = glob(os.path.join(dbs_dir, 'cat_*.hdf5'))

    assert cats_dbs

    n_cats = len(cats_dbs)
    n_cpus = min(n_cats, n_cpus)

    n_cpus = min(n_cats, n_cpus)

    const_args = (compare_ann_cyc_flag, hgs_db_path)
    cats_paths_gen = ((cat_db, const_args) for cat_db in cats_dbs)

    if n_cpus > 1:
        mp_pool = ProcessPool(n_cpus)
        mp_pool.restart(True)
        try:
            print(list(mp_pool.uimap(plot_cat_kfold_effs, cats_paths_gen)))
            mp_pool.clear()
        except Exception as msg:
            mp_pool.close()
            mp_pool.join()
            print('Error in plot_k_fold_effs:', msg)
    else:
        for cat_paths in cats_paths_gen:
            plot_cat_kfold_effs(cat_paths)
    return


def plot_kfolds_best_prms(dbs_dir, n_cpus):

    cats_dbs = glob(os.path.join(dbs_dir, 'cat_*.hdf5'))

    assert cats_dbs
    n_cats = len(cats_dbs)
    n_cpus = min(n_cats, n_cpus)

    cats_paths_gen = (cat_db for cat_db in cats_dbs)

    if n_cpus > 1:
        mp_pool = ProcessPool(n_cpus)
        mp_pool.restart(True)
        try:
            print(list(mp_pool.uimap(_kfold_best_prms, cats_paths_gen)))
            mp_pool.clear()
        except Exception as msg:
            mp_pool.close()
            mp_pool.join()
            print('Error in plot_kfolds_best_prms:', msg)
    else:
        for cat_paths in cats_paths_gen:
            _kfold_best_prms(cat_paths)
    return


def plot_prm_vecs(dbs_dir, n_cpus):
    '''Plot the population
    '''
    cats_dbs = glob(os.path.join(dbs_dir, 'cat_*.hdf5'))

    assert cats_dbs

    n_cats = len(cats_dbs)
    n_cpus = min(n_cats, n_cpus)

    n_cpus = min(n_cats, n_cpus)

    opt_res_gen = (cat_db for cat_db in cats_dbs)

    if n_cpus > 1:
        mp_pool = ProcessPool(n_cpus)
        mp_pool.restart(True)
        try:
            print(list(mp_pool.uimap(_plot_prm_vecs, opt_res_gen)))
            mp_pool.clear()
        except Exception as msg:
            mp_pool.close()
            mp_pool.join()
            print('Error in plot_pops:', msg)
    else:
        for opt_res in opt_res_gen:
            _plot_prm_vecs(opt_res)
    return


def plot_vars(
        dbs_dir,
        valid_flag,
        water_bal_step_size,
        plot_simple_opt_flag=False,
        plot_wat_bal_flag=False,
        n_cpus=1):
    '''Plot the optimization results
    '''

    cats_dbs = glob(os.path.join(dbs_dir, 'cat_*.hdf5'))

    assert cats_dbs

    n_cats = len(cats_dbs)
    n_cpus = min(n_cats, n_cpus)

    const_args = (water_bal_step_size, plot_simple_opt_flag, plot_wat_bal_flag, valid_flag)

    plot_gen = ((cat_db, const_args) for cat_db in cats_dbs)

    if n_cpus > 1:
        mp_pool = ProcessPool(n_cpus)
        mp_pool.restart(True)
        try:
            print(list(mp_pool.uimap(plot_hbv, plot_gen)))
            mp_pool.clear()
        except Exception as msg:
            mp_pool.close()
            mp_pool.join()
            print('Error in plot_vars:', msg)
    else:
        for plot_args in plot_gen:
            plot_hbv(plot_args)
    return


def plot_prm_trans_perfs(dbs_dir, n_cpus=1):
    '''Plot the optimization results
    '''

    cats_dbs = glob(os.path.join(dbs_dir, 'cat_*.hdf5'))

    assert cats_dbs

    n_cats = len(cats_dbs)
    n_cpus = min(n_cats, n_cpus)

    kf_prms_dict = {}
    cats_vars_dict = {}
    for cat_db in cats_dbs:
        with h5py.File(cat_db, 'r') as db:
            kfolds = db['data'].attrs['kfolds']
            cat = db.attrs['cat']

            f_var_infos = db['cdata/aux_var_infos'][...]
            prms_idxs = db['cdata/use_prms_idxs'][...]
            f_vars = db['cdata/aux_vars'][...]
            prms_flags = db['cdata/all_prms_flags'][...]
            bds_arr = db['cdata/bds_arr'][...]

            cat_vars_dict = {}
            cat_vars_dict['f_var_infos'] = f_var_infos
            cat_vars_dict['prms_idxs'] = prms_idxs
            cat_vars_dict['f_vars'] = f_vars
            cat_vars_dict['prms_flags'] = prms_flags
            cat_vars_dict['bds_arr'] = bds_arr

            cats_vars_dict[cat] = cat_vars_dict

            for i in range(1, kfolds + 1):
                kf_str = f'kf_{i:02d}'
                cd_db = db[f'calib/{kf_str}']

                opt_prms = cd_db['opt_prms'][...]

                if i not in kf_prms_dict:
                    kf_prms_dict[i] = {}

                kf_prms_dict[i][cat] = {}

                kf_prms_dict[i][cat]['opt_prms'] = opt_prms

    const_args = (kf_prms_dict, cats_vars_dict)
    plot_gen = ((cat_db, const_args) for cat_db in cats_dbs)

    if n_cpus > 1:
        mp_pool = ProcessPool(n_cpus)
        mp_pool.restart(True)
        try:
            print(list(mp_pool.uimap(plot_prm_trans, plot_gen)))
            mp_pool.clear()
        except Exception as msg:
            mp_pool.close()
            mp_pool.join()
            print('Error in plot_prm_trans:', msg)
    else:
        for plot_args in plot_gen:
            plot_prm_trans(plot_args)
    return

def plot_error_stats(dbs_dir,
                          valid_flag,
                          n_cpus=1):
    cats_dbs = glob(os.path.join(dbs_dir, 'cat_*.hdf5'))

    assert cats_dbs

    plot_gen = ((cat_db, valid_flag) for cat_db in cats_dbs)

    if n_cpus > 1:
        mp_pool = ProcessPool(n_cpus)
        mp_pool.restart(True)
        try:
            print(list(
                mp_pool.uimap(plot_prm_trans, plot_gen)))
            mp_pool.clear()
        except Exception as msg:
            mp_pool.close()
            mp_pool.join()
            print('Error in plot_prm_trans:', msg)
    else:
        for plot_args in plot_gen:
            plot_error(plot_args)

    return

def plot_conv_hull(dbs_dir,
                     valid_flag,
                     n_cpus=1):
    cats_dbs = glob(os.path.join(dbs_dir, 'cat_*.hdf5'))

    assert cats_dbs

    plot_gen = ((cat_db) for cat_db in cats_dbs)

    if n_cpus > 1:
        mp_pool = ProcessPool(n_cpus)
        mp_pool.restart(True)
        try:
            print(list(
                mp_pool.uimap(plot_prm_trans, plot_gen)))
            mp_pool.clear()
        except Exception as msg:
            mp_pool.close()
            mp_pool.join()
            print('Error in plot_prm_trans:', msg)
    else:
        for plot_args in plot_gen:
            plot_hull(plot_args)

    return