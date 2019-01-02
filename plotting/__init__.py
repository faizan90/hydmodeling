import os
from glob import glob

import h5py
import pandas as pd
from pathos.multiprocessing import ProcessPool

from .sims import plot_cat_hbv_sim
from .eval import plot_cat_qsims
from .prms import (
    plot_cat_prm_vecs,
    plot_cat_prm_vecs_evo,
    plot_cat_best_prms_1d,
    plot_cats_best_prms_2d,)
from .perfs import (
    plot_cat_kfold_effs,
    plot_cats_ann_cycs_fdcs_comp,
    plot_cat_prms_transfer_perfs,
    plot_cat_vars_errors)


def plot_cats_vars_errors(dbs_dir, err_var_labs, n_cpus):

    cats_dbs = glob(os.path.join(dbs_dir, 'cat_*.hdf5'))

    assert cats_dbs

    n_cats = len(cats_dbs)
    n_cpus = min(n_cats, n_cpus)

    n_cpus = min(n_cats, n_cpus)

    cats_paths_gen = ((cat_db, err_var_labs) for cat_db in cats_dbs)

    if (n_cpus > 1) and (n_cats > 1):
        mp_pool = ProcessPool(n_cpus)
        mp_pool.restart(True)

        try:
            print(list(mp_pool.uimap(plot_cat_vars_errors, cats_paths_gen)))
            mp_pool.clear()

        except Exception as msg:
            mp_pool.close()
            mp_pool.join()
            print('Error in plot_cat_vars_errors:', msg)

    else:
        for cat_paths in cats_paths_gen:
            plot_cat_vars_errors(cat_paths)

    return


def plot_cats_kfold_effs(dbs_dir, hgs_db_path, compare_ann_cyc_flag, n_cpus):

    '''Plot the k-fold efficiency results.'''

    cats_dbs = glob(os.path.join(dbs_dir, 'cat_*.hdf5'))

    assert cats_dbs

    n_cats = len(cats_dbs)
    n_cpus = min(n_cats, n_cpus)

    n_cpus = min(n_cats, n_cpus)

    const_args = (compare_ann_cyc_flag, hgs_db_path)
    cats_paths_gen = ((cat_db, const_args) for cat_db in cats_dbs)

    if (n_cpus > 1) and (n_cats > 1):
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


def plot_cats_best_prms_1d(dbs_dir, n_cpus):

    '''Plot every best kfold parameter set for all catchments.'''

    cats_dbs = glob(os.path.join(dbs_dir, 'cat_*.hdf5'))

    assert cats_dbs
    n_cats = len(cats_dbs)
    n_cpus = min(n_cats, n_cpus)

    cats_paths_gen = (cat_db for cat_db in cats_dbs)

    if (n_cpus > 1) and (n_cats > 1):
        mp_pool = ProcessPool(n_cpus)
        mp_pool.restart(True)

        try:
            print(list(mp_pool.uimap(plot_cat_best_prms_1d, cats_paths_gen)))
            mp_pool.clear()

        except Exception as msg:
            mp_pool.close()
            mp_pool.join()
            print('Error in plot_kfolds_best_prms:', msg)

    else:
        for cat_paths in cats_paths_gen:
            plot_cat_best_prms_1d(cat_paths)

    return


def plot_cats_prm_vecs(dbs_dir, n_cpus):

    '''Plot final parameter set from kfold for every catchments along with
    objective function value distribution.
    '''

    cats_dbs = glob(os.path.join(dbs_dir, 'cat_*.hdf5'))

    assert cats_dbs

    n_cats = len(cats_dbs)
    n_cpus = min(n_cats, n_cpus)

    n_cpus = min(n_cats, n_cpus)

    opt_res_gen = (cat_db for cat_db in cats_dbs)

    if (n_cpus > 1) and (n_cats > 1):
        mp_pool = ProcessPool(n_cpus)
        mp_pool.restart(True)

        try:
            print(list(mp_pool.uimap(plot_cat_prm_vecs, opt_res_gen)))
            mp_pool.clear()

        except Exception as msg:
            mp_pool.close()
            mp_pool.join()
            print('Error in plot_cat_prm_vecs:', msg)

    else:
        for opt_res in opt_res_gen:
            plot_cat_prm_vecs(opt_res)

    return


def plot_cats_prm_vecs_evo(
        dbs_dir,
        save_obj_flag,
        save_png_flag,
        save_gif_flag,
        anim_secs,
        n_cpus=1):

    '''Plot the evolution of parameter vectors and convex hull for every
    catchment for every kfold.
    '''

    cats_dbs = glob(os.path.join(dbs_dir, 'cat_*.hdf5'))

    assert cats_dbs

    n_cats = len(cats_dbs)
    n_cpus = min(n_cats, n_cpus)

    n_cpus = min(n_cats, n_cpus)

    opt_res_gen = (
        (cat_db, save_obj_flag, save_png_flag, save_gif_flag, anim_secs)
        for cat_db in cats_dbs)

    if (n_cpus > 1) and (n_cats > 1):
        mp_pool = ProcessPool(n_cpus)
        mp_pool.restart(True)

        try:
            print(list(mp_pool.uimap(plot_cat_prm_vecs_evo, opt_res_gen)))
            mp_pool.clear()

        except Exception as msg:
            mp_pool.close()
            mp_pool.join()
            print('Error in plot_cat_prm_vecs_evo:', msg)

    else:
        for opt_res in opt_res_gen:
            plot_cat_prm_vecs_evo(opt_res)

    return


def plot_cats_hbv_sim(
        dbs_dir,
        water_bal_step_size,
        full_flag=False,
        wat_bal_flag=False,
        n_cpus=1):

    '''Plot hbv simulations for every catchment for every kfold.'''

    cats_dbs = glob(os.path.join(dbs_dir, 'cat_*.hdf5'))

    assert cats_dbs

    n_cats = len(cats_dbs)
    n_cpus = min(n_cats, n_cpus)

    const_args = (water_bal_step_size, full_flag, wat_bal_flag)

    plot_gen = ((cat_db, const_args) for cat_db in cats_dbs)

    if (n_cpus > 1) and (n_cats > 1):
        mp_pool = ProcessPool(n_cpus)
        mp_pool.restart(True)

        try:
            print(list(mp_pool.uimap(plot_cat_hbv_sim, plot_gen)))
            mp_pool.clear()

        except Exception as msg:
            mp_pool.close()
            mp_pool.join()
            print('Error in plot_cat_hbv_sim:', msg)

    else:
        for plot_args in plot_gen:
            plot_cat_hbv_sim(plot_args)

    return


def plot_cats_qsims(dbs_dir, n_cpus=1):

    '''Plot discharge simulations for every catchment for every
    kfold using its prm_vecs.'''

    cats_dbs = glob(os.path.join(dbs_dir, 'cat_*.hdf5'))

    assert cats_dbs

    n_cats = len(cats_dbs)
    n_cpus = min(n_cats, n_cpus)

    plot_gen = (cat_db for cat_db in cats_dbs)

    if (n_cpus > 1) and (n_cats > 1):
        mp_pool = ProcessPool(n_cpus)
        mp_pool.restart(True)

        try:
            print(list(mp_pool.uimap(plot_cat_qsims, plot_gen)))
            mp_pool.clear()

        except Exception as msg:
            mp_pool.close()
            mp_pool.join()
            print('Error in plot_cat_rope_q_sims:', msg)

    else:
        for plot_args in plot_gen:
            plot_cat_qsims(plot_args)

    return


def plot_cats_prms_transfer_perfs(dbs_dir, n_cpus=1):

    '''Plot catchments performances' by using parameters from other
    catchment.
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

            cv_flag = db['data'].attrs['cv_flag']

            if cv_flag:
                print('plot_prm_trans_perfs not possible with cv_flag!')
                return

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

    if (n_cpus > 1) and (n_cats > 1):
        mp_pool = ProcessPool(n_cpus)
        mp_pool.restart(True)

        try:
            print(list(mp_pool.uimap(plot_cat_prms_transfer_perfs, plot_gen)))
            mp_pool.clear()

        except Exception as msg:
            mp_pool.close()
            mp_pool.join()
            print('Error in plot_cat_prms_transfer_perfs:', msg)

    else:
        for plot_args in plot_gen:
            plot_cat_prms_transfer_perfs(plot_args)

    return
