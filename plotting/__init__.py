import os
from glob import glob
import pandas as pd
from pathos.multiprocessing import ProcessPool

from .sims import plot_hbv, plot_pop, _plot_hbv_kf
from .k_folds import (
    plot_cat_kfold_effs,
    _kfold_best_prms,
    plot_kfolds_best_hbv_prms_2d)


def plot_vars(
        dbs_dir,
        water_bal_step_size,
        plot_simple_opt_flag=False,
        plot_wat_bal_flag=False,
        n_cpus=1):
    '''Plot the optimization results
    '''

    cats_dbs = glob(os.path.join(dbs_dir, 'cat_*.bak'))

    assert cats_dbs

    n_cats = len(cats_dbs)
    n_cpus = min(n_cats, n_cpus)

    const_args = (water_bal_step_size, plot_simple_opt_flag, plot_wat_bal_flag)

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


def plot_kfold_effs(dbs_dir, hgs_db_path, compare_ann_cyc_flag, n_cpus):
    '''Plot the k-fold efficiency results
    '''

    cats_dbs = glob(os.path.join(dbs_dir, 'cat_*.bak'))

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

    cats_dbs = glob(os.path.join(dbs_dir, 'cat_*.bak'))

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


def plot_pops(dbs_dir, n_cpus):
    '''Plot the population
    '''
    cats_dbs = glob(os.path.join(dbs_dir, 'cat_*.bak'))

    assert cats_dbs

    n_cats = len(cats_dbs)
    n_cpus = min(n_cats, n_cpus)

    n_cpus = min(n_cats, n_cpus)

    opt_res_gen = (cat_db for cat_db in cats_dbs)

    if n_cpus > 1:
        mp_pool = ProcessPool(n_cpus)
        mp_pool.restart(True)
        try:
            print(list(mp_pool.uimap(plot_pop, opt_res_gen)))
            mp_pool.clear()
        except Exception as msg:
            mp_pool.close()
            mp_pool.join()
            print('Error in plot_pops:', msg)
    else:
        for opt_res in opt_res_gen:
            plot_pop(opt_res)
    return

