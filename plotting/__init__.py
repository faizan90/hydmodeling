import pickle

import pandas as pd
from pathos.multiprocessing import ProcessPool

from .sims import plot_hbv, plot_pop
from .k_folds import plot_cat_kfold_effs


def plot_vars(
        path_to_opt_res_pkl,
        n_cpus,
        plot_simple_opt_flag=False,
        plot_wat_bal_flag=False):
    '''Plot the optimization results
    '''
    pkl_cur = open(path_to_opt_res_pkl, 'rb')
    opt_results_dict = pickle.load(pkl_cur)
    pkl_cur.close()

    cats_list = [_ for _ in opt_results_dict.keys() if isinstance(_, int)]
    assert cats_list

    n_cats = len(cats_list)
    n_cpus = min(n_cats, n_cpus)

    opt_res_gen = (opt_results_dict[cat] for cat in cats_list)
    simple_opt_flag_gen = (plot_simple_opt_flag for i in range(n_cats))
    wat_bal_flag_gen = (plot_wat_bal_flag for i in range(n_cats))

    print('\n\nPlotting opt_results...')

    if n_cpus > 1:
        mp_pool = ProcessPool(n_cpus)
        mp_pool.restart(True)
        try:
            print(list(mp_pool.uimap(plot_hbv,
                                     opt_res_gen,
                                     simple_opt_flag_gen,
                                     wat_bal_flag_gen)))
            mp_pool.clear()
        except Exception as msg:
            mp_pool.close()
            mp_pool.join()
            print('Error in plot_vars:', msg)
    else:
        for opt_res in opt_res_gen:
            plot_hbv(opt_res, plot_simple_opt_flag, plot_wat_bal_flag)
    return


def plot_pops(path_to_opt_res_pkl, n_cpus):
    '''Plot the population
    '''
    pkl_cur = open(path_to_opt_res_pkl, 'rb')
    opt_results_dict = pickle.load(pkl_cur)
    pkl_cur.close()

    cats_list = [_ for _ in opt_results_dict.keys() if isinstance(_, int)]
    assert cats_list

    n_cpus = min(len(cats_list), n_cpus)

    opt_res_gen = (opt_results_dict[cat] for cat in cats_list)

    print('\n\nPlotting DE population...')

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


def plot_kfold_effs(kfold_opt_res_paths, compare_ann_cyc_flag, n_cpus):
    '''Plot the k-fold efficiency results
    '''

    with open(kfold_opt_res_paths[0], 'rb') as _hdl:
        _cats_dict = pickle.load(_hdl)

    cats_list = [_ for _ in _cats_dict.keys() if isinstance(_, int)]
    assert cats_list

    n_cpus = min(len(cats_list), n_cpus)

    const_args = (kfold_opt_res_paths, compare_ann_cyc_flag)
    cats_paths_gen = ((cat, const_args) for cat in cats_list)

    print('\n\nPlotting kfold results...')

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
