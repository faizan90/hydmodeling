import pickle

from pathos.multiprocessing import ProcessPool

from .sims import plot_hbv, plot_pop
from .k_folds import plot_cat_k_fold_effs


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

    cats_list = list(opt_results_dict.keys())
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

    cats_list = list(opt_results_dict.keys())

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


def plot_k_fold_effs(
        path_to_opt_res_pkl,
        path_to_params_pkl,
        n_cpus):
    '''Plot the k-fold efficiency results
    '''
    pkl_cur = open(path_to_opt_res_pkl, 'rb')
    opt_results_dict = pickle.load(pkl_cur)
    pkl_cur.close()

    prms_cur = open(path_to_params_pkl, 'rb')
    prms_dict = pickle.load(prms_cur)
    prms_cur.close()

    cats_list = list(opt_results_dict.keys())
    n_cpus = min(len(cats_list), n_cpus)
    assert cats_list

    opt_res_gen = (opt_results_dict[cat] for cat in cats_list)
    kfold_res_gen = (prms_dict[cat] for cat in cats_list)

    print('\n\nPlotting kfold results...')

    if n_cpus > 1:
        mp_pool = ProcessPool(n_cpus)
        mp_pool.restart(True)
        try:
            print(list(mp_pool.uimap(plot_cat_k_fold_effs,
                                     opt_res_gen,
                                     kfold_res_gen)))
            mp_pool.clear()
        except Exception as msg:
            mp_pool.close()
            mp_pool.join()
            print('Error in plot_k_fold_effs:', msg)
    else:
        for opt_res in zip(opt_res_gen, kfold_res_gen):
            plot_cat_k_fold_effs(*opt_res)
    return
