'''
@author: Faizan-Uni-Stuttgart

'''

import os
import timeit
import time
import configparser as cfpm

from hydmodeling import (
    TauDEMAnalysis,
    get_stms,
    crt_strms_rltn_tree,
    plot_strm_rltn,
    get_cumm_cats)


def main():
    cfp = cfpm.ConfigParser(interpolation=cfpm.ExtendedInterpolation())
    cfp.read('config_hydmodeling_template.ini')

    n_cpus = cfp['DEFAULT'].getint('n_cpus')

    main_dir = cfp['DEFAULT']['main_dir']
    os.chdir(main_dir)

    hyd_analysis_flag = False
    get_stms_flag = False
    create_stms_rels = False
    create_cumm_cats = False

#     hyd_analysis_flag = True
#     get_stms_flag = True
    create_stms_rels = True
#     create_cumm_cats = True

    # =============================================================================
    # This performs the hydrological preprocessing
    # =============================================================================
    show_ansys_stdout = cfp['HYD_ANSYS'].getboolean('show_ansys_stdout')
    hyd_ansys_runtype = cfp['HYD_ANSYS']['hyd_ansys_runtype']
    calc_for_cats_only = cfp['HYD_ANSYS'].getboolean('show_ansys_stdout')
    max_cell_move = cfp['HYD_ANSYS'].getint('max_cell_move')
    strm_strt_thresh = cfp['HYD_ANSYS'].getint('strm_strt_thresh')

    out_pre_proc_dir = cfp['HYD_ANSYS']['out_pre_proc_dir']
    in_dem_loc = cfp['HYD_ANSYS']['in_dem_loc']
    in_gage_shp_loc = cfp['HYD_ANSYS']['in_gage_shp_loc']

    hyd_ansys = TauDEMAnalysis(in_dem_loc,
                               in_gage_shp_loc,
                               out_pre_proc_dir,
                               n_cpus=n_cpus)
    hyd_ansys.run_type = hyd_ansys_runtype
    hyd_ansys.strm_orign_thresh = strm_strt_thresh
    hyd_ansys.max_cell_move = max_cell_move
    hyd_ansys.verbose = show_ansys_stdout
    hyd_ansys.area_flag = calc_for_cats_only
    if hyd_analysis_flag:
        hyd_ansys()

    #=========================================================================
    # This extracts the required streams for catchments from the shapefiles
    # that we get from TauDEM
    #=========================================================================
    in_dem_net_shp_file = hyd_ansys.dem_net
    in_wat_ids_file = hyd_ansys.watersheds_ids
    out_dem_net_shp_file = cfp['GET_STMS']['out_dem_net_shp_file']
    in_dem_file = hyd_ansys.fil
    in_cats_file = hyd_ansys.watersheds_shp
    in_gauges_coords_file = hyd_ansys.gage_shp_moved
    gauge_coords_field_name = cfp['GET_STMS']['gauge_coords_field_name']
    out_df_file = cfp['GET_STMS']['dem_net_file']
    out_wat_ids_file = cfp['GET_STMS']['out_wat_ids_file']
    sep = cfp['DEFAULT']['sep']

    if get_stms_flag:
        get_stms(in_dem_net_shp_file,
                 in_wat_ids_file,
                 in_dem_file,
                 in_cats_file,
                 in_gauges_coords_file,
                 out_dem_net_shp_file,
                 out_df_file,
                 out_wat_ids_file,
                 sep,
                 gauge_coords_field_name)

    #=========================================================================
    # This creates a stream relationship tree based on their order of
    # occurrence in the out_df_file
    #=========================================================================
    prcss_cats_list = cfp['CREATE_STM_RELS']['prcss_cats_list'].split(',')

    out_hyd_mod_dir = cfp['CREATE_STM_RELS']['hyd_mod_dir']
    out_cats_prcssed_file = cfp['CREATE_STM_RELS']['cats_prcssed_file']
    out_stms_prcssed_file = cfp['CREATE_STM_RELS']['stms_prcssed_file']
    watershed_field_name = cfp['CREATE_STM_RELS']['watershed_field_name']
    out_cats_rel_fig_path = cfp['CREATE_STM_RELS']['out_cats_rel_fig_path']

    if not os.path.exists(out_hyd_mod_dir):
        os.mkdir(out_hyd_mod_dir)

    if create_stms_rels:
        crt_strms_rltn_tree(
            prcss_cats_list,
            out_df_file,
            in_cats_file,
            out_cats_prcssed_file,
            out_stms_prcssed_file,
            sep,
            watershed_field_name)

        plot_strm_rltn(
            hyd_ansys.watersheds_shp,
            hyd_ansys.gage_shp_moved,
            out_dem_net_shp_file,
            out_df_file,
            out_cats_prcssed_file,
            out_stms_prcssed_file,
            prcss_cats_list,
            out_cats_rel_fig_path,
            sep=sep)

    out_cumm_cat_shp = cfp['CUMM_CATS']['out_cumm_cat_shp']
    out_cumm_cat_descrip_file = cfp['CUMM_CATS']['out_cumm_cat_descrip_file']

    if create_cumm_cats:
        get_cumm_cats(
            in_cats_file,
            watershed_field_name,
            out_wat_ids_file,
            sep,
            out_cumm_cat_shp,
            out_cumm_cat_descrip_file,
            sep)

    return


if __name__ == '__main__':
    _save_log_ = False
    if _save_log_:
        from datetime import datetime
        from std_logger import StdFileLoggerCtrl

        # save all console activity to out_log_file
        out_log_file = os.path.join(r'P:\\',
                                    r'Synchronize',
                                    r'python_script_logs',
                                    ('xxx_log_%s.log' %
                                     datetime.now().strftime('%Y%m%d%H%M%S')))
        log_link = StdFileLoggerCtrl(out_log_file)

    print('#### Started on %s ####\n' % time.asctime())
    START = timeit.default_timer()  # to get the runtime of the program

    main()

    STOP = timeit.default_timer()  # Ending time
    print(('\n#### Done with everything on %s.\nTotal run time was'
           ' about %0.4f seconds ####' % (time.asctime(), STOP - START)))

    if _save_log_:
        log_link.stop()
