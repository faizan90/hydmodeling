'''
@author: Faizan-Uni-Stuttgart

'''

import os
import timeit
import time
import shelve
from pathlib import Path

import numpy as np

from hydmodeling import hbv_mult_cat_loop_py


def main():
    main_dir = Path(r'P:\Synchronize\IWS\QGIS_Neckar\horb_dist_speed_test\01_database')
    os.chdir(main_dir)

    with shelve.open(r'cat_411', 'r') as db:
#         print(list(db['calib']['kf_01'].keys()))
#         print(list(db['data'].keys()))
        # print(db['valid']['kf_01']['out_cats_flow_df'].max())

#         print(list(db['valid']['kf_02'].keys()))

        valid_dict = db['calib']['kf_02']
        calib_dict = db['calib']['kf_02']
        data_dict = db['data']

        hbv_prms, route_prms = calib_dict['hbv_prms'], calib_dict['route_prms']
        inis_arr, temp_arr, prec_arr, petn_arr, qact_arr = (valid_dict['ini_arr'],
                                                            valid_dict['tem_arr'],
                                                            valid_dict['ppt_arr'],
                                                            valid_dict['pet_arr'],
                                                            valid_dict['qact_arr'])

        (curr_us_stm,
         stm_idxs,
         cat_to_idx_dict,
         stm_to_idx_dict) = -2, np.array([0], dtype=np.int32), {411: 0}, {}

        (dem_net_arr,
         cats_outflow_arr,
         stms_inflow_arr,
         stms_outflow_arr) = (np.array([[1.0]], dtype=float),
                              np.zeros((qact_arr.shape[0], 1), order='f'),
                              np.array([[1.0]], order='f'),
                              np.array([[1.0]], order='f'))

        (off_idx,
         rnof_q_conv,
         route_type,
         cat_no,
         n_cpus,
         n_stms,
         n_cells,
         use_obs_flow_flag,
         n_hm_prms,
         use_step_flag,
         use_step_arr) = (data_dict['off_idx'],
                          data_dict['conv_ratio'],
                          data_dict['route_type'],
                          411,
                          1,
                          0,
                          temp_arr.shape[0],
                          data_dict['use_obs_flow_flag'],
                          calib_dict['opt_prms'].shape[0],
                          valid_dict['use_step_flag'],
                          valid_dict['use_step_arr'])

        area_arr = data_dict['area_arr']

    args = []
#     hbv_prms[:, 3] += 70
#     hbv_prms[:, 5] += 30
    args.append((hbv_prms, route_prms))
    args.append([])
    args.append([])
    args.append((inis_arr, temp_arr, prec_arr, petn_arr, qact_arr))
    args.append((curr_us_stm, stm_idxs, cat_to_idx_dict, stm_to_idx_dict))
    args.append((dem_net_arr, cats_outflow_arr, stms_inflow_arr, stms_outflow_arr))
    args.append(
        (off_idx,
         rnof_q_conv,
         route_type,
         cat_no,
         n_cpus,
         n_stms,
         n_cells,
         use_obs_flow_flag,
         n_hm_prms,
         use_step_flag,
         use_step_arr))
    args.append(area_arr)

    res = hbv_mult_cat_loop_py(args)
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
