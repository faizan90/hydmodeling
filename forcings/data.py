'''
Created on May 2, 2019

@author: Faizan-Uni

Under construction. Do not use!
'''
from pathlib import Path

import pandas as pd


class ReferenceSimPath:

    def __init__(self):

        self._ref_paths_set_flag = False

        self._ref_inputs_vrfy_flag = False
        return

    def set_ref_sim_path(self, ref_sim_dir, config_file):

        assert isinstance(ref_sim_dir, (str, Path))
        assert isinstance(config_file, (str, Path))

        ref_sim_dir = Path(ref_sim_dir).absolute()
        config_file = Path(config_file).absolute()

        cats_prcssed_df_file = ref_sim_dir / r'cats_prcssed_df.csv'
        stms_prcssed_df_file = ref_sim_dir / r'stms_prcssed_df.csv'

        assert ref_sim_dir.exists()
        assert ref_sim_dir.is_dir()

        assert config_file.exists()
        assert config_file.is_file()

        assert cats_prcssed_df_file.exists()
        assert cats_prcssed_df_file.is_file()

        assert stms_prcssed_df_file.exists()
        assert stms_prcssed_df_file.is_file()

        self.ref_sim_dir = ref_sim_dir
        self.ref_config_file = config_file

        self._ref_paths_set_flag = True
        return

    def verify(self):

        assert self._ref_paths_set_flag

        self._ref_inputs_vrfy_flag = True
        return

    __verify = verify


class InitCalibData(ReferenceSimPath):

    def __init__(self):

        ReferenceSimPath.__init__(self)

        self._calib_data_init_flag = False

        self._calib_data_vrfy_flag = False
        return

    def init_calib_data(self):

        self._calib_data_init_flag = True
        return

    def verify(self):

        ReferenceSimPath._ReferenceSimPath__verify(self)

        assert self._calib_data_init_flag

        self._calib_data_vrfy_flag = True
        return


class DestinationData:

    def __init__(self):

        self._ppt_data_set_flag = False
        self._tem_data_set_flag = False
        self._pet_data_set_flag = False

        self._dis_data_set_flag = False

        self._dest_sim_dir_set_flag = False

        self._dst_inputs_vrfy_flag = False

        self.dppd = None
        self.dted = None
        self.dped = None
        return

    def _set_dest_var_data(self, data, var):

        assert isinstance(data, dict)

        assert all([isinstance(df, pd.DataFrame) for df in data])

        setattr(self, var, data)
        return

    def set_ppt_data(self, ppt_dfs_dict):

        self._set_dest_var_data(ppt_dfs_dict, 'dppd')

        self._ppt_data_set_flag = True
        return

    def set_tem_data(self, tem_dfs_dict):

        self._set_dest_var_data(tem_dfs_dict, 'dted')

        self._tem_data_set_flag = True
        return

    def set_pet_data(self, pet_dfs_dict):

        self._set_dest_var_data(pet_dfs_dict, 'dped')

        self._pet_data_set_flag = True
        return

    def set_dest_dir(self, dest_dir):

        assert isinstance(dest_dir, (str, Path))

        dest_dir = Path(dest_dir).absolute()

        assert dest_dir.parent[0].exists()

        dest_dir.mkdir(exist_ok=True)

        self.dstd = dest_dir

        self._dest_sim_dir_set_flag = True
        return

    def set_dis_data(self, dis_df):

        assert isinstance(dis_df, pd.DataFrame)

        self._dis_data_set_flag = True
        return

    def verify(self):

        assert self._ppt_data_set_flag
        assert self._tem_data_set_flag
        assert self._pet_data_set_flag

        assert self._dest_sim_dir_set_flag

        self._dst_inputs_vrfy_flag = True
        return

    __verify = verify


class VerifyAndInitializeData:

    def __init__(self, ref_data_cls, dst_data_cls):

        assert ref_data_cls._ref_inputs_vrfy_flag
        assert dst_data_cls._dst_inputs_vrfy_flag

        self.refc = ref_data_cls
        self.dstc = dst_data_cls

        self._ref_data_vrfy_init_flag = False
        self._dst_data_vrfy_init_flag = False

        self._data_vrfy_init_flag = False
        return

