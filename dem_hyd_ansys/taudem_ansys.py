'''
Created on Oct 8, 2017

@author: Faizan
'''
import time
import timeit
import subprocess
from os import mkdir
from platform import architecture
from os.path import (
	join as os_join,
	exists as os_exists,
	abspath,
	basename,
	dirname)

import ogr

from .misc import get_ras_props, get_vec_props
from .merge_polys import merge_same_id_shp_poly


class TauDEMAnalysis:

	def __init__(self, raw_dem_path, gage_shp_path, outputs_dir, n_cpus=1):

		self.raw_dem_path = abspath(raw_dem_path)
		self.gage_shp_path = abspath(gage_shp_path)
		self.outputs_dir = abspath(outputs_dir)
		self.n_cpus = n_cpus

		assert n_cpus > 0

		if not os_exists(self.raw_dem_path):
			raise IOError(
				'raw_dem file does not exist at the given location!')

		if not os_exists(self.gage_shp_path):
			raise IOError(
				'gage_shp (%s) file does not exist at the given location!' % (
					self.gage_shp_path))

		self.fil = os_join(self.outputs_dir, 'fil.tif')
		self.fdr = os_join(self.outputs_dir, 'fdr.tif')
		self.sd8 = os_join(self.outputs_dir, 'sd8.tif')
		self.fac = os_join(self.outputs_dir, 'fac.tif')
		self.strm = os_join(self.outputs_dir, 'strm.tif')
		self.strm_dist = os_join(self.outputs_dir, 'strm_dist.tif')
		self.log_file = os_join(self.outputs_dir, 'log_file.txt')

		self.watersheds = os_join(self.outputs_dir, 'watersheds.tif')
		self.watersheds_all = os_join(self.outputs_dir, 'watersheds_all.tif')
		self.watersheds_shp = os_join(self.outputs_dir, 'watersheds.shp')
		self.watersheds_ids = os_join(self.outputs_dir, 'watersheds_id.txt')

		_ = basename(self.gage_shp_path).rsplit('.', 1)[0]
		self.gage_shp_moved = os_join(self.outputs_dir, _ + '_moved.shp')

		self.dem_ord = os_join(self.outputs_dir, 'dem_ord.tif')
		self.dem_tree = os_join(self.outputs_dir, 'dem_tree.dat')
		self.dem_coords = os_join(self.outputs_dir, 'dem_coords.dat')
		self.dem_net = os_join(self.outputs_dir, 'dem_net.shp')

		self.watersheds_flag = True
		self.area_flag = False
		self.strm_dists_flag = False
		self.verbose = True

		self.run_type = 'before'  # can be 'before' or 'after'
		self.strm_orign_thresh = 1000
		self.max_cell_move = 15  # grid cells

		_bitness = architecture()[0]
		if _bitness == '32bit':
			raise NotImplementedError('To be downloaded!')
			self.exes_dir = os_join(dirname(abspath(__file__)),
									'TauDEM537exeWin32')

		elif _bitness == '64bit':
			self.exes_dir = os_join(dirname(abspath(__file__)),
									'TauDEM535exeWin64')

		else:
			raise RuntimeError('Could not get the bitness of the system!')

		self.polygonize_file = os_join(
			dirname(abspath(__file__)), 'gdal_polygonize.py')

		self.fil_exe = os_join(self.exes_dir, 'PitRemove')
		self.fdr_exe = os_join(self.exes_dir, 'D8FlowDir')
		self.fac_exe = os_join(self.exes_dir, 'AreaD8')
		self.thresh_exe = os_join(self.exes_dir, 'Threshold')
		self.snap_pp_exe = os_join(self.exes_dir, 'MoveOutletsToStreams')
		self.strm_net_exe = os_join(self.exes_dir, 'StreamNet')
		self.gage_watershed_exe = os_join(self.exes_dir, 'GageWatershed')
		self.strm_dist_exe = os_join(self.exes_dir, 'D8HDistToStrm')
		return

	def _prepare(self):

		self.dem_coord_sys = get_ras_props(self.raw_dem_path)[8]
		self.gage_shp_coord_sys = get_vec_props(self.gage_shp_path, 0)[4]

		assert 'PROJCS' in self.dem_coord_sys, (
			'DEM coordinate system not projected!')

		assert 'PROJCS' in self.gage_shp_coord_sys, (
			'Gage shapefile coordinate system not projected!')

		print('\n', 'Raw DEM coord. sys.:\n', self.dem_coord_sys, sep='')
		print('\n\n',
			  'Gage shp. coord. sys.:\n',
			  self.gage_shp_coord_sys,
			  '\n\n',
			  sep='')

		try:
			in_shp_vec = ogr.Open(self.gage_shp_path)
			in_shp_lyr = in_shp_vec.GetLayer(0)
			in_lyr_defn = in_shp_lyr.GetLayerDefn()
			in_field_cnt = in_lyr_defn.GetFieldCount()
			id_in_fields = False

			for i in range(in_field_cnt):
				if str(in_lyr_defn.GetFieldDefn(i).name) == str('id'):
					id_in_fields = True
					break

			assert id_in_fields

		except AssertionError:
			raise AssertionError('Field \'id\' does not exist in gage_shp!')

		finally:
			in_shp_vec.Destroy()

		if not os_exists(self.outputs_dir):
			mkdir(self.outputs_dir)
		return

	def __call__(self):

		self._prepare()

		fil_cmd = '"%s" -z "%s" -fel "%s"' % (
			self.fil_exe, self.raw_dem_path, self.fil)

		fdr_cmd = '"%s" -fel "%s" -p "%s" -sd8 "%s"' % (
			self.fdr_exe, self.fil, self.fdr, self.sd8)

		if self.area_flag:
			# if gage_shp is the original one, then stream net might be wrong.
			# Run once without the area_flag. Rename gage_shp_moved to
			# gage_shp_path
			fac_cmd = '"%s" -p "%s" -ad8 "%s" -o "%s"' % (
				self.fac_exe, self.fdr, self.fac, self.gage_shp_path)

		else:
			fac_cmd = '"%s" -p "%s" -ad8 "%s"' % (
				self.fac_exe, self.fdr, self.fac)

		thresh_cmd = ('"%s" -ssa "%s" -src "%s" -thresh %d' %
					  (self.thresh_exe,
					   self.fac,
					   self.strm,
					   self.strm_orign_thresh))

		snap_pp_cmd = ('"%s" -p "%s" -src "%s" -o "%s" -om "%s" -md %d' %
					   (self.snap_pp_exe,
						self.fdr,
						self.strm,
						self.gage_shp_path,
						self.gage_shp_moved,
						self.max_cell_move))

		streamnet_cmd = (
			('"%s" -fel "%s" -p "%s" -ad8 "%s" -src "%s" -ord "%s" '
			 '-tree "%s" -coord "%s" -net "%s" -w "%s" -o "%s" -sw') %
	 		(self.strm_net_exe,
  			 self.fil,
		  	 self.fdr,
	  		 self.fac,
		  	 self.strm,
		  	 self.dem_ord,
		  	 self.dem_tree,
		  	 self.dem_coords,
		  	 self.dem_net,
		  	 self.watersheds_all,
		  	 self.gage_shp_moved))

		if self.n_cpus > 1:
			fil_cmd = 'mpiexec -n %d %s' % (self.n_cpus, fil_cmd)
			fdr_cmd = 'mpiexec -n %d %s' % (self.n_cpus, fdr_cmd)
			fac_cmd = 'mpiexec -n %d %s' % (self.n_cpus, fac_cmd)
			thresh_cmd = 'mpiexec -n %d %s' % (self.n_cpus, thresh_cmd)
			snap_pp_cmd = 'mpiexec -n %d %s' % (self.n_cpus, snap_pp_cmd)
			streamnet_cmd = 'mpiexec -n %d %s' % (self.n_cpus, streamnet_cmd)

		out_shps = [self.gage_shp_moved, self.dem_net, self.watersheds_shp]

		for shp in out_shps:
			if not os_exists(shp):
				continue

			in_vec = ogr.Open(shp)
			out_driver = in_vec.GetDriver()
			in_vec.Destroy()
			out_driver.DeleteDataSource(shp)

		if self.run_type == 'before':
			cmd_list = [
				fil_cmd,
				fdr_cmd,
				fac_cmd,
				thresh_cmd,
				snap_pp_cmd,
				streamnet_cmd]

		elif self.run_type == 'after':
			cmd_list = [fac_cmd, thresh_cmd, snap_pp_cmd, streamnet_cmd]

			assert os_exists(self.fil), '%s does not exist!' % self.fil
			assert os_exists(self.fdr), '%s does not exist!' % self.fdr
			assert os_exists(self.sd8), '%s does not exist!' % self.sd8

		else:
			raise NameError(
				'\RUN_TYPE\' can only be \'before\' or \'after\'!')

		if self.watersheds_flag:
			gage_watershed_cmd = (
				'"%s" -p "%s" -o "%s" -gw "%s" -id "%s"' % (
					self.gage_watershed_exe,
					self.fdr,
					self.gage_shp_moved,
					self.watersheds,
					self.watersheds_ids))

			if self.n_cpus > 1:
				gage_watershed_cmd = (
					'mpiexec -n %d %s' % (self.n_cpus, gage_watershed_cmd))

			cmd_list.append(gage_watershed_cmd)

		if self.strm_dists_flag:
			strm_dist_cmd = (
				'"%s" -p "%s" -src "%s" -dist "%s" -thresh "%d"' % (
					self.strm_dist_exe,
					self.fdr,
					self.fac,
					self.strm_dist,
					self.strm_orign_thresh))

			if self.n_cpus > 1:
				strm_dist_cmd = (
					'mpiexec -n %d %s' % (self.n_cpus, strm_dist_cmd))

			cmd_list.append(strm_dist_cmd)

		if self.verbose:
			for cmd in cmd_list:
				print('\nExecuting: %s' % cmd)
				proc = subprocess.Popen(cmd, shell=False)
				proc.wait()

		else:
			log_file_cur = open(self.log_file, 'w')
			for cmd in cmd_list:
				# print activitities to LOG_FILE
				print('\nExecuting: %s' % cmd)
				proc = subprocess.Popen(
					cmd,
					shell=False,
					stdout=log_file_cur,
					stderr=log_file_cur)

				proc.wait()

			log_file_cur.close()

		if self.watersheds_flag:
			assert os_exists(self.watersheds), (
				'watersheds file does not exist!')

			temp_shp = dirname(self.watersheds_shp)
			temp_shp = os_join(temp_shp, 'temp_')
			temp_shp += basename(self.watersheds_shp)

			fmt = 'ESRI Shapefile'
			cmd = 'python "%s" "%s" -f "%s" "%s"' % (
				self.polygonize_file, self.watersheds, fmt, temp_shp)

			print('\nExecuting: %s' % cmd)

			if self.verbose:
				proc = subprocess.Popen(cmd, shell=False)
				proc.wait()

			else:
				log_file_cur = open(self.log_file, 'a')
				proc = subprocess.Popen(
					cmd,
					shell=False,
					stdout=log_file_cur,
					stderr=log_file_cur)

				proc.wait()
				log_file_cur.close()

			merge_same_id_shp_poly(
				temp_shp, self.watersheds_shp, field='DN')

			driver = ogr.GetDriverByName(fmt)
			if os_exists(temp_shp):
				driver.DeleteDataSource(temp_shp)
		return


if __name__ == '__main__':

	print('\a\a\a\a Started on %s \a\a\a\a\n' % time.asctime())
	START = timeit.default_timer()  # to get the runtime of the program

	main_dir = r'P:\Synchronize\IWS\QGIS_Neckar'

	dem = os_join(main_dir,
				  r'raster',
				  r'lower_de_gauss_z3_100m.tif')  # Input DEM

	# Input, get catchments only for these drainage points
	gage_shp = os_join(main_dir,
					   r'vector',
					   r'neckar_infilled_46_stns_july_2017.shp')

	# Output dir, everything goes in here
	out_dir = os_join(main_dir,
					  r'raster',
					  r'taudem_out_test')

	n_cpus = 1

	taudem_analysis = TauDEMAnalysis(dem, gage_shp, out_dir, n_cpus=n_cpus)
	taudem_analysis.run_type = 'after'
# 	 taudem_analysis.verbose = False
	taudem_analysis()

	STOP = timeit.default_timer()  # Ending time
	print(('\n\a\a\a Done with everything on %s. Total run time was'
		   ' about %0.4f seconds \a\a\a' % (time.asctime(), STOP - START)))
