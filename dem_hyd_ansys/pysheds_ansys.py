'''
Created on Oct 8, 2017

@author: Faizan

'''
#==============================================================================
# set PATH=PATH;C:\Program Files\GDAL
# set GDAL_DATA=C:\Program Files\GDAL\gdal-data
# set GDAL_DRIVER_PATH=C:\Program Files\GDAL\gdalplugins
# set PROJ_LIB=C:\Program Files\GDAL\projlib
#==============================================================================

import time
import timeit
import subprocess
from platform import architecture
from os import mkdir
from os.path import (
	join as os_join,
	exists as os_exists,
	abspath,
	basename,
	dirname)

import numpy as np
from osgeo import ogr, gdal
from pysheds.grid import Grid
import matplotlib.pyplot as plt

plt.ioff()

from .misc import get_ras_props, get_vec_props
# from .merge_polys import merge_same_id_shp_poly

import os

subproc_env = os.environ.copy()
subproc_env['PATH'] = r'C:\Program Files\GDAL'
subproc_env['GDAL_DATA'] = r'C:\Program Files\GDAL\gdal-data'
subproc_env['GDAL_DRIVER_PATH'] = r'C:\Program Files\GDAL\gdalplugins'
subproc_env['PROJ_LIB'] = r'C:\Program Files\GDAL\projlib'


class PyShedsAnalysis:

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
		self.strm_orign_thresh = 1000  # grid cells
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

		self._grid = None
		self._dirmap = (64, 128, 1, 2, 4, 8, 16, 32)  # Don't change.
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

	def _deps(self):

		if self.verbose:
			print('Filling depressions...')

# 		self._grid.fill_depressions(data='dem', out_name='flooded_dem')
		return

	def _flats(self):

		if self.verbose:
			print('Resolving flats...')

# 		assert hasattr(self._grid, 'flooded_dem'), 'Call _deps first!'
#
# 		self._grid.resolve_flats('flooded_dem', out_name='fil')
#
# 		self._grid.to_raster('fil', self.fil)

		cmd = '"%s" -z "%s" -fel "%s"' % (
			self.fil_exe, self.raw_dem_path, self.fil)

		cmd = 'mpiexec -n %d %s' % (self.n_cpus, cmd)

		if self.verbose:
			print(cmd)

		proc = subprocess.Popen(cmd, shell=False, env=subproc_env)
		proc.wait()
		return

	def _fdrs(self):

		if self.verbose:
			print('Computing FDR...')

# 		self._grid.flowdir(data='fil', out_name='fdr', dirmap=self._dirmap)
#
# 		self._grid.to_raster('fdr', self.fdr)

		cmd = '"%s" -fel "%s" -p "%s" -sd8 "%s"' % (
			self.fdr_exe, self.fil, self.fdr, self.sd8)

		cmd = 'mpiexec -n %d %s' % (self.n_cpus, cmd)

		if self.verbose:
			print(cmd)

		proc = subprocess.Popen(cmd, shell=False, env=subproc_env)
		proc.wait()
		return

	def _slopes(self):

		if self.verbose:
			print('Computing cell slopes...')

# 		assert hasattr(self._grid, 'fdr'), 'Call _fdrs first!'
#
# 		self._grid.cell_slopes('fdr', 'fil', out_name='sd8')
#
# 		self._grid.to_raster('sd8', self.sd8)
		return

	def _facs(self):

		if self.verbose:
			print('Computing FAC...')

# 		assert hasattr(self._grid, 'fdr'), 'Call _fdrs first!'
#
# 		self._grid.accumulation(
# 			data='fdr', dirmap=self._dirmap, out_name='fac')
#
# 		self._grid.to_raster('fac', self.fac)

		cmd = '"%s" -p "%s" -ad8 "%s"' % (
			self.fac_exe, self.fdr, self.fac)

		cmd = 'mpiexec -n %d %s' % (self.n_cpus, cmd)

		if self.verbose:
			print(cmd)

		proc = subprocess.Popen(cmd, shell=False, env=subproc_env)
		proc.wait()
		return

	def _threshs(self):

		if self.verbose:
			print('Defining streams...')

# 		assert hasattr(self._grid, 'fac'), 'Call _facs first!'

		cmd = ('"%s" -ssa "%s" -src "%s" -thresh %d' %
					  (self.thresh_exe,
					   self.fac,
					   self.strm,
					   self.strm_orign_thresh))

		cmd = 'mpiexec -n %d %s' % (self.n_cpus, cmd)

		if self.verbose:
			print(cmd)

		proc = subprocess.Popen(cmd, shell=False, env=subproc_env)
		proc.wait()
		return

	def _snaps(self):

		if self.verbose:
			print('Snapping pour points...')

		cmd = ('"%s" -p "%s" -src "%s" -o "%s" -om "%s" -md %d' %
					   (self.snap_pp_exe,
						self.fdr,
						self.strm,
						self.gage_shp_path,
						self.gage_shp_moved,
						self.max_cell_move))

		cmd = 'mpiexec -n %d %s' % (self.n_cpus, cmd)

		if self.verbose:
			print(cmd)

		proc = subprocess.Popen(cmd, shell=False, env=subproc_env)
		proc.wait()
		return

	def _streams_net(self):

		if self.verbose:
			print('Delineating streams network...')

		# -sw means single watershed per outlet.

		cmd = (
			('"%s" -fel "%s" -p "%s" -ad8 "%s" -src "%s" -ord "%s" '
			 '-tree "%s" -coord "%s" -net "%s" -netlyr dem_net -w "%s" -o "%s" -sw') %
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

# 		cmd = (
# 			('"%s" -fel "%s" -p "%s" -ad8 "%s" -src "%s" -ord "%s" '
# 			 '-tree "%s" -coord "%s" -net "%s" -netlyr dem_net -w "%s"') %
# 	 		(self.strm_net_exe,
#   			 self.fil,
# 		  	 self.fdr,
# 	  		 self.fac,
# 		  	 self.strm,
# 		  	 self.dem_ord,
# 		  	 self.dem_tree,
# 		  	 self.dem_coords,
# 		  	 self.dem_net,
# 		  	 self.watersheds_all))

		cmd = 'mpiexec -n %d %s' % (self.n_cpus, cmd)

		if self.verbose:
			print(cmd)

		proc = subprocess.Popen(cmd, shell=False, env=subproc_env)
		proc.wait()
		return

# 	def _streams(self):
#
# 		if self.verbose:
# 			print('Extracting river network...')
#
# 		assert hasattr(self._grid, 'fdr'), 'Call _fdrs first!'
# 		assert hasattr(self._grid, 'fac'), 'Call _facs first!'
#
# 		branches = self._grid.extract_river_network(
# 		    fdir='fdr', acc='fac', threshold=10, dirmap=self._dirmap)
#
# 		print('Plotting branches...')
# 		for branch in branches['features']:
# 			line = np.asarray(branch['geometry']['coordinates'])
# 			plt.plot(line[:, 0], line[:, 1])
#
# # 		plt.show()
# 		plt.savefig(
# 		os_join(self.outputs_dir, 'branches.png'), bbox_inches='tight')
#
# 		plt.close()
# 		return

	def __call__(self):

		self._prepare()

		out_shps = [self.gage_shp_moved, self.dem_net, self.watersheds_shp]

		for shp in out_shps:
			if not os_exists(shp):
				continue

			in_vec = ogr.Open(shp)
			out_driver = in_vec.GetDriver()
			in_vec.Destroy()
			out_driver.DeleteDataSource(shp)

		self._grid = Grid.from_raster(self.raw_dem_path, data_name='dem')

		ftns = [
			self._deps,
			self._flats,
			self._fdrs,
			self._slopes,
			self._facs,
			self._threshs,
			self._snaps,
			self._streams_net,
# 			self._streams,
			]

		for ftn in ftns:
			beg_time = timeit.default_timer()

			ftn()

			end_time = timeit.default_timer()

			if self.verbose:
				print(f'Took {end_time - beg_time:0.2f} seconds.\n')

		return


if __name__ == '__main__':

	print('\a\a\a\a Started on %s \a\a\a\a\n' % time.asctime())
	START = timeit.default_timer()  # to get the runtime of the program

# 	main_dir = r'P:\Synchronize\IWS\QGIS_Neckar'
#
# 	dem = os_join(main_dir,
# 				  r'raster',
# 				  r'lower_de_gauss_z3_100m.tif')  # Input DEM
#
# 	# Input, get catchments only for these drainage points
# 	gage_shp = os_join(main_dir,
# 					   r'vector',
# 					   r'neckar_infilled_46_stns_july_2017.shp')
#
# 	# Output dir, everything goes in here
# 	out_dir = os_join(main_dir,
# 					  r'raster',
# 					  r'taudem_out_test')
#
# 	n_cpus = 1
#
# 	taudem_analysis = PyShedsAnalysis(dem, gage_shp, out_dir, n_cpus=n_cpus)
# 	taudem_analysis.run_type = 'after'
# # 	 taudem_analysis.verbose = False
# 	taudem_analysis()

	STOP = timeit.default_timer()  # Ending time
	print(('\n\a\a\a Done with everything on %s. Total run time was'
		   ' about %0.4f seconds \a\a\a' % (time.asctime(), STOP - START)))
