cd %~dp0
set PATH=GDAL
set GDAL_DATA=GDAL\gdal-data
ECHO %PATH%
ECHO %GDAL_DATA%
%1 -z %2 -fel %3

	