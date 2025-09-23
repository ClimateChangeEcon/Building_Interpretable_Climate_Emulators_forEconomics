#
# Figure 9
#
# python script to illustrate ERA5 temperatures on a global map, as well as
# deviation of CMIP5 models HadGEM2-ES and MPI-ESM-LR from ERA5
# all temperatures are means 1991-2020
#
# Input:
#   tas_climmean_era5_1991-2020_g025.nc                         ERA5 data
#   tas_climmean_HadGEM2-ES_minus_era5_1991-2020_g025.nc        HadGEM2-ES data minus ERA5 data
#   tas_climmean_MPI-ESM-LR_minus_era5_1991-2020_g025.nc        MPI-ESM-LR data minus ERA5 data
#
#   difference between maps was computed offline using cdo, climat data operators, https://code.mpimet.mpg.de/projects/cdo
#
# Output: Figure 9, 3 panels
#
# To run: exec(open("./Figure9.py").read())
#------------------------------------------------------------------------------------------------------------

import os
import re
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import xarray as xr
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.util import add_cyclic_point
import glob
from matplotlib.colors import LinearSegmentedColormap, ListedColormap

#font size for annotation
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
fsl  = 14  #axis labels
fscb = 14  #colorbar labels
fsti = 14  #title 
figname1 = 'tas_clim_1991-2020_'
figname2 = 'ClimPatMap_'
proj    = 'PlateCarree'
figx    = 12
figy    = 10
GMTC    = 1.
levels  = 60
addcont = 'no'
addcb   = 'yes'

#define color maps to be used (for list of colors with names: https://matplotlib.org/stable/gallery/color/named_colors.html)
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
icmap = 3  #which of the colormaps to use? cmap1 (1) or cmap2 (2)?

vmin1  = 0.5;   vmax1 = 3.5;   dv1   = vmax1-vmin1
colors1 = ["darkblue",            "white",           "green",              "yellow",        "darkorange",                "red",   "darkred"]
nodes1  = [       0.0,    (1.0-vmin1)/dv1,    (1.5-vmin1)/dv1,      (2.0-vmin1)/dv1,     (2.5-vmin1)/dv1,      (3.0-vmin1)/dv1,         1.0]
cmap1   = LinearSegmentedColormap.from_list("mycmap", list(zip(nodes1, colors1)))

vmin2  = 0.5;   vmax2 = 3.0;   dv2   = vmax2-vmin2
colors2 = ["darkblue",            "white",           "green",              "yellow",               "red",   "darkred"]
nodes2  = [       0.0,    (1.0-vmin2)/dv2,    (1.5-vmin2)/dv2,      (2.0-vmin2)/dv2,     (2.5-vmin2)/dv2,         1.0]
cmap2   = LinearSegmentedColormap.from_list("mycmap", list(zip(nodes2, colors2)))

cmap3 = 'bwr';  vmin3=-5.1;   vmax3=+5.1;

#get a listing of your files, sorted by file name
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
folder = "./"                                              # where are the data to be processed?
fileid = "tas_climmean_*_minus_era5_1991-2020_g025.nc"   # list the files to be processed
files  = sorted(glob.glob(os.path.join(folder,fileid)))
nfiles = np.size(files)                                    #number of files
print('number of models is',nfiles)

folder2 = "./"                                              # where are the data to be processed?
fileid2 = "tas_climmean_era5_1991-2020_g025.nc"   # list the files to be processed
files2  = sorted(glob.glob(os.path.join(folder2,fileid2)))
nfiles2 = np.size(files2)                                    #number of files
print('number of models is',nfiles2)

#go through list of files and read the data
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
models = []
for ifile in range(nfiles):
    fname = files[ifile]
    print( " processing file : ",fname)
    s    = re.split('_',fname)
    momo = s[2]
    M4P  = s[2]
    models.append(momo)
    ds       = xr.open_dataset(fname)
    T2D_tas  = ds.tas
    lons     = ds.lon
    lats     = ds.lat
    
    if (icmap==1): cmap=cmap1; vmin=vmin1; vmax=vmax1
    if (icmap==2): cmap=cmap2; vmin=vmin2; vmax=vmax2
    if (icmap==3): cmap=cmap3; vmin=vmin3; vmax=vmax3
    #pattern data needs adding one set of longitudes, to 'close' the map at longitud = 0 degrees
    #tell python / cartopy to add a cylclic entry to longitudes, to get rid of white line at lon=0
    T2D_tas, lons = add_cyclic_point(T2D_tas, axis=1, coord=lons)
    T2D_tas[T2D_tas>vmax]=vmax
    T2D_tas[T2D_tas<vmin]=vmin
    
    #what projection?
    data_crs=ccrs.PlateCarree()  #default projection
    proj_crs=ccrs.PlateCarree()
    if (proj=='PlateCarree'):                  proj_crs=ccrs.PlateCarree()
    elif (proj=='Mercator'):                   proj_crs=ccrs.Mercator()
    elif (proj=='Miller'):                     proj_crs=ccrs.Miller()
    elif (proj=='Mollweide'):                  proj_crs=ccrs.Mollweide()
    elif (proj=='Robinson'):                   proj_crs=ccrs.Robinson()
    elif (proj=='InterruptedGoodeHomolosine'): proj_crs=ccrs.InterruptedGoodeHomolosine()
    elif (proj=='RotatedPole'):                proj_crs=ccrs.RotatedPole()
    else: print("WARNING! Desired projection "+ proj +" not found, using PlateCarree instead")
    print("projection is "+ proj)

    #plot change pattern
    fig = plt.figure(figsize=(figx, figy))
    ax = fig.add_subplot(1, 1, 1, projection=proj_crs)
    ax.set_global()
    ax.coastlines()
    cf = ax.contourf(lons, lats, T2D_tas, levels=levels, cmap=cmap, transform=data_crs) #plot filled contours, 60 levels
    if (addcont=='yes'): cl = ax.contour(lons, lats, T2D_tas, levels=cf.levels, colors=['black'], transform=data_crs) #add black contour lines
    if (addcb=='yes'):
        cb = plt.colorbar(cf, shrink=0.5)
        cb.ax.tick_params(labelsize=fscb)
        cb.ax.set_title('$^{o}$Celsius',fontsize=fscb)
        cb.ax.locator_params(nbins=11)
    ax.set_title('1991-2020 climatological mean temperature on 2.5x2.5 grid, '+M4P+' minus ERA5',fontsize=fsti)
    plt.tight_layout(); 
    #plt.show(block=False)
    #fout = figname1+M4P+'_minus_ERA5.png'
    if (ifile==0): fout = 'figure_9_bottom_right.png'
    if (ifile==1): fout = 'figure_9_bottom_left.png'
    plt.savefig(fout,format='png')


#go through list of files2 and read the data
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
models2 = []
for ifile in range(nfiles2):
    fname = files2[ifile]
    print( " processing file : ",fname)
    s    = re.split('_',fname)
    momo = s[2]
    M4P  = s[2]
    models2.append(momo)
    ds       = xr.open_dataset(fname)
    T2D_tas  = ds.tas[0,:,:]
    lons     = ds.lon
    lats     = ds.lat
    
    if (icmap==1): cmap=cmap1; vmin=vmin1; vmax=vmax1
    if (icmap==2): cmap=cmap2; vmin=vmin2; vmax=vmax2
    if (icmap==3): cmap=cmap3; vmin=vmin3; vmax=vmax3
    cmap='viridis'; vmin=-30.; vmax=+30;
    
    #pattern data needs adding one set of longitudes, to 'close' the map at longitud = 0 degrees
    #tell python / cartopy to add a cylclic entry to longitudes, to get rid of white line at lon=0
    T2D_tas, lons = add_cyclic_point(T2D_tas, axis=1, coord=lons)
    T2D_tas[T2D_tas>vmax]=vmax
    T2D_tas[T2D_tas<vmin]=vmin
    
    #what projection?
    data_crs=ccrs.PlateCarree()  #default projection
    proj_crs=ccrs.PlateCarree()
    if (proj=='PlateCarree'):                  proj_crs=ccrs.PlateCarree()
    elif (proj=='Mercator'):                   proj_crs=ccrs.Mercator()
    elif (proj=='Miller'):                     proj_crs=ccrs.Miller()
    elif (proj=='Mollweide'):                  proj_crs=ccrs.Mollweide()
    elif (proj=='Robinson'):                   proj_crs=ccrs.Robinson()
    elif (proj=='InterruptedGoodeHomolosine'): proj_crs=ccrs.InterruptedGoodeHomolosine()
    elif (proj=='RotatedPole'):                proj_crs=ccrs.RotatedPole()
    else: print("WARNING! Desired projection "+ proj +" not found, using PlateCarree instead")
    print("projection is "+ proj)

    #plot change pattern
    fig = plt.figure(figsize=(figx, figy))
    ax = fig.add_subplot(1, 1, 1, projection=proj_crs)
    ax.set_global()
    ax.coastlines()
    cf = ax.contourf(lons, lats, T2D_tas, levels=levels, cmap=cmap, transform=data_crs) #plot filled contours, 60 levels
    if (addcont=='yes'): cl = ax.contour(lons, lats, T2D_tas, levels=cf.levels, colors=['black'], transform=data_crs) #add black contour lines
    if (addcb=='yes'):
        cb = plt.colorbar(cf, shrink=0.5)
        cb.ax.tick_params(labelsize=fscb)
        cb.ax.set_title('$^{o}$Celsius',fontsize=fscb)
        cb.ax.locator_params(nbins=11)
    if (ifile==0): ax.set_title('ERA5 1991-2020 climatological mean temperature on 2.5x2.5 grid',fontsize=fsti)
    plt.tight_layout(); 
    #plt.show(block=False)
    #if (ifile==0): fout = 'ERA5_1991-2020_tas_climmean.png'
    if (ifile==0): fout = 'figure_9_top.png'
    plt.savefig(fout,format='png')

