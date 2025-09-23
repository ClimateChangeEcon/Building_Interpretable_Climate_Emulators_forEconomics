#
# python script for Figure 33
#
# python script to loade patterns of 31 models from Lynch et al. (2017), aggregated into 46 regions as given in  Iturbide et al. (2020) [land only]
# missing models are: CESM1-WACCM, EC_Earth, FIO-ESM, GFDL-CM3, GFDL-ESM2G, GFDL-ESM2M, GISS-ES_R, bcc-csm-1-m, bcc-csm-1, inmcm
#
# produce a plot that shows in color coding the warming for each model and region
#
# regions:
# ['ARP', 'CAF', 'CAR', 'CAU', 'CNA', 'EAN', 'EAS', 'EAU', 'ECA', 'EEU', 'ENA', 'ESAF', 'ESB', 'GIC', 'MDG', 'MED',
#  'NAU', 'NCA', 'NEAF', 'NEN', 'NES', 'NEU', 'NSA', 'NWN', 'NWS', 'NZ', 'RAR', 'RAR*', 'RFE', 'SAH', 'SAM', 'SAS',
#  'SAU', 'SCA', 'SEA', 'SEAF', 'SES', 'SSA', 'SWS', 'TIB', 'WAF', 'WAN', 'WCA', 'WCE', 'WNA', 'WSAF']
#
# models:
# ['ACCESS1-0', 'ACCESS1-3', 'BNU-ESM', 'CCSM4', 'CESM1-BGC', 'CESM1-CAM5', 'CMCC-CESM', 'CMCC-CMS', 'CMCC-CM', 'CNRM-CM5',
# 'CSIRO-Mk3-6-0', 'CanESM2', 'FGOALS-g2', 'GISS-E2-H-CC', 'GISS-E2-H', 'GISS-E2-R-CC', 'HadGEM2-AO', 'HadGEM2-CC', 'HadGEM2-ES',
# 'IPSL-CM5A-LR', 'IPSL-CM5A-MR', 'IPSL-CM5B-LR', 'MIROC-ESM-CHEM', 'MIROC-ESM', 'MIROC5', 'MPI-ESM-LR', 'MPI-ESM-MR', 'MRI-CGCM3',
# 'MRI-ESM1', 'NorESM1-ME', 'NorESM1-M']
#
# To run: exec(open("./Figure33.py").read())
#------------------------------------------------------------------------------------------------------------

import re            #module for regular expression pattern matching
import glob
import os            #module for operating system stuff
import csv
import importlib
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap, ListedColormap

#font size for annotation
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
fsl  = 14  #axis labels
fscb = 14  #colorbar labels
figname = 'figure_32.png'

#define color maps to be used (for list of colors with names: https://matplotlib.org/stable/gallery/color/named_colors.html)
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
icmap = 2  #which of the colormaps to use? cmap1 (1) or cmap2 (2)?

vmin1  = 0.5;   vmax1 = 3.5;   dv1   = vmax1-vmin1
colors1 = ["darkblue",            "white",           "green",              "yellow",        "darkorange",                "red",   "darkred"]
nodes1  = [       0.0,    (1.0-vmin1)/dv1,    (1.5-vmin1)/dv1,      (2.0-vmin1)/dv1,     (2.5-vmin1)/dv1,      (3.0-vmin1)/dv1,         1.0]
cmap1   = LinearSegmentedColormap.from_list("mycmap", list(zip(nodes1, colors1)))

vmin2  = 0.5;   vmax2 = 3.0;   dv2   = vmax2-vmin2
colors2 = ["darkblue",            "white",           "green",              "yellow",               "red",   "darkred"]
nodes2  = [       0.0,    (1.0-vmin2)/dv2,    (1.5-vmin2)/dv2,      (2.0-vmin2)/dv2,     (2.5-vmin2)/dv2,         1.0]
cmap2   = LinearSegmentedColormap.from_list("mycmap", list(zip(nodes2, colors2)))


#get a listing of your files, sorted by file name
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
folder = "./"                                              # where are the data to be processed?
fileid = "temp_region_patternonly_PATTERN_tas_ANN_*.csv"   # list the files to be processed
files  = sorted(glob.glob(os.path.join(folder,fileid)))
nfiles = np.size(files)                                    #number of files
print('number of models is',nfiles)

#get number of lines in file, subtract number of header lines, to get number of data lines
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
cmd    = 'wc -l '+files[0]
cou    = os.popen(cmd).read()
nlt    = re.split('\s+', cou)
n_head = 1
n_data = np.int64( int(nlt[0]) - n_head )
print('number of regions is',n_data)

#initialize data structures
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
nregions = n_data-1
nmodels  = nfiles
regions  = []
models   = []
dtreg    = np.zeros([nregions,nmodels+3],dtype=np.float64)  #add three more 'models' for multimodel min / max / range

#go through list of files and read the data
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
for ifile in range(nmodels):
    f = open(files[ifile], 'r')
    print( " processing file : ",files[ifile])
    #get model name
    s    = re.split('_',files[ifile])
    momo = s[6]
    models.append(momo)
    #read the data and range it
    data = list(csv.reader(f, delimiter=","))
    f.close()
    for ireg in range(0,nregions):
        dtreg[ireg,ifile] = float(data[ireg+1][1])
        if (ifile==0): regions.append(data[ireg+1][0])

#file additional two 'model entries' with multimodel min / max
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
imo=nmodels
models.append('MMMin')
for ireg in range(0,nregions): dtreg[ireg,imo] = np.min(dtreg[ireg,0:nmodels])
imo=nmodels+1
models.append('MMMax')
for ireg in range(0,nregions): dtreg[ireg,imo] = np.max(dtreg[ireg,0:nmodels])
imo=nmodels+2
models.append('MMRange')
for ireg in range(0,nregions): dtreg[ireg,imo] = np.max(dtreg[ireg,0:nmodels]) - np.min(dtreg[ireg,0:nmodels])


#plot what you read
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
if (icmap==1): cmap = cmap1;   vmin = vmin1;   vmax = vmax1
if (icmap==2): cmap = cmap2;   vmin = vmin2;   vmax = vmax2
fig,ax = plt.subplots(1,1,figsize=(12,10))
plt.subplots_adjust(left=0.08, bottom=0.15, right=0.98, top=0.97)
#psm    = ax.pcolormesh(dtreg, cmap=cmap, rasterized=True, vmin=vmin, vmax=vmax)
psm    = ax.pcolormesh(dtreg, cmap=cmap, vmin=vmin, vmax=vmax)
cbar   = fig.colorbar(psm, ax=ax)
cbar.ax.tick_params(labelsize=fscb)
cbar.ax.set_ylabel('regional temperature change [degree Celsius]', fontsize=fscb+2)
ax.set_xticks(np.arange(len(models))+0.5, labels=models, rotation=45, ha="right", rotation_mode="anchor", fontsize=fsl)
ax.set_yticks(np.arange(len(regions))+0.5, labels=regions, fontsize=fsl)
#plt.show(block=False)
plt.savefig(figname,format='png')


#produce table for paper: region name, full description, warming in MPI-ESM-LR, in HadGem-ES, min_over_ESMs, max_over_ESMs
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
acr=[];  fre=[]; 
acr.append('GIC');   fre.append('Greenland/Iceland');
acr.append('NWN');   fre.append('N.W.North-America');
acr.append('NEN');   fre.append('N.E.North-America');
acr.append('WNA');   fre.append('W.North-America');
acr.append('CNA');   fre.append('C.North-America');
acr.append('ENA');   fre.append('E.North-America');
acr.append('NCA');   fre.append('N.Central-America');
acr.append('SCA');   fre.append('S.Central-America');
acr.append('CAR');   fre.append('Caribbean');
acr.append('NWS');   fre.append('N.W.South-America');
acr.append('NSA');   fre.append('N.South-America');
acr.append('NES');   fre.append('N.E.South-America');
acr.append('SAM');   fre.append('South-American-Monsoon');
acr.append('SWS');   fre.append('S.W.South-America');
acr.append('SES');   fre.append('S.E.South-America');
acr.append('SSA');   fre.append('S.South-America');
acr.append('NEU');   fre.append('N.Europe');
acr.append('WCE');   fre.append('West\&Central-Europe');
acr.append('EEU');   fre.append('E.Europe');
acr.append('MED');   fre.append('Mediterranean');
acr.append('SAH');   fre.append('Sahara');
acr.append('WAF');   fre.append('Western-Africa');
acr.append('CAF');   fre.append('Central-Africa');
acr.append('NEAF');  fre.append('N.Eastern-Africa');
acr.append('SEAF');  fre.append('S.Eastern-Africa');
acr.append('WSAF');  fre.append('W.Southern-Africa');
acr.append('ESAF');  fre.append('E.Southern-Africa');
acr.append('MDG');   fre.append('Madagascar');
acr.append('RAR');   fre.append('Russian-Arctic');
acr.append('RAR*');  fre.append('Russian-Arctic');
acr.append('WSB');   fre.append('W.Siberia');
acr.append('ESB');   fre.append('E.Siberia');
acr.append('RFE');   fre.append('Russian-Far-East');
acr.append('WCA');   fre.append('W.C.Asia');
acr.append('ECA');   fre.append('E.C.Asia');
acr.append('TIB');   fre.append('Tibetan-Plateau');
acr.append('EAS');   fre.append('E.Asia');
acr.append('ARP');   fre.append('Arabian-Peninsula');
acr.append('SAS');   fre.append('S.Asia');
acr.append('SEA');   fre.append('S.E.Asia');
acr.append('NAU');   fre.append('N.Australia');
acr.append('CAU');   fre.append('C.Australia');
acr.append('EAU');   fre.append('E.Australia');
acr.append('SAU');   fre.append('S.Australia');
acr.append('NZ');    fre.append('New-Zealand');
acr.append('EAN');   fre.append('E.Antarctica');
acr.append('WAN');   fre.append('W.Antarctica');

#find index imo1 of model MPI-ESM-LR
for imo in range(len(models)):
    if (models[imo]=='MPI-ESM-LR'):
        imo1 = imo
                 
#find index imo2 of model HadGEM2-ES
for imo in range(len(models)):
    if (models[imo]=='HadGEM2-ES'):
        imo2 = imo
#go through list of regions as listed in the table in the paper
#search dtreg[iregon,imodel] to find regional warming for that region for models momo1 and momo2, as well as min and max over all models
f = open('TableBody4PaperBetaOnly.txt','w')
nacr=len(acr)
for ir1 in range(nacr):
    for ir2 in range(len(regions)):
        if (regions[ir2]==acr[ir1]): irget=ir2
    dtmo1 = dtreg[irget,imo1]
    dtmo2 = dtreg[irget,imo2]
    dtmin = np.min(dtreg[irget,0:nmodels])
    dtmax = np.max(dtreg[irget,0:nmodels])
    f.write( acr[ir1] + ' & ' + fre[ir1] + ' & ')                                    # col  1 & 2
    f.write( '   %6.4f   &   %6.4f   &    %6.4f   &   %6.4f   \\\ \n' % ( dtmo1,dtmo2,dtmin,dtmax ) )    # col  2
    
f.close()




