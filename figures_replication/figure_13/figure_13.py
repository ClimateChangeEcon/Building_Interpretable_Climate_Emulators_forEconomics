#
# Figure 13 
#
# python script to illustrate change in TFP (red-white-blue colors, red indicating increas in TFP) as function of
# present (x-axis 1991-2020 mean) and future (y-axis, 2100) temperature for selected South American cities.
#
# - script is stand alone, data is hard wired in the script
# - damage function inspired by Krusell & Smith (2022) [KS]
# - temperature change from CMIP5 models HadGem and MPI, anchored at ERA5 data
# - actual temperatures today (1991 - 2020 mean) and in the future (2100) hard wired in this plotting script
#
# output: Figure 13
#
# To run: exec(open("./Figure13.py").read())
#------------------------------------------------------------------------------------------------------------

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

#font size for annotation
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
fsl  = 20                                    #axis labels
fscb = 20                                    #colorbar labels
fsa  = 20                                    #ticks
figname = 'figure_13.png'                     #name of output figure file
cmap = 'bwr'                                 #colormap
vmin = -0.3                                  #colormap minimum
vmax = +0.3                                  #colormap maximum

#temperature grid
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
tmin =  4.  #minumum temperature [must comprise today and future]
tmax = 31.  #maximum temperature [must comprise today and future]
tcha =  3.  #maximum temperature change [assuming there is no negative temperature change...]
dt   = 0.1  #temperature step size [resolution of temperature change signal]

#KS damages, formulas and constants:
#D(T) = (1.-d)*np.exp(-1.0*kap_p*np.square(T-Tstar)))+d  #if T>T*
#D(T) = (1.-d)*np.exp(-1.0*kap_m*np.square(T-Tstar)))+d  #if T>T*
#RD(T) = D(T)/D(T0)  #damage ratio, 'future divided by today'
kap_p = 0.00311  #kappa plus
kap_m = 0.00456  #kappa minus
ddd   =  0.02    #lower bound on D(T)
Tstar = 11.58    #opt.temp

#specify city temperatures, TN_* temperature now, TF_* temperature in the future
TC_Era=[];  TC_Had=[];  TC_MPI=[];  TC_name=[]
TC_Era.append('21.21');    TC_Had.append('23.04');   TC_MPI.append('23.19');   TC_name.append('Sao Paulo')
TC_Era.append('16.61');    TC_Had.append('17.73');   TC_MPI.append('17.97');   TC_name.append('Buenos Aires')
TC_Era.append('22.24');    TC_Had.append('23.60');   TC_MPI.append('23.71');   TC_name.append('Rio de Janeiro')
TC_Era.append('17.34');    TC_Had.append('18.74');   TC_MPI.append('19.43');   TC_name.append('Lima')
TC_Era.append('20.45');    TC_Had.append('22.00');   TC_MPI.append('22.44');   TC_name.append('Bogota')
TC_Era.append('9.67');     TC_Had.append('11.13');   TC_MPI.append('11.54');   TC_name.append('Santiago')
TC_Era.append('27.03');    TC_Had.append('28.80');   TC_MPI.append('29.17');   TC_name.append('Caracas')
TC_Era.append('21.94');    TC_Had.append('23.60');   TC_MPI.append('24.03');   TC_name.append('Quito')
TC_Era.append('11.63');    TC_Had.append('13.43');   TC_MPI.append('13.97');   TC_name.append('La Paz')
TC_Era.append('24.44');    TC_Had.append('26.38');   TC_MPI.append('26.34');   TC_name.append('Brasilia')
TC_Era.append('20.45');    TC_Had.append('22.00');   TC_MPI.append('22.44');   TC_name.append('Medellin')
TC_Era.append('20.45');    TC_Had.append('21.87');   TC_MPI.append('22.17');   TC_name.append('Guayaquil')
TC_Era.append('22.84');    TC_Had.append('24.51');   TC_MPI.append('25.36');   TC_name.append('Asuncion')
TC_Era.append('16.45');    TC_Had.append('17.34');   TC_MPI.append('17.75');   TC_name.append('Montevideo')
TC_Era.append('18.94');    TC_Had.append('20.52');   TC_MPI.append('20.85');   TC_name.append('Curitiba')
#TC_Era.append('');    TC_Had.append('');   TC_MPI.append('');   TC_name.append('')

#set up matrix for later use with pcolor - the background red-white-blue color of the plot
temp_beg = np.arange(tmin,tmax,dt)  #temperature 'today', base temperature for damage ratio
temp_cha = np.arange(0,tcha,dt)     #future temperature change 
temp_end = np.arange(tmin,tmax,dt)  #temperature in the future', target temperature for damage ratio, temp_beg+temp_cha
ntb      = len(temp_beg)
ntc      = len(temp_cha)
nte      = len(temp_end)

damara_mat = np.zeros((ntc,ntb),dtype=float)  #dimensions are such that I get a plot with ntb entries (present day temp) on the x-axis and ntc temp change on the y-axis

for ib in range(ntb):
    for ie in range(ntc):
        T = temp_beg[ib]
        if (T>=Tstar): DT_b = (1.-ddd)*np.exp(-1.0*kap_p*np.square(T-Tstar))+ddd 
        if (T<Tstar): DT_b = (1.-ddd)*np.exp(-1.0*kap_m*np.square(T-Tstar))+ddd
        T = temp_beg[ib]+temp_cha[ie]
        if (T>=Tstar): DT_e = (1.-ddd)*np.exp(-1.0*kap_p*np.square(T-Tstar))+ddd
        if (T<Tstar): DT_e = (1.-ddd)*np.exp(-1.0*kap_m*np.square(T-Tstar))+ddd
        damara_mat[ie,ib] = DT_e/DT_b-1.

#plot stuff
plt.figure(figsize=(16,8))
plt.pcolor(temp_beg,temp_cha,damara_mat, cmap=cmap, vmin=vmin, vmax=vmax)  #red-white-blue background temperature change coloring
cbar = plt.colorbar()
cbar.ax.tick_params(labelsize=fscb)
cbar.ax.set_ylabel('damages, relative change', fontsize=fscb+2)
#overplot cities
for ic in range(len(TC_Era)):
    plt.scatter(np.array(TC_Era,dtype=float)[ic],np.array(TC_Had,dtype=float)[ic]-np.array(TC_Era,dtype=float)[ic], color = 'black', s=150)  #temperatures from HadGem
    plt.scatter(np.array(TC_Era,dtype=float)[ic],np.array(TC_MPI,dtype=float)[ic]-np.array(TC_Era,dtype=float)[ic], color = '#ffcc00', s=150)  #temperatures from MPI
ilb = 5
ile = 25

plt.xticks(fontsize=fsa)
plt.yticks(fontsize=fsa)
plt.xlabel('mean temperature 1991-2020 [$^\circ$C]',fontsize=fsl)
plt.ylabel('temperature change by 2100 [$^\circ$C]',fontsize=fsl)
plt.title('damages, relative change, selected S. American cities',fontsize=fsl)
#plt.show(block=False)
plt.savefig(figname,format='png')

