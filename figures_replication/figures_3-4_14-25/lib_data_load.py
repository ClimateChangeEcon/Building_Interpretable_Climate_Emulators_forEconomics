import numpy as np
import netCDF4 as nc
import glob
import os
from typing import List, Dict, Tuple, Union



def land_emission(folder_name='/files/Replication_Building_Emulators/data/land_emission/'):

    file_name = folder_name + 'Gasser_et_al_2020_best_guess.nc'
    data = nc.Dataset(file_name)

    return data

def pulse_fraction(test_type, T, conditions:Union['PD','PI']='PD', folder_name:str = '/files/Replication_Building_Emulators/data/pulse/'):


    name = ''

    if conditions =='PD': 
        data = np.loadtxt(folder_name+'pulse_IRF_PD/IRF_PD100_SMOOTHED_CO2.dat') 

        # -----------------------------------
        # PD
        # -----------------------------------
        # COLUMN 1 : YEAR
        # COLUMN 2 : NCAR CSM1.4
        # COLUMN 3 : HadGEM2-ES
        # COLUMN 4 : MPI-ESM
        # COLUMN 5 : Bern3D-LPJ (reference)
        # COLUMN 6 : Bern3D-LPJ (ensemble median)
        # COLUMN 7 : Bern2.5D-LPJ
        # COLUMN 8 : CLIMBER2-LPJ
        # COLUMN 9 : DCESS
        # COLUMN 10 : GENIE (ensemble median)
        # COLUMN 11 : LOVECLIM
        # COLUMN 12 : MESMO
        # COLUMN 13 : UVic2.9
        # COLUMN 14 : ACC2
        # COLUMN 15 : Bern-SAR
        # COLUMN 16 : MAGICC6 (ensemble median)
        # COLUMN 17 : TOTEM2
        # COLUMN 18 : MULTI-MODEL MEAN
        # COLUMN 19 : MULTI-MODEL STDEV


        # [NCAR,HADGEM2,MPIESM,BERN3DR,BERN3DE,BERN25D,CLIMBER2,DCESS,GENIE,LOVECLIM,MESMO,UVIC29,ACC2,BERNSAR,MAGICC6,TOTEM2,MMM,MMMU,MMMD]

        if test_type == 'NCAR':
            pulse_frac = data[:,1]
            name = 'NCAR CSM1.4'
        elif test_type == 'HADGEM2':
            pulse_frac = data[:,2]
            name = 'HadGEM2-ES'
        elif test_type == 'MPIESM':
            pulse_frac = data[:,3]
            name = 'MPI-ESM'
        elif test_type == 'BERN3DR':
            pulse_frac = data[:,4]
            name = 'Bern3D-LPJ Reference'
        elif test_type == 'BERN3DE':
            pulse_frac = data[:,5]
            name = 'Bern3D-LPJ Ensemble'
        elif test_type == 'BERN25D':
            pulse_frac = data[:,6]
            name = 'Bern2.5D-LPJ '
        elif test_type == 'CLIMBER2':
            pulse_frac = data[:,7]
            name = 'CLIMBER2-LPJ'
        elif test_type == 'DCESS':
            pulse_frac = data[:,8]
            name = 'DCESS'
        elif test_type == 'GENIE':
            pulse_frac = data[:,9]
            name = 'GENIE'
        elif test_type == 'LOVECLIM':
            pulse_frac = data[:,10]
            name = 'LOVECLIM'        
        elif test_type == 'MESMO':
            pulse_frac = data[:,11]
            name = 'MESMO'
        elif test_type == 'UVIC29':
            pulse_frac = data[:,12]
            name = 'UVic2.9'
        elif test_type == 'ACC2':
            pulse_frac = data[:,13]
            name = 'ACC2'
        elif test_type == 'BERNSAR':
            pulse_frac = data[:,14]
            name = 'Bern-SAR'
        elif test_type == 'MAGICC6':
            pulse_frac = data[:,15]
            name = 'MAGICC6'
        elif test_type == 'TOTEM2':
            pulse_frac = data[:,16]
            name = 'TOTEM2'
        elif test_type == 'MMM':
            pulse_frac = data[:,17]
            name = '$\mu$'
        elif test_type == 'MMMU':
            pulse_frac = data[:,17] + data[:,18]*2
            name = '$\mu^+$'
        elif test_type == 'MMMD':
            pulse_frac = data[:,17] - data[:,18]*2
            name = '$\mu^-$'
        else:
            raise Exception("Invalid test_type")


    elif conditions =='PI': 
        data = np.loadtxt(folder_name+'pulse_IRF_PI/IRF_PI100_SMOOTHED_CO2.dat') 

        # -----------------------------------
        # PI
        # -----------------------------------
        # COLUMN 1 : YEAR
        # -----------------------------------
        # COLUMN 2 : NCAR CSM1.4
        # COLUMN 3 : Bern3D-LPJ (reference)
        # COLUMN 4 : Bern2.5D-LPJ
        # COLUMN 5 : CLIMBER2-LPJ
        # COLUMN 6 : DCESS
        # COLUMN 7 : GENIE (ensemble median)
        # COLUMN 8 : LOVECLIM
        # COLUMN 9 : MESMO
        # COLUMN 10 : UVic2.9
        # COLUMN 11 : Bern-SAR
        # -----------------------------------
        # COLUMN 12 : MULTI-MODEL MEAN
        # COLUMN 13 : MULTI-MODEL STDEV


        # [NCAR,BERN3D,BERN25D,CLIMBER2,DCESS,GENIE,LOVECLIM,MESMO,UVIC29,BERNSAR,MMM,MMMU,MMMD]

        if test_type == 'NCAR':
            pulse_frac = data[:,1]
            name = 'NCAR CSM1.4'
        elif test_type == 'BERN3D':
            pulse_frac = data[:,2]
            name = 'Bern3D-LPJ'
        elif test_type == 'BERN25D':
            pulse_frac = data[:,3]
            name = 'Bern2.5D-LPJ'
        elif test_type == 'CLIMBER2':
            pulse_frac = data[:,4]
            name = 'CLIMBER2-LPJ'
        elif test_type == 'DCESS':
            pulse_frac = data[:,5]
            name = 'DCESS'
        elif test_type == 'GENIE':
            pulse_frac = data[:,6]
            name = 'GENIE '
            # (ensemble median)
        elif test_type == 'LOVECLIM':
            pulse_frac = data[:,7]
            name = 'LOVECLIM'
        elif test_type == 'MESMO':
            pulse_frac = data[:,8]
            name = 'MESMO'
        elif test_type == 'UVIC29':
            pulse_frac = data[:,9]
            name = 'UVic2.9'
        elif test_type == 'BERNSAR':
            pulse_frac = data[:,10]
            name = 'Bern-SAR'
        elif test_type == 'MMM':
            pulse_frac = data[:,11]
            name = '$\mu$'
        elif test_type == 'MMMU':
            pulse_frac = data[:,11] + data[:,12]*2
            name = '$\mu^+$'
        elif test_type == 'MMMD':
            pulse_frac = data[:,11] - data[:,12]*2
            name = '$\mu^-$'
        else:
            raise Exception("Invalid test_type")

    
    pulse_frac = pulse_frac[pulse_frac<1e10]

    #convert to GTC from concentraitonst
    pulse_frac = pulse_frac *2.12

    # add 100 GtC in the first index
    pulse_frac = np.append([100],pulse_frac) 

    #scale everyting back to %
    pulse_frac = pulse_frac/100

    pulse_frac = pulse_frac[0:T]

    return [pulse_frac,name] 

def cmip_emission(scenerio_name: str, T_start: int = 1765, T_end: int = 2500, folder_name: str = '/files/Replication_Building_Emulators/data/emission/',emission_type:Union['fossil','fossil+land','land'] = 'fossil+land' ):

    assert T_start >= 1765
    assert T_end   <= 2500

    temp = np.loadtxt(folder_name + scenerio_name + '.txt')

    # 0. Year
    # 1. FossilCO2        - Fossil & Industrial CO2 (Fossil	Cement	Gas Flaring & Bunker Fuels)				
    # 2. OtherCO2         - Landuse related CO2 Emissions						
    # 3. CH4              - Methane						
    # 4. N2O              - Nitrous Oxide						
    # 5. - 11.            - Tropospheric ozone precursors	aerosols and reactive gas emissions					
    # 12. - 23.           - Flourinated gases controlled under the Kyoto Protocol	(HFCs	PFCs	SF6)			
    # 24. - 39.           - Ozone Depleting Substances controlled under the Montreal Protocol (CFCs	HFCFC	Halons	CCl4	MCF	CH3Br	CH3Cl)

    inx_time=0
    inx_co2_emission = [None]

    if emission_type=='fossil':
        inx_co2_emission=[1]
    elif emission_type=='fossil+land':
        inx_co2_emission=[1,2]
    elif emission_type=='land':
        inx_co2_emission=[2]
    else:
        raise Exception("Invalid emission_type")

    data_val = np.array(np.sum(temp[:, inx_co2_emission], 1), dtype='float')
    data_year = np.array(temp[:, inx_time], dtype='int')

    # shortcut for finding index of given value
    inx_find = lambda x, val: int(np.where(x == val)[0])
    i_start = inx_find(data_year, int(T_start))
    i_end = inx_find(data_year, int(T_end)) + 1

    data_val = data_val[i_start:i_end]
    data_year = data_year[i_start:i_end]

    return [data_val, data_year]

def cmip_concentration(scenerio_name: str, T_start: int = 1765, T_end: int = 2500,  concentration_type:Union['CO2EQ','KYOTO-CO2EQ','CO2'] = 'CO2' , folder_name: str = '/files/Replication_Building_Emulators/data/concentration/' ):

    assert T_start >= 1765
    assert T_end <= 2500

    temp = np.loadtxt(folder_name + scenerio_name + '.txt')

    #COLUMN_DESCRIPTION________________________________________
    #0. year
    #1. CO2EQ            - CO2 equivalence concentrations using CO2 radiative forcing relationship Q = 3.71/ln(2)*ln(C/278), aggregating all anthropogenic forcings, including greenhouse gases listed below (i.e. columns 3,4,5 and 8-35), and aerosols, trop. ozone etc. (not listed below).
    #2. KYOTO-CO2EQ      - As column 1, but only aggregating greenhouse gases controlled under the Kyoto Protocol (columns 3,4,5 and 8-19).
    #3. CO2              - Atmospheric CO2 concentrations
    #4. CH4              - Atmospheric CH4 concentrations
    #5. N2O              - Atmospheric N2O concentrations
    #6. FGASSUMHFC134AEQ - All flourinated gases controlled under the Kyoto Protocol, i.e. HFCs, PFCs, and SF6 (columns 8-19) expressed as HFC134a equivalence concentrations.
    #7. MHALOSUMCFC12EQ  - All flourinated gases controlled under the Montreal Protocol, i.e. CFCs, HCFCs, Halons, CCl4, CH3Br, CH3Cl (columns 20-35) expressed as CFC-12 equivalence concentrations.
    #8. - 19.            - Flourinated Gases controlled under the Kyoto Protocol
    #20. - 35.           - Ozone Depleting Substances controlled under the Montreal Protocol

    inx_time = 0

    if concentration_type == 'CO2EQ':
        inx_co2_conc = 1
    elif concentration_type == 'KYOTO-CO2EQ':
        inx_co2_conc = 2
    elif concentration_type == 'CO2':
        inx_co2_conc = 3
    else:
        raise Exception("Invalid concentration_type")

    data_val = np.array(temp[:, inx_co2_conc], dtype='float')
    data_year = np.array(temp[:, inx_time], dtype='int')

    # shortcut for finding index of given value
    inx_find = lambda x, val: int(np.where(x == val)[0])

    i_start = inx_find(data_year, int(T_start))
    i_end = inx_find(data_year, int(T_end)) + 1

    data_val = data_val[i_start:i_end]
    data_year = data_year[i_start:i_end]

    return [data_val, data_year]

def cmip_temperature(scenerio_name: str, T_start: int = 1765, T_end: int = 2500, folder_name: str = '/files/Replication_Building_Emulators/data/temperature/'):

    assert T_start >= 1765
    assert T_end <= 2500

    year_min = 1765
    year_max = 2500

    file_name_set = glob.glob(folder_name + scenerio_name + '/' + '*_am_*.nc')

    year_set = []
    val_set  = []
    t_min    = []

    for file_name in file_name_set:
        data = nc.Dataset(file_name)

        year_set += [np.array(np.array(data['time']).flatten() / 1e4, dtype='int')]

        temp =  np.array(data['ts']).flatten()
        temp =  temp - np.mean(temp[0:5])

        val_set  += [temp]

    data_val_full = np.zeros(shape=(len(val_set), year_max - year_min + 1))
    data_year_full = np.array(range(year_min, year_max + 1))

    #print(data_year_full)

    # shortcut for finding index of given value
    inx_find = lambda x, val: int(np.where(x == val)[0])

    for i in range(0, len(data_val_full)):

        i_start = inx_find(data_year_full, min(year_set[i]))
        i_end   = inx_find(data_year_full, max(year_set[i])) + 1

        data_val_full[i, i_start:i_end] = val_set[i]

    i_start = inx_find(data_year_full, int(T_start))
    i_end   = inx_find(data_year_full, int(T_end)) + 1

    data_val  = data_val_full[:, i_start:i_end]
    data_year = data_year_full[i_start:i_end]

    return [data_val, data_year]


def reactive_forcing(T_start: int = 1765, T_end: int = 2500, folder_name: str = '/files/Replication_Building_Emulators/data/forcing/'):


    radiative_forcing_co2_data   = np.loadtxt(folder_name + '/radiative_forcing_co2.txt').T
    radiative_forcing_total_data = np.loadtxt(folder_name + '/radiative_forcing_total.txt').T

    year = radiative_forcing_co2_data[0,:]

    radiative_forcing_co2={}
    radiative_forcing_total={}
    radiative_forcing_ratio={}


    temp_1=0
    temp_2=0
    temp_3=0

    for i,scenerio_name in enumerate(['RCP2.6', 'RCP4.5', 'RCP6.0', 'RCP8.5']):
        
        temp_1 += radiative_forcing_co2_data[i+1,:]
        temp_2 += radiative_forcing_total_data[i+1,:]
        temp_3 += radiative_forcing_total_data[i+1,:]/radiative_forcing_co2_data[i+1,:]

        radiative_forcing_co2[scenerio_name]   = np.interp(np.arange(T_start,T_end+1), year, radiative_forcing_co2_data[i+1,:])
        radiative_forcing_total[scenerio_name] = np.interp(np.arange(T_start,T_end+1), year, radiative_forcing_total_data[i+1,:])
        radiative_forcing_ratio[scenerio_name] = np.interp(np.arange(T_start,T_end+1), year, radiative_forcing_total_data[i+1,:]/radiative_forcing_co2_data[i+1,:])


    radiative_forcing_co2['avg']   =np.interp(np.arange(T_start,T_end+1), year, temp_1/4)
    radiative_forcing_total['avg'] =np.interp(np.arange(T_start,T_end+1), year, temp_2/4)
    radiative_forcing_ratio['avg'] =np.interp(np.arange(T_start,T_end+1), year, temp_3/4)


    return[radiative_forcing_co2,radiative_forcing_ratio,np.arange(T_start,T_end+1)]



# model_name_set=[ 'CanESM5','GFDL-ESM2M','CESM', 'ACCESS','MIROC-ES2L', 'UKESM1', 'MPIESM','NorESM2','CNRM-ESM2-1','UVIC_ESCM','LOVECLIM','MIROC-lite', 'DCESS','CLIMBER2', 'MESM', 'PLASIM-GENIE','Bern','IAPRAS']
def zec_1000(T_start: int = 0, T_end: int = 500, var='co2', folder_name: str = '/files/Replication_Building_Emulators/data/ZEC/', model_name_set=['GFDL-ESM2M', 'ACCESS','MIROC-ES2L', 'UKESM1', 'MPIESM','NorESM2','CNRM-ESM2-1','UVIC_ESCM','LOVECLIM','MIROC-lite', 'DCESS','CLIMBER2', 'MESM', 'PLASIM-GENIE','Bern','IAPRAS']):

    experiment_id = "1pct-brch-1000PgC"

    data=[]

    for model_name in model_name_set:
        file_name = glob.glob(folder_name+model_name+'/' + var + '_*' +experiment_id+ '*')[0]

        try:
            temp = np.loadtxt(file_name)
        except:
            try:
                temp = np.loadtxt(file_name,delimiter=',')
            except:
                temp = np.loadtxt(file_name,skiprows=1,delimiter=',')


        temp = temp[T_start:T_end,1]

        #if var=='tas' or var=='tos':
        temp = temp - temp[0]

        data.append(temp)
   
    return [data,model_name_set]



# model_name_set=[ 'CanESM5','GFDL-ESM2M','CESM', 'ACCESS','MIROC-ES2L', 'UKESM1', 'MPIESM','NorESM2','CNRM-ESM2-1','UVIC_ESCM','LOVECLIM','MIROC-lite', 'DCESS','CLIMBER2', 'MESM', 'PLASIM-GENIE','Bern','IAPRAS']

def zec_1000_cess(T_start: int = 0, T_end: int = 100, var='co2', folder_name: str = '/files/Replication_Building_Emulators/data/ZEC/',model_name_set=['GFDL-ESM2M', 'ACCESS','MIROC-ES2L', 'UKESM1', 'MPIESM','NorESM2','CNRM-ESM2-1','UVIC_ESCM','LOVECLIM','MIROC-lite', 'DCESS','CLIMBER2', 'MESM', 'PLASIM-GENIE','Bern','IAPRAS']):

    # find cecation point
    [data_co2,_]=zec_1000(T_start=0, T_end=500, var='co2', folder_name=folder_name,model_name_set=model_name_set)

    # find cecation point
    [data,_]=zec_1000(T_start=0, T_end=500, var=var, folder_name=folder_name,model_name_set=model_name_set)

    t_cess=[]

    for i in range(0,len(data)):
        temp = np.diff(data_co2[i])[50:]
        t_cess.append(np.argmax(temp<0)+50)

        data[i] = data[i][t_cess[i]:]
        data[i] = data[i][T_start:T_end]


    return [data,t_cess,model_name_set]



def check_error():

    [data_val, data_year] = cmip_temperature(scenerio_name='RCP2.6', T_start=1850, T_end=2023)


if __name__ == "__main__":
    check_error()
