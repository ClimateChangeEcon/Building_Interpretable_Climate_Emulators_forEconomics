CDF   �   
      time       bnds      lon       lat             CDI       ?Climate Data Interface version 1.9.6 (http://mpimet.mpg.de/cdi)    Conventions       CF-1.4     history      jMon Mar 01 11:24:51 2021: cdo -s -a yearmean -fldmean /echam/folini/cmip5/1pctCO2//MPI-ESM-LR_r1i1p1_ts_185001-199912_fulldata.nc /echam/folini/cmip5/1pctCO2//MPI-ESM-LR_r1i1p1_ts_185001-199912_globalmean_am_timeseries.nc
Mon Mar 01 11:24:47 2021: cdo -s -a selvar,ts /echam/folini/cmip5/1pctCO2//tmp_01.nc /echam/folini/cmip5/1pctCO2//tmp_11.nc
Mon Mar 01 11:24:44 2021: cdo -s -a mergetime /net/atmos/data/cmip5/1pctCO2/Amon/ts/MPI-ESM-LR/r1i1p1/ts_Amon_MPI-ESM-LR_1pctCO2_r1i1p1_185001-198912.nc /net/atmos/data/cmip5/1pctCO2/Amon/ts/MPI-ESM-LR/r1i1p1/ts_Amon_MPI-ESM-LR_1pctCO2_r1i1p1_199001-199912.nc /echam/folini/cmip5/1pctCO2//tmp_01.nc
Model raw output postprocessing with modelling environment (IMDI) at DKRZ: URL: http://svn-mad.zmaw.de/svn/mad/Model/IMDI/trunk, REV: 3208 2011-05-31T14:07:28Z CMOR rewrote data to comply with CF standards and CMIP5 requirements.     source        �MPI-ESM-LR 2011; URL: http://svn.zmaw.de/svn/cosmos/branches/releases/mpi-esm-cmip5/src/mod; atmosphere: ECHAM6 (REV: 4571), T63L47; land: JSBACH (REV: 4571); ocean: MPIOM (REV: 4571), GR15L40; sea ice: 4571; marine bgc: HAMOCC (REV: 4571);   institution       $Max Planck Institute for Meteorology   institute_id      MPI-M      experiment_id         1pctCO2    model_id      
MPI-ESM-LR     forcing       N/A    parent_experiment_id      	piControl      parent_experiment_rip         r1i1p1     branch_time       @�f�       contact       cmip5-mpi-esm@dkrz.de      
references       �ECHAM6: n/a; JSBACH: Raddatz et al., 2007. Will the tropical land biosphere dominate the climate-carbon cycle feedback during the twenty first century? Climate Dynamics, 29, 565-574, doi 10.1007/s00382-007-0247-8;  MPIOM: Marsland et al., 2003. The Max-Planck-Institute global ocean/sea ice model with orthogonal curvilinear coordinates. Ocean Modelling, 5, 91-127;  HAMOCC: http://www.mpimet.mpg.de/fileadmin/models/MPIOM/HAMOCC5.1_TECHNICAL_REPORT.pdf;     initialization_method               physics_version             tracking_id       $db1259bb-66f8-45ba-9841-1bacd1c95716   product       output     
experiment        1 percent per year CO2     	frequency         year   creation_date         2011-05-31T14:07:35Z   
project_id        CMIP5      table_id      ;Table Amon (27 April 2011) a5a1c518f52ae340313ba0aada03f862    title         AMPI-ESM-LR model output prepared for CMIP5 1 percent per year CO2      parent_experiment         pre-industrial control     modeling_realm        atmos      realization             cmor_version      2.5.9      CDO       ?Climate Data Operators version 1.9.6 (http://mpimet.mpg.de/cdo)          time                standard_name         time   	long_name         time   bounds        	time_bnds      units         day as %Y%m%d.%f   calendar      proleptic_gregorian    axis      T           �   	time_bnds                             �   lon                standard_name         	longitude      	long_name         	longitude      units         degrees_east   axis      X           �   lat                standard_name         latitude   	long_name         latitude   units         degrees_north      axis      Y           �   ts                     	   standard_name         surface_temperature    	long_name         Surface Temperature    units         K      
_FillValue        `�x�   missing_value         `�x�   comment       0""skin"" temperature (i.e., SST for open ocean)    cell_methods      time: mean     cell_measures         area: areacella    associated_files      �baseURL: http://cmip-pcmdi.llnl.gov/CMIP5/dataLocation gridspecFile: gridspec_atmos_fx_MPI-ESM-LR_1pctCO2_r0i0p0.nc areacella: areacella_fx_MPI-ESM-LR_1pctCO2_r0i0p0.nc                             Aq���   Aq��P   Aq�P   C��Aq�6�   Aq�P   Aq��P   C��Aq���   Aq��P   Aq��P   C���Aq��   Aq��P   Aq�dP   C��Aq���   Aq�dP   Aq��P   C��zAq���   Aq��P   Aq�FP   C�ˁAq�k�   Aq�FP   Aq��P   C��'Aq���   Aq��P   Aq�(P   C��+Aq�M�   Aq�(P   Aq��P   C��Aq���   Aq��P   Aq�
P   C��xAq�/�   Aq�
P   Aq�{P   C��WAq���   Aq�{P   Aq��P   C���Aq��   Aq��P   Aq�]P   C���AqĂ�   Aq�]P   Aq��P   C��EAq���   Aq��P   Aq�?P   C���Aq�d�   Aq�?P   Aq˰P   C���Aq���   Aq˰P   Aq�!P   C�ɣAq�F�   Aq�!P   AqВP   C���Aqз�   AqВP   Aq�P   C���Aq�(�   Aq�P   Aq�tP   C��QAqՙ�   Aq�tP   Aq��P   C�پAq�
�   Aq��P   Aq�VP   C��xAq�{�   Aq�VP   Aq��P   C��Aq���   Aq��P   Aq�8P   C��Aq�]�   Aq�8P   Aq�P   C� Aq���   Aq�P   Aq�P   C�yAq�?�   Aq�P   Aq�P   C�8Aq��   Aq�P   Aq��P   C���Aq�!�   Aq��P   Aq�mP   C��ZAq��   Aq�mP   Aq��P   C��Aq��   Aq��P   Aq�OP   C�@Aq�t�   Aq�OP   Aq��P   C��Aq���   Aq��P   Aq�1P   C�Aq�V�   Aq�1P   Aq��P   C��Aq���   Aq��P   Aq�P   C�aAq�8�   Aq�P   Aq��P   C��Aq���   Aq��P   Aq��P   C�!�Aq��   Aq��P   ArfP   C�=�Ar��   ArfP   Ar�P   C�/�Ar��   Ar�P   ArHP   C�-�Arm�   ArHP   Ar�P   C�B�Ar��   Ar�P   Ar*P   C�A4ArO�   Ar*P   Ar�P   C�@�Ar��   Ar�P   ArP   C�Q Ar1�   ArP   Ar}P   C�9cAr��   Ar}P   Ar�P   C�!�Ar�   Ar�P   Ar_P   C�7�Ar��   Ar_P   Ar�P   C�F<Ar��   Ar�P   ArAP   C�IRArf�   ArAP   Ar�P   C�mAr��   Ar�P   Ar!#P   C�|QAr!H�   Ar!#P   Ar#�P   C�q�Ar#��   Ar#�P   Ar&P   C�x�Ar&*�   Ar&P   Ar(vP   C�u%Ar(��   Ar(vP   Ar*�P   C�}	Ar+�   Ar*�P   Ar-XP   C�m0Ar-}�   Ar-XP   Ar/�P   C�{@Ar/��   Ar/�P   Ar2:P   C�z�Ar2_�   Ar2:P   Ar4�P   C�fjAr4��   Ar4�P   Ar7P   C��?Ar7A�   Ar7P   Ar9�P   C��cAr9��   Ar9�P   Ar;�P   C��%Ar<#�   Ar;�P   Ar>oP   C���Ar>��   Ar>oP   Ar@�P   C���ArA�   Ar@�P   ArCQP   C��ArCv�   ArCQP   ArE�P   C���ArE��   ArE�P   ArH3P   C��mArHX�   ArH3P   ArJ�P   C��OArJ��   ArJ�P   ArMP   C��oArM:�   ArMP   ArO�P   C��,ArO��   ArO�P   ArQ�P   C��=ArR�   ArQ�P   ArThP   C���ArT��   ArThP   ArV�P   C��QArV��   ArV�P   ArYJP   C�±ArYo�   ArYJP   Ar[�P   C��GAr[��   Ar[�P   Ar^,P   C��PAr^Q�   Ar^,P   Ar`�P   C���Ar`��   Ar`�P   ArcP   C��#Arc3�   ArcP   AreP   C��6Are��   AreP   Arg�P   C��Arh�   Arg�P   ArjaP   C���Arj��   ArjaP   Arl�P   C�� Arl��   Arl�P   AroCP   C��Aroh�   AroCP   Arq�P   C��PArq��   Arq�P   Art%P   C���ArtJ�   Art%P   Arv�P   C��kArv��   Arv�P   AryP   C�5Ary,�   AryP   Ar{xP   C�Ar{��   Ar{xP   Ar}�P   C���Ar~�   Ar}�P   Ar�ZP   C� JAr��   Ar�ZP   Ar��P   C��Ar���   Ar��P   Ar�<P   C�'5Ar�a�   Ar�<P   Ar��P   C�%]Ar���   Ar��P   Ar�P   C�!GAr�C�   Ar�P   Ar��P   C��Ar���   Ar��P   Ar� P   C�.mAr�%�   Ar� P   Ar�qP   C�68Ar���   Ar�qP   Ar��P   C�@�Ar��   Ar��P   Ar�SP   C�2�Ar�x�   Ar�SP   Ar��P   C�CEAr���   Ar��P   Ar�5P   C�d�Ar�Z�   Ar�5P   Ar��P   C�}kAr���   Ar��P   Ar�P   C�yuAr�<�   Ar�P   Ar��P   C�rAr���   Ar��P   Ar��P   C�U4Ar��   Ar��P   Ar�jP   C�OAr���   Ar�jP   Ar��P   C�V�Ar� �   Ar��P   Ar�LP   C�j�Ar�q�   Ar�LP   Ar��P   C���Ar���   Ar��P   Ar�.P   C���Ar�S�   Ar�.P   Ar��P   C���Ar���   Ar��P   Ar�P   C���Ar�5�   Ar�P   Ar��P   C�qzAr���   Ar��P   Ar��P   C�nbAr��   Ar��P   Ar�cP   C�kAr���   Ar�cP   Ar��P   C��cAr���   Ar��P   Ar�EP   C���Ar�j�   Ar�EP   ArĶP   C��Ar���   ArĶP   Ar�'P   C���Ar�L�   Ar�'P   ArɘP   C���Arɽ�   ArɘP   Ar�	P   C���Ar�.�   Ar�	P   Ar�zP   C���ArΟ�   Ar�zP   Ar��P   C��'Ar��   Ar��P   Ar�\P   C�ӥArӁ�   Ar�\P   Ar��P   C��OAr���   Ar��P   Ar�>P   C���Ar�c�   Ar�>P   ArگP   C���Ar���   ArگP   Ar� P   C��KAr�E�   Ar� P   ArߑP   C��Ar߶�   ArߑP   Ar�P   C���Ar�'�   Ar�P   Ar�sP   C��2Ar��   Ar�sP   Ar��P   C��Ar�	�   Ar��P   Ar�UP   C��Ar�z�   Ar�UP   Ar��P   C��Ar���   Ar��P   Ar�7P   C�
�Ar�\�   Ar�7P   Ar�P   C�	2Ar���   Ar�P   Ar�P   C��sAr�>�   Ar�P   Ar��P   C�Ar���   Ar��P   Ar��P   C�*�Ar� �   Ar��P   Ar�lP   C��Ar���   Ar�lP   Ar��P   C�-!Ar��   Ar��P   Ar�NP   C�9�Ar�s�   Ar�NP   As�P   C�PfAs��   As�P   As0P   C�C}AsU�   As0P   As�P   C�@TAs��   As�P   As	P   C�0�As	7�   As	P   As�P   C�<As��   As�P   As�P   C�gMAs�   As�P   AseP   C�\�As��   AseP   As�P   C�W