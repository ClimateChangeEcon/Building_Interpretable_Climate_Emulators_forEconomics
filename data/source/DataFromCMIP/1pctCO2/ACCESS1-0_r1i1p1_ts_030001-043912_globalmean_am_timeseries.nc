CDF   �   
      time       bnds      lon       lat             CDI       ?Climate Data Interface version 1.9.6 (http://mpimet.mpg.de/cdi)    Conventions       CF-1.4     history      �Mon Mar 01 11:20:37 2021: cdo -s -a yearmean -fldmean /echam/folini/cmip5/1pctCO2//ACCESS1-0_r1i1p1_ts_030001-043912_fulldata.nc /echam/folini/cmip5/1pctCO2//ACCESS1-0_r1i1p1_ts_030001-043912_globalmean_am_timeseries.nc
Mon Mar 01 11:20:32 2021: cdo -s -a selvar,ts /echam/folini/cmip5/1pctCO2//tmp_01.nc /echam/folini/cmip5/1pctCO2//tmp_11.nc
Mon Mar 01 11:20:29 2021: cdo -s -a mergetime /net/atmos/data/cmip5/1pctCO2/Amon/ts/ACCESS1-0/r1i1p1/ts_Amon_ACCESS1-0_1pctCO2_r1i1p1_030001-043912.nc /echam/folini/cmip5/1pctCO2//tmp_01.nc
CMIP5 compliant file produced from raw ACCESS model output using the ACCESS Post-Processor and CMOR2. 2012-01-15T09:56:34Z CMOR rewrote data to comply with CF standards and CMIP5 requirements.     source       ACCESS1-0 2011. Atmosphere: AGCM v1.0 (N96 grid-point, 1.875 degrees EW x approx 1.25 degree NS, 38 levels); ocean: NOAA/GFDL MOM4p1 (nominal 1.0 degree EW x 1.0 degrees NS, tripolar north of 65N, equatorial refinement to 1/3 degree from 10S to 10 N, cosine dependent NS south of 25S, 50 levels); sea ice: CICE4.1 (nominal 1.0 degree EW x 1.0 degrees NS, tripolar north of 65N, equatorial refinement to 1/3 degree from 10S to 10 N, cosine dependent NS south of 25S); land: MOSES2 (1.875 degree EW x 1.25 degree NS, 4 levels    institution       {CSIRO (Commonwealth Scientific and Industrial Research Organisation, Australia), and BOM (Bureau of Meteorology, Australia)    institute_id      	CSIRO-BOM      experiment_id         1pctCO2    model_id      	ACCESS1-0      forcing      GHG, Oz, SA, Sl, Vl, BC, OC, (GHG = CO2, N2O, CH4, CFC11, CFC12, CFC113, HCFC22, HFC125, HFC134a), (Constant Pre-Industrial (1850) forcings for GHGs and aerosols, except with CO2 concentration increasing by 1 percent each year. Using the 1844-1856 average value for solar forcing)   parent_experiment_id      	piControl      parent_experiment_rip         r1i1p1     branch_time       @��p       contact       �The ACCESS wiki: http://wiki.csiro.au/confluence/display/ACCESS/Home. Contact Tony.Hirst@csiro.au regarding the ACCESS coupled climate model. Contact Peter.Uhe@csiro.au regarding ACCESS coupled climate model CMIP5 datasets.    
references        FSee http://wiki.csiro.au/confluence/display/ACCESS/ACCESS+Publications     initialization_method               physics_version             tracking_id       $4c8e74c8-ab7e-4ec5-9b44-1eaca2cf893e   version_number        	v20120115      product       output     
experiment        1 percent per year CO2     	frequency         year   creation_date         2012-01-15T09:56:34Z   
project_id        CMIP5      table_id      ;Table Amon (27 April 2011) 9c851218e3842df9a62ef38b1e2575bb    title         @ACCESS1-0 model output prepared for CMIP5 1 percent per year CO2   parent_experiment         pre-industrial control     modeling_realm        atmos      realization             cmor_version      2.8.0      CDO       ?Climate Data Operators version 1.9.6 (http://mpimet.mpg.de/cdo)          time                standard_name         time   	long_name         time   bounds        	time_bnds      units         day as %Y%m%d.%f   calendar      proleptic_gregorian    axis      T           �   	time_bnds                             �   lon                standard_name         	longitude      	long_name         	longitude      units         degrees_east   axis      X           �   lat                standard_name         latitude   	long_name         latitude   units         degrees_north      axis      Y           �   ts                     
   standard_name         surface_temperature    	long_name         Surface Temperature    units         K      
_FillValue        `�x�   missing_value         `�x�   comment       0""skin"" temperature (i.e., SST for open ocean)    cell_methods      time: mean     cell_measures         area: areacella    history       v2012-01-15T09:56:34Z altered by CMOR: replaced missing value flag (-1.07374e+09) with standard missing value (1e+20).      associated_files      �baseURL: http://cmip-pcmdi.llnl.gov/CMIP5/dataLocation gridspecFile: gridspec_atmos_fx_ACCESS1-0_1pctCO2_r0i0p0.nc areacella: areacella_fx_ACCESS1-0_1pctCO2_r0i0p0.nc          �                AF侠   AF㒀   AF��   C��9AF�F�   AF��   AG
��   C���AGΠ   AG
��   AG*�   C��;AGV�   AG*�   AG1��   C���AG2ޠ   AG1��   AGE:�   C��AGFf�   AGE:�   AGX   C���AGY�   AGX   AGlJ�   C���AGmv�   AGlJ�   AGҀ   C�`AG���   AGҀ   AG�Z�   C�uAG���   AG�Z�   AG��   C��AG��   AG��   AG�j�   C�AG���   AG�j�   AG��   C��AG��   AG��   AG�z�   C��AG⦠   AG�z�   AG��   C��AG�.�   AG��   AH��   C�EAH	��   AH��   AH�   C�PAH>�   AH�   AH/��   C��AH0Ơ   AH/��   AHC"�   C��AHDN�   AHC"�   AHV��   C�DAHW֠   AHV��   AHj2�   C�(AHk^�   AHj2�   AH}��   C� AH~�   AH}��   AH�B�   C�%�AH�n�   AH�B�   AH�ʀ   C�-AH���   AH�ʀ   AH�R�   C�*�AH�~�   AH�R�   AH�ڀ   C�+�AH��   AH�ڀ   AH�b�   C�#�AH���   AH�b�   AH��   C�(�AH��   AH��   AIr�   C�/DAI��   AIr�   AI��   C�/[AI&�   AI��   AI-��   C�;�AI.��   AI-��   AIA
�   C�<�AIB6�   AIA
�   AIT��   C�B�AIU��   AIT��   AIh�   C�&�AIiF�   AIh�   AI{��   C�:�AI|Π   AI{��   AI�*�   C�C�AI�V�   AI�*�   AI���   C�W�AI�ޠ   AI���   AI�:�   C�[RAI�f�   AI�:�   AI�   C�^�AI��   AI�   AI�J�   C�`�AI�v�   AI�J�   AI�Ҁ   C�r�AI���   AI�Ҁ   AJZ�   C�m�AJ��   AJZ�   AJ�   C�w:AJ�   AJ�   AJ+j�   C�p{AJ,��   AJ+j�   AJ>�   C�o�AJ@�   AJ>�   AJRz�   C�d�AJS��   AJRz�   AJf�   C�w	AJg.�   AJf�   AJy��   C�m�AJz��   AJy��   AJ��   C���AJ�>�   AJ��   AJ���   C�|�AJ�Ơ   AJ���   AJ�"�   C��eAJ�N�   AJ�"�   AJǪ�   C��[AJ�֠   AJǪ�   AJ�2�   C�{�AJ�^�   AJ�2�   AJ   C���AJ��   AJ   AKB�   C���AKn�   AKB�   AKʀ   C���AK��   AKʀ   AK)R�   C���AK*~�   AK)R�   AK<ڀ   C���AK>�   AK<ڀ   AKPb�   C���AKQ��   AKPb�   AKc�   C���AKe�   AKc�   AKwr�   C��VAKx��   AKwr�   AK���   C���AK�&�   AK���   AK���   C���AK���   AK���   AK�
�   C���AK�6�   AK�
�   AKŒ�   C���AKƾ�   AKŒ�   AK��   C���AK�F�   AK��   AK좀   C�ȍAK�Π   AK좀   AL *�   C��ALV�   AL *�   AL��   C��LALޠ   AL��   AL':�   C��=AL(f�   AL':�   AL:   C��AL;�   AL:   ALNJ�   C���ALOv�   ALNJ�   ALaҀ   C�ۯALb��   ALaҀ   ALuZ�   C��yALv��   ALuZ�   AL��   C��TAL��   AL��   AL�j�   C���AL���   AL�j�   AL��   C���AL��   AL��   AL�z�   C��ALĦ�   AL�z�   AL��   C�%sAL�.�   AL��   ALꊀ   C��AL붠   ALꊀ   AL��   C��AL�>�   AL��   AM��   C�6AMƠ   AM��   AM%"�   C�	�AM&N�   AM%"�   AM8��   C�AM9֠   AM8��   AML2�   C�?AMM^�   AML2�   AM_��   C�HAM`�   AM_��   AMsB�   C�3�AMtn�   AMsB�   AM�ʀ   C�5[AM���   AM�ʀ   AM�R�   C�*�AM�~�   AM�R�   AM�ڀ   C�O�AM��   AM�ڀ   AM�b�   C�FLAM�   AM�b�   AM��   C�FNAM��   AM��   AM�r�   C�P)AM鞠   AM�r�   AM���   C�])AM�&�   AM���   AN��   C�FAAN��   AN��   AN#
�   C�X�AN$6�   AN#
�   AN6��   C�h�AN7��   AN6��   ANJ�   C�^5ANKF�   ANJ�   AN]��   C�m^AN^Π   AN]��   ANq*�   C�d�ANrV�   ANq*�   AN���   C�k�AN�ޠ   AN���   AN�:�   C�s$AN�f�   AN�:�   AN�   C�~�AN��   AN�   AN�J�   C��xAN�v�   AN�J�   AN�Ҁ   C�}�AN���   AN�Ҁ   AN�Z�   C���AN熠   AN�Z�   AN��   C���AN��   AN��   AOj�   C���AO��   AOj�   AO �   C��>AO"�   AO �   AO4z�   C��AO5��   AO4z�   AOH�   C��lAOI.�   AOH�   AO[��   C��sAO\��   AO[��   AOo�   C��HAOp>�   AOo�   AO���   C���AO�Ơ   AO���   AO�"�   C��-AO�N�   AO�"�   AO���   C��qAO�֠   AO���   AO�2�   C���AO�^�   AO�2�   AOк�   C���AO��   AOк�   AO�B�   C��AO�n�   AO�B�   AO�ʀ   C��}AO���   AO�ʀ   AP�@   C��/AP?P   AP�@   APm@   C���APP   APm@   AP1@   C���AP�P   AP1@   AP"�@   C��*AP#�P   AP"�@   AP,�@   C��GAP-OP   AP,�@   AP6}@   C���AP7P   AP6}@   AP@A@   C��AP@�P   AP@A@   APJ@   C���APJ�P   APJ@   APS�@   C���APT_P   APS�@   AP]�@   C��AP^#P   AP]�@   APgQ@   C��APg�P   APgQ@   APq@   C�:APq�P   APq@   APz�@   C��AP{oP   APz�@   AP��@   C�tAP�3P   AP��@   AP�a@   C�'�AP��P   AP�a@   AP�%@   C�(AP��P   AP�%@   AP��@   C�,�AP�P   AP��@   AP��@   C�!�AP�CP   AP��@   AP�q@   C�EWAP�P   AP�q@   AP�5@   C�I+AP��P   AP�5@   AP��@   C�Q�