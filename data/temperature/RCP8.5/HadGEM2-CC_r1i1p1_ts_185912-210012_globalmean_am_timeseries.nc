CDF   �   
      time       bnds      lon       lat             CDI       ?Climate Data Interface version 1.9.6 (http://mpimet.mpg.de/cdi)    Conventions       CF-1.4     history      1Sun Feb 28 11:39:21 2021: cdo -s -a yearmean /echam/folini/cmip5/historical_rcp85/full_data//HadGEM2-CC_r1i1p1_ts_185912-210012_globalmean_mm_timeseries.nc /echam/folini/cmip5/historical_rcp85/full_data//HadGEM2-CC_r1i1p1_ts_185912-210012_globalmean_am_timeseries.nc
Sun Feb 28 11:39:18 2021: cdo -s -a fldmean /echam/folini/cmip5/historical_rcp85/full_data//HadGEM2-CC_r1i1p1_ts_185912-210012_fulldata.nc /echam/folini/cmip5/historical_rcp85/full_data//HadGEM2-CC_r1i1p1_ts_185912-210012_globalmean_mm_timeseries.nc
Sun Feb 28 11:39:14 2021: cdo -s mergetime /echam/folini/cmip5/historical/full_data//HadGEM2-CC_r1i1p1_ts_185912-200511_fulldata.nc /echam/folini/cmip5/historical_rcp85/full_data//tmp_merge.nc /echam/folini/cmip5/historical_rcp85/full_data//HadGEM2-CC_r1i1p1_ts_185912-210012_fulldata.nc
Sat May 06 11:36:38 2017: cdo -s -a selvar,ts /echam/folini/cmip5/historical//tmp_01.nc /echam/folini/cmip5/historical//tmp_11.nc
Sat May 06 11:36:34 2017: cdo -s -a mergetime /net/atmos/data/cmip5/historical/Amon/ts/HadGEM2-CC/r1i1p1/ts_Amon_HadGEM2-CC_historical_r1i1p1_185912-188411.nc /net/atmos/data/cmip5/historical/Amon/ts/HadGEM2-CC/r1i1p1/ts_Amon_HadGEM2-CC_historical_r1i1p1_188412-190911.nc /net/atmos/data/cmip5/historical/Amon/ts/HadGEM2-CC/r1i1p1/ts_Amon_HadGEM2-CC_historical_r1i1p1_190912-193411.nc /net/atmos/data/cmip5/historical/Amon/ts/HadGEM2-CC/r1i1p1/ts_Amon_HadGEM2-CC_historical_r1i1p1_193412-195911.nc /net/atmos/data/cmip5/historical/Amon/ts/HadGEM2-CC/r1i1p1/ts_Amon_HadGEM2-CC_historical_r1i1p1_195912-198411.nc /net/atmos/data/cmip5/historical/Amon/ts/HadGEM2-CC/r1i1p1/ts_Amon_HadGEM2-CC_historical_r1i1p1_198412-200511.nc /echam/folini/cmip5/historical//tmp_01.nc
MOHC pp to CMOR/NetCDF convertor (version 1.10.1) 2011-09-12T09:20:24Z CMOR rewrote data to comply with CF standards and CMIP5 requirements.      source        �HadGEM2-CC (2011) atmosphere: HadGAM2(N96L60); ocean: HadGOM2 (lat: 1.0-0.3 lon: 1.0 L40); land-surface/vegetation: MOSES2 and TRIFFID; ocean biogeochemistry: diat-HadOCC     institution       aMet Office Hadley Centre, Fitzroy Road, Exeter, Devon, EX1 3PB, UK, (http://www.metoffice.gov.uk)      institute_id      MOHC   experiment_id         
historical     model_id      
HadGEM2-CC     forcing       <GHG, Oz, SA, LU, Sl, Vl, BC, OC, (GHG = CO2, N2O, CH4, CFCs)   parent_experiment_id      	piControl      parent_experiment_rip         r1i1p1     branch_time                  contact       @chris.d.jones@metoffice.gov.uk, steven.hardiman@metoffice.gov.uk   
references       FJones, C.D. et al. (2011) The HadGEM2-ES implementation of CMIP5 centennial simulations. Geosci. Model Dev., 4, 543-570, http://www.geosci-model-dev.net/4/543/2011/gmd-4-543-2011.html; Martin G.M. et al. (2011) The HadGEM2 family of Met Office Unified Model climate configurations, Geosci. Model Dev., 4, 723-757, http://www.geosci-model-dev.net/4/723/2011/gmd-4-723-2011.html; Collins, W.J. et al. (2011) Development and evaluation of an Earth-system model - HadGEM2, Geosci. Model Dev. Discuss., 4, 997-1062, http://www.geosci-model-dev-discuss.net/4/997/2011/gmdd-4-997-2011.html     initialization_method               physics_version             tracking_id       $7865d408-0b5d-47a5-8992-7dfa9e8f3a24   mo_runid      akgid      product       output     
experiment        
historical     	frequency         year   creation_date         2011-09-12T09:21:47Z   
project_id        CMIP5      table_id      :Table Amon (26 July 2011) 976b7fd1d9e1be31dddd28f5dc79b7a1     title         5HadGEM2-CC model output prepared for CMIP5 historical      parent_experiment         pre-industrial control     modeling_realm        atmos      realization             cmor_version      2.7.1      CDO       ?Climate Data Operators version 1.9.6 (http://mpimet.mpg.de/cdo)          time                standard_name         time   	long_name         time   bounds        	time_bnds      units         day as %Y%m%d.%f   calendar      360_day    axis      T              	time_bnds                                lon                standard_name         	longitude      	long_name         	longitude      units         degrees_east   axis      X              lat                standard_name         latitude   	long_name         latitude   units         degrees_north      axis      Y              ts                     
   standard_name         surface_temperature    	long_name         Surface Temperature    units         K      
_FillValue        `�x�   missing_value         `�x�   comment       0""skin"" temperature (i.e., SST for open ocean)    original_name         mo: m01s00i024     cell_methods      time: mean     history       v2011-09-12T09:21:47Z altered by CMOR: replaced missing value flag (-1.07374e+09) with standard missing value (1e+20).      associated_files      �baseURL: http://cmip-pcmdi.llnl.gov/CMIP5/dataLocation gridspecFile: gridspec_atmos_fx_HadGEM2-CC_historical_r0i0p0.nc areacella: areacella_fx_HadGEM2-CC_historical_r0i0p0.nc          ,                Aq��    Aq��   Aq�
P   C��^Aq�/�   Aq�
P   Aq�{P   C��Aq���   Aq�{P   Aq��P   C��4Aq��   Aq��P   Aq�]P   C��#AqĂ�   Aq�]P   Aq��P   C�ԗAq���   Aq��P   Aq�?P   C��(Aq�d�   Aq�?P   Aq˰P   C�ݛAq���   Aq˰P   Aq�!P   C��KAq�F�   Aq�!P   AqВP   C��Aqз�   AqВP   Aq�P   C��FAq�(�   Aq�P   Aq�tP   C���Aqՙ�   Aq�tP   Aq��P   C��Aq�
�   Aq��P   Aq�VP   C���Aq�{�   Aq�VP   Aq��P   C��5Aq���   Aq��P   Aq�8P   C��Aq�]�   Aq�8P   Aq�P   C���Aq���   Aq�P   Aq�P   C���Aq�?�   Aq�P   Aq�P   C��yAq��   Aq�P   Aq��P   C���Aq�!�   Aq��P   Aq�mP   C��.Aq��   Aq�mP   Aq��P   C��5Aq��   Aq��P   Aq�OP   C��Aq�t�   Aq�OP   Aq��P   C���Aq���   Aq��P   Aq�1P   C���Aq�V�   Aq�1P   Aq��P   C���Aq���   Aq��P   Aq�P   C��\Aq�8�   Aq�P   Aq��P   C��Aq���   Aq��P   Aq��P   C���Aq��   Aq��P   ArfP   C��Ar��   ArfP   Ar�P   C��	Ar��   Ar�P   ArHP   C��Arm�   ArHP   Ar�P   C���Ar��   Ar�P   Ar*P   C��tArO�   Ar*P   Ar�P   C���Ar��   Ar�P   ArP   C��"Ar1�   ArP   Ar}P   C�ϗAr��   Ar}P   Ar�P   C��lAr�   Ar�P   Ar_P   C���Ar��   Ar_P   Ar�P   C�¤Ar��   Ar�P   ArAP   C���Arf�   ArAP   Ar�P   C���Ar��   Ar�P   Ar!#P   C���Ar!H�   Ar!#P   Ar#�P   C���Ar#��   Ar#�P   Ar&P   C��Ar&*�   Ar&P   Ar(vP   C��YAr(��   Ar(vP   Ar*�P   C��iAr+�   Ar*�P   Ar-XP   C��7Ar-}�   Ar-XP   Ar/�P   C��Ar/��   Ar/�P   Ar2:P   C�ՎAr2_�   Ar2:P   Ar4�P   C��_Ar4��   Ar4�P   Ar7P   C���Ar7A�   Ar7P   Ar9�P   C��OAr9��   Ar9�P   Ar;�P   C���Ar<#�   Ar;�P   Ar>oP   C��Ar>��   Ar>oP   Ar@�P   C��eArA�   Ar@�P   ArCQP   C���ArCv�   ArCQP   ArE�P   C��RArE��   ArE�P   ArH3P   C��.ArHX�   ArH3P   ArJ�P   C��~ArJ��   ArJ�P   ArMP   C��#ArM:�   ArMP   ArO�P   C��{ArO��   ArO�P   ArQ�P   C���ArR�   ArQ�P   ArThP   C��ArT��   ArThP   ArV�P   C��kArV��   ArV�P   ArYJP   C���ArYo�   ArYJP   Ar[�P   C��HAr[��   Ar[�P   Ar^,P   C���Ar^Q�   Ar^,P   Ar`�P   C��HAr`��   Ar`�P   ArcP   C��ZArc3�   ArcP   AreP   C��DAre��   AreP   Arg�P   C��Arh�   Arg�P   ArjaP   C��WArj��   ArjaP   Arl�P   C��2Arl��   Arl�P   AroCP   C��%Aroh�   AroCP   Arq�P   C���Arq��   Arq�P   Art%P   C��GArtJ�   Art%P   Arv�P   C�úArv��   Arv�P   AryP   C���Ary,�   AryP   Ar{xP   C���Ar{��   Ar{xP   Ar}�P   C���Ar~�   Ar}�P   Ar�ZP   C���Ar��   Ar�ZP   Ar��P   C��dAr���   Ar��P   Ar�<P   C�ҟAr�a�   Ar�<P   Ar��P   C���Ar���   Ar��P   Ar�P   C��2Ar�C�   Ar�P   Ar��P   C��rAr���   Ar��P   Ar� P   C�ֶAr�%�   Ar� P   Ar�qP   C�ÞAr���   Ar�qP   Ar��P   C��=Ar��   Ar��P   Ar�SP   C��(Ar�x�   Ar�SP   Ar��P   C�̶Ar���   Ar��P   Ar�5P   C�ӽAr�Z�   Ar�5P   Ar��P   C��Ar���   Ar��P   Ar�P   C���Ar�<�   Ar�P   Ar��P   C���Ar���   Ar��P   Ar��P   C���Ar��   Ar��P   Ar�jP   C�²Ar���   Ar�jP   Ar��P   C���Ar� �   Ar��P   Ar�LP   C��sAr�q�   Ar�LP   Ar��P   C��~Ar���   Ar��P   Ar�.P   C�ɦAr�S�   Ar�.P   Ar��P   C��Ar���   Ar��P   Ar�P   C��Ar�5�   Ar�P   Ar��P   C��GAr���   Ar��P   Ar��P   C�նAr��   Ar��P   Ar�cP   C���Ar���   Ar�cP   Ar��P   C��OAr���   Ar��P   Ar�EP   C���Ar�j�   Ar�EP   ArĶP   C���Ar���   ArĶP   Ar�'P   C���Ar�L�   Ar�'P   ArɘP   C��Arɽ�   ArɘP   Ar�	P   C��?Ar�.�   Ar�	P   Ar�zP   C���ArΟ�   Ar�zP   Ar��P   C��:Ar��   Ar��P   Ar�\P   C��PArӁ�   Ar�\P   Ar��P   C��^Ar���   Ar��P   Ar�>P   C��0Ar�c�   Ar�>P   ArگP   C��*Ar���   ArگP   Ar� P   C���Ar�E�   Ar� P   ArߑP   C��Ar߶�   ArߑP   Ar�P   C��Ar�'�   Ar�P   Ar�sP   C��Ar��   Ar�sP   Ar��P   C���Ar�	�   Ar��P   Ar�UP   C���Ar�z�   Ar�UP   Ar��P   C��Ar���   Ar��P   Ar�7P   C��nAr�\�   Ar�7P   Ar�P   C��@Ar���   Ar�P   Ar�P   C���Ar�>�   Ar�P   Ar��P   C��\Ar���   Ar��P   Ar��P   C�ʪAr� �   Ar��P   Ar�lP   C��8Ar���   Ar�lP   Ar��P   C��5Ar��   Ar��P   Ar�NP   C��eAr�s�   Ar�NP   As�P   C���As��   As�P   As0P   C�ؖAsU�   As0P   As�P   C��As��   As�P   As	P   C�ߵAs	7�   As	P   As�P   C��iAs��   As�P   As�P   C��oAs�   As�P   AseP   C���As��   AseP   As�P   C���As��   As�P   AsGP   C��Asl�   AsGP   As�P   C�As��   As�P   As)P   C���AsN�   As)P   As�P   C��4As��   As�P   AsP   C���As0�   AsP   As!|P   C��BAs!��   As!|P   As#�P   C���As$�   As#�P   As&^P   C��#As&��   As&^P   As(�P   C���As(��   As(�P   As+@P   C�	<As+e�   As+@P   As-�P   C�{As-��   As-�P   As0"P   C�YAs0G�   As0"P   As2�P   C�"�As2��   As2�P   As5P   C�7�As5)�   As5P   As7uP   C�P.As7��   As7uP   As9�P   C�W`As:�   As9�P   As<WP   C�\"As<|�   As<WP   As>�P   C�KAs>��   As>�P   AsA9P   C�?�AsA^�   AsA9P   AsC�P   C�L�AsC��   AsC�P   AsFP   C�TwAsF@�   AsFP   AsH�P   C�]^AsH��   AsH�P   AsJ�P   C�rAAsK"�   AsJ�P   AsMnP   C���AsM��   AsMnP   AsO�P   C���AsP�   AsO�P   AsRPP   C��AsRu�   AsRPP   AsT�P   C���AsT��   AsT�P   AsW2P   C�~�AsWW�   AsW2P   AsY�P   C�~�AsY��   AsY�P   As\P   C���As\9�   As\P   As^�P   C���As^��   As^�P   As`�P   C��tAsa�   As`�P   AscgP   C��3Asc��   AscgP   Ase�P   C��Ase��   Ase�P   AshIP   C���Ashn�   AshIP   Asj�P   C��4Asj��   Asj�P   Asm+P   C��<AsmP�   Asm+P   Aso�P   C���Aso��   Aso�P   AsrP   C��^Asr2�   AsrP   Ast~P   C�åAst��   Ast~P   Asv�P   C���Asw�   Asv�P   Asy`P   C��ZAsy��   Asy`P   As{�P   C��As{��   As{�P   As~BP   C���As~g�   As~BP   As��P   C���As���   As��P   As�$P   C��"As�I�   As�$P   As��P   C�٣As���   As��P   As�P   C��As�+�   As�P   As�wP   C���As���   As�wP   As��P   C���As��   As��P   As�YP   C�As�~�   As�YP   As��P   C�.As���   As��P   As�;P   C�$As�`�   As�;P   As��P   C�!�As���   As��P   As�P   C��As�B�   As�P   As��P   C��As���   As��P   As��P   C�!�As�$�   As��P   As�pP   C�-�As���   As�pP   As��P   C�E+As��   As��P   As�RP   C�T_As�w�   As�RP   As��P   C�b	As���   As��P   As�4P   C�t)As�Y�   As�4P   As��P   C��^As���   As��P   As�P   C���As�;�   As�P   As��P   C���As���   As��P   As��P   C���As��   As��P   As�iP   C���As���   As�iP   As��P   C���As���   As��P   As�KP   C��As�p�   As�KP   As��P   C��As���   As��P   As�-P   C���As�R�   As�-P   AsP   C��/As���   AsP   As�P   C�ҝAs�4�   As�P   AsǀP   C�ΰAsǥ�   AsǀP   As��P   C�ƬAs��   As��P   As�bP   C���Aṡ�   As�bP   As��P   C���As���   As��P   As�DP   C��As�i�   As�DP   AsӵP   C���As���   AsӵP   As�&P   C�۽As�K�   As�&P   AsؗP   C�� Asؼ�   AsؗP   As�P   C���As�-�   As�P   As�yP   C��Asݞ�   As�yP   As��P   C���As��   As��P   As�[P   C���As��   As�[P   As��P   C���As���   As��P   As�=P   C���As�b�   As�=P   As�P   C�	As���   As�P   As�P   C�"�As�D�   As�P   As�P   C�rAs��   As�P   As�P   C�(7As�&�   As�P   As�rP   C�8<As��   As�rP   As��P   C�4PAs��   As��P   As�TP   C�I�As�y�   As�TP   As��P   C�_�As���   As��P   As�6P   C�\MAs�[�   As�6P   As��P   C�W As���   As��P   AtP   C�T�At=�   AtP   At�P   C�W�At��   At�P   At�P   C�`$At�   At�P   At	kP   C�z'