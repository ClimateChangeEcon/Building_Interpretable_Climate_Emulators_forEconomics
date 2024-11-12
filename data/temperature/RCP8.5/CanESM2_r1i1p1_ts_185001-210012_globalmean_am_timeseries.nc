CDF   �   
      time       bnds      lon       lat          !   CDI       ?Climate Data Interface version 1.9.6 (http://mpimet.mpg.de/cdi)    Conventions       CF-1.4     history      �Sun Feb 28 11:28:15 2021: cdo -s -a yearmean /echam/folini/cmip5/historical_rcp85/full_data//CanESM2_r1i1p1_ts_185001-210012_globalmean_mm_timeseries.nc /echam/folini/cmip5/historical_rcp85/full_data//CanESM2_r1i1p1_ts_185001-210012_globalmean_am_timeseries.nc
Sun Feb 28 11:28:14 2021: cdo -s -a fldmean /echam/folini/cmip5/historical_rcp85/full_data//CanESM2_r1i1p1_ts_185001-210012_fulldata.nc /echam/folini/cmip5/historical_rcp85/full_data//CanESM2_r1i1p1_ts_185001-210012_globalmean_mm_timeseries.nc
Sun Feb 28 11:28:13 2021: cdo -s mergetime /echam/folini/cmip5/historical/full_data//CanESM2_r1i1p1_ts_185001-200512_fulldata.nc /echam/folini/cmip5/historical_rcp85/full_data//tmp_merge.nc /echam/folini/cmip5/historical_rcp85/full_data//CanESM2_r1i1p1_ts_185001-210012_fulldata.nc
Sat May 06 11:34:06 2017: cdo -s -a selvar,ts /echam/folini/cmip5/historical//tmp_01.nc /echam/folini/cmip5/historical//tmp_11.nc
Sat May 06 11:34:04 2017: cdo -s -a mergetime /net/atmos/data/cmip5/historical/Amon/ts/CanESM2/r1i1p1/ts_Amon_CanESM2_historical_r1i1p1_185001-200512.nc /echam/folini/cmip5/historical//tmp_01.nc
2011-03-16T18:50:42Z CMOR rewrote data to comply with CF standards and CMIP5 requirements.     source        �CanESM2 2010 atmosphere: CanAM4 (AGCM15i, T63L35) ocean: CanOM4 (OGCM4.0, 256x192L40) and CMOC1.2 sea ice: CanSIM1 (Cavitating Fluid, T63 Gaussian Grid) land: CLASS2.7 and CTEM1      institution       PCCCma (Canadian Centre for Climate Modelling and Analysis, Victoria, BC, Canada)   institute_id      CCCma      experiment_id         
historical     model_id      CanESM2    forcing       IGHG,Oz,SA,BC,OC,LU,Sl,Vl (GHG includes CO2,CH4,N2O,CFC11,effective CFC12)      parent_experiment_id      	piControl      parent_experiment_rip         r1i1p1     branch_time       A�X       contact       cccma_info@ec.gc.ca    
references         http://www.cccma.ec.gc.ca/models   initialization_method               physics_version             tracking_id       $1490335b-daaf-42dc-b307-759309980803   branch_time_YMDH      2321:01:01:00      CCCma_runid       IGM    CCCma_parent_runid        IGA    CCCma_data_licence       �1) GRANT OF LICENCE - The Government of Canada (Environment Canada) is the 
owner of all intellectual property rights (including copyright) that may exist in this Data 
product. You (as "The Licensee") are hereby granted a non-exclusive, non-assignable, 
non-transferable unrestricted licence to use this data product for any purpose including 
the right to share these data with others and to make value-added and derivative 
products from it. This licence is not a sale of any or all of the owner's rights.
2) NO WARRANTY - This Data product is provided "as-is"; it has not been designed or 
prepared to meet the Licensee's particular requirements. Environment Canada makes no 
warranty, either express or implied, including but not limited to, warranties of 
merchantability and fitness for a particular purpose. In no event will Environment Canada 
be liable for any indirect, special, consequential or other damages attributed to the 
Licensee's use of the Data product.    product       output     
experiment        
historical     	frequency         year   creation_date         2011-03-16T18:50:42Z   
project_id        CMIP5      table_id      =Table Amon (31 January 2011) 53b766a395ac41696af40aab76a49ae5      title         2CanESM2 model output prepared for CMIP5 historical     parent_experiment         pre-industrial control     modeling_realm        atmos      realization             cmor_version      2.5.4      CDO       ?Climate Data Operators version 1.9.6 (http://mpimet.mpg.de/cdo)          time                standard_name         time   	long_name         time   bounds        	time_bnds      units         day as %Y%m%d.%f   calendar      365_day    axis      T           �   	time_bnds                             �   lon                standard_name         	longitude      	long_name         	longitude      units         degrees_east   axis      X           t   lat                standard_name         latitude   	long_name         latitude   units         degrees_north      axis      Y           |   ts                     
   standard_name         surface_temperature    	long_name         Surface Temperature    units         K      
_FillValue        `�x�   missing_value         `�x�   comment       0""skin"" temperature (i.e., SST for open ocean)    original_name         GT     cell_methods      "time: mean (interval: 15 minutes)      history       o2011-03-16T18:50:42Z altered by CMOR: replaced missing value flag (1e+38) with standard missing value (1e+20).     associated_files      �baseURL: http://cmip-pcmdi.llnl.gov/CMIP5/dataLocation gridspecFile: gridspec_atmos_fx_CanESM2_historical_r0i0p0.nc areacella: areacella_fx_CanESM2_historical_r0i0p0.nc            �                Aq���   Aq��P   Aq�P   C�s�Aq�6�   Aq�P   Aq��P   C��Aq���   Aq��P   Aq��P   C��Aq��   Aq��P   Aq�dP   C��OAq���   Aq�dP   Aq��P   C���Aq���   Aq��P   Aq�FP   C��gAq�k�   Aq�FP   Aq��P   C���Aq���   Aq��P   Aq�(P   C�nPAq�M�   Aq�(P   Aq��P   C�yoAq���   Aq��P   Aq�
P   C�e�Aq�/�   Aq�
P   Aq�{P   C�|XAq���   Aq�{P   Aq��P   C���Aq��   Aq��P   Aq�]P   C�~�AqĂ�   Aq�]P   Aq��P   C�k�Aq���   Aq��P   Aq�?P   C�j�Aq�d�   Aq�?P   Aq˰P   C�mAq���   Aq˰P   Aq�!P   C�i(Aq�F�   Aq�!P   AqВP   C���Aqз�   AqВP   Aq�P   C��Aq�(�   Aq�P   Aq�tP   C�w?Aqՙ�   Aq�tP   Aq��P   C���Aq�
�   Aq��P   Aq�VP   C���Aq�{�   Aq�VP   Aq��P   C���Aq���   Aq��P   Aq�8P   C��zAq�]�   Aq�8P   Aq�P   C���Aq���   Aq�P   Aq�P   C���Aq�?�   Aq�P   Aq�P   C��Aq��   Aq�P   Aq��P   C���Aq�!�   Aq��P   Aq�mP   C���Aq��   Aq�mP   Aq��P   C��kAq��   Aq��P   Aq�OP   C���Aq�t�   Aq�OP   Aq��P   C��hAq���   Aq��P   Aq�1P   C��TAq�V�   Aq�1P   Aq��P   C�w�Aq���   Aq��P   Aq�P   C�^�Aq�8�   Aq�P   Aq��P   C�P�Aq���   Aq��P   Aq��P   C�F�Aq��   Aq��P   ArfP   C�_-Ar��   ArfP   Ar�P   C�XXAr��   Ar�P   ArHP   C�^�Arm�   ArHP   Ar�P   C�\�Ar��   Ar�P   Ar*P   C�l1ArO�   Ar*P   Ar�P   C��Ar��   Ar�P   ArP   C���Ar1�   ArP   Ar}P   C�{dAr��   Ar}P   Ar�P   C���Ar�   Ar�P   Ar_P   C���Ar��   Ar_P   Ar�P   C��NAr��   Ar�P   ArAP   C�n!Arf�   ArAP   Ar�P   C��/Ar��   Ar�P   Ar!#P   C���Ar!H�   Ar!#P   Ar#�P   C�{nAr#��   Ar#�P   Ar&P   C�hAr&*�   Ar&P   Ar(vP   C�q�Ar(��   Ar(vP   Ar*�P   C�k�Ar+�   Ar*�P   Ar-XP   C�c8Ar-}�   Ar-XP   Ar/�P   C��Ar/��   Ar/�P   Ar2:P   C���Ar2_�   Ar2:P   Ar4�P   C���Ar4��   Ar4�P   Ar7P   C�u�Ar7A�   Ar7P   Ar9�P   C�q&Ar9��   Ar9�P   Ar;�P   C���Ar<#�   Ar;�P   Ar>oP   C���Ar>��   Ar>oP   Ar@�P   C���ArA�   Ar@�P   ArCQP   C�~]ArCv�   ArCQP   ArE�P   C��8ArE��   ArE�P   ArH3P   C��)ArHX�   ArH3P   ArJ�P   C��ArJ��   ArJ�P   ArMP   C��ArM:�   ArMP   ArO�P   C��ArO��   ArO�P   ArQ�P   C��(ArR�   ArQ�P   ArThP   C��RArT��   ArThP   ArV�P   C���ArV��   ArV�P   ArYJP   C��ArYo�   ArYJP   Ar[�P   C���Ar[��   Ar[�P   Ar^,P   C��Ar^Q�   Ar^,P   Ar`�P   C��NAr`��   Ar`�P   ArcP   C��cArc3�   ArcP   AreP   C���Are��   AreP   Arg�P   C��Arh�   Arg�P   ArjaP   C��YArj��   ArjaP   Arl�P   C���Arl��   Arl�P   AroCP   C���Aroh�   AroCP   Arq�P   C���Arq��   Arq�P   Art%P   C��lArtJ�   Art%P   Arv�P   C���Arv��   Arv�P   AryP   C���Ary,�   AryP   Ar{xP   C���Ar{��   Ar{xP   Ar}�P   C��rAr~�   Ar}�P   Ar�ZP   C���Ar��   Ar�ZP   Ar��P   C��/Ar���   Ar��P   Ar�<P   C��*Ar�a�   Ar�<P   Ar��P   C���Ar���   Ar��P   Ar�P   C��~Ar�C�   Ar�P   Ar��P   C��Ar���   Ar��P   Ar� P   C���Ar�%�   Ar� P   Ar�qP   C�~�Ar���   Ar�qP   Ar��P   C���Ar��   Ar��P   Ar�SP   C��9Ar�x�   Ar�SP   Ar��P   C��lAr���   Ar��P   Ar�5P   C��HAr�Z�   Ar�5P   Ar��P   C���Ar���   Ar��P   Ar�P   C��;Ar�<�   Ar�P   Ar��P   C��Ar���   Ar��P   Ar��P   C���Ar��   Ar��P   Ar�jP   C��6Ar���   Ar�jP   Ar��P   C��;Ar� �   Ar��P   Ar�LP   C��7Ar�q�   Ar�LP   Ar��P   C��:Ar���   Ar��P   Ar�.P   C���Ar�S�   Ar�.P   Ar��P   C���Ar���   Ar��P   Ar�P   C��bAr�5�   Ar�P   Ar��P   C���Ar���   Ar��P   Ar��P   C��0Ar��   Ar��P   Ar�cP   C��Ar���   Ar�cP   Ar��P   C��mAr���   Ar��P   Ar�EP   C���Ar�j�   Ar�EP   ArĶP   C�|PAr���   ArĶP   Ar�'P   C���Ar�L�   Ar�'P   ArɘP   C��PArɽ�   ArɘP   Ar�	P   C���Ar�.�   Ar�	P   Ar�zP   C���ArΟ�   Ar�zP   Ar��P   C��&Ar��   Ar��P   Ar�\P   C��FArӁ�   Ar�\P   Ar��P   C���Ar���   Ar��P   Ar�>P   C��!Ar�c�   Ar�>P   ArگP   C���Ar���   ArگP   Ar� P   C���Ar�E�   Ar� P   ArߑP   C��2Ar߶�   ArߑP   Ar�P   C���Ar�'�   Ar�P   Ar�sP   C��xAr��   Ar�sP   Ar��P   C���Ar�	�   Ar��P   Ar�UP   C�ЪAr�z�   Ar�UP   Ar��P   C��Ar���   Ar��P   Ar�7P   C��XAr�\�   Ar�7P   Ar�P   C��|Ar���   Ar�P   Ar�P   C��\Ar�>�   Ar�P   Ar��P   C��vAr���   Ar��P   Ar��P   C���Ar� �   Ar��P   Ar�lP   C�־Ar���   Ar�lP   Ar��P   C��-Ar��   Ar��P   Ar�NP   C�ӳAr�s�   Ar�NP   As�P   C��0As��   As�P   As0P   C��PAsU�   As0P   As�P   C��KAs��   As�P   As	P   C��As	7�   As	P   As�P   C���As��   As�P   As�P   C�FAs�   As�P   AseP   C��As��   AseP   As�P   C��As��   As�P   AsGP   C�� Asl�   AsGP   As�P   C�� As��   As�P   As)P   C��AsN�   As)P   As�P   C�SAs��   As�P   AsP   C�As0�   AsP   As!|P   C�0jAs!��   As!|P   As#�P   C�A�As$�   As#�P   As&^P   C�%�As&��   As&^P   As(�P   C��As(��   As(�P   As+@P   C�G�As+e�   As+@P   As-�P   C�NcAs-��   As-�P   As0"P   C�E�As0G�   As0"P   As2�P   C�!jAs2��   As2�P   As5P   C�%VAs5)�   As5P   As7uP   C�$As7��   As7uP   As9�P   C�I�As:�   As9�P   As<WP   C�^`As<|�   As<WP   As>�P   C�UiAs>��   As>�P   AsA9P   C�iiAsA^�   AsA9P   AsC�P   C�G�AsC��   AsC�P   AsFP   C�X|AsF@�   AsFP   AsH�P   C�^AsH��   AsH�P   AsJ�P   C�oUAsK"�   AsJ�P   AsMnP   C�k�AsM��   AsMnP   AsO�P   C�w�AsP�   AsO�P   AsRPP   C�ywAsRu�   AsRPP   AsT�P   C�~�AsT��   AsT�P   AsW2P   C��PAsWW�   AsW2P   AsY�P   C�r�AsY��   AsY�P   As\P   C�ktAs\9�   As\P   As^�P   C���As^��   As^�P   As`�P   C��xAsa�   As`�P   AscgP   C��VAsc��   AscgP   Ase�P   C��iAse��   Ase�P   AshIP   C���Ashn�   AshIP   Asj�P   C���Asj��   Asj�P   Asm+P   C���AsmP�   Asm+P   Aso�P   C��;Aso��   Aso�P   AsrP   C��Asr2�   AsrP   Ast~P   C��kAst��   Ast~P   Asv�P   C��|Asw�   Asv�P   Asy`P   C��:Asy��   Asy`P   As{�P   C���As{��   As{�P   As~BP   C�ɿAs~g�   As~BP   As��P   C���As���   As��P   As�$P   C���As�I�   As�$P   As��P   C���As���   As��P   As�P   C���As�+�   As�P   As�wP   C�PAs���   As�wP   As��P   C��As��   As��P   As�YP   C�	�As�~�   As�YP   As��P   C�
rAs���   As��P   As�;P   C�As�`�   As�;P   As��P   C�GAs���   As��P   As�P   C��As�B�   As�P   As��P   C�)�As���   As��P   As��P   C��As�$�   As��P   As�pP   C�*As���   As�pP   As��P   C�DGAs��   As��P   As�RP   C�DoAs�w�   As�RP   As��P   C�D*As���   As��P   As�4P   C�W�As�Y�   As�4P   As��P   C�lAs���   As��P   As�P   C�k�As�;�   As�P   As��P   C�o�As���   As��P   As��P   C�[�As��   As��P   As�iP   C�Y1As���   As�iP   As��P   C�~$As���   As��P   As�KP   C���As�p�   As�KP   As��P   C�sXAs���   As��P   As�-P   C�m�As�R�   As�-P   AsP   C���As���   AsP   As�P   C���As�4�   As�P   AsǀP   C��$Asǥ�   AsǀP   As��P   C���As��   As��P   As�bP   C���Aṡ�   As�bP   As��P   C���As���   As��P   As�DP   C��As�i�   As�DP   AsӵP   C��As���   AsӵP   As�&P   C��As�K�   As�&P   AsؗP   C���Asؼ�   AsؗP   As�P   C��As�-�   As�P   As�yP   C��Asݞ�   As�yP   As��P   C���As��   As��P   As�[P   C���As��   As�[P   As��P   C��As���   As��P   As�=P   C��As�b�   As�=P   As�P   C��As���   As�P   As�P   C��As�D�   As�P   As�P   C�3�As��   As�P   As�P   C�1�As�&�   As�P   As�rP   C�&IAs��   As�rP   As��P   C�8As��   As��P   As�TP   C�I�As�y�   As�TP   As��P   C�BAs���   As��P   As�6P   C�HpAs�[�   As�6P   As��P   C�(�As���   As��P   AtP   C�*�At=�   AtP   At�P   C�NtAt��   At�P   At�P   C�p�At�   At�P   At	kP   C��