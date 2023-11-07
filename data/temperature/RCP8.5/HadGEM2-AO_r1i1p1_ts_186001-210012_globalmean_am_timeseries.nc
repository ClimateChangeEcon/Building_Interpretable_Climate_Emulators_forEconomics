CDF   �   
      time       bnds      lon       lat             CDI       ?Climate Data Interface version 1.9.6 (http://mpimet.mpg.de/cdi)    Conventions       CF-1.4     history      �Sun Feb 28 11:38:57 2021: cdo -s -a yearmean /echam/folini/cmip5/historical_rcp85/full_data//HadGEM2-AO_r1i1p1_ts_186001-210012_globalmean_mm_timeseries.nc /echam/folini/cmip5/historical_rcp85/full_data//HadGEM2-AO_r1i1p1_ts_186001-210012_globalmean_am_timeseries.nc
Sun Feb 28 11:38:54 2021: cdo -s -a fldmean /echam/folini/cmip5/historical_rcp85/full_data//HadGEM2-AO_r1i1p1_ts_186001-210012_fulldata.nc /echam/folini/cmip5/historical_rcp85/full_data//HadGEM2-AO_r1i1p1_ts_186001-210012_globalmean_mm_timeseries.nc
Sun Feb 28 11:38:50 2021: cdo -s mergetime /echam/folini/cmip5/historical/full_data//HadGEM2-AO_r1i1p1_ts_186001-200512_fulldata.nc /echam/folini/cmip5/historical_rcp85/full_data//tmp_merge.nc /echam/folini/cmip5/historical_rcp85/full_data//HadGEM2-AO_r1i1p1_ts_186001-210012_fulldata.nc
Sat May 06 11:36:26 2017: cdo -s -a selvar,ts /echam/folini/cmip5/historical//tmp_01.nc /echam/folini/cmip5/historical//tmp_11.nc
Sat May 06 11:36:22 2017: cdo -s -a mergetime /net/atmos/data/cmip5/historical/Amon/ts/HadGEM2-AO/r1i1p1/ts_Amon_HadGEM2-AO_historical_r1i1p1_186001-200512.nc /echam/folini/cmip5/historical//tmp_01.nc
Output from archive/hadgem2-ao. 2012-08-10T06:32:42Z CMOR rewrote data to comply with CF standards and CMIP5 requirements.     source        �HadGEM2-AO r6.6.3 (2010): atmosphere:  HadGAM (HadGAM2, N96L38); atm: HadGOM (HadGOM2, 1x1L40, increased res at Equator); sea ice: part of HadGOM2; land: MOSES-2      institution       HNIMR (National Institute of Meteorological Research, Seoul, South Korea)   institute_id      NIMR-KMA   experiment_id         
historical     model_id      
HadGEM2-AO     forcing       5Nat, Ant, GHG, SA, Oz, LU, Sl, Vl, SS, Ds, BC, MD, OC      parent_experiment_id      	piControl      branch_time                  contact       Hyo-Shin Lee   
references        �Model described by Johns et al. (J. Clim., 2006, 1327-1353).  Also see http://www.metoffice.gov.uk/research/hadleycentre/pubs/HCTN/HCTN_54.pdf  transient experiments described in Stott et al. (J. Clim., 2006, 2763-2782.)   initialization_method               physics_version             tracking_id       $b57ffce3-f63a-44ca-a31f-6ece1a07b0f0   product       output     
experiment        
historical     	frequency         year   creation_date         2012-08-10T06:32:42Z   
project_id        CMIP5      table_id      :Table Amon (08 July 2010) 0f29d14a72ad86a2466d830072657eac     title         5HadGEM2-AO model output prepared for CMIP5 historical      parent_experiment         pre-industrial control     modeling_realm        atmos      realization             cmor_version      2.0.1      CDO       ?Climate Data Operators version 1.9.6 (http://mpimet.mpg.de/cdo)          time                standard_name         time   	long_name         time   bounds        	time_bnds      units         day as %Y%m%d.%f   calendar      360_day    axis      T           �   	time_bnds                             �   lon                standard_name         	longitude      	long_name         	longitude      units         degrees_east   axis      X           �   lat                standard_name         latitude   	long_name         latitude   units         degrees_north      axis      Y           �   ts                     
   standard_name         surface_temperature    	long_name         Surface Temperature    units         K      
_FillValue        `�x�   missing_value         `�x�   comment       0""skin"" temperature (i.e., SST for open ocean)    original_name         TS     cell_methods      time: mean (interval: 1 month)     history       v2012-08-10T06:32:42Z altered by CMOR: replaced missing value flag (-1.07374e+09) with standard missing value (1e+20).      associated_files      �baseURL: http://cmip-pcmdi.llnl.gov/CMIP5/dataLocation gridspecFile: gridspec_fx_HadGEM2-AO_historical_r0i0p0.nc areacella: areacella_fx_HadGEM2-AO_historical_r0i0p0.nc            �                Aq�/�   Aq�
P   Aq�{P   C�	:Aq���   Aq�{P   Aq��P   C��Aq��   Aq��P   Aq�]P   C��AqĂ�   Aq�]P   Aq��P   C��Aq���   Aq��P   Aq�?P   C�;Aq�d�   Aq�?P   Aq˰P   C��Aq���   Aq˰P   Aq�!P   C��Aq�F�   Aq�!P   AqВP   C�Aqз�   AqВP   Aq�P   C��Aq�(�   Aq�P   Aq�tP   C��Aqՙ�   Aq�tP   Aq��P   C�BAq�
�   Aq��P   Aq�VP   C�Aq�{�   Aq�VP   Aq��P   C��Aq���   Aq��P   Aq�8P   C�Aq�]�   Aq�8P   Aq�P   C��Aq���   Aq�P   Aq�P   C�zAq�?�   Aq�P   Aq�P   C��dAq��   Aq�P   Aq��P   C�7Aq�!�   Aq��P   Aq�mP   C�Aq��   Aq�mP   Aq��P   C�!Aq��   Aq��P   Aq�OP   C��Aq�t�   Aq�OP   Aq��P   C�%�Aq���   Aq��P   Aq�1P   C�$�Aq�V�   Aq�1P   Aq��P   C�"Aq���   Aq��P   Aq�P   C��0Aq�8�   Aq�P   Aq��P   C���Aq���   Aq��P   Aq��P   C�dAq��   Aq��P   ArfP   C��Ar��   ArfP   Ar�P   C���Ar��   Ar�P   ArHP   C���Arm�   ArHP   Ar�P   C��Ar��   Ar�P   Ar*P   C���ArO�   Ar*P   Ar�P   C���Ar��   Ar�P   ArP   C��bAr1�   ArP   Ar}P   C���Ar��   Ar}P   Ar�P   C� �Ar�   Ar�P   Ar_P   C��GAr��   Ar_P   Ar�P   C��Ar��   Ar�P   ArAP   C��Arf�   ArAP   Ar�P   C��Ar��   Ar�P   Ar!#P   C�Ar!H�   Ar!#P   Ar#�P   C��Ar#��   Ar#�P   Ar&P   C���Ar&*�   Ar&P   Ar(vP   C���Ar(��   Ar(vP   Ar*�P   C��(Ar+�   Ar*�P   Ar-XP   C���Ar-}�   Ar-XP   Ar/�P   C� >Ar/��   Ar/�P   Ar2:P   C��Ar2_�   Ar2:P   Ar4�P   C�JAr4��   Ar4�P   Ar7P   C�RAr7A�   Ar7P   Ar9�P   C� Ar9��   Ar9�P   Ar;�P   C�Ar<#�   Ar;�P   Ar>oP   C��Ar>��   Ar>oP   Ar@�P   C���ArA�   Ar@�P   ArCQP   C��]ArCv�   ArCQP   ArE�P   C�)ArE��   ArE�P   ArH3P   C���ArHX�   ArH3P   ArJ�P   C��ArJ��   ArJ�P   ArMP   C��ArM:�   ArMP   ArO�P   C�%�ArO��   ArO�P   ArQ�P   C�/�ArR�   ArQ�P   ArThP   C��ArT��   ArThP   ArV�P   C�'"ArV��   ArV�P   ArYJP   C�(�ArYo�   ArYJP   Ar[�P   C�;Ar[��   Ar[�P   Ar^,P   C��Ar^Q�   Ar^,P   Ar`�P   C��Ar`��   Ar`�P   ArcP   C�+6Arc3�   ArcP   AreP   C��Are��   AreP   Arg�P   C� �Arh�   Arg�P   ArjaP   C��Arj��   ArjaP   Arl�P   C�
Arl��   Arl�P   AroCP   C�&�Aroh�   AroCP   Arq�P   C�JArq��   Arq�P   Art%P   C�AArtJ�   Art%P   Arv�P   C�_Arv��   Arv�P   AryP   C�&WAry,�   AryP   Ar{xP   C�2Ar{��   Ar{xP   Ar}�P   C�,sAr~�   Ar}�P   Ar�ZP   C�/�Ar��   Ar�ZP   Ar��P   C�)�Ar���   Ar��P   Ar�<P   C�6�Ar�a�   Ar�<P   Ar��P   C�6)Ar���   Ar��P   Ar�P   C�#Ar�C�   Ar�P   Ar��P   C�xAr���   Ar��P   Ar� P   C�)?Ar�%�   Ar� P   Ar�qP   C�/�Ar���   Ar�qP   Ar��P   C�7�Ar��   Ar��P   Ar�SP   C�3�Ar�x�   Ar�SP   Ar��P   C�=�Ar���   Ar��P   Ar�5P   C�<.Ar�Z�   Ar�5P   Ar��P   C�2�Ar���   Ar��P   Ar�P   C�.�Ar�<�   Ar�P   Ar��P   C�,(Ar���   Ar��P   Ar��P   C��Ar��   Ar��P   Ar�jP   C��Ar���   Ar�jP   Ar��P   C�%8Ar� �   Ar��P   Ar�LP   C�1bAr�q�   Ar�LP   Ar��P   C�/�Ar���   Ar��P   Ar�.P   C�2�Ar�S�   Ar�.P   Ar��P   C�+�Ar���   Ar��P   Ar�P   C�#,Ar�5�   Ar�P   Ar��P   C�(}Ar���   Ar��P   Ar��P   C��Ar��   Ar��P   Ar�cP   C���Ar���   Ar�cP   Ar��P   C��Ar���   Ar��P   Ar�EP   C��Ar�j�   Ar�EP   ArĶP   C��Ar���   ArĶP   Ar�'P   C�
PAr�L�   Ar�'P   ArɘP   C�Arɽ�   ArɘP   Ar�	P   C�'Ar�.�   Ar�	P   Ar�zP   C�ArΟ�   Ar�zP   Ar��P   C��Ar��   Ar��P   Ar�\P   C��ArӁ�   Ar�\P   Ar��P   C��Ar���   Ar��P   Ar�>P   C��Ar�c�   Ar�>P   ArگP   C��Ar���   ArگP   Ar� P   C��Ar�E�   Ar� P   ArߑP   C��Ar߶�   ArߑP   Ar�P   C� �Ar�'�   Ar�P   Ar�sP   C�,�Ar��   Ar�sP   Ar��P   C�4SAr�	�   Ar��P   Ar�UP   C� �Ar�z�   Ar�UP   Ar��P   C�lAr���   Ar��P   Ar�7P   C�Ar�\�   Ar�7P   Ar�P   C�E�Ar���   Ar�P   Ar�P   C�;/Ar�>�   Ar�P   Ar��P   C�8�Ar���   Ar��P   Ar��P   C�H�Ar� �   Ar��P   Ar�lP   C�H�Ar���   Ar�lP   Ar��P   C�J#Ar��   Ar��P   Ar�NP   C�6VAr�s�   Ar�NP   As�P   C�$�As��   As�P   As0P   C�0�AsU�   As0P   As�P   C�-As��   As�P   As	P   C�*�As	7�   As	P   As�P   C�@�As��   As�P   As�P   C�YxAs�   As�P   AseP   C�WnAs��   AseP   As�P   C�`As��   As�P   AsGP   C�a�Asl�   AsGP   As�P   C�n�As��   As�P   As)P   C�z�AsN�   As)P   As�P   C���As��   As�P   AsP   C���As0�   AsP   As!|P   C���As!��   As!|P   As#�P   C���As$�   As#�P   As&^P   C�ylAs&��   As&^P   As(�P   C�q�As(��   As(�P   As+@P   C�b�As+e�   As+@P   As-�P   C�ulAs-��   As-�P   As0"P   C�!As0G�   As0"P   As2�P   C��SAs2��   As2�P   As5P   C�~�As5)�   As5P   As7uP   C���As7��   As7uP   As9�P   C���As:�   As9�P   As<WP   C�}As<|�   As<WP   As>�P   C�p�As>��   As>�P   AsA9P   C�v�AsA^�   AsA9P   AsC�P   C��'AsC��   AsC�P   AsFP   C��VAsF@�   AsFP   AsH�P   C���AsH��   AsH�P   AsJ�P   C��AsK"�   AsJ�P   AsMnP   C���AsM��   AsMnP   AsO�P   C��[AsP�   AsO�P   AsRPP   C��AsRu�   AsRPP   AsT�P   C���AsT��   AsT�P   AsW2P   C��AsWW�   AsW2P   AsY�P   C��wAsY��   AsY�P   As\P   C���As\9�   As\P   As^�P   C��RAs^��   As^�P   As`�P   C���Asa�   As`�P   AscgP   C���Asc��   AscgP   Ase�P   C��Ase��   Ase�P   AshIP   C���Ashn�   AshIP   Asj�P   C��UAsj��   Asj�P   Asm+P   C�ʤAsmP�   Asm+P   Aso�P   C��"Aso��   Aso�P   AsrP   C���Asr2�   AsrP   Ast~P   C���Ast��   Ast~P   Asv�P   C� �Asw�   Asv�P   Asy`P   C��Asy��   Asy`P   As{�P   C��tAs{��   As{�P   As~BP   C��As~g�   As~BP   As��P   C���As���   As��P   As�$P   C�.As�I�   As�$P   As��P   C�4As���   As��P   As�P   C�As�+�   As�P   As�wP   C�!�As���   As�wP   As��P   C��As��   As��P   As�YP   C��As�~�   As�YP   As��P   C�)VAs���   As��P   As�;P   C�-�As�`�   As�;P   As��P   C�C�As���   As��P   As�P   C�JlAs�B�   As�P   As��P   C�b�As���   As��P   As��P   C�X~As�$�   As��P   As�pP   C�h	As���   As�pP   As��P   C�f�As��   As��P   As�RP   C�e�As�w�   As�RP   As��P   C�xYAs���   As��P   As�4P   C���As�Y�   As�4P   As��P   C��KAs���   As��P   As�P   C���As�;�   As�P   As��P   C��As���   As��P   As��P   C��WAs��   As��P   As�iP   C���As���   As�iP   As��P   C���As���   As��P   As�KP   C��TAs�p�   As�KP   As��P   C���As���   As��P   As�-P   C���As�R�   As�-P   AsP   C��[As���   AsP   As�P   C��FAs�4�   As�P   AsǀP   C���Asǥ�   AsǀP   As��P   C���As��   As��P   As�bP   C��Aṡ�   As�bP   As��P   C��,As���   As��P   As�DP   C��As�i�   As�DP   AsӵP   C�GAs���   AsӵP   As�&P   C�uAs�K�   As�&P   AsؗP   C�$Asؼ�   AsؗP   As�P   C� hAs�-�   As�P   As�yP   C��Asݞ�   As�yP   As��P   C� �As��   As��P   As�[P   C�+oAs��   As�[P   As��P   C�%DAs���   As��P   As�=P   C�7LAs�b�   As�=P   As�P   C�)yAs���   As�P   As�P   C�)-As�D�   As�P   As�P   C�1�As��   As�P   As�P   C�N�As�&�   As�P   As�rP   C�bEAs��   As�rP   As��P   C�fzAs��   As��P   As�TP   C�i�As�y�   As�TP   As��P   C�j�As���   As��P   As�6P   C�v�As�[�   As�6P   As��P   C��
As���   As��P   AtP   C���At=�   AtP   At�P   C���At��   At�P   At�P   C��At�   At�P   At	kP   C��f