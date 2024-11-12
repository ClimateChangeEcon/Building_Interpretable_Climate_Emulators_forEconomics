CDF   �   
      time       bnds      lon       lat             CDI       ?Climate Data Interface version 1.9.6 (http://mpimet.mpg.de/cdi)    Conventions       CF-1.4     history      �Sun Feb 28 11:42:40 2021: cdo -s -a yearmean /echam/folini/cmip5/historical_rcp85/full_data//MIROC5_r1i1p1_ts_185001-210012_globalmean_mm_timeseries.nc /echam/folini/cmip5/historical_rcp85/full_data//MIROC5_r1i1p1_ts_185001-210012_globalmean_am_timeseries.nc
Sun Feb 28 11:42:36 2021: cdo -s -a fldmean /echam/folini/cmip5/historical_rcp85/full_data//MIROC5_r1i1p1_ts_185001-210012_fulldata.nc /echam/folini/cmip5/historical_rcp85/full_data//MIROC5_r1i1p1_ts_185001-210012_globalmean_mm_timeseries.nc
Sun Feb 28 11:42:31 2021: cdo -s mergetime /echam/folini/cmip5/historical/full_data//MIROC5_r1i1p1_ts_185001-201212_fulldata.nc /echam/folini/cmip5/historical_rcp85/full_data//tmp_merge.nc /echam/folini/cmip5/historical_rcp85/full_data//MIROC5_r1i1p1_ts_185001-210012_fulldata.nc
Sat May 06 11:38:02 2017: cdo -s -a selvar,ts /echam/folini/cmip5/historical//tmp_01.nc /echam/folini/cmip5/historical//tmp_11.nc
Sat May 06 11:37:56 2017: cdo -s -a mergetime /net/atmos/data/cmip5/historical/Amon/ts/MIROC5/r1i1p1/ts_Amon_MIROC5_historical_r1i1p1_185001-201212.nc /echam/folini/cmip5/historical//tmp_01.nc
2011-04-23T09:00:51Z CMOR rewrote data to comply with CF standards and CMIP5 requirements.     source        �MIROC5 2010 atmosphere: MIROC-AGCM6 (T85L40); ocean: COCO (COCO4.5, 256x224 L50); sea ice: COCO (COCO4.5); land: MATSIRO (MATSIRO, L6); aerosols: SPRINTARS (SPRINTARS 5.00, T85L40)   institution       �AORI (Atmosphere and Ocean Research Institute, The University of Tokyo, Chiba, Japan), NIES (National Institute for Environmental Studies, Ibaraki, Japan), JAMSTEC (Japan Agency for Marine-Earth Science and Technology, Kanagawa, Japan)    institute_id      MIROC      experiment_id         
historical     model_id      MIROC5     forcing       �GHG, SA, Oz, LU, Sl, Vl, SS, Ds, BC, MD, OC (GHG includes CO2, N2O, methane, and fluorocarbons; Oz includes OH and H2O2; LU excludes change in lake fraction)      parent_experiment_id      	piControl      parent_experiment_rip         r1i1p1     branch_time       AO�       contact       �Masahiro Watanabe (hiro@aori.u-tokyo.ac.jp), Seita Emori (emori@nies.go.jp), Masayoshi Ishii (ism@jamstec.go.jp), Masahide Kimoto (kimoto@aori.u-tokyo.ac.jp)      
references        �Watanabe et al., 2010: Improved climate simulation by MIROC5: Mean states, variability, and climate sensitivity. J. Climate, 23, 6312-6335     initialization_method               physics_version             tracking_id       $079c2104-a2d0-46c9-8958-89f5ab465a40   product       output     
experiment        
historical     	frequency         year   creation_date         2011-04-23T09:00:51Z   
project_id        CMIP5      table_id      =Table Amon (31 January 2011) 53b766a395ac41696af40aab76a49ae5      title         1MIROC5 model output prepared for CMIP5 historical      parent_experiment         pre-industrial control     modeling_realm        atmos      realization             cmor_version      2.5.6      CDO       ?Climate Data Operators version 1.9.6 (http://mpimet.mpg.de/cdo)          time                standard_name         time   	long_name         time   bounds        	time_bnds      units         day as %Y%m%d.%f   calendar      365_day    axis      T           (   	time_bnds                             0   lon                standard_name         	longitude      	long_name         	longitude      units         degrees_east   axis      X              lat                standard_name         latitude   	long_name         latitude   units         degrees_north      axis      Y               ts                     
   standard_name         surface_temperature    	long_name         Surface Temperature    units         K      
_FillValue        `�x�   missing_value         `�x�   comment       0""skin"" temperature (i.e., SST for open ocean)    original_name         GRTS       cell_methods      time: mean     history       �2011-04-23T09:00:51Z altered by CMOR: replaced missing value flag (-999) with standard missing value (1e+20). 2011-04-23T09:00:51Z altered by CMOR: Inverted axis: lat.    associated_files      �baseURL: http://cmip-pcmdi.llnl.gov/CMIP5/dataLocation gridspecFile: gridspec_atmos_fx_MIROC5_historical_r0i0p0.nc areacella: areacella_fx_MIROC5_historical_r0i0p0.nc          @                Aq���   Aq��P   Aq�P   C�*Aq�6�   Aq�P   Aq��P   C�)�Aq���   Aq��P   Aq��P   C�3�Aq��   Aq��P   Aq�dP   C�=/Aq���   Aq�dP   Aq��P   C�)�Aq���   Aq��P   Aq�FP   C�"�Aq�k�   Aq�FP   Aq��P   C�(Aq���   Aq��P   Aq�(P   C�"Aq�M�   Aq�(P   Aq��P   C�"IAq���   Aq��P   Aq�
P   C�Aq�/�   Aq�
P   Aq�{P   C�(6Aq���   Aq�{P   Aq��P   C�'�Aq��   Aq��P   Aq�]P   C�/$AqĂ�   Aq�]P   Aq��P   C�8Aq���   Aq��P   Aq�?P   C��Aq�d�   Aq�?P   Aq˰P   C�5Aq���   Aq˰P   Aq�!P   C��Aq�F�   Aq�!P   AqВP   C�)�Aqз�   AqВP   Aq�P   C�@yAq�(�   Aq�P   Aq�tP   C�@�Aqՙ�   Aq�tP   Aq��P   C�*dAq�
�   Aq��P   Aq�VP   C�4@Aq�{�   Aq�VP   Aq��P   C�I,Aq���   Aq��P   Aq�8P   C�A�Aq�]�   Aq�8P   Aq�P   C�"Aq���   Aq�P   Aq�P   C�"�Aq�?�   Aq�P   Aq�P   C�(sAq��   Aq�P   Aq��P   C�D�Aq�!�   Aq��P   Aq�mP   C�E�Aq��   Aq�mP   Aq��P   C��Aq��   Aq��P   Aq�OP   C��Aq�t�   Aq�OP   Aq��P   C�RAq���   Aq��P   Aq�1P   C�'�Aq�V�   Aq�1P   Aq��P   C�1�Aq���   Aq��P   Aq�P   C�&	Aq�8�   Aq�P   Aq��P   C��Aq���   Aq��P   Aq��P   C�`Aq��   Aq��P   ArfP   C�AAr��   ArfP   Ar�P   C��Ar��   Ar�P   ArHP   C�A7Arm�   ArHP   Ar�P   C�PiAr��   Ar�P   Ar*P   C�%�ArO�   Ar*P   Ar�P   C��Ar��   Ar�P   ArP   C��Ar1�   ArP   Ar}P   C�&Ar��   Ar}P   Ar�P   C�4�Ar�   Ar�P   Ar_P   C�7�Ar��   Ar_P   Ar�P   C�&�Ar��   Ar�P   ArAP   C��Arf�   ArAP   Ar�P   C�*�Ar��   Ar�P   Ar!#P   C�/�Ar!H�   Ar!#P   Ar#�P   C�(�Ar#��   Ar#�P   Ar&P   C��Ar&*�   Ar&P   Ar(vP   C��Ar(��   Ar(vP   Ar*�P   C��Ar+�   Ar*�P   Ar-XP   C�*Ar-}�   Ar-XP   Ar/�P   C�(�Ar/��   Ar/�P   Ar2:P   C�9�Ar2_�   Ar2:P   Ar4�P   C�;�Ar4��   Ar4�P   Ar7P   C�5ZAr7A�   Ar7P   Ar9�P   C�)�Ar9��   Ar9�P   Ar;�P   C�1�Ar<#�   Ar;�P   Ar>oP   C�7xAr>��   Ar>oP   Ar@�P   C�(wArA�   Ar@�P   ArCQP   C�ArCv�   ArCQP   ArE�P   C�-�ArE��   ArE�P   ArH3P   C�-rArHX�   ArH3P   ArJ�P   C�&ArJ��   ArJ�P   ArMP   C�2�ArM:�   ArMP   ArO�P   C�1ArO��   ArO�P   ArQ�P   C�-�ArR�   ArQ�P   ArThP   C�@<ArT��   ArThP   ArV�P   C�L/ArV��   ArV�P   ArYJP   C�BbArYo�   ArYJP   Ar[�P   C�2�Ar[��   Ar[�P   Ar^,P   C�-XAr^Q�   Ar^,P   Ar`�P   C�5LAr`��   Ar`�P   ArcP   C�6WArc3�   ArcP   AreP   C�3�Are��   AreP   Arg�P   C�=qArh�   Arg�P   ArjaP   C�KWArj��   ArjaP   Arl�P   C�fArl��   Arl�P   AroCP   C�o�Aroh�   AroCP   Arq�P   C�+QArq��   Arq�P   Art%P   C��ArtJ�   Art%P   Arv�P   C�AVArv��   Arv�P   AryP   C�?:Ary,�   AryP   Ar{xP   C�H�Ar{��   Ar{xP   Ar}�P   C�JZAr~�   Ar}�P   Ar�ZP   C�FAr��   Ar�ZP   Ar��P   C�EAr���   Ar��P   Ar�<P   C�4�Ar�a�   Ar�<P   Ar��P   C�3�Ar���   Ar��P   Ar�P   C�G-Ar�C�   Ar�P   Ar��P   C�G�Ar���   Ar��P   Ar� P   C�=#Ar�%�   Ar� P   Ar�qP   C�D{Ar���   Ar�qP   Ar��P   C�S�Ar��   Ar��P   Ar�SP   C�RuAr�x�   Ar�SP   Ar��P   C�C�Ar���   Ar��P   Ar�5P   C�RnAr�Z�   Ar�5P   Ar��P   C�lAr���   Ar��P   Ar�P   C�u�Ar�<�   Ar�P   Ar��P   C�B
Ar���   Ar��P   Ar��P   C�1_Ar��   Ar��P   Ar�jP   C�CAr���   Ar�jP   Ar��P   C�KmAr� �   Ar��P   Ar�LP   C�O�Ar�q�   Ar�LP   Ar��P   C�YdAr���   Ar��P   Ar�.P   C�h_Ar�S�   Ar�.P   Ar��P   C�y0Ar���   Ar��P   Ar�P   C�` Ar�5�   Ar�P   Ar��P   C�)�Ar���   Ar��P   Ar��P   C�'�Ar��   Ar��P   Ar�cP   C�"�Ar���   Ar�cP   Ar��P   C�GLAr���   Ar��P   Ar�EP   C�_�Ar�j�   Ar�EP   ArĶP   C�-Ar���   ArĶP   Ar�'P   C��Ar�L�   Ar�'P   ArɘP   C��Arɽ�   ArɘP   Ar�	P   C�+�Ar�.�   Ar�	P   Ar�zP   C�)�ArΟ�   Ar�zP   Ar��P   C�%~Ar��   Ar��P   Ar�\P   C�!�ArӁ�   Ar�\P   Ar��P   C�3�Ar���   Ar��P   Ar�>P   C�X<Ar�c�   Ar�>P   ArگP   C�`Ar���   ArگP   Ar� P   C�,mAr�E�   Ar� P   ArߑP   C�%LAr߶�   ArߑP   Ar�P   C�9bAr�'�   Ar�P   Ar�sP   C�H�Ar��   Ar�sP   Ar��P   C�I�Ar�	�   Ar��P   Ar�UP   C�1�Ar�z�   Ar�UP   Ar��P   C��Ar���   Ar��P   Ar�7P   C�'�Ar�\�   Ar�7P   Ar�P   C�6�Ar���   Ar�P   Ar�P   C�D�Ar�>�   Ar�P   Ar��P   C�N�Ar���   Ar��P   Ar��P   C�A�Ar� �   Ar��P   Ar�lP   C�G�Ar���   Ar�lP   Ar��P   C�P�Ar��   Ar��P   Ar�NP   C�\FAr�s�   Ar�NP   As�P   C�S�As��   As�P   As0P   C�vNAsU�   As0P   As�P   C�S�As��   As�P   As	P   C�@8As	7�   As	P   As�P   C�Q}As��   As�P   As�P   C�r{As�   As�P   AseP   C�s�As��   AseP   As�P   C�v|As��   As�P   AsGP   C�sqAsl�   AsGP   As�P   C�sOAs��   As�P   As)P   C���AsN�   As)P   As�P   C���As��   As�P   AsP   C��
As0�   AsP   As!|P   C���As!��   As!|P   As#�P   C��XAs$�   As#�P   As&^P   C���As&��   As&^P   As(�P   C��AAs(��   As(�P   As+@P   C���As+e�   As+@P   As-�P   C���As-��   As-�P   As0"P   C���As0G�   As0"P   As2�P   C���As2��   As2�P   As5P   C��2As5)�   As5P   As7uP   C�ġAs7��   As7uP   As9�P   C���As:�   As9�P   As<WP   C�|�As<|�   As<WP   As>�P   C��As>��   As>�P   AsA9P   C��nAsA^�   AsA9P   AsC�P   C���AsC��   AsC�P   AsFP   C��$AsF@�   AsFP   AsH�P   C���AsH��   AsH�P   AsJ�P   C��DAsK"�   AsJ�P   AsMnP   C��6AsM��   AsMnP   AsO�P   C��_AsP�   AsO�P   AsRPP   C���AsRu�   AsRPP   AsT�P   C�� AsT��   AsT�P   AsW2P   C���AsWW�   AsW2P   AsY�P   C��AsY��   AsY�P   As\P   C��AAs\9�   As\P   As^�P   C��0As^��   As^�P   As`�P   C��eAsa�   As`�P   AscgP   C���Asc��   AscgP   Ase�P   C���Ase��   Ase�P   AshIP   C��Ashn�   AshIP   Asj�P   C��Asj��   Asj�P   Asm+P   C��@AsmP�   Asm+P   Aso�P   C�ޙAso��   Aso�P   AsrP   C��\Asr2�   AsrP   Ast~P   C�*Ast��   Ast~P   Asv�P   C��Asw�   Asv�P   Asy`P   C��Asy��   Asy`P   As{�P   C�9As{��   As{�P   As~BP   C�2nAs~g�   As~BP   As��P   C�NAs���   As��P   As�$P   C���As�I�   As�$P   As��P   C��As���   As��P   As�P   C�!As�+�   As�P   As�wP   C�1mAs���   As�wP   As��P   C��As��   As��P   As�YP   C��As�~�   As�YP   As��P   C�As���   As��P   As�;P   C�9,As�`�   As�;P   As��P   C�Z�As���   As��P   As�P   C�T�As�B�   As�P   As��P   C�U�As���   As��P   As��P   C�H�As�$�   As��P   As�pP   C�L�As���   As�pP   As��P   C�`xAs��   As��P   As�RP   C�r�As�w�   As�RP   As��P   C���As���   As��P   As�4P   C�\�As�Y�   As�4P   As��P   C�MsAs���   As��P   As�P   C�e�As�;�   As�P   As��P   C��@As���   As��P   As��P   C��?As��   As��P   As�iP   C��As���   As�iP   As��P   C��As���   As��P   As�KP   C�|!As�p�   As�KP   As��P   C�}�As���   As��P   As�-P   C���As�R�   As�-P   AsP   C��As���   AsP   As�P   C��As�4�   As�P   AsǀP   C���Asǥ�   AsǀP   As��P   C���As��   As��P   As�bP   C���Aṡ�   As�bP   As��P   C��,As���   As��P   As�DP   C��^As�i�   As�DP   AsӵP   C���As���   AsӵP   As�&P   C��fAs�K�   As�&P   AsؗP   C�ƘAsؼ�   AsؗP   As�P   C���As�-�   As�P   As�yP   C��0Asݞ�   As�yP   As��P   C���As��   As��P   As�[P   C��wAs��   As�[P   As��P   C�ӒAs���   As��P   As�=P   C��3As�b�   As�=P   As�P   C��As���   As�P   As�P   C�
HAs�D�   As�P   As�P   C��=As��   As�P   As�P   C�<As�&�   As�P   As�rP   C�!.As��   As�rP   As��P   C�MAs��   As��P   As�TP   C��@As�y�   As�TP   As��P   C���As���   As��P   As�6P   C�NAs�[�   As�6P   As��P   C�)As���   As��P   AtP   C�C�At=�   AtP   At�P   C�Z�At��   At�P   At�P   C�7�At�   At�P   At	kP   C�8M