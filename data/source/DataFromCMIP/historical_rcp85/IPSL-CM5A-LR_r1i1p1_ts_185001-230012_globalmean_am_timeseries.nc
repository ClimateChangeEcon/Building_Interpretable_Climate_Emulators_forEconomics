CDF  �   
      time       bnds      lon       lat             CDI       ?Climate Data Interface version 1.9.6 (http://mpimet.mpg.de/cdi)    Conventions       CF-1.4     history      �Sun Feb 28 11:41:45 2021: cdo -s -a yearmean /echam/folini/cmip5/historical_rcp85/full_data//IPSL-CM5A-LR_r1i1p1_ts_185001-230012_globalmean_mm_timeseries.nc /echam/folini/cmip5/historical_rcp85/full_data//IPSL-CM5A-LR_r1i1p1_ts_185001-230012_globalmean_am_timeseries.nc
Sun Feb 28 11:41:43 2021: cdo -s -a fldmean /echam/folini/cmip5/historical_rcp85/full_data//IPSL-CM5A-LR_r1i1p1_ts_185001-230012_fulldata.nc /echam/folini/cmip5/historical_rcp85/full_data//IPSL-CM5A-LR_r1i1p1_ts_185001-230012_globalmean_mm_timeseries.nc
Sun Feb 28 11:41:40 2021: cdo -s mergetime /echam/folini/cmip5/historical/full_data//IPSL-CM5A-LR_r1i1p1_ts_185001-200512_fulldata.nc /echam/folini/cmip5/historical_rcp85/full_data//tmp_merge.nc /echam/folini/cmip5/historical_rcp85/full_data//IPSL-CM5A-LR_r1i1p1_ts_185001-230012_fulldata.nc
Sat May 06 11:36:58 2017: cdo -s -a selvar,ts /echam/folini/cmip5/historical//tmp_01.nc /echam/folini/cmip5/historical//tmp_11.nc
Sat May 06 11:36:57 2017: cdo -s -a mergetime /net/atmos/data/cmip5/historical/Amon/ts/IPSL-CM5A-LR/r1i1p1/ts_Amon_IPSL-CM5A-LR_historical_r1i1p1_185001-200512.nc /echam/folini/cmip5/historical//tmp_01.nc
2011-02-23T17:52:43Z CMOR rewrote data to comply with CF standards and CMIP5 requirements.     source        �IPSL-CM5A-LR (2010) : atmos : LMDZ4 (LMDZ4_v5, 96x95x39); ocean : ORCA2 (NEMOV2_3, 2x2L31); seaIce : LIM2 (NEMOV2_3); ocnBgchem : PISCES (NEMOV2_3); land : ORCHIDEE (orchidee_1_9_4_AR5)      institution       3IPSL (Institut Pierre Simon Laplace, Paris, France)    institute_id      IPSL   experiment_id         
historical     model_id      IPSL-CM5A-LR   forcing       &Nat,Ant,GHG,SA,Oz,LU,SS,Ds,BC,MD,OC,AA     parent_experiment_id      	piControl      parent_experiment_rip         r1i1p1     branch_time       @��        contact       ?ipsl-cmip5 _at_ ipsl.jussieu.fr Data manager : Sebastien Denvil    comment       HThis 20th century simulation include natural and anthropogenic forcings.   
references        NModel documentation and further reference available here : http://icmc.ipsl.fr     initialization_method               physics_version             tracking_id       $f0677dd3-71ce-4f37-83bb-43e50472a4cc   product       output     
experiment        
historical     	frequency         year   creation_date         2011-02-23T17:52:43Z   
project_id        CMIP5      table_id      =Table Amon (31 January 2011) 53b766a395ac41696af40aab76a49ae5      title         7IPSL-CM5A-LR model output prepared for CMIP5 historical    parent_experiment         pre-industrial control     modeling_realm        atmos      realization             cmor_version      2.5.1      CDO       ?Climate Data Operators version 1.9.6 (http://mpimet.mpg.de/cdo)          time                standard_name         time   	long_name         time   bounds        	time_bnds      units         day as %Y%m%d.%f   calendar      365_day    axis      T               	time_bnds                             (   lon                standard_name         	longitude      	long_name         	longitude      units         degrees_east   axis      X              lat                standard_name         latitude   	long_name         latitude   units         degrees_north      axis      Y              ts                     
   standard_name         surface_temperature    	long_name         Surface Temperature    units         K      
_FillValue        `�x�   missing_value         `�x�   comment       0""skin"" temperature (i.e., SST for open ocean)    original_name         tsol       cell_methods      "time: mean (interval: 30 minutes)      history       �2011-02-23T17:52:42Z altered by CMOR: replaced missing value flag (9.96921e+36) with standard missing value (1e+20). 2011-02-23T17:52:43Z altered by CMOR: Inverted axis: lat.     associated_files      �baseURL: http://cmip-pcmdi.llnl.gov/CMIP5/dataLocation gridspecFile: gridspec_atmos_fx_IPSL-CM5A-LR_historical_r0i0p0.nc areacella: areacella_fx_IPSL-CM5A-LR_historical_r0i0p0.nc          8                Aq���   Aq��P   Aq�P   C�;`Aq�6�   Aq�P   Aq��P   C�,nAq���   Aq��P   Aq��P   C�/rAq��   Aq��P   Aq�dP   C�0�Aq���   Aq�dP   Aq��P   C�/�Aq���   Aq��P   Aq�FP   C�+�Aq�k�   Aq�FP   Aq��P   C�XAq���   Aq��P   Aq�(P   C�	ZAq�M�   Aq�(P   Aq��P   C��{Aq���   Aq��P   Aq�
P   C�Aq�/�   Aq�
P   Aq�{P   C��Aq���   Aq�{P   Aq��P   C�(�Aq��   Aq��P   Aq�]P   C�!�AqĂ�   Aq�]P   Aq��P   C��Aq���   Aq��P   Aq�?P   C�2�Aq�d�   Aq�?P   Aq˰P   C�A�Aq���   Aq˰P   Aq�!P   C�2MAq�F�   Aq�!P   AqВP   C��Aqз�   AqВP   Aq�P   C��Aq�(�   Aq�P   Aq�tP   C�*�Aqՙ�   Aq�tP   Aq��P   C�'�Aq�
�   Aq��P   Aq�VP   C�.lAq�{�   Aq�VP   Aq��P   C�$�Aq���   Aq��P   Aq�8P   C�6:Aq�]�   Aq�8P   Aq�P   C�;�Aq���   Aq�P   Aq�P   C�EqAq�?�   Aq�P   Aq�P   C�.�Aq��   Aq�P   Aq��P   C�4�Aq�!�   Aq��P   Aq�mP   C�1�Aq��   Aq�mP   Aq��P   C��Aq��   Aq��P   Aq�OP   C�.#Aq�t�   Aq�OP   Aq��P   C�G�Aq���   Aq��P   Aq�1P   C�0�Aq�V�   Aq�1P   Aq��P   C��Aq���   Aq��P   Aq�P   C��Aq�8�   Aq�P   Aq��P   C���Aq���   Aq��P   Aq��P   C�Aq��   Aq��P   ArfP   C��Ar��   ArfP   Ar�P   C�Ar��   Ar�P   ArHP   C�@Arm�   ArHP   Ar�P   C�Ar��   Ar�P   Ar*P   C��ArO�   Ar*P   Ar�P   C��Ar��   Ar�P   ArP   C��Ar1�   ArP   Ar}P   C�@Ar��   Ar}P   Ar�P   C�+Ar�   Ar�P   Ar_P   C��Ar��   Ar_P   Ar�P   C��Ar��   Ar�P   ArAP   C��Arf�   ArAP   Ar�P   C�
�Ar��   Ar�P   Ar!#P   C��Ar!H�   Ar!#P   Ar#�P   C�VAr#��   Ar#�P   Ar&P   C�4Ar&*�   Ar&P   Ar(vP   C�"fAr(��   Ar(vP   Ar*�P   C�)Ar+�   Ar*�P   Ar-XP   C�5�Ar-}�   Ar-XP   Ar/�P   C�/�Ar/��   Ar/�P   Ar2:P   C�2�Ar2_�   Ar2:P   Ar4�P   C�.�Ar4��   Ar4�P   Ar7P   C�,kAr7A�   Ar7P   Ar9�P   C�+Ar9��   Ar9�P   Ar;�P   C�:�Ar<#�   Ar;�P   Ar>oP   C�+UAr>��   Ar>oP   Ar@�P   C�*dArA�   Ar@�P   ArCQP   C�0:ArCv�   ArCQP   ArE�P   C�23ArE��   ArE�P   ArH3P   C�#dArHX�   ArH3P   ArJ�P   C�3�ArJ��   ArJ�P   ArMP   C�?2ArM:�   ArMP   ArO�P   C�2%ArO��   ArO�P   ArQ�P   C�.7ArR�   ArQ�P   ArThP   C�5�ArT��   ArThP   ArV�P   C�G�ArV��   ArV�P   ArYJP   C�@�ArYo�   ArYJP   Ar[�P   C�=Ar[��   Ar[�P   Ar^,P   C�I�Ar^Q�   Ar^,P   Ar`�P   C�`6Ar`��   Ar`�P   ArcP   C�]Arc3�   ArcP   AreP   C�X�Are��   AreP   Arg�P   C�W�Arh�   Arg�P   ArjaP   C�L�Arj��   ArjaP   Arl�P   C�UArl��   Arl�P   AroCP   C�`�Aroh�   AroCP   Arq�P   C�9�Arq��   Arq�P   Art%P   C�,%ArtJ�   Art%P   Arv�P   C��Arv��   Arv�P   AryP   C�K7Ary,�   AryP   Ar{xP   C�T�Ar{��   Ar{xP   Ar}�P   C�K�Ar~�   Ar}�P   Ar�ZP   C�@�Ar��   Ar�ZP   Ar��P   C�K&Ar���   Ar��P   Ar�<P   C�RLAr�a�   Ar�<P   Ar��P   C�XLAr���   Ar��P   Ar�P   C�]ZAr�C�   Ar�P   Ar��P   C�U�Ar���   Ar��P   Ar� P   C�N�Ar�%�   Ar� P   Ar�qP   C�U�Ar���   Ar�qP   Ar��P   C�X-Ar��   Ar��P   Ar�SP   C�V�Ar�x�   Ar�SP   Ar��P   C�wCAr���   Ar��P   Ar�5P   C�r5Ar�Z�   Ar�5P   Ar��P   C�mAr���   Ar��P   Ar�P   C�P�Ar�<�   Ar�P   Ar��P   C�fsAr���   Ar��P   Ar��P   C�\�Ar��   Ar��P   Ar�jP   C�M�Ar���   Ar�jP   Ar��P   C�X�Ar� �   Ar��P   Ar�LP   C�T�Ar�q�   Ar�LP   Ar��P   C�A�Ar���   Ar��P   Ar�.P   C�X�Ar�S�   Ar�.P   Ar��P   C�Ar���   Ar��P   Ar�P   C�y�Ar�5�   Ar�P   Ar��P   C�X�Ar���   Ar��P   Ar��P   C�J�Ar��   Ar��P   Ar�cP   C�3"Ar���   Ar�cP   Ar��P   C�'�Ar���   Ar��P   Ar�EP   C�;XAr�j�   Ar�EP   ArĶP   C�WkAr���   ArĶP   Ar�'P   C�I�Ar�L�   Ar�'P   ArɘP   C�KcArɽ�   ArɘP   Ar�	P   C�[�Ar�.�   Ar�	P   Ar�zP   C�Z]ArΟ�   Ar�zP   Ar��P   C�YAr��   Ar��P   Ar�\P   C�c�ArӁ�   Ar�\P   Ar��P   C�f�Ar���   Ar��P   Ar�>P   C�]Ar�c�   Ar�>P   ArگP   C�h�Ar���   ArگP   Ar� P   C�k�Ar�E�   Ar� P   ArߑP   C�^ Ar߶�   ArߑP   Ar�P   C�XBAr�'�   Ar�P   Ar�sP   C�Y[Ar��   Ar�sP   Ar��P   C�mpAr�	�   Ar��P   Ar�UP   C�t>Ar�z�   Ar�UP   Ar��P   C�g�Ar���   Ar��P   Ar�7P   C�naAr�\�   Ar�7P   Ar�P   C�]/Ar���   Ar�P   Ar�P   C�v~Ar�>�   Ar�P   Ar��P   C���Ar���   Ar��P   Ar��P   C���Ar� �   Ar��P   Ar�lP   C���Ar���   Ar�lP   Ar��P   C���Ar��   Ar��P   Ar�NP   C��Ar�s�   Ar�NP   As�P   C�{�As��   As�P   As0P   C��AsU�   As0P   As�P   C��}As��   As�P   As	P   C��&As	7�   As	P   As�P   C���As��   As�P   As�P   C��As�   As�P   AseP   C���As��   AseP   As�P   C��SAs��   As�P   AsGP   C��xAsl�   AsGP   As�P   C��`As��   As�P   As)P   C���AsN�   As)P   As�P   C���As��   As�P   AsP   C��fAs0�   AsP   As!|P   C��}As!��   As!|P   As#�P   C��XAs$�   As#�P   As&^P   C��-As&��   As&^P   As(�P   C��9As(��   As(�P   As+@P   C��OAs+e�   As+@P   As-�P   C��As-��   As-�P   As0"P   C���As0G�   As0"P   As2�P   C���As2��   As2�P   As5P   C���As5)�   As5P   As7uP   C�ƧAs7��   As7uP   As9�P   C���As:�   As9�P   As<WP   C�	�As<|�   As<WP   As>�P   C�,As>��   As>�P   AsA9P   C���AsA^�   AsA9P   AsC�P   C�
OAsC��   AsC�P   AsFP   C��@AsF@�   AsFP   AsH�P   C���AsH��   AsH�P   AsJ�P   C���AsK"�   AsJ�P   AsMnP   C�	AsM��   AsMnP   AsO�P   C�AsP�   AsO�P   AsRPP   C��AsRu�   AsRPP   AsT�P   C��AsT��   AsT�P   AsW2P   C��AsWW�   AsW2P   AsY�P   C��AsY��   AsY�P   As\P   C�$dAs\9�   As\P   As^�P   C�.tAs^��   As^�P   As`�P   C�<Asa�   As`�P   AscgP   C�O�Asc��   AscgP   Ase�P   C�8�Ase��   Ase�P   AshIP   C�DgAshn�   AshIP   Asj�P   C�[jAsj��   Asj�P   Asm+P   C�^ AsmP�   Asm+P   Aso�P   C�UgAso��   Aso�P   AsrP   C�V(Asr2�   AsrP   Ast~P   C�T1Ast��   Ast~P   Asv�P   C�Q�Asw�   Asv�P   Asy`P   C�gAsy��   Asy`P   As{�P   C�t�As{��   As{�P   As~BP   C��(As~g�   As~BP   As��P   C���As���   As��P   As�$P   C���As�I�   As�$P   As��P   C��As���   As��P   As�P   C���As�+�   As�P   As�wP   C��dAs���   As�wP   As��P   C���As��   As��P   As�YP   C���As�~�   As�YP   As��P   C��xAs���   As��P   As�;P   C��As�`�   As�;P   As��P   C�ƞAs���   As��P   As�P   C���As�B�   As�P   As��P   C��TAs���   As��P   As��P   C��LAs�$�   As��P   As�pP   C�۔As���   As�pP   As��P   C��rAs��   As��P   As�RP   C���As�w�   As�RP   As��P   C���As���   As��P   As�4P   C���As�Y�   As�4P   As��P   C���As���   As��P   As�P   C�dAs�;�   As�P   As��P   C�	�As���   As��P   As��P   C�)As��   As��P   As�iP   C�*�As���   As�iP   As��P   C��As���   As��P   As�KP   C�(�As�p�   As�KP   As��P   C�H�As���   As��P   As�-P   C�J�As�R�   As�-P   AsP   C�@SAs���   AsP   As�P   C�DfAs�4�   As�P   AsǀP   C�c�Asǥ�   AsǀP   As��P   C�i�As��   As��P   As�bP   C�R[Aṡ�   As�bP   As��P   C�e�As���   As��P   As�DP   C�}�As�i�   As�DP   AsӵP   C��WAs���   AsӵP   As�&P   C��OAs�K�   As�&P   AsؗP   C�z�Asؼ�   AsؗP   As�P   C�}�As�-�   As�P   As�yP   C��9Asݞ�   As�yP   As��P   C���As��   As��P   As�[P   C�ʰAs��   As�[P   As��P   C��}As���   As��P   As�=P   C���As�b�   As�=P   As�P   C��nAs���   As�P   As�P   C���As�D�   As�P   As�P   C��^As��   As�P   As�P   C��As�&�   As�P   As�rP   C���As��   As�rP   As��P   C��As��   As��P   As�TP   C���As�y�   As�TP   As��P   C���As���   As��P   As�6P   C��As�[�   As�6P   As��P   C�"�As���   As��P   AtP   C� �At=�   AtP   At�P   C��At��   At�P   At�P   C�'tAt�   At�P   At	kP   C�%�At	��   At	kP   At�P   C�5�At�   At�P   AtMP   C�1}Atr�   AtMP   At�P   C�$/At��   At�P   At/P   C�2�AtT�   At/P   At�P   C�<At��   At�P   AtP   C�E�At6�   AtP   At�P   C�ZWAt��   At�P   At�P   C�?At�   At�P   AtdP   C�E�At��   AtdP   At!�P   C�P"At!��   At!�P   At$FP   C�`�At$k�   At$FP   At&�P   C�^+At&��   At&�P   At)(P   C�w�At)M�   At)(P   At+�P   C�byAt+��   At+�P   At.
P   C�ipAt./�   At.
P   At0{P   C�j4At0��   At0{P   At2�P   C�|�At3�   At2�P   At5]P   C��\At5��   At5]P   At7�P   C���At7��   At7�P   At:?P   C���At:d�   At:?P   At<�P   C��dAt<��   At<�P   At?!P   C��NAt?F�   At?!P   AtA�P   C��6AtA��   AtA�P   AtDP   C��uAtD(�   AtDP   AtFtP   C���AtF��   AtFtP   AtH�P   C��$AtI
�   AtH�P   AtKVP   C��-AtK{�   AtKVP   AtM�P   C��4AtM��   AtM�P   AtP8P   C��bAtP]�   AtP8P   AtR�P   C�ɅAtR��   AtR�P   AtUP   C�݃AtU?�   AtUP   AtW�P   C�ھAtW��   AtW�P   AtY�P   C��~AtZ!�   AtY�P   At\mP   C��WAt\��   At\mP   At^�P   C��>At_�   At^�P   AtaOP   C���Atat�   AtaOP   Atc�P   C��SAtc��   Atc�P   Atf1P   C�rAtfV�   Atf1P   Ath�P   C��Ath��   Ath�P   AtkP   C�sAtk8�   AtkP   Atm�P   C�%Atm��   Atm�P   Ato�P   C��6Atp�   Ato�P   AtrfP   C��Atr��   AtrfP   Att�P   C�Att��   Att�P   AtwHP   C�RAtwm�   AtwHP   Aty�P   C�#iAty��   Aty�P   At|*P   C�7kAt|O�   At|*P   At~�P   C�O�At~��   At~�P   At�P   C�FsAt�1�   At�P   At�}P   C�ZsAt���   At�}P   At��P   C�^GAt��   At��P   At�_P   C�KAt���   At�_P   At��P   C�dAt���   At��P   At�AP   C�k	At�f�   At�AP   At��P   C�oAt���   At��P   At�#P   C�y�At�H�   At�#P   At��P   C��^At���   At��P   At�P   C�n�At�*�   At�P   At�vP   C���At���   At�vP   At��P   C���At��   At��P   At�XP   C��6At�}�   At�XP   At��P   C���At���   At��P   At�:P   C���At�_�   At�:P   At��P   C���At���   At��P   At�P   C���At�A�   At�P   At��P   C���At���   At��P   At��P   C��*At�#�   At��P   At�oP   C��dAt���   At�oP   At��P   C��AAt��   At��P   At�QP   C�ȈAt�v�   At�QP   At��P   C��At���   At��P   At�3P   C�ͮAt�X�   At�3P   At��P   C���At���   At��P   At�P   C��#At�:�   At�P   At��P   C�ڸAt���   At��P   At��P   C��;At��   At��P   At�hP   C��Atō�   At�hP   At��P   C��,At���   At��P   At�JP   C���At�o�   At�JP   At̻P   C���At���   At̻P   At�,P   C��At�Q�   At�,P   AtѝP   C��At���   AtѝP   At�P   C�At�3�   At�P   At�P   C�FAt֤�   At�P   At��P   C���At��   At��P   At�aP   C�	�Atۆ�   At�aP   At��P   C��At���   At��P   At�CP   C��At�h�   At�CP   At�P   C��At���   At�P   At�%P   C�/aAt�J�   At�%P   At�P   C�)�At��   At�P   At�P   C�.At�,�   At�P   At�xP   C�+UAt��   At�xP   At��P   C�&�At��   At��P   At�ZP   C�7�At��   At�ZP   At��P   C�4gAt���   At��P   At�<P   C�EZAt�a�   At�<P   At��P   C�N�At���   At��P   At�P   C�a�At�C�   At�P   At��P   C�c�At���   At��P   Au  P   C�[�Au %�   Au  P   AuqP   C�\�Au��   AuqP   Au�P   C�e�Au�   Au�P   AuSP   C�d*Aux�   AuSP   Au	�P   C�i=Au	��   Au	�P   Au5P   C�l�AuZ�   Au5P   Au�P   C�eAu��   Au�P   AuP   C�V-Au<�   AuP   Au�P   C�x1Au��   Au�P   Au�P   C�wnAu�   Au�P   AujP   C�tAu��   AujP   Au�P   C�gAu �   Au�P   AuLP   C�tjAuq�   AuLP   Au�P   C��TAu��   Au�P   Au".P   C��bAu"S�   Au".P   Au$�P   C���Au$��   Au$�P   Au'P   C��AAu'5�   Au'P   Au)�P   C��$Au)��   Au)�P   Au+�P   C���Au,�   Au+�P   Au.cP   C���Au.��   Au.cP   Au0�P   C���Au0��   Au0�P   Au3EP   C���Au3j�   Au3EP   Au5�P   C��YAu5��   Au5�P   Au8'P   C��Au8L�   Au8'P   Au:�P   C��Au:��   Au:�P   Au=	P   C��;Au=.�   Au=	P   Au?zP   C�ĮAu?��   Au?zP   AuA�P   C���AuB�   AuA�P   AuD\P   C��AuD��   AuD\P   AuF�P   C���AuF��   AuF�P   AuI>P   C��VAuIc�   AuI>P   AuK�P   C��AuK��   AuK�P   AuN P   C��vAuNE�   AuN P   AuP�P   C��AuP��   AuP�P   AuSP   C��AuS'�   AuSP   AuUsP   C���AuU��   AuUsP   AuW�P   C���AuX	�   AuW�P   AuZUP   C��QAuZz�   AuZUP   Au\�P   C���Au\��   Au\�P   Au_7P   C��Au_\�   Au_7P   Aua�P   C���Aua��   Aua�P   AudP   C��QAud>�   AudP   Auf�P   C��XAuf��   Auf�P   Auh�P   C��lAui �   Auh�P   AuklP   C��Auk��   AuklP   Aum�P   C��bAun�   Aum�P   AupNP   C���Aups�   AupNP   Aur�P   C��OAur��   Aur�P   Auu0P   C�AuuU�   Auu0P   Auw�P   C��Auw��   Auw�P   AuzP   C��Auz7�   AuzP   Au|�P   C�Au|��   Au|�P   Au~�P   C��Au�   Au~�P   Au�eP   C�Au���   Au�eP   Au��P   C��Au���   Au��P   Au�GP   C�$^Au�l�   Au�GP   Au��P   C�$8Au���   Au��P   Au�)P   C�#�Au�N�   Au�)P   Au��P   C��Au���   Au��P   Au�P   C��Au�0�   Au�P   Au�|P   C�*�Au���   Au�|P   Au��P   C�6qAu��   Au��P   Au�^P   C�0BAu���   Au�^P   Au��P   C�&0Au���   Au��P   Au�@P   C�yAu�e�   Au�@P   Au��P   C�5�Au���   Au��P   Au�"P   C�N�Au�G�   Au�"P   Au��P   C�>�Au���   Au��P   Au�P   C�/�Au�)�   Au�P   Au�uP   C�=@Au���   Au�uP   Au��P   C�Q�Au��   Au��P   Au�WP   C�?�Au�|�   Au�WP   Au��P   C�I|Au���   Au��P   Au�9P   C�D�Au�^�   Au�9P   Au��P   C�F�Au���   Au��P   Au�P   C�K�Au�@�   Au�P   Au��P   C�F�Au���   Au��P   Au��P   C�@�Au�"�   Au��P   Au�nP   C�49Au���   Au�nP   Au��P   C�1�Au��   Au��P   Au�PP   C�2�Au�u�   Au�PP   Au��P   C�N�Au���   Au��P   Au�2P   C�H4Au�W�   Au�2P   AuʣP   C�<`Au���   AuʣP   Au�P   C�X�Au�9�   Au�P   AuυP   C�T9AuϪ�   AuυP   Au��P   C�C�Au��   Au��P   Au�gP   C�_pAuԌ�   Au�gP   Au��P   C�j�Au���   Au��P   Au�IP   C�QcAu�n�   Au�IP   AuۺP   C�^RAu���   AuۺP   Au�+P   C�\�Au�P�   Au�+P   Au��P   C�T�Au���   Au��P   Au�P   C�_`Au�2�   Au�P   Au�~P   C�fAu��   Au�~P   Au��P   C�n^Au��   Au��P   Au�`P   C�j?Au��   Au�`P   Au��P   C�j1Au���   Au��P   Au�BP   C�e%Au�g�   Au�BP   Au�P   C�g�