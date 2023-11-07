CDF   �   
      time       bnds      lon       lat             CDI       ?Climate Data Interface version 1.9.6 (http://mpimet.mpg.de/cdi)    Conventions       CF-1.4     history      USun Feb 28 12:28:14 2021: cdo -s -a yearmean /echam/folini/cmip5/historical_rcp45/full_data//MPI-ESM-MR_r1i1p1_ts_185001-210012_globalmean_mm_timeseries.nc /echam/folini/cmip5/historical_rcp45/full_data//MPI-ESM-MR_r1i1p1_ts_185001-210012_globalmean_am_timeseries.nc
Sun Feb 28 12:28:13 2021: cdo -s -a fldmean /echam/folini/cmip5/historical_rcp45/full_data//MPI-ESM-MR_r1i1p1_ts_185001-210012_fulldata.nc /echam/folini/cmip5/historical_rcp45/full_data//MPI-ESM-MR_r1i1p1_ts_185001-210012_globalmean_mm_timeseries.nc
Sun Feb 28 12:28:10 2021: cdo -s mergetime /echam/folini/cmip5/historical/full_data//MPI-ESM-MR_r1i1p1_ts_185001-200512_fulldata.nc /echam/folini/cmip5/historical_rcp45/full_data//tmp_merge.nc /echam/folini/cmip5/historical_rcp45/full_data//MPI-ESM-MR_r1i1p1_ts_185001-210012_fulldata.nc
Sat May 06 11:38:22 2017: cdo -s -a selvar,ts /echam/folini/cmip5/historical//tmp_01.nc /echam/folini/cmip5/historical//tmp_11.nc
Sat May 06 11:38:18 2017: cdo -s -a mergetime /net/atmos/data/cmip5/historical/Amon/ts/MPI-ESM-MR/r1i1p1/ts_Amon_MPI-ESM-MR_historical_r1i1p1_185001-200512.nc /echam/folini/cmip5/historical//tmp_01.nc
Model raw output postprocessing with modelling environment (IMDI) at DKRZ: URL: http://svn-mad.zmaw.de/svn/mad/Model/IMDI/trunk, REV: 3911 2011-10-08T16:30:02Z CMOR rewrote data to comply with CF standards and CMIP5 requirements.      source        �MPI-ESM-MR 2011; URL: http://svn.zmaw.de/svn/cosmos/branches/releases/mpi-esm-cmip5/src/mod; atmosphere: ECHAM6 (REV: 4936), T63L47; land: JSBACH (REV: 4936); ocean: MPIOM (REV: 4936), GR15L40; sea ice: 4936; marine bgc: HAMOCC (REV: 4936);   institution       $Max Planck Institute for Meteorology   institute_id      MPI-M      experiment_id         
historical     model_id      
MPI-ESM-MR     forcing       GHG,Oz,SD,Sl,Vl,LU     parent_experiment_id      N/A    parent_experiment_rip         N/A    branch_time                  contact       cmip5-mpi-esm@dkrz.de      
references       �ECHAM6: n/a; JSBACH: Raddatz et al., 2007. Will the tropical land biosphere dominate the climate-carbon cycle feedback during the twenty first century? Climate Dynamics, 29, 565-574, doi 10.1007/s00382-007-0247-8;  MPIOM: Marsland et al., 2003. The Max-Planck-Institute global ocean/sea ice model with orthogonal curvilinear coordinates. Ocean Modelling, 5, 91-127;  HAMOCC: Technical Documentation, http://www.mpimet.mpg.de/fileadmin/models/MPIOM/HAMOCC5.1_TECHNICAL_REPORT.pdf;    initialization_method               physics_version             tracking_id       $dc554603-a326-4e33-8086-00a8a230e0a9   product       output     
experiment        
historical     	frequency         year   creation_date         2011-10-08T16:30:06Z   
project_id        CMIP5      table_id      ;Table Amon (27 April 2011) a5a1c518f52ae340313ba0aada03f862    title         5MPI-ESM-MR model output prepared for CMIP5 historical      parent_experiment         N/A    modeling_realm        atmos      realization             cmor_version      2.6.0      CDO       ?Climate Data Operators version 1.9.6 (http://mpimet.mpg.de/cdo)          time                standard_name         time   	long_name         time   bounds        	time_bnds      units         day as %Y%m%d.%f   calendar      proleptic_gregorian    axis      T           �   	time_bnds                             �   lon                standard_name         	longitude      	long_name         	longitude      units         degrees_east   axis      X           �   lat                standard_name         latitude   	long_name         latitude   units         degrees_north      axis      Y           �   ts                        standard_name         surface_temperature    	long_name         Surface Temperature    units         K      
_FillValue        `�x�   missing_value         `�x�   comment       0""skin"" temperature (i.e., SST for open ocean)    cell_methods      time: mean     associated_files      �baseURL: http://cmip-pcmdi.llnl.gov/CMIP5/dataLocation gridspecFile: gridspec_atmos_fx_MPI-ESM-MR_historical_r0i0p0.nc areacella: areacella_fx_MPI-ESM-MR_historical_r0i0p0.nc          �                Aq���   Aq��P   Aq�P   C�ӬAq�6�   Aq�P   Aq��P   C��YAq���   Aq��P   Aq��P   C��$Aq��   Aq��P   Aq�dP   C���Aq���   Aq�dP   Aq��P   C�ސAq���   Aq��P   Aq�FP   C��Aq�k�   Aq�FP   Aq��P   C��-Aq���   Aq��P   Aq�(P   C��Aq�M�   Aq�(P   Aq��P   C��Aq���   Aq��P   Aq�
P   C��5Aq�/�   Aq�
P   Aq�{P   C��]Aq���   Aq�{P   Aq��P   C��Aq��   Aq��P   Aq�]P   C���AqĂ�   Aq�]P   Aq��P   C��Aq���   Aq��P   Aq�?P   C��Aq�d�   Aq�?P   Aq˰P   C��KAq���   Aq˰P   Aq�!P   C��Aq�F�   Aq�!P   AqВP   C��^Aqз�   AqВP   Aq�P   C��tAq�(�   Aq�P   Aq�tP   C��=Aqՙ�   Aq�tP   Aq��P   C���Aq�
�   Aq��P   Aq�VP   C���Aq�{�   Aq�VP   Aq��P   C�ɨAq���   Aq��P   Aq�8P   C���Aq�]�   Aq�8P   Aq�P   C��Aq���   Aq�P   Aq�P   C��VAq�?�   Aq�P   Aq�P   C��Aq��   Aq�P   Aq��P   C���Aq�!�   Aq��P   Aq�mP   C���Aq��   Aq�mP   Aq��P   C��QAq��   Aq��P   Aq�OP   C�ЪAq�t�   Aq�OP   Aq��P   C��^Aq���   Aq��P   Aq�1P   C��QAq�V�   Aq�1P   Aq��P   C�ʠAq���   Aq��P   Aq�P   C��iAq�8�   Aq�P   Aq��P   C���Aq���   Aq��P   Aq��P   C��YAq��   Aq��P   ArfP   C���Ar��   ArfP   Ar�P   C��2Ar��   Ar�P   ArHP   C���Arm�   ArHP   Ar�P   C��9Ar��   Ar�P   Ar*P   C���ArO�   Ar*P   Ar�P   C���Ar��   Ar�P   ArP   C��`Ar1�   ArP   Ar}P   C��%Ar��   Ar}P   Ar�P   C�ʈAr�   Ar�P   Ar_P   C�уAr��   Ar_P   Ar�P   C��cAr��   Ar�P   ArAP   C���Arf�   ArAP   Ar�P   C��nAr��   Ar�P   Ar!#P   C��mAr!H�   Ar!#P   Ar#�P   C�ٹAr#��   Ar#�P   Ar&P   C���Ar&*�   Ar&P   Ar(vP   C�ǪAr(��   Ar(vP   Ar*�P   C��VAr+�   Ar*�P   Ar-XP   C���Ar-}�   Ar-XP   Ar/�P   C��Ar/��   Ar/�P   Ar2:P   C���Ar2_�   Ar2:P   Ar4�P   C��>Ar4��   Ar4�P   Ar7P   C��Ar7A�   Ar7P   Ar9�P   C���Ar9��   Ar9�P   Ar;�P   C��Ar<#�   Ar;�P   Ar>oP   C�ШAr>��   Ar>oP   Ar@�P   C��AArA�   Ar@�P   ArCQP   C���ArCv�   ArCQP   ArE�P   C��!ArE��   ArE�P   ArH3P   C��eArHX�   ArH3P   ArJ�P   C���ArJ��   ArJ�P   ArMP   C�ÂArM:�   ArMP   ArO�P   C�ՙArO��   ArO�P   ArQ�P   C��"ArR�   ArQ�P   ArThP   C��ArT��   ArThP   ArV�P   C���ArV��   ArV�P   ArYJP   C�� ArYo�   ArYJP   Ar[�P   C��hAr[��   Ar[�P   Ar^,P   C��Ar^Q�   Ar^,P   Ar`�P   C��	Ar`��   Ar`�P   ArcP   C���Arc3�   ArcP   AreP   C���Are��   AreP   Arg�P   C��Arh�   Arg�P   ArjaP   C��3Arj��   ArjaP   Arl�P   C���Arl��   Arl�P   AroCP   C��Aroh�   AroCP   Arq�P   C���Arq��   Arq�P   Art%P   C���ArtJ�   Art%P   Arv�P   C��]Arv��   Arv�P   AryP   C��>Ary,�   AryP   Ar{xP   C���Ar{��   Ar{xP   Ar}�P   C��Ar~�   Ar}�P   Ar�ZP   C��;Ar��   Ar�ZP   Ar��P   C��qAr���   Ar��P   Ar�<P   C��\Ar�a�   Ar�<P   Ar��P   C��&Ar���   Ar��P   Ar�P   C��YAr�C�   Ar�P   Ar��P   C��Ar���   Ar��P   Ar� P   C��Ar�%�   Ar� P   Ar�qP   C��UAr���   Ar�qP   Ar��P   C��Ar��   Ar��P   Ar�SP   C��[Ar�x�   Ar�SP   Ar��P   C��SAr���   Ar��P   Ar�5P   C��Ar�Z�   Ar�5P   Ar��P   C��2Ar���   Ar��P   Ar�P   C��8Ar�<�   Ar�P   Ar��P   C��|Ar���   Ar��P   Ar��P   C���Ar��   Ar��P   Ar�jP   C��wAr���   Ar�jP   Ar��P   C���Ar� �   Ar��P   Ar�LP   C��Ar�q�   Ar�LP   Ar��P   C���Ar���   Ar��P   Ar�.P   C���Ar�S�   Ar�.P   Ar��P   C��dAr���   Ar��P   Ar�P   C��Ar�5�   Ar�P   Ar��P   C�
yAr���   Ar��P   Ar��P   C��Ar��   Ar��P   Ar�cP   C�ͫAr���   Ar�cP   Ar��P   C��^Ar���   Ar��P   Ar�EP   C���Ar�j�   Ar�EP   ArĶP   C��vAr���   ArĶP   Ar�'P   C���Ar�L�   Ar�'P   ArɘP   C��jArɽ�   ArɘP   Ar�	P   C��BAr�.�   Ar�	P   Ar�zP   C���ArΟ�   Ar�zP   Ar��P   C��Ar��   Ar��P   Ar�\P   C�:ArӁ�   Ar�\P   Ar��P   C��Ar���   Ar��P   Ar�>P   C��Ar�c�   Ar�>P   ArگP   C��&Ar���   ArگP   Ar� P   C���Ar�E�   Ar� P   ArߑP   C�Ar߶�   ArߑP   Ar�P   C���Ar�'�   Ar�P   Ar�sP   C��Ar��   Ar�sP   Ar��P   C�
�Ar�	�   Ar��P   Ar�UP   C�wAr�z�   Ar�UP   Ar��P   C���Ar���   Ar��P   Ar�7P   C���Ar�\�   Ar�7P   Ar�P   C�'Ar���   Ar�P   Ar�P   C�8Ar�>�   Ar�P   Ar��P   C�(Ar���   Ar��P   Ar��P   C�YAr� �   Ar��P   Ar�lP   C�&xAr���   Ar�lP   Ar��P   C�0Ar��   Ar��P   Ar�NP   C�.rAr�s�   Ar�NP   As�P   C�zAs��   As�P   As0P   C��AsU�   As0P   As�P   C�#As��   As�P   As	P   C�#>As	7�   As	P   As�P   C�/gAs��   As�P   As�P   C�K�As�   As�P   AseP   C�;�As��   AseP   As�P   C�>�As��   As�P   AsGP   C�I�Asl�   AsGP   As�P   C�P�As��   As�P   As)P   C�K�AsN�   As)P   As�P   C�A>As��   As�P   AsP   C�GLAs0�   AsP   As!|P   C�IqAs!��   As!|P   As#�P   C�LAs$�   As#�P   As&^P   C�X�As&��   As&^P   As(�P   C�i�As(��   As(�P   As+@P   C�ZaAs+e�   As+@P   As-�P   C�aGAs-��   As-�P   As0"P   C�Z�As0G�   As0"P   As2�P   C�U�As2��   As2�P   As5P   C�TAs5)�   As5P   As7uP   C�[�As7��   As7uP   As9�P   C�b0As:�   As9�P   As<WP   C�lEAs<|�   As<WP   As>�P   C�w�As>��   As>�P   AsA9P   C�p�AsA^�   AsA9P   AsC�P   C�eDAsC��   AsC�P   AsFP   C�`�AsF@�   AsFP   AsH�P   C�sXAsH��   AsH�P   AsJ�P   C�|7AsK"�   AsJ�P   AsMnP   C�~AsM��   AsMnP   AsO�P   C�FAsP�   AsO�P   AsRPP   C�}�AsRu�   AsRPP   AsT�P   C���AsT��   AsT�P   AsW2P   C���AsWW�   AsW2P   AsY�P   C���AsY��   AsY�P   As\P   C��nAs\9�   As\P   As^�P   C���As^��   As^�P   As`�P   C��Asa�   As`�P   AscgP   C���Asc��   AscgP   Ase�P   C���Ase��   Ase�P   AshIP   C���Ashn�   AshIP   Asj�P   C��sAsj��   Asj�P   Asm+P   C��+AsmP�   Asm+P   Aso�P   C��:Aso��   Aso�P   AsrP   C���Asr2�   AsrP   Ast~P   C���Ast��   Ast~P   Asv�P   C��Asw�   Asv�P   Asy`P   C��uAsy��   Asy`P   As{�P   C��KAs{��   As{�P   As~BP   C���As~g�   As~BP   As��P   C���As���   As��P   As�$P   C��qAs�I�   As�$P   As��P   C���As���   As��P   As�P   C��As�+�   As�P   As�wP   C��xAs���   As�wP   As��P   C�̲As��   As��P   As�YP   C��uAs�~�   As�YP   As��P   C���As���   As��P   As�;P   C�ڥAs�`�   As�;P   As��P   C�ЛAs���   As��P   As�P   C��rAs�B�   As�P   As��P   C��=As���   As��P   As��P   C��As�$�   As��P   As�pP   C�̎As���   As�pP   As��P   C���As��   As��P   As�RP   C���As�w�   As�RP   As��P   C���As���   As��P   As�4P   C��As�Y�   As�4P   As��P   C���As���   As��P   As�P   C���As�;�   As�P   As��P   C�� As���   As��P   As��P   C��As��   As��P   As�iP   C��As���   As�iP   As��P   C� �As���   As��P   As�KP   C��As�p�   As�KP   As��P   C���As���   As��P   As�-P   C���As�R�   As�-P   AsP   C��rAs���   AsP   As�P   C��9As�4�   As�P   AsǀP   C��%Asǥ�   AsǀP   As��P   C��pAs��   As��P   As�bP   C��GAṡ�   As�bP   As��P   C��nAs���   As��P   As�DP   C��As�i�   As�DP   AsӵP   C���As���   AsӵP   As�&P   C���As�K�   As�&P   AsؗP   C��8Asؼ�   AsؗP   As�P   C�fAs�-�   As�P   As�yP   C��Asݞ�   As�yP   As��P   C�*As��   As��P   As�[P   C��As��   As�[P   As��P   C��bAs���   As��P   As�=P   C���As�b�   As�=P   As�P   C��bAs���   As�P   As�P   C��As�D�   As�P   As�P   C���As��   As�P   As�P   C�?As�&�   As�P   As�rP   C��As��   As�rP   As��P   C�
�As��   As��P   As�TP   C��As�y�   As�TP   As��P   C���As���   As��P   As�6P   C��As�[�   As�6P   As��P   C��sAs���   As��P   AtP   C��At=�   AtP   At�P   C�%9At��   At�P   At�P   C���At�   At�P   At	kP   C��