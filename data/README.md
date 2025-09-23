# Calibration Data for Climate Emulators

The folder contains data used for the fitting procedure of the CE, arranged in several sub-folders:

- ['concentration'](concentration) contains concentration data see from ['folder'](source/EmiAndConcData). Please refer to the resepctive file RCP*_MIDYR_CONC.* for the original data.
- ['emission'](emission) contains emissions data from ['folder'](source/EmiAndConcData). Please refer to the resepctive file RCP*_EMISSIONS.* for the original data.
- ['forcing'](forcing) contains radiative forcing data from ['folder'](source/forcing).
- ['land_emission'](land_emission) contains land emission data from Gasser et al 2020.
- ['pulse'](pulse) contains data on CO2 pulse response function for the calculation of Global Warming Potentials from Joos et al. (2013).
- ['source'](source) contains emission data for the different RCPs ('EmiAndConcData'), CMIP5 benchmark data for the different RCPs and the 1pctCO2 ('DataFromCMIP') and forcing data from Taylor et al. (2012) and Meinshausen et al. (2011).
- ['temperature'](temperature) contains temperature data from source/DataFromCMIP.
- ['ZEC'](ZEC) contains the results from experiments that were part of the contribution of the Zero Emissions Commitment ModelIntercomparison Project (ZECMIP) to the Coupled Climate Carbon Cycle Model Intercomparison Project (C4MIP) from Jones et al. (2020) and MacDougall et al. (2020).

Please refer to the READMEs of each data folder for more information about the data.

## Usage

The data is used to claibrate the CE and for plotting. Python scripts that use the data are provided in the folder ['figures_3-4_14-25'](../figures_replication/figures_3-4_14-25).

Technical requirements and description of each script are given in the repsective README ['README'](../figures_replication/README.md).


