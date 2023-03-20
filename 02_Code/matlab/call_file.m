addpath('/Users/roschkoenig/Desktop/2303_si-DCM/02_Code/matlab'); 
addpath('/Users/roschkoenig/Desktop/2303_si-DCM/03_Output'); 
addpath('/Users/roschkoenig/Desktop/2303_si-DCM/01_Data'); 
load('/Users/roschkoenig/Desktop/2303_si-DCM/si_seeg_scripts/testdata.mat')
options.path = '/Users/roschkoenig/Desktop/2303_si-DCM/03_Output'; 
%%
cfg.task        = 'init_siDCM';
cfg.dcm_path    = [options.path filesep 'DCM'];
cfg.data        = data;
cfg.options     = options;
pdcm(cfg);

%%
cfg.task        = 'csd_siDCM'; pdcm(cfg);

%%
cfg.task        = 'inv_siDCM'; pdcm(cfg); 