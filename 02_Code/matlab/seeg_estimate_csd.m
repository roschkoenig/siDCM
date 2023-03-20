function dcm_path = seeg_estimate_csd(dcm_path)
load(dcm_path)
DCM             = pdcm_dcm_fix(DCM);
DCM             = spm_dcm_erp_data(DCM);
DCM             = spm_dcm_erp_dipfit(DCM, 1); 
DCM             = spm_dcm_csd_data(DCM);
DCM.options.DATA        = 0; 
DCM.name        = dcm_path;   
save(dcm_path, 'DCM', '-v7')
disp('CSD estimation complete')
