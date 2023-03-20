function pdcm(cfg)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
disp('SPM initialising')
spm('defaults', 'eeg')
disp('SPM initialised')

% If called with path to configure file
%--------------------------------------------------------------------------
if ischar(cfg), load(cfg); end 

% Check whether this is running in princicfgple
%--------------------------------------------------------------------------
if strcmp(cfg.task, 'test'), disp('The App is working'), end


%% Standard DCM pathyway
%==========================================================================
% Call cross-spectral density estimation (will update DCM path)
%--------------------------------------------------------------------------
if strcmp(cfg.task,'estimate_csd')
    dcm_path     = pdcm_estimate_csd(cfg.specfile);
    cfg.dcm_path = dcm_path;
end

% Call DCM inversion (will update DCM path)
%--------------------------------------------------------------------------
if strcmp(cfg.task,'run_dcm') 
    dcm_path        = pdcm_run_dcm(cfg.dcm_path); 
    cfg.dcm_path    = dcm_path; 
end 

%% SI DCM Pathway
%==========================================================================
% Specify si-DCM (will update DCM path)
%--------------------------------------------------------------------------
if strcmp(cfg.task, 'init_siDCM')
    disp('Setting up DCM structure')
    DCM = seeg_spm_dcm_si(cfg.data, cfg.options);
    save(cfg.dcm_path, 'DCM', '-v7'); 
end

% Call cross-spectral density estimation (will update DCM path)
%--------------------------------------------------------------------------
if strcmp(cfg.task,'csd_siDCM')
    disp('Estimating CSD for structurally informed DCM')
    dcm_path     = seeg_estimate_csd(cfg.dcm_path);
    cfg.dcm_path = dcm_path;
end

% Invert si-DCM (will update DCM path)
%--------------------------------------------------------------------------
if strcmp(cfg.task, 'inv_siDCM')
    disp('Inverting prepared DCM')
    load(cfg.dcm_path);  % loads DCM structure previously specified
    DCM = pdcm_dcm_fix(DCM);        % fix some python conversion issues
    DCM = seeg_spm_dcm_si_inv(DCM); % run actual inversion
    save(cfg.dcm_path, 'DCM', '-v7'); 
end