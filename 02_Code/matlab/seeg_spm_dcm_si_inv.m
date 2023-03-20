function INV = seeg_spm_dcm_si_inv(DCM,options)
% Function that inverts model with or without user specified structural priors
% data      - DCM struct containing prior and model information
% options   - struct containing the mandatory and optional fields
% (currently not in use)
%           - fs: sampling frequency
%           - label: cell containing M label string cells
%           - structure: struct with fields smax, alpha, delta (hyperparameters) and net (network matrix)
%           - freqth: frequency threshold for spectral dcm

%% Model Inversion Code
TMP         = seeg_spm_dcm_csd(DCM);
TMP.xY.R    = diag(TMP.xY.R);
INV         = TMP;
end