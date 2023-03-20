function DCM = seeg_spm_dcm_si(data,options)
% Function that defines DCM model with or without user specified structural priors
% data      - FT SEEG object or ExMxN SEEG trial matrix. Each E row contains a matrix
%             of SEEG data for each channel (M) and epoch sample (N) of no more
%             than 10 channels
% options   - struct containing the mandatory and optional fields
%           - fs: sampling frequency
%           - label: cell containing M label string cells
%           - structure: struct with fields smax, alpha, delta (hyperparameters) and net (network matrix)
%           - freqth: frequency threshold for spectral dcm

%% Convert Raw Data to SPM format
if ~isfield(options,'fs')
    options.fs = 1024;
else % to fix python conversion issues                                      %% RR edit 
    options.fs = double(options.fs); 
end

structure = 1;

if iscell(data)
    E = length(data);
    M = size(data{1},1);
    N = size(data{1},2);
else   
    E = size(data,1);
    M = size(data,2);
    N = size(data,3);
end

if ~isfield(options,'label')
    options.label = arrayfun(@num2str, 1:M, 'UniformOutput', 0);
else % to fix python conversion issues
    options.label = cellstr(options.label);
end

if ~isfield(options,'freqth')
    sprintf('WARNING: Defaulting to 1Hz to 32Hz band.');
    options.freqth = 32;
end

if isfield(options,'structure')
       if ~isfield(options.structure,'alpha') || ~isfield(options.structure,'net') || ~isfield(options.structure,'delta') || ~isfield(options.structure,'smax')
            sprintf('WARNING: Structural params missing. Setting defaults.');
            options.structure.smax = 0.5;
            options.structure.alpha = 0;
            options.structure.delta = 8;
            net = ones([M M]);
            net(1:1+size(net,1):end) = 0;
            options.structure.net = net;
       end
else
    sprintf('WARNING: No structural information found. Using standard DCM priors.');
    structure = 0;
end

if M>10
    sprintf('WARNING: Large models may require additional resources');
end

seeg = {};
seeg.fsample = options.fs;
if iscell(data)
    seeg.trial = data;
    seeg.time = cell([E 1]);
    seeg.label = options.label;
    for i=1:E
        seeg.time{i} = (1:N)./options.fs;
    end
else
    seeg.trial = cell([E 1]);
    seeg.time = cell([E 1]);
    seeg.label = options.label;
    for i=1:E
        seeg.time{i} = (1:N)./options.fs; % RR edit - back and forth import to python changes fs to uint16 
        seeg.trial{i} = squeeze(data(i,:,:));
    end
end

spmfile = [options.path filesep 'spmtemp' datestr(datetime('now'), 'yymmdd') '.mat'];
D       = spm_eeg_ft2spm(seeg,spmfile);

S = [];
S.ind = 1:size(D,1);
S.D = spmfile;
S.task = 'settype';
S.type = 'LFP';
D = spm_eeg_prep(S);
save(spmfile,'D','-v7.3');

%% Define DCM
Fs                  = fsample(D);
smpls               = size(D,2);
w                   = 1;
DCM                  = [];
DCM.options.analysis = 'CSD';       % cross-spectral density 
DCM.options.model    = 'CMC';      	% structure canonical microcircuit (for now)
DCM.options.spatial  = 'LFP'; 
timax               = linspace(0, smpls/Fs, smpls);
DCM.options.Tdcm     = [timax(1) timax(end)] * 1000;     

DCM.options.Fdcm    = [1 options.freqth];     	% frequency range  
DCM.options.D       = 1;         	% frequency bin, 1 = no downsampling
DCM.options.Nmodes  = 8;          	% cosine reduction components used 
DCM.options.han     = 0;         	% no hanning 
DCM.options.trials  = w;            % index of ERPs within file (FLAG: AM I TRAINING ON A SINGLE TRIAL)?

DCM.Sname           = chanlabels(D);
DCM.M.Hz            = DCM.options.Fdcm(1):DCM.options.D:DCM.options.Fdcm(2);
DCM.xY.Hz           = DCM.M.Hz;

DCM.xY.Dfile        = spmfile;

% Define different connection types
%==========================================================================
F      = tril(ones(size(D,1)),-1);    % Forward triangle (lower)
B      = triu(ones(size(D,1)),1);     % Backward triangle (upper)
S      = zeros(size(D,1));   for s = 1:length(S); S(s,s) = 1; end

% Define model arcitecture (A), conditional effects (B) and input (C) 
%==========================================================================
DCM.A{1}    =   F + B;
DCM.A{2}    =   F + B;
DCM.A{3}    =   S;

DCM.B           = {};
DCM.C           = sparse(length(DCM.A{1}),0); 

% Reorganise model parameters in specific structure
%==========================================================================
DCM.M.dipfit.Nm    = DCM.options.Nmodes;
DCM.M.dipfit.model = DCM.options.model;
DCM.M.dipfit.type  = DCM.options.spatial;

DCM.M.dipfit.Nc    = size(D,1);
DCM.M.dipfit.Ns    = length(DCM.A{1});
% Define priors
%==========================================================================
% Load standard neural priors (including for structural case)
%--------------------------------------------------------------------------
if structure
    options.structure.net(1:1+size(options.structure.net,1):end) = 0;
    DCM.structure.net    = options.structure.net;
    DCM.structure.net    = DCM.structure.net;
    DCM.structure.net    = (DCM.structure.net+DCM.structure.net')/2;
    DCM.structure.net    = DCM.structure.net/(max(DCM.structure.net(:)));
    DCM.structure.alpha = options.structure.alpha;
    DCM.structure.smax = options.structure.smax;
    DCM.structure.delta = options.structure.delta;

    [pEsi,pCsi]  = spm_si_neural_priors(DCM.A,DCM.B,DCM.C,DCM.options.model,DCM.structure);
    [pEsi,pCsi]  = spm_L_priors(DCM.M.dipfit,pEsi,pCsi);
    [pEsi,pCsi]  = spm_ssr_priors(pEsi,pCsi);

    % Switch off variations in spatial parameters
    %--------------------------------------------------------------------------

    pCsi.L        = pCsi.L * 0;
    pCsi.J        = pCsi.J * 0;

    DCM.M.pE   = pEsi;
    DCM.M.pC   = pCsi;
else
    
    [pE,pC]  = spm_dcm_neural_priors(DCM.A,DCM.B,DCM.C,DCM.options.model);
    [pE,pC]  = spm_L_priors(DCM.M.dipfit,pE,pC);
    [pE,pC]  = spm_ssr_priors(pE,pC);

    % Switch off variations in spatial parameters
    %--------------------------------------------------------------------------
    pC.L        = pC.L * 0;
    pC.J        = pC.J * 0;

    DCM.M.pE   = pE;
    DCM.M.pC   = pC;
end
% 
% DCM.name = 'dcmtemp.mat';
% save(DCM.name,'DCM')

% load([Fdcm fs 'emp_priors.mat']);
% pE.T        = P.T;
% pE.L        = ones(length(pE.L),1) .* P.L;
% pE.J        = P.J;
% 
% for n = 1:length(P.name)
%     thisname        = P.name{n};
%     nameid          = find(strcmp(chanlab, thisname));
%     pE.G(nameid,:)  = P.G(n,:);
% end