% SNR_TEST      A test of the sparse time-series signal recovery algorithm,
% that, for a fixed choice of signal model parameters, will sweep the
% SNR over a range of values.
%
%
% SYNTAX:
% snr_test(N, M, lambda_0, T, N_trials)
%
% INPUTS:
%   N               Number of unknowns at each timestep
%   M               Number of measurements at each timestep
%   T              	Number of timesteps - 1
%   lambda_0        IID activity probability, Pr{s_n(0) = 1}
%   N_trials     	Number of trials to average results over
%
% OUTPUTS:
% This function will save the timestep-averaged NMSEs (TNMSEs) and
% time-averaged bit error rates (TBERs) for a support-aware genie smoother,
% a naive BP recovery scheme, a naive support-aware MMSE estimator, and the
% proposed multi-timestep recovery algorithm in a .MAT file whose filename
% is created at runtime based on input parameters and the time/date of the
% start of execution.
% Suffix keywords:
%   _smooth     The support-aware timestep-aware genie smoother
%   _iMMSE      The timestep-independent, support-aware MMSE estimator
%   _naive      The support-unaware, timestep-unaware naive BP estimator
%   _frame      The proposed multi-timestep BP algorithm (sp_multi_frame_fxn)
%

%
% Coded by: Justin Ziniel, The Ohio State Univ.
% E-mail: ziniel.1@osu.edu
% Last change: 12/07/15
% Change summary: 
%		- Created (03/14/10; JAZ)
%       - Fixed bugs introduced in v1.1 due to migration to SigGenParams
%           and DCSModelParams objects (12/07/15; JAZ)
% Version 1.2
%

function snr_test(N, M, lambda_0, T, N_trials)

%% Declare the parameters for the signal model and test setup

if nargin == 0
    N = 512;                % # of parameters                           [dflt=256]
    M = round(N/4);         % # of measurements                         [dflt=N/3]
    T = 25;                 % # of timesteps - 1                        [dflt=25]
    lambda_0 = 0.20;        % Pr{s_n(0) = 1}                            [dflt=.15]
    N_trials = 50;          % Number of trials to average results over  [dflt=50]
end

% Declare static signal model parameters
A_type = 1;             % 1=iid CN, 2=rademacher, 3=subsampled DFT	[dflt=1]
eta_0 = 0;            	% Complex mean of active coefficients       [dflt=0]
kappa_0 = 1;        	% Circular variance of active coefficients	[dflt=1]
eps = 1e-7;             % "Squelch" parameter for inactives         [dflt=1e-4]

% Algorithm execution parameters
smooth_iter = 5;        % Number of forward/backward "smoothing" passes 
                        % to perform [dflt: 5]
eq_iter = 50;          	% Number of equalization iterations to perform,
                        % per forward/backward pass, at each timestep [dflt: 25]
alg = 2;                % Type of BP algorithm to use during equalization 
                        % at each timestep: (1) for standard BP, 
                        % (2) for AMP [dflt: 2]
warning('SKS is being run using AMP instead of ABP')
update = false;       	% Update DCS-AMP parameters using EM learning?

% Test setup parameters
Q = 20;                       	% Sampling density
SNRmdB = linspace(-5, 45, Q);  	% Sparsity ratio E[K]/M
N_snr = length(SNRmdB);
pz1 =   [.01 .01 .05 .05];      % Pairwise combos of pz1 and alpha over
alpha = [.01 .10 .01 .10];      % which to run the SNR sweep
N_confs = length(pz1)*(length(pz1) == length(alpha));
if numel(lambda_0) ~= 1, error('Please input scalar lambda_0'); end

% Filename for saving data
savefile = [datestr(now, 'ddmmmyyyy_HHMMSSFFF') '_SNR_test_N_' ...
    num2str(N) '_M_' num2str(M(1)) '_T_' num2str(T) '_lambda0_' ...
    num2str(lambda_0) '_eta0_' num2str(eta_0) '_kappa0_' ...
    num2str(kappa_0) '.mat'];

% Randomly seed the RNG and save its state for reference purposes
NewStream = RandStream.create('mt19937ar', 'Seed', sum(100*clock));
RandStream.setGlobalStream(NewStream);
savedState = NewStream.State;

%% Execute the phase plane test

% Add necessary path
path(path, '../../Functions')

% Create arrays to store the NMSEs and BERs
TNMSE_smooth = NaN*ones(N_confs, N_snr, N_trials);   % TNMSE for genie smoother
TNMSE_iMMSE = NaN*ones(N_confs, N_snr, N_trials);    % TNMSE for TI, SA MMSE estimator
TNMSE_naive = NaN*ones(N_confs, N_snr, N_trials);    % TNMSE for the naive BP recovery
TNMSE_frame = NaN*ones(N_confs, N_snr, N_trials);    % TNMSE for the proposed algorithm
TBER_naive = NaN*ones(N_confs, N_snr, N_trials);     % TBER for the naive BP recovery
TBER_frame = NaN*ones(N_confs, N_snr, N_trials);     % TBER for the proposed algorithm

try % Matlabpool opening
    % Open a matlabpool, if one is not already open
    if matlabpool('size') == 0
        matlabpool open
        fprintf('Opened a matlabpool with %d workers', matlabpool('size'));
    end
catch
    disp('Error in opening a matlabpool')
end

time = clock;   % Start the stopwatch
for n = 1:N_trials
    for c = 1:N_confs     % Chosen (pz1,alpha) pair
        parfor s = 1:N_snr   % Chosen SNR
            % First determine the value of rho
            rho = (2 - alpha(c))*kappa_0/alpha(c);  % Driving noise variance (leads
                                                    % to constant amplitude
                                                    % variance
            
            % *************************************************************
            % Generate a signal/measurement pair realization for this suite
            % *************************************************************         
            % Create an object to hold the signal model parameters
            signalParams = SigGenParams('N', N, 'M', M, 'T', T, 'A_type', ...
                A_type, 'lambda', lambda_0, 'zeta', eta_0, 'sigma2', kappa_0, ...
                'alpha', alpha(c), 'p01', pz1(c), 'SNRmdB', SNRmdB(s));
            
            % Create the signal, measurements, matrices, etc.
            [x_true, y, A, support, ~, sig2e] = dcs_signal_gen_fxn(signalParams);


            % *************************************************************
            % Solve using the TI, SA MMSE estimator and the naive BP alg.
            % *************************************************************
            NMSE_tmp_iMMSE = NaN*ones(1,T);   % Will hold NMSE of each timestep
            NMSE_tmp_naive = NaN*ones(1,T);   % Will hold NMSE of each timestep
            BER_tmp_naive = NaN*ones(1,T);    % Will hold BER of each timestep
            x_iMMSE = cell(1,T);              % Holds iMMSE recovery
            x_naive = cell(1,T);              % Holds independent CS recovery
            var_tmp = [0; (kappa_0 - alpha(c)^2*rho)/(1 - alpha(c))^2];
            for t = 1:T
                % First the timestep-independent (TI), support-aware (SA)
                % MMSE estimate
                s_mod = ones(N,1); s_mod(support{t}) = 2;
                % Compute MoG means and variances for current timestep
                mean_tmp = [0; eta_0];	 
                var_tmp = [0; (1-alpha(c))^2*var_tmp(2) + alpha(c)^2*rho];
                R = A{t}*diag(var_tmp(s_mod))*A{t}' + sig2e*eye(M);
                x_iMMSE{t} = mean_tmp(s_mod) + ...
                    diag(var_tmp(s_mod))*A{t}'*(R\(y{t} - A{t}*mean_tmp(s_mod)));
                NMSE_tmp_iMMSE(t) = (norm(x_true{t} - x_iMMSE{t})/...
                    norm(x_true{t}))^2;
                if NMSE_tmp_iMMSE(t) == inf
                    % True signal currently has no active coefficients,
                    % thus store a time-series averaged NSME here
                    NMSE_tmp_iMMSE(t) = norm(x_true{t} - x_iMMSE{t})^2/...
                        (norm([x_true{:}], 'fro')^2/(T));
                end

                % Next, the naive BP recovery
                [x_naive{t}, OutMessages] = sp_frame_fxn(y{t}, A{t}, ...
                    struct('pi', lambda_0, 'xi', mean_tmp(2), 'psi', ...
                    var_tmp(2), 'eps', eps, 'sig2e', sig2e), ...
                    struct('iter', eq_iter, 'alg', alg));
                NMSE_tmp_naive(t) = (norm(x_true{t} - x_naive{t})/...
                    norm(x_true{t}))^2;
                if NMSE_tmp_naive(t) == inf
                    % True signal currently has no active coefficients,
                    % thus store a time-series averaged NSME here
                    NMSE_tmp_naive(t) = norm(x_true{t} - x_naive{t})^2/...
                        (norm([x_true{:}], 'fro')^2/(T));
                end
                s_mod = zeros(N,1); s_mod(support{t}) = 1;
                BER_tmp_naive(t) = sum(s_mod ~= (OutMessages.pi > 0.5))/N;
            end

            % Save the time-averaged NMSE and BER for the genie and naive BP
            TNMSE_iMMSE(c,s,n) = sum(NMSE_tmp_iMMSE)/T;    % Genie TNMSE
            TNMSE_naive(c,s,n) = sum(NMSE_tmp_naive)/T;    % Naive TNMSE
            TBER_naive(c,s,n) = sum(BER_tmp_naive)/T;      % Naive TBER


            % *************************************************************
            % Solve using the support-aware, timestep-aware genie smoother
            % *************************************************************
            if alg == 1, SKSalg = 'BP'; else SKSalg = 'AMP'; end
            
            % Construct class objects for model parameters, copying over
            % the true signal generating parameters
            SKS_ModelParams = DCSModelParams(signalParams, sig2e);
            SKS_RunOptions = Options('smooth_iters', smooth_iter, ...
                'inner_iters', eq_iter, 'alg', SKSalg, 'eps', eps);
            
            % Solve using the proposed algorithm
            x_smooth = genie_dcs_fxn(y, A, support, SKS_ModelParams, ...
                SKS_RunOptions);

            % Compute NMSE for the genie smoother
            NMSE_tmp_smooth = NaN*ones(1,T);	% Will hold NMSE of each timestep
            for t = 1:T
                NMSE_tmp_smooth(t) = (norm(x_true{t} - x_smooth{t})/...
                    norm(x_true{t}))^2;
                if isnan(NMSE_tmp_smooth(t))
                    % True signal currently has no active coefficients,
                    % thus we must correct a 0/0 NaN error
                    NMSE_tmp_smooth(t) = 0;
                end
            end
            TNMSE_smooth(c,s,n) = sum(NMSE_tmp_smooth)/T;	% Smoother TNMSE


            % *************************************************************
            % Solve using the multi-timestep sparse recovery algorithm
            % *************************************************************
            % Create a structure to hold the signal model parameters
            dcsSignalParams = DCSModelParams(signalParams, sig2e);
            
            % Create an object to hold the algorithm execution parameters
            DCSAMPOptions = Options('smooth_iters', smooth_iter, ...
                'inner_iters', eq_iter, 'alg', alg, 'eps', eps, ...
                'update', update);
            
            % Solve using the proposed algorithm
            [x_frame, ~, lambda_frame] = sp_multi_frame_fxn(y, A, ...
                dcsSignalParams, DCSAMPOptions);

            % Compute NMSE and BER for the proposed algorithm
            NMSE_tmp_frame = NaN*ones(1,T);   % Will hold NMSE of each timestep
            BER_tmp_frame = NaN*ones(1,T);    % Will hold BER of each timestep
            for t = 1:T
                NMSE_tmp_frame(t) = (norm(x_true{t} - x_frame{t})/...
                    norm(x_true{t}))^2;
                if NMSE_tmp_frame(t) == inf
                    % True signal currently has no active coefficients,
                    % thus store a time-series averaged NSME here
                    NMSE_tmp_frame(t) = norm(x_true{t} - x_frame{t})^2/...
                        (norm([x_true{:}], 'fro')^2/(T));
                elseif isnan(NMSE_tmp_frame(t))
                    NMSE_tmp_frame(t) = 0;
                end
                s_mod = zeros(N,1); s_mod(support{t}) = 1;
                BER_tmp_frame(t) = sum(s_mod ~= (lambda_frame{t} > 0.5))/N;
            end
            TNMSE_frame(c,s,n) = sum(NMSE_tmp_frame)/T;    % Alg. TNMSE
            TBER_frame(c,s,n) = sum(BER_tmp_frame)/T;      % Alg. TBER
            
        end     % SNR
    end     % (pz1,alpha)
    
    % Having completed one sweep of the (beta, delta) grid, estimate the
    % time remaining and save the temporary data
    est_time = (N_trials - n)*(etime(clock, time)/n)/60/60;
    fprintf('(%d/%d) Estimated time remaining: %3.2f hours', ...
        n, N_trials, est_time);
    clear A R
    save(savefile);
end

fprintf('Total time elapsed for test: %3.2f hours', ...
    etime(clock, time)/60/60);

try
    % Close the matlabpool
    matlabpool close
catch
    disp('No matlabpool to close')
end

% Final save of data
clear A R
save(savefile);