% ELL1_PHASE_TEST	A test of the sparse time-series signal recovery 
% algorithm, that, for a fixed choice of signal model parameters, will 
% sweep the expected-number-of-active-coefficients-per-measurement ratio, 
% (a.k.a. the normalized sparsity ratio (E[K]/M)), beta, and the
% number-of-measurements-per-number-of-unknowns ratio, (a.k.a. the
% undersampling ratio (M/N)), delta over ranges of values, while storing
% the timestep-averaged normalized mean squared error (t-avg. NMSE) at each
% (beta, delta) pair.  The t-avg. NMSE will also be recorded for a 
% support-aware genie smoother using genie_multi_frame_fxn, as well
% as a basis pursuit denoising (BPDN) estimator that independently recovers
% the unknown signals at each timestep without considering any relationship
% between the recovered signals of adjacent timesteps, and a similar
% timestep-independent but support-aware conditional MMSE estimator.  In
% the BPDN solver, there is a user-specified parameter sigma that must be
% set in order to solve min ||x||_1 s.t. ||Ax - y||_2 < sigma.  Since the
% optimal choice of sigma is not known, for each (delta, beta) pair, a
% cross-validation procedure will run multiple recoveries using the BPDN
% solver with different choices of sigma.  A genie will then select the
% value of sigma that minimizes NMSE for that signal realization.
%
%
% SYNTAX:
% ell1_phase_test(N, T, pz1, alpha, SNRmdB, N_trials)
%
% INPUTS:
%   N               Number of unknowns at each timestep
%   T              	Number of timesteps - 1
%   pz1             Pr{s_n(t) = 0 | s_n(t-1) = 1}
%   alpha         	Innovation rate of thetas (1 = total)
%   SNRmdB         	Per-measurement SNR (in dB)
%   N_trials     	Number of trials to avg. results over per (beta,delta)
%
% OUTPUTS:
% This function will save the timestep-averaged NMSEs and BERs for a
% support-aware genie smoother, an independent BPDN estimator, a naive 
% support-aware MMSE estimator, and the proposed multi-timestep recovery 
% algorithm in a .MAT file whose filename is created at runtime based on 
% input parameters and the time/date of the start of execution
% Suffix keywords:
%   _smooth     The support-aware timestep-aware genie smoother
%   _iMMSE      The timestep-independent, support-aware MMSE estimator
%   _bpdn       The timestep-independent BPDN estimator
%   _frame      The proposed multi-timestep BP algorithm (sp_multi_frame_fxn)
%

%
% Coded by: Justin Ziniel, The Ohio State Univ.
% E-mail: ziniel.1@osu.edu
% Last change: 12/07/15
% Change summary: 
%		- Created from ell1_phase_test v0.2 (12/09/10; JAZ)
%       - Fixed bugs introduced in v1.1 due to migration to SigGenParams
%           and DCSModelParams objects (12/07/15; JAZ)
% Version 1.2
%

function ell1_phase_test(N, T, pz1, alpha, SNRmdB, N_trials)

%% Declare the parameters for the signal model and test setup

if nargin == 0
    N = 512;                % # parameters                              [dflt=256]
    T = 25;                 % # of timesteps - 1                        [dflt=9]
    pz1 = 0.05;             % Pr{s_n(t) = eps | s_n(t-1) = 1}           [dflt=0.05]
    alpha = 0.10;           % Innovation rate of thetas (1 = total)     [dflt=0.10]
    SNRmdB = 15;           	% Per-measurement SNR (in dB)               [dflt=15]
    N_trials = 5;           % Number of trials to avg. results over per (beta,delta)
end

% Declare additional static signal model parameters
A_type = 1;             % 1=iid CN, 2=rademacher, 3=subsampled DFT	[dflt=1]
eta_0 = 0;            	% Complex mean of active coefficients       [dflt=0]
kappa_0 = 1;        	% Circular variance of active coefficients	[dflt=1]
rho = (2 - alpha)*kappa_0/alpha;        % Driving noise variance    
                                        % [dflt=(2 - alpha)*kappa_0/alpha]
eps = 1e-5;             % "Squelch" parameter for inactives         [dflt=1e-4]
update = false;       	% Update DCS-AMP parameters using EM learning?

% Algorithm execution parameters
smooth_iter = 5;        % Number of forward/backward "smoothing" passes 
                        % to perform [dflt: 5]
eq_iter = 25;          	% Number of equalization iterations to perform,
                        % per forward/backward pass, at each timestep [dflt: 25]
alg = 2;                % Type of BP algorithm to use during equalization 
                        % at each timestep: (1) for standard BP, 
                        % (2) for AMP [dflt: 2]

% Test setup parameters
Q = 30;                            	% Grid sampling density
beta = linspace(0.05, 0.95, Q);  	% Sparsity ratio E[K]/M
delta = linspace(0.05, 0.95, Q);  	% Undersampling ratio M/N
N_beta = length(beta);
N_delta = length(delta);

% Filename for saving data
savefile = [datestr(now, 'ddmmmyyyy_HHMMSSFFF') '_ell1_phase_test_N_' ...
    num2str(N) '_T_' num2str(T) '_pz1_' num2str(pz1) '_alpha_' ...
    num2str(alpha) '_SNRm_' num2str(SNRmdB) '.mat'];

% Randomly seed the RNG and save its state for reference purposes
NewStream = RandStream.create('mt19937ar', 'Seed', sum(100*clock));
RandStream.setGlobalStream(NewStream);
savedState = NewStream.State;


%% Execute the phase plane test

% Add necessary path
path(path, '../../Functions')
% path(path, '../../../../../SparseSolvers/spgl1-1.7')

% Create arrays to store the NMSEs and BERs
NMSE_smooth = NaN*ones(N_beta, N_delta, N_trials);	% NMSE for genie smoother
NMSE_iMMSE = NaN*ones(N_beta, N_delta, N_trials);   % NMSE for TI, SA MMSE estimator
NMSE_bpdn = NaN*ones(N_beta, N_delta, N_trials);    % NMSE for the BPDN estimator
NMSE_frame = NaN*ones(N_beta, N_delta, N_trials);   % NMSE for the proposed algorithm
BER_bpdn = NaN*ones(N_beta, N_delta, N_trials);     % BER for the BPDN estimator
BER_frame = NaN*ones(N_beta, N_delta, N_trials);    % BER for the proposed algorithm

time = clock;   % Start the stopwatch
for n = 1:N_trials
    for b = 1:N_beta
        for d = 1:N_delta
            % First determine the values of E[K] and M, and things that
            % depend on them
            M = round(delta(d)*N);      % Number of measurements at each timestep
            lambda_0 = beta(b)*M/N;     % Pr{s_n(t) = 1}
            
            % *************************************************************
            % Generate a signal/measurement pair realization for this suite
            % *************************************************************
            % Create an object to hold the signal model parameters           
            signalParams = SigGenParams('N', N, 'M', M, 'T', T, 'A_type', ...
                A_type, 'lambda', lambda_0, 'zeta', eta_0, 'sigma2', kappa_0, ...
                'alpha', alpha, 'p01', pz1, 'SNRmdB', SNRmdB);

            % Create the signal, measurements, matrices, etc.
            [x_true, y, A, support, ~, sig2e] = dcs_signal_gen_fxn(signalParams);
            
            
            % *************************************************************
            % Solve using the timestep-independent, support aware MMSE estimator
            % *************************************************************
            NMSE_tmp_iMMSE = NaN*ones(1,T);   % Will hold NMSE of each timestep
            x_iMMSE = cell(1,T);              % Holds iMMSE recovery
            for t = 1:T
                % First the timestep-independent (TI), support-aware (SA)
                % MMSE estimate
                s_mod = ones(N,1); s_mod(support{t}) = 2;
                % Compute MoG means and variances for current timestep
                mean_tmp = [0; (1-alpha)^(t-1)*eta_0];	 
                var_tmp = [0; kappa_0];
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
            end
            % Save the time-averaged NMSE and BER for the iMMSE genie
            NMSE_iMMSE(b,d,n) = sum(NMSE_tmp_iMMSE)/T;    % iMMSE t-avg. NMSE
            
            % Uncomment the following if you have SPGL1
%             % *************************************************************
%             % Solve using BPDN, with genie-aided parameter selection
%             % *************************************************************
%             sigma_BPDN = sqrt(2)*sqrt(sig2e)*exp(gammaln((mean(M)+1)/2) - ...
%                 gammaln(mean(M)/2));   % Default parameter (E[||y - Ax_true||_2])
%             mult = 0.1 : 0.1 : 2;   % Multiples of sigma_BPDN for x-val
%             NMSE_bpdn(b,d,n) = inf; % Initialize NMSE
%             x_bpdn = cell(1,T);
%             for i = 1:length(mult)  % Sweep across parameter range
%                 NMSE_tmp_bpdn = NaN*ones(1,T);   % Will hold NMSE of each timestep
%                 BER_tmp_bpdn = NaN*ones(1,T);    % Will hold BER of each timestep
%                 
%                 % Solve all timesteps with current parameter choice
%                 for t = 1:T
%                     % Solve BPDN using SPGL1 code package
%                     x_bpdn{t} = spg_bpdn(A{t}, y{t}, mult(i)*sigma_BPDN, ...
%                         spgSetParms('verbosity', 0));
%                     
%                     % Compute NMSE for current timestep
%                     NMSE_tmp_bpdn(t) = (norm(x_true{t} - x_bpdn{t})/...
%                         norm(x_true{t}))^2;
%                     if NMSE_tmp_bpdn(t) == inf
%                         % True signal currently has no active coefficients,
%                         % thus store a time-series averaged NSME here
%                         NMSE_tmp_bpdn(t) = norm(x_true{t} - x_bpdn{t})^2/...
%                             (norm([x_true{:}], 'fro')^2/(T));
%                     end
%                     s_mod = zeros(N,1); s_mod(support{t}) = 1;
%                     BER_tmp_bpdn(t) = sum(s_mod ~= (x_bpdn{t} ~= 0))/N;
%                 end
%                 
%                 % Now see if this parameter choice is superior to all
%                 % previous choices.  If so, update NMSE and BER
%                 if sum(NMSE_tmp_bpdn)/(T) < NMSE_bpdn(b,d,n)
%                     NMSE_bpdn(b,d,n) = sum(NMSE_tmp_bpdn)/(T);
%                     BER_bpdn(b,d,n) = sum(BER_tmp_bpdn)/(T);
%                     i_min = i;
%                 end
%             end
%             if i_min == 1 || i_min == length(mult)
%                 fprintf('Endpoint minimizer: beta = %f, delta = %f', ...
%                     beta(b), delta(d));
%             end
                    
            
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
            
            % Clear the Params and Options variables for re-use
            clear Params Options
            
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
            NMSE_smooth(b,d,n) = sum(NMSE_tmp_smooth)/T;	% Smoother t-avg. NMSE
            
            
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
            NMSE_frame(b,d,n) = sum(NMSE_tmp_frame)/T;    % Alg. t-avg. NMSE
            BER_frame(b,d,n) = sum(BER_tmp_frame)/T;      % Alg. t-avg. BER
            
        end     % delta (M/N)
    end         % beta (E[K]/M)
    
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

% Final save of data
clear A R
save(savefile);