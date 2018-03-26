% DYNAMICS_TEST      A test of the sparse time-series signal recovery 
% algorithm, that, for a fixed choice of signal model parameters, will 
% sweep the Markov transition probability, pz1 = Pr{s_n(t)=0 | s_n(t-1)=1},
% and the Gauss-Markov amplitude process correlation coefficient, alpha,
% over ranges of values, while storing the timestep-averaged normalized
% mean squared error (TNMSE) at each (pz1,alpha) pair.  The TNMSE will also
% be recorded for a support-aware genie smoother using 
% genie_multi_frame_fxn, as well as a naive belief propagation (BP) 
% estimator that independently recovers the unknown signals at each 
% timestep without considering any relationship between the recovered
% signals of adjacent timesteps, and a similar timestep-independent but 
% support-aware conditional MMSE estimator.
%
%
% SYNTAX:
% dynamics_test(N, M, T, lambda_0, SNRmdB, N_trials)
%
% INPUTS:
%   N               Number of unknowns at each timestep
%   M               Number of measurements at each timestep
%   T              	Number of timesteps - 1
%   lambda_0        IID activity probability, Pr{s_n(0) = 1}
%   SNRmdB          Per-measurement SNR in dB
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
%		- Created (03/24/11; JAZ)
%       - Fixed bugs introduced in v1.1 due to migration to SigGenParams
%           and DCSModelParams objects (12/07/15; JAZ)
% Version 1.2
%

function dynamics_test(N, M, T, lambda_0, SNRmdB, N_trials, fid)

%% Declare the parameters for the signal model and test setup

if nargin == 0
    N = 512;                % # parameters                              [dflt=256]
    M = round(N/3);         % # of measurements at each timestep        [dflt=N/3]
    T = 24;                 % # of timesteps - 1                        [dflt=9]
    lambda_0 = 0.15;        % Pr{s_n(0) = 1}                            [dflt=0.15]
    SNRmdB = 15;           	% Per-measurement SNR (in dB)               [dflt=15]
    N_trials = 50;          % Number of trials to avg. results over
    fid = 1;                % Write progress to screen?                 [dflt=1]
elseif nargin < 7
    fid = 1;    % Write to screen by default
end

warning('Smoother being run with AMP, not ABP')

% Declare additional static signal model parameters
A_type = 1;             % 1=iid CN, 2=rademacher, 3=subsampled DFT	[dflt=1]
eta_0 = 0;            	% Complex mean of active coefficients       [dflt=0]
kappa_0 = 1;        	% Circular variance of active coefficients	[dflt=1]
eps = 1e-7;             % "Squelch" parameter for inactives         [dflt=1e-4]

% Algorithm execution parameters
smooth_iter = 5;        % Number of forward/backward "smoothing" passes 
                        % to perform [dflt: 5]
eq_iter = 25;          	% Number of equalization iterations to perform,
                        % per forward/backward pass, at each timestep [dflt: 25]
alg = 2;                % Type of BP algorithm to use during equalization 
                        % at each timestep: (1) for standard BP, 
                        % (2) for AMP [dflt: 2]
update = false;       	% Update DCS-AMP parameters using EM learning?
tau = 1 - 1e-2;         % tau-thresholding parameter

% Test setup parameters
Q = 5;                                 % Grid sampling density
pz1 = linspace(0, .15, Q);           	% Pr{s_n(t) = 0 | s_n(t-1) = 1}
alpha = logspace(-3, log10(0.95), Q);	% Gauss-Markov process correlation
N_pz1 = length(pz1);
N_alpha = length(alpha);
if numel(lambda_0) ~= 1, error('Please input scalar lambda_0'); end

% Filename for saving data
savefile = [datestr(now, 'ddmmmyyyy_HHMMSSFFF') '_dynamics_test_N_' ...
    num2str(N) '_M_' num2str(M) '_T_' num2str(T) '_lambda0_' ...
    num2str(lambda_0) '_eta0_' num2str(eta_0) '_kappa0_' ...
    num2str(kappa_0) '.mat'];

% Randomly seed the RNG and save its state for reference purposes
NewStream = RandStream.create('mrg32k3a', 'Seed', sum(100*clock));
RandStream.setGlobalStream(NewStream);
savedState = NewStream.State;


%% Execute the phase plane test

% Add necessary path
path(path, '../../Functions')
path(path, '../../ClassDefs')

% Create arrays to store the NMSEs and BERs
TNMSE_smooth = NaN*ones(N_pz1, N_alpha, N_trials);   % TNMSE for genie smoother
TNMSE_iMMSE = NaN*ones(N_pz1, N_alpha, N_trials);    % TNMSE for TI, SA MMSE estimator
TNMSE_naive = NaN*ones(N_pz1, N_alpha, N_trials);    % TNMSE for the naive BP recovery
TNMSE_frame = NaN*ones(N_pz1, N_alpha, N_trials);    % TNMSE for the proposed algorithm
TBER_naive = NaN*ones(N_pz1, N_alpha, N_trials);     % TBER for the naive BP recovery
TBER_frame = NaN*ones(N_pz1, N_alpha, N_trials);     % TBER for the proposed algorithm

try % Matlabpool opening
    % Open a matlabpool, if one is not already open
    if matlabpool('size') == 0
        matlabpool open
        fprintf('Opened a matlabpool with %d workers', matlabpool('size'));
    end
catch
    error('Error in opening a matlabpool')
end
% Set the RNGs of the parallel workers based on the common random seed
% obtained above
parfor i = 1:matlabpool('size')
    RandStream.setGlobalStream(NewStream);
    pause(3);
end

time = clock;   % Start the stopwatch
for n = 1:N_trials
    for i = 1:N_pz1     % Chosen pz1
        parfor j = 1:N_alpha	% Chosen alpha
            % If running in parallel, for reproducibility we should have
            % each worker move the RNG stream to a predefined substream
            % based on the iteration index
            Stream = RandStream.getGlobalStream;
            Stream.Substream = j;   % Set substream to iteration index
            
            % First determine the values of p1z and rho
            rho = (2 - alpha(j))*kappa_0/alpha(j);  % Driving noise variance (leads
                                                    % to constant amplitude
                                                    % variance
            
            % *************************************************************
            % Generate a signal/measurement pair realization for this suite
            % *************************************************************
            % Create an object to hold the signal model parameters           
            signalParams = SigGenParams('N', N, 'M', M, 'T', T, 'A_type', ...
                A_type, 'lambda', lambda_0, 'zeta', eta_0, 'sigma2', kappa_0, ...
                'alpha', alpha(j), 'p01', pz1(i), 'SNRmdB', SNRmdB);

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
            var_tmp = [0; (kappa_0 - alpha(j)^2*rho)/(1 - alpha(j))^2];
            for t = 1:T
                % First the timestep-independent (TI), support-aware (SA)
                % MMSE estimate
                s_mod = ones(N,1); s_mod(support{t}) = 2;
                % Compute MoG means and variances for current timestep
                mean_tmp = [0; eta_0];	 
                var_tmp = [0; (1-alpha(j))^2*var_tmp(2) + alpha(j)^2*rho];
                R = A{t}*diag(var_tmp(s_mod))*A{t}' + sig2e*eye(M);
                x_iMMSE{t} = mean_tmp(s_mod) + ...
                    diag(var_tmp(s_mod))*A{t}'*(R\(y{t} - A{t}*mean_tmp(s_mod)));
                NMSE_tmp_iMMSE(t) = (norm(x_true{t} - x_iMMSE{t})/...
                    norm(x_true{t}))^2;
                if NMSE_tmp_iMMSE(t) == inf
                    % True signal currently has no active coefficients,
                    % thus store a time-series averaged NSME here
                    NMSE_tmp_iMMSE(t) = norm(x_true{t} - x_iMMSE{t})^2/...
                        (norm([x_true{:}], 'fro')^2/T);
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
                        (norm([x_true{:}], 'fro')^2/(T+1));
                end
                s_mod = zeros(N,1); s_mod(support{t}) = 1;
                BER_tmp_naive(t) = sum(s_mod ~= (OutMessages.pi > 0.5))/N;
            end

            % Save the time-averaged NMSE and BER for the genie and naive BP
            TNMSE_iMMSE(i,j,n) = sum(NMSE_tmp_iMMSE)/T;    % Genie TNMSE
            TNMSE_naive(i,j,n) = sum(NMSE_tmp_naive)/T;    % Naive TNMSE
            TBER_naive(i,j,n) = sum(BER_tmp_naive)/T;      % Naive TBER


            % *************************************************************
            % Solve using the support-aware, timestep-aware genie smoother
            % *************************************************************
            
            if alg == 1, SKSalg = 'BP'; else SKSalg = 'AMP'; end
            
            % Construct class objects for model parameters, copying over
            % the true signal generating parameters
            SKS_ModelParams = DCSModelParams(signalParams, sig2e);
            SKS_RunOptions = Options('smooth_iters', smooth_iter, ...
                'inner_iters', eq_iter, 'alg', SKSalg, 'eps', eps, ...
                'tau', tau);

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
            TNMSE_smooth(i,j,n) = sum(NMSE_tmp_smooth)/T;	% Smoother TNMSE


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
                        (norm([x_true{:}], 'fro')^2/(T+1));
                elseif isnan(NMSE_tmp_frame(t))
                    NMSE_tmp_frame(t) = 0;
                end
                s_mod = zeros(N,1); s_mod(support{t}) = 1;
                BER_tmp_frame(t) = sum(s_mod ~= (lambda_frame{t} > 0.5))/N;
            end
            TNMSE_frame(i,j,n) = sum(NMSE_tmp_frame)/T;    % Alg. TNMSE
            TBER_frame(i,j,n) = sum(BER_tmp_frame)/T;      % Alg. TBER
            
            
%             fprintf('alpha: %d of %d\n', j, N_alpha)
            if fid ~= 1, fid2 = fopen([savefile(1:end-3) 'txt'], 'w'); 
            elseif fid == 1, fid2 = 1; end
            fprintf(fid2, 'Progress: %g\n', (N_pz1*(i-1) + j) / N_alpha / N_pz1);
            if fid ~= 1, fclose(fid2); end
        end     % alpha
%         fprintf('p01: %d of %d\n', i, N_pz1)
    end     % p01
    
    % Having completed one sweep of the (beta, delta) grid, estimate the
    % time remaining and save the temporary data
    est_time = (N_trials - n)*(etime(clock, time)/n)/60/60;
    fprintf('(%d/%d) Estimated time remaining: %3.2f hours', ...
        n, N_trials, est_time);
    clear A R
    save(savefile);
end

fprintf('Total time elapsed for test: %3.2f hours', etime(clock, time)/60/60);

try
    % Close the matlabpool
    matlabpool close
catch
    disp('No matlabpool to close')
end

% Final save of data
clear A R
save(savefile);