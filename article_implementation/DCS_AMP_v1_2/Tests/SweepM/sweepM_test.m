% SWEEPM_TEST	A test of the sparse time-series signal recovery 
% algorithm, that, for a fixed choice of signal model parameters, will 
% sweep the number-of-measurements-per-number-of-unknowns ratio, (a.k.a. 
% the undersampling ratio (M/N)), delta over a range of values, while 
% storing the timestep-averaged normalized mean squared error (t-avg. NMSE)
% at each delta.  The t-avg. NMSE will also be recorded for a 
% support-aware genie smoother using genie_multi_frame_fxn, as well
% as a basis pursuit denoising (BPDN) estimator that independently recovers
% the unknown signals at each timestep without considering any relationship
% between the recovered signals of adjacent timesteps, a similar
% timestep-independent but support-aware conditional MMSE estimator, and a
% timestep-independent BP estimator.  In the BPDN solver, there is a 
% user-specified parameter sigma that must be set in order to solve 
% min ||x||_1 s.t. ||Ax - y||_2 < sigma.  Since the optimal choice of sigma 
% is not known, for each realization a cross-validation procedure will run 
% multiple recoveries using the BPDN solver with different choices of 
% sigma.  A genie will then select the value of sigma that minimizes t-avg. 
% NMSE for that signal realization.
%
%
% SYNTAX:
% sweepM_test(N, T, lambda, pz1, alpha, SNRmdB, N_trials)
%
% INPUTS:
%   N               Number of unknowns at each timestep
%   T              	Number of timesteps - 1
%   lambda          Pr{s_n(0) = 1}
%   pz1             Pr{s_n(t) = 0 | s_n(t-1) = 1}
%   alpha         	Innovation rate of thetas (1 = total)
%   SNRmdB         	Per-measurement SNR (in dB)
%   N_trials     	Number of trials to avg. results over per (beta,delta)
%
% OUTPUTS:
% This function will save the timestep-averaged NMSEs and support error
% rates for a support-aware genie smoother, an independent BPDN estimator, 
% a naive support-aware MMSE estimator, an independent BP estimator, and 
% the proposed multi-timestep recovery algorithm in a .MAT file whose 
% filename is created at runtime based on input parameters and the 
% time/date of the start of execution
% Suffix keywords:
%   _smooth     The support-aware timestep-aware genie smoother
%   _iMMSE      The timestep-independent, support-aware MMSE estimator
%   _bpdn       The timestep-independent BPDN estimator
%   _bp         The timestep-independent BP estimator
%   _frame      The proposed multi-timestep BP algorithm (sp_multi_frame_fxn)
%

%
% Coded by: Justin Ziniel, The Ohio State Univ.
% E-mail: ziniel.1@osu.edu
% Last change: 12/07/15
% Change summary: 
%		- Created from sweepM_test v0.2 (12/09/10; JAZ)
%       - Fixed bugs introduced in v1.1 due to migration to SigGenParams
%           and DCSModelParams objects (12/07/15; JAZ)
% Version 1.2
%

function sweepM_test(N, T, lambda, pz1, alpha, SNRmdB, N_trials)

%% Declare the parameters for the signal model and test setup

if nargin == 0
    N = 512;                % # parameters                              [dflt=256]
    T = 25;                 % # of timesteps - 1                        [dflt=9]
    lambda = 0.20;          % prior probability of non-zero tap         [dflt=0.10]
    pz1 = 0.05;             % Pr{s_n(t) = eps | s_n(t-1) = 1}           [dflt=0.05]
    alpha = 0.01;           % Innovation rate of thetas (1 = total)     [dflt=0.10]
    SNRmdB = 15;           	% Per-measurement SNR (in dB)               [dflt=15]
    N_trials = 100;           % Number of trials to avg. results over per delta
end

% Declare additional static signal model parameters
A_type = 1;             % 1=iid CN, 2=rademacher, 3=subsampled DFT	[dflt=1]
eta_0 = 0;            	% Complex mean of active coefficients       [dflt=0]
kappa_0 = 1;        	% Circular variance of active coefficients	[dflt=1]
eps = 1e-5;             % "Squelch" parameter for inactives         [dflt=1e-4]

% Algorithm execution parameters
smooth_iter = 5;        % Number of forward/backward "smoothing" passes 
                        % to perform [dflt: 5]
eq_iter = 25;          	% Number of equalization iterations to perform,
                        % per forward/backward pass, at each timestep [dflt: 25]
alg = 2;                % Type of BP algorithm to use during equalization 
                        % at each timestep: (1) for standard BP, 
                        % (2) for AMP [dflt: 2]
update = false;       	% Update DCS-AMP parameters using EM learning?

% Test setup parameters
Q = 30;                            	% Grid sampling density
delta = linspace(0.05, 0.95, Q);  	% Undersampling ratio M/N
N_delta = length(delta);

% Filename for saving data
savefile = [datestr(now, 'ddmmmyyyy_HHMMSSFFF') '_sweepM_test_N_' ...
    num2str(N) '_T_' num2str(T) '_lambda_' num2str(lambda) '_pz1_' ...
    num2str(pz1) '_alpha_' num2str(alpha) '_SNRm_' num2str(SNRmdB) '.mat'];

% Randomly seed the RNG and save its state for reference purposes
NewStream = RandStream.create('mt19937ar', 'Seed', sum(100*clock));
RandStream.setGlobalStream(NewStream);
savedState = NewStream.State;


%% Execute the phase plane test

% Add necessary path
path(path, '../../Functions')
% path(path, '../../../../../SparseSolvers/spgl1-1.7')

% Open a Matlab worker pool for parallel computation
try % Matlabpool opening
    % Open a matlabpool, if one is not already open
    if matlabpool('size') == 0
        matlabpool open
        fprintf('Opened a matlabpool with %d workers', matlabpool('size'));
    end
catch
    error('Error in opening a matlabpool')
end

% Create arrays to store the NMSEs and BERs
NMSE_smooth = NaN*ones(N_trials, N_delta);	% NMSE for genie smoother
NMSE_iMMSE = NaN*ones(N_trials, N_delta);   % NMSE for TI, SA MMSE estimator
NMSE_bpdn = NaN*ones(N_trials, N_delta);    % NMSE for the BPDN estimator
NMSE_bp = NaN*ones(N_trials, N_delta);      % NMSE for the TI BP estimator
NMSE_frame = NaN*ones(N_trials, N_delta);   % NMSE for the proposed algorithm
SER_bpdn = NaN*ones(N_trials, N_delta);     % SER for the BPDN estimator
SER_bp = NaN*ones(N_trials, N_delta);       % SER for the BP estimator
SER_frame = NaN*ones(N_trials, N_delta);    % SER for the proposed algorithm

time = clock;   % Start the stopwatch
for n = 1:N_trials
    for d = 1:N_delta
        % First determine the value M, and things that depend on it
        M = round(delta(d)*N);      % Number of measurements at each timestep

        % *************************************************************
        % Generate a signal/measurement pair realization for this suite
        % *************************************************************        
        % Create an object to hold the signal model parameters
        signalParams = SigGenParams('N', N, 'M', M, 'T', T, 'A_type', ...
            A_type, 'lambda', lambda, 'zeta', eta_0, 'sigma2', kappa_0, ...
            'alpha', alpha, 'p01', pz1, 'SNRmdB', SNRmdB);
        
        % Create the signal, measurements, matrices, etc.
        [x_true, y, A, support, NNZ, sig2e] = dcs_signal_gen_fxn(signalParams);


        % *************************************************************
        % Solve using the TI, SA MMSE estimator and the naive BP alg.
        % *************************************************************
        NMSE_tmp_iMMSE = NaN*ones(1,T);% Will hold NMSE of each timestep
        NMSE_tmp_bp = NaN*ones(1,T);   % Will hold NMSE of each timestep
        SER_tmp_bp = NaN*ones(1,T);    % Will hold BER of each timestep
        parfor t = 1:T
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

            % Next, the naive BP recovery
            [x_bp_tmp, OutMessages] = sp_frame_fxn(y{t}, A{t}, ...
                struct('pi', lambda, 'xi', mean_tmp(2), 'psi', ...
                var_tmp(2), 'eps', eps, 'sig2e', sig2e), ...
                struct('iter', eq_iter, 'alg', alg));
            x_bp{t} = x_bp_tmp;
            NMSE_tmp_bp(t) = (norm(x_true{t} - x_bp{t})/...
                norm(x_true{t}))^2;
            if NMSE_tmp_bp(t) == inf
                % True signal currently has no active coefficients,
                % thus store a time-series averaged NSME here
                NMSE_tmp_bp(t) = norm(x_true{t} - x_bp{t})^2/...
                    (norm([x_true{:}], 'fro')^2/(T)); %#ok<*PFBNS>
            end
            s_mod = zeros(N,1); s_mod(support{t}) = 1;
            SER_tmp_bp(t) = sum(s_mod ~= (OutMessages.pi > 0.5))/NNZ(t);
            if SER_tmp_bp(t) == inf
                % There were no active coefficients at this time, so just
                % use the average number of active coefficients when
                % calculating SER at this timestep
                SER_tmp_bp(t) = sum(s_mod ~= (OutMessages.pi > 0.5))/...
                    mean(NNZ);
            end
        end

        % Save the time-averaged NMSE and BER for the genie and naive BP
        NMSE_iMMSE(n,d) = sum(NMSE_tmp_iMMSE)/T;	% Genie t-avg. NMSE
        NMSE_bp(n,d) = sum(NMSE_tmp_bp)/T;        % Naive t-avg. NMSE
        SER_bp(n,d) = sum(SER_tmp_bp)/T;          % TI BP t-avg. SER


        % Uncomment these lines if you have SPGL1
%         % *************************************************************
%         % Solve using BPDN, with genie-aided parameter selection
%         % *************************************************************
%         sigma_BPDN = sqrt(2)*sqrt(sig2e)*exp(gammaln((mean(M)+1)/2) - ...
%             gammaln(mean(M)/2));   % Default parameter (E[||y - Ax_true||_2])
%         mult = 0.1 : 0.1 : 2;   % Multiples of sigma_BPDN for x-val
%         x_bpdn = cell(length(mult));    % Init. storage for x_bpdn
%         NMSE_tmp_bpdn = cell(length(mult),1);
%         SER_tmp_bpdn = cell(length(mult),1);
%         parfor i = 1:length(mult)  % Sweep across parameter range
%             % Solve all timesteps with current parameter choice
%             for t = 1:T
%                 % Solve BPDN using SPGL1 code package
%                 x_bpdn{i}{t} = spg_bpdn(A{t}, y{t}, mult(i)*sigma_BPDN, ...
%                     spgSetParms('verbosity', 0));
% 
%                 % Compute NMSE for current timestep
%                 NMSE_tmp_bpdn{i}(t) = (norm(x_true{t} - x_bpdn{i}{t})/...
%                     norm(x_true{t}))^2;
%                 if NMSE_tmp_bpdn{i}(t) == inf
%                     % True signal currently has no active coefficients,
%                     % thus store a time-series averaged NSME here
%                     NMSE_tmp_bpdn{i}(t) = norm(x_true{t} - x_bpdn{i}{t})^2/...
%                         (norm([x_true{:}], 'fro')^2/(T));
%                 end
%                 s_mod = zeros(N,1); s_mod(support{t}) = 1;
%                 SER_tmp_bpdn{i}(t) = sum(s_mod ~= (x_bpdn{i}{t} ~= 0))/NNZ(t);
%                 if SER_tmp_bpdn{i}(t) == inf
%                     % There were no active coefficients at this time, so just
%                     % use the average number of active coefficients when
%                     % calculating SER at this timestep
%                     SER_tmp_bpdn{i}(t) = sum(s_mod ~= (x_bpdn{i}{t} ~= 0))/...
%                         mean(NNZ);
%                 end
%             end
%         end
%         % Now, sum up the NMSEs and SERs across time for each choice of the
%         % BPDN sigma parameter.  Select and store the one that minimizes
%         % NMSE
%         NMSE_tmp2_bpdn = NaN*ones(length(mult),T);
%         SER_tmp2_bpdn = NaN*ones(length(mult),T);
%         for i = 1:length(mult)
%             NMSE_tmp2_bpdn(i,:) = NMSE_tmp_bpdn{i};
%             SER_tmp2_bpdn(i,:) = SER_tmp_bpdn{i};
%         end
%         [NMSE_bpdn(n,d), i_min] = min(sum(NMSE_tmp2_bpdn, 2)/(T));
%         SER_bpdn(n,d) = sum(SER_tmp2_bpdn(i_min,:))/(T);
%         if i_min == 1 || i_min == length(mult)
%             disp(sprintf('Endpoint minimizer: delta = %f', delta(d)));
%         end


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
        NMSE_smooth(n,d) = sum(NMSE_tmp_smooth)/T;	% Smoother t-avg. NMSE


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
        SER_tmp_frame = NaN*ones(1,T);    % Will hold BER of each timestep
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
            SER_tmp_frame(t) = sum(s_mod ~= (lambda_frame{t} > 0.5))/NNZ(t);
            if SER_tmp_frame(t) == inf
                % There were no active coefficients at this time, so just
                % use the average number of active coefficients when
                % calculating SER at this timestep
                SER_tmp_frame(t) = sum(s_mod ~= (lambda_frame{t} > 0.5))/...
                    mean(NNZ);
            end
        end
        NMSE_frame(n,d) = sum(NMSE_tmp_frame)/T;    % Alg. t-avg. NMSE
        SER_frame(n,d) = sum(SER_tmp_frame)/T;      % Alg. t-avg. BER
        
        disp(sprintf('Finished delta iter: %d', d))

    end     % delta (M/N)
    
    % Having completed one sweep of the (beta, delta) grid, estimate the
    % time remaining and save the temporary data
    est_time = (N_trials - n)*(etime(clock, time)/n)/60/60;
    disp(sprintf('(%d/%d) Estimated time remaining: %3.2f hours', ...
        n, N_trials, est_time));
    clear A R
    save(savefile);
end

% Clsoe the Matlab worker pool
matlabpool close

disp(sprintf('Total time elapsed for test: %3.2f hours', ...
    etime(clock, time)/60/60));

% Final save of data
clear A R
save(savefile);