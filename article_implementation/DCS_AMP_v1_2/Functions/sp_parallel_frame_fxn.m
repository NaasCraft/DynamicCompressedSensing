% SP_PARALLEL_FRAME_FXN     Function for performing soft-input, soft-output 
% sparse reconstruction of a time-series of complex-valued signals, {x(t)}, 
% t = 0, 1, ..., T, with each x(t) being N-dimensional.  Complex-valued
% observations, {y(t)}, are obtained through a noisy linear combination,
% y(t) = A(t)x(t) + e(t), with y(t) being an M(t) dimensional vector of
% observations, A(t) being an M(t)-by-N complex-valued measurement
% matrix, and e(t) being circular, complex, additive Gaussian noise with
% covariance matrix sig2e*eye(M(t)).
%
% At any given time, t, it is assumed that the coefficients of x(t),
% x_n(t), can be represented by the product of a discrete variable, s_n(t),
% and a continuous-valued variable, theta_n(t), that is, x_n(t) =
% s_n(t)*theta_n(t).  s_n(t) either takes the value 0 or 1, and 
% follows a Bernoulli distribution with Pr{s_n(t) = 1} = lambda(n).  
% theta_n(t) follows a complex Gaussian distribution with mean eta(n) 
% and variance kappa(n).
%
% Across frames (timesteps), s_n(t) evolves according to a Markov chain,
% characterized by the transition probability, p01 == Pr{s_n(t) = 0 | 
% s_n(t-1) = 1}.  theta_n(t) evolves according to a Gauss-Markov process, 
% i.e., theta_n(t) = (1 - alpha)*theta_n(t-1) + alpha*w_n(t), where alpha 
% is a scalar between 0 and 1, and w_n(t) is complex Gaussian driving noise 
% with zero mean and variance chosen to maintain a steady-state amplitude 
% variance of kappa(n).
%
% Inference is performed using belief propagation (BP) and approximate 
% message passing (AMP).  At each iteration, messages are passed in
% parallel into each frame, and a recovery of x(t) is produced for each
% frame.  Then, messages are passed out of the frame, and a
% forward/backward pass is conducted to update beliefs on the values of
% {s(t)} and {theta(t)}.  This message update schedule is not recommended
% when one wishes to perform (causal) filtering.
%
% SYNTAX:
% [x_hat, v_hat, lambda_hat, StateOut] = ...
%       sp_parallel_frame_fxn(y, A, Params, RunOptions)
%
% INPUTS:
% y                 A 1-by-T+1 cell array of observation vectors.  Each
%                   complex-valued observation vector is defined to be of
%                   length M(t), t = 0, 1, ..., T
% A                 A 1-by-T+1 cell array of complex-valued measurement
%                   matrices, with each matrix of dimension M(t)-by-N, or a
%                   1-by-T+1 cell array of function handles which implement
%                   matrix-vector multiplications.  See below for the form 
%                   of these function handles
% Params            Either an object of the DCSModelParams class, or
%                   (now deprecated) a structure containing signal model 
%                   parameters with the following fields:
%   .lambda         N-by-1 vector of activity probabilities for s(t),
%                   i.e., Pr{s_n(t) = 1} = Params.lambda(n)
%   .p01            Scalar Markov chain transition probability,
%                   Pr{s_n(t) = 0 | s_n(t-1) = 1}
%   .eta        	N-by-1 complex vector of the means of the amplitude
%                   evolution process, i.e., E[theta(n,t)] = eta(n)
%   .kappa          N-by-1 vector of theta variances, i.e.,
%                   var{theta_n(t)} = Params.kappa(n)
%   .alpha          Scalar Gauss-Markov "innovation rate" parameter, btwn.
%                   0 and 1, (see above). Can also be N-by-1 if coefficents
%                   are being segregated into groups that each have
%                   different alpha values (see Options.upd_groups)
%   .sig2e          Scalar variance of circular, complex, additive Gaussian
%                   measurement noise
% RunOptions       	An *optional* argument consisting of an object of the
%                   Options class (see Options.m in the ClassDefs folder),
%                   or a (now deprecated) structure of runtime 
%                   configurations with the following fields:
%   .smooth_iter    Maximum number of smoothing iterations to
%                   perform.  If one wishes to perform filtering, i.e.,
%                   causal message passing only, then set this field to
%                   -1 [dflt: 5]
%   .min_iters      Minimum number of smoothing iterations to perform
%                   [dflt: 5]
%   .eq_iter        Maximum number of inner AMP iterations to perform, per
%                   forward/backward pass, at each timestep [dflt: 25]
%   .alg            Type of BP algorithm to use during equalization at each
%                   timestep: (1) for standard BP, (2) for AMP [dflt: 2]
%   .update         Treat the hyperparameters given in Params as random
%                   quantities, and update after each forward\backward
%                   pass (1), or treat them as fixed (0) [dflt: 0]
%   .update_list    A cell array of strings indicating which parameters (in
%                   the Params structure) should be updated, if
%                   RunOptions.update = 1.  By default, all parameters are
%                   updated, i.e., RunOptions.update_list = {'lambda', ...
%                   'p01', 'p10', 'eta', 'kappa', 'alpha', 'rho', ...
%                   'sig2e'} ('eps' is not learnable).  Removing any
%                   parameters from the cell array will keep them fixed at
%                   their initial value
%   .upd_groups     This field can be used when updating hyperparameters to
%                   force the algorithm to estimate different values of the
%                   hyperparameters for different subsets of the
%                   coefficients.  If specified, RunOptions.upd_groups must be
%                   a cell array in which each cell contains indices of
%                   coefficients whose parameters are to be estimated
%                   simultaneously.  The total number of cells determines
%                   the total number of unique subsets, and hence, unique
%                   parameter estimates. [dflt: {1:N}, (i.e. single group)]
%   .verbose        Output information during execution (1), or display
%                   nothing (0) [dflt: 0]
%   .StateIn        An optional structure that contains variables needed to
%                   warm-start this function.  Can be obtained initially
%                   from the 'StateOut' output of this function.
%   .eps            Real-valued scalar, (<< 1), needed to provide
%                   approximations of outgoing BP messages of active
%                   coefficient means (xi's) and variances (psi's) [dflt:
%                   1e-5]
%   .tau            Optional parameter.  If tau = -1, a Taylor series
%                   approximation is used to determine the outgoing AMP
%                   messages OutMessages.xi and OutMessages.psi.  If a
%                   positive value between zero and one is passed for tau, 
%                   e.g., tau = 1 - 1e-4, it will be used as a threshold
%                   on the incoming AMP activity probabilities to
%                   determine whether to pass an informative or 
%                   non-informative message via OutMessages.xi and
%                   OutMessages.psi [dflt: -1]
%
% OUTPUTS:
% x_hat             A 1-by-T+1 cell array of length-N MMSE estimates of the
%                   unknown time series
% v_hat             A 1-by-T+1 cell array of length-N estimates of the
%                   variances of the coefficient estimates in x_hat
% lambda_hat        A 1-by-T+1 cell array of length-N estimates of the
%                   probabilities of activity for {s(t)}
% StateOut          A structure containing AMP state outputs, useful for
%                   warm-starting this function if running again under
%                   different parameter configurations (pass this to
%                   RunOptions.StateIn)
%
%
% *** If input A is a cell array of function handles, each handle should be
%   of the form @(x,mode). If mode == 1, then the function handle returns 
%   an M(t)-by-1 vector that is the result of A(t)*x.  If mode == 2, then 
%   the function handle returns an N-by-1 vector that is the result of 
%   A(t)'*x. ***
%

%
% Coded by: Justin Ziniel, The Ohio State Univ.
% E-mail: ziniel.1@osu.edu
% Last change: 12/05/12
% Change summary: 
%		- Created from sp_multi_frame_fxn v0.3 (04/26/11; JAZ)
%       - Added option to warm-start the algorithm (06/20/11; JAZ; v1.0)
%       - Added ability to specify which parameters to update, and which to
%         leave fixed in RunOptions.update_list; removed p10 and rho from
%         list of user-specified parameters; added DCSModelParams class
%         parameterization as preferred format henceforth 
%         (12/05/12; JAZ; v1.1)
% Version 1.2
%

function [x_hat, v_hat, lambda_hat, StateOut] = ...
    sp_parallel_frame_fxn(y, A, Params, RunOptions)

%% Begin by creating a default test case matched to the signal model

if nargin == 0      % No user-provided variables, create default test
    
%     rand('twister',5489);
%     randn('state',0);
    
    % Signal generation parameters
    N = 1024;               % # parameters                              [dflt=256]
    T = 10;                 % # of timesteps - 1                       [dflt=9]
    M = round(N/4*ones(1,T+1));	% # observations                            [dflt=N/4]
    lambda = 0.12;          % prior probability of non-zero tap         [dflt=0.04]
    p01 = 0.05;             % Pr{s_n(t) = 0 | s_n(t-1) = 1}           [dflt=0.05]
    p10 = lambda*p01/(1 - lambda);      % Pr{s_n(t) = 1 | s_n(t-1) = 0}   
                                            % [dflt=lambda*p01/(1 - lambda)]
    A_type = 4;             % 1=iid CN, 2=rademacher, 3=subsampled DFT, 4=identical CN [dflt=1]
    explicitA = 1;          % Matrix is explicitly given
    eta = 0;            	% Complex mean of active coefficients       [dflt=0]
    kappa = 1;              % Circular variance of active coefficients	[dflt=1]
    alpha = 0.10;           % Innovation rate of thetas (1 = total)     [dflt=0.10]
    rho = (2 - alpha)*kappa/alpha;        % Driving noise variance    
                                            % [dflt=(2 - alpha)*kappa/alpha]
    eps = 1e-7;             % "Squelch" parameter for inactives         [dflt=1e-7]
    SNRmdB = 25;           	% Per-measurement SNR (in dB)               [dflt=15]
    version = 'R';          % Complex-valued ('C') or real-valued ('R') signal  [dflt=1]
    
    % Algorithm parameters
    smooth_iter = 25;       % Number of forward/backward "smoothing" passes 
                            % to perform [dflt: 5]
	eq_iter = 50;          	% Number of equalization iterations to perform,
                            % per forward/backward pass, at each timestep [dflt: 25]
	alg = 2;                % Type of BP algorithm to use during equalization 
                            % at each timestep: (1) for standard BP, 
                            % (2) for AMP [dflt: 2]
    
    update = 1;
    update_list = {'lambda', 'p01', 'p10', 'eta', 'kappa', ...
        'alpha', 'rho', 'sig2e'};
    verbose = 1;
    
    for l = 1       % (Generate time series from parameters)
        % Construct a SigGenParams object for use in generating a sample 
        % time series
        SignalParams = SigGenParams(N, M(1), T+1, A_type, lambda, eta, ...
            kappa, alpha, p01, SNRmdB, version);

        % ***********************************************************
        % Generate the time series according to the above parameters
        [x_true, y, A, support, NNZ, sig2e] = ...
            dcs_signal_gen_fxn(SignalParams);
        % Plot the true signal
        figure(1); clf; imagesc(abs([x_true{:}])); xlabel('Timestep (t)');
        ylabel('|x_n^{(t)}|'); title('True signal magnitude trajectories');
        colorbar;
        % ***********************************************************
        
        % ***********************************************************
        % Compute the support-aware smoothed genie estimate
        disp('Warning: Using AMP instead of ABP for support-aware smoother');
        GenieModelParams = DCSModelParams(SignalParams, sig2e);
        RunOptions = Options('smooth_iters', 10, 'inner_iters', ...
            eq_iter, 'alg', 'AMP', 'update', false, 'eps', eps);
        [x_genie, ~] = genie_dcs_fxn(y, A, support, GenieModelParams, ...
            RunOptions);
        % ***********************************************************
        
        % ***********************************************************
        % Compute the multi-timestep AMP solution using the alternative
        % message passing schedule, for comparison
        AltSchedModelParams = DCSModelParams(SignalParams, sig2e);
        [x_damp, ~] = sp_multi_frame_fxn(y, A, AltSchedModelParams, ...
            RunOptions);
        % ***********************************************************
        
        
        % Compute the support-aware, timestep-independent genie MMSE 
        % estimate and NMSE, as well as a BP recovery w/o any 
        % multi-timestep information
        [x_indgenie, x_bp] = deal(cell(1,T+1));
        [NMSE_indgenie, NMSE_bp, NMSE_genie, NMSE_damp, SNRdB_emp] = ...
            deal(NaN(1,T+1));
        for t = 1:T+1
            % First the support-aware, timestep-independent genie MMSE 
            % estimate
            s_mod = ones(N,1); s_mod(support{t}) = 2;
            % Compute MoG means and variances for current timestep
            mean_tmp = [0; eta];	 
            var_tmp = [0; ((1 - (1-alpha)^(2*(t-1))) / (1 - (1-alpha)^2)) * ...
                alpha^2*rho + (1-alpha)^(2*(t-1))*kappa];
            R = A{t}*diag(var_tmp(s_mod))*A{t}' + sig2e*eye(M(t));
            x_indgenie{t} = mean_tmp(s_mod) + diag(var_tmp(s_mod))*A{t}'*...
                (R\(y{t} - A{t}*mean_tmp(s_mod)));
            NMSE_indgenie(t) = (norm(x_true{t} - x_indgenie{t})/norm(x_true{t}))^2;
            
            % Next, the BP recovery
            x_bp{t} = sp_frame_fxn(y{t}, A{t}, struct('pi', lambda, ...
                'xi', mean_tmp(2), 'psi', var_tmp(2), 'eps', eps, ...
                'sig2e', sig2e));
            NMSE_bp(t) = (norm(x_true{t} - x_bp{t})/norm(x_true{t}))^2;
            
            % Smoothed genie NMSE computation
            NMSE_genie(t) = (norm(x_true{t} - x_genie{t})/norm(x_true{t}))^2;
            
            % Alternative message scheduling AMP method NMSE computation
            NMSE_damp(t) = (norm(x_true{t} - x_damp{t})/norm(x_true{t}))^2;

            % Lastly, compute the empirical SNR of the true signal (in dB)
            % Store the empirical SNR at this timestep
            SNRdB_emp(t) = 20*log10(norm(A{t}*x_true{t})/...
                norm(y{t} - A{t}*x_true{t}));
        end
        % Support-aware smoothed genie avg NMSE (dB)
        NMSEavg_dB_genie = 10*log10(sum(NMSE_genie)/(T + 1));
        % Timestep-independent genie avg NMSE (dB)
        NMSEavg_dB_indgenie = 10*log10(sum(NMSE_indgenie)/(T + 1));
        % BP avg NMSE (dB)
        NMSEavg_dB_bp = 10*log10(sum(NMSE_bp)/(T + 1));
        % Alternative message schedule avg NMSE (dB)
        NMSEavg_dB_damp = 10*log10(sum(NMSE_damp)/(T + 1));
        
        % Plot the support-aware genie MMSE recovery
        figure(2); imagesc(abs([x_genie{:}])); colorbar; 
        title('Support-aware smoothed genie MMSE estimate');
        xlabel('Timestep (t)'); ylabel('|x_{genie}(t)|');
        K_handle = line(1:T+1, NNZ); 
        set(K_handle, 'Color', 'Cyan', 'LineStyle', '-.'); 
        M_handle = line(1:T+1, M); 
        set(M_handle, 'Color', 'White', 'LineStyle', '--'); 
        legend_handle = legend('K(t)', 'M(t)');
        set(legend_handle, 'Color', [.392 .475 .635])
%         pause(5)
        figure(2); imagesc(abs([x_bp{:}])); colorbar; 
        title('Independent BP recoveries');
        xlabel('Timestep (t)'); ylabel('|x_{BP}(t)|');
        K_handle = line(1:T+1, NNZ); 
        set(K_handle, 'Color', 'Cyan', 'LineStyle', '-.'); 
        M_handle = line(1:T+1, M); 
        set(M_handle, 'Color', 'White', 'LineStyle', '--'); 
        legend_handle = legend('K(t)', 'M(t)');
        set(legend_handle, 'Color', [.392 .475 .635])
%         pause(3)
        
        % Vectorize certain scalar parameters
        lambda = lambda*ones(N,1);
        eta = eta*ones(N,1);
        kappa = kappa*ones(N,1);
        alpha = alpha*ones(N,1);
        rho = rho*ones(N,1);
        
        % Give a default group for updating values
        upd_groups{1} = 1:N;
        StateIn = [];   % No pre-existing AMP state
    end
end


%% If user is providing variables, do some basic error checking

if nargin < 3 && nargin > 0
    error('sp_parallel_frame_fxn: Insufficient number of input arguments')
elseif nargin == 0
    if smooth_iter == -1, filter = 1; smooth_iter = 1;
    else filter = 0; end
    min_iters = min(5, smooth_iter);
elseif nargin > 3       % Correct number of input arguments
    
    for l = 1       % (Check errors and unpack variables)
        % Verify dimensionality agreement between y(t) and A(t)
        T = length(y) - 1;      % (Total # of timesteps = T + 1)
        if isa(A{1}, 'function_handle')
            explicitA = 0;      % Matrix A is not given explicitly
            try
                N = length(A{1}(y{1},2));
            catch ME
                error(['sp_parallel_frame_fxn: Improperly designed ' ...
                    'function handle (%s)'], ME.message)
            end
            for t = 1:T+1
                M(t) = length(y{t});
            end
        else    % Matrix is provided explicitly
            for t = 1:T+1
                M(t) = length(y{t});
            end
            explicitA = 1;
            N = size(A{1}, 2);
        end
        
        
        % Unpack the runtime options.  The preferred method of passing
        % runtime options is through an Options object, and the use
        % of a structure is now deprecated.  However, for legacy code, we
        % will still support the structure method for now.
        if nargin >= 4      % 'RunOptions' structure passed to function
            if isa(RunOptions, 'struct')
                % Deprecated format.  Migrate parameters from RunOptions 
                % structure to an Options object
                TmpOptions = RunOptions;    % Copy over
                RunOptions = Options();     % Create empty object
                names = fieldnames(TmpOptions);
                for i = 1:numel(names);
                    try
                        set(RunOptions, lower(names{i}), TmpOptions.(names{i}));
                    catch ME
                        fprintf('Problem setting option %s: %s\n', ...
                            names{i}, ME.message);
                    end
                end
            end
        else
            % Create an Options object with all default settings
            RunOptions = Options();
        end
        
        % Now extract the options
        [smooth_iter, min_iters, eq_iter, alg, update, upd_groups, ...
            verbose, StateIn, eps, Eq_Params.tau, update_list] = ...
            RunOptions.getOptions();
        if smooth_iter == -1    % User wishes to filter, not smooth
            filter = 1;
            smooth_iter = 1;
        else
            filter = 0;
        end
        if ~isempty(upd_groups) && ~numel(setdiff(1:N, cat(1,upd_groups{:}))) ~= 0
            error(['sp_parallel_frame_fxn: ''RunOptions.upd_groups'' ' ...
                'does not partition every coefficient into a ' ...
                'unique group'])
        end
        
        
        % Unpack the model parameters.  The preferred method of passing
        % model parameters is through a DCSModelParams object, and the use
        % of a structure is now deprecated.  However, for legacy code, we
        % will still support the structure method for now.
        if isa(Params, 'struct')
            % Deprecated format.  Migrate parameters from Params structure
            % to a DCSModelParams object
            TmpOptions = Params;         % Copy over
            Params = DCSModelParams();  % Create empty object
            names = fieldnames(TmpOptions);
            for i = 1:numel(names);
                % Weird special case
                if strcmpi(names{i}, 'pz1')
                    names{i} = 'p01';
                    TmpOptions.(names{i}) = TmpOptions.pz1;
                end
                try
                    set(Params, lower(names{i}), TmpOptions.(names{i}));
                catch ME
                    fprintf('Problem setting model param. %s: %s\n', ...
                        names{i}, ME.message);
                end
            end
            % Check for any missing properties
            props = properties(Params);
            for i = 1:numel(props)
                if isempty(Params.(props{i}))
                    error('Model parameter missing!')
                end
            end
        end
        % Great, now we just need to extract the parameters...
        [lambda, p01, eta, kappa, alpha, sig2e, p10, rho] = ...
            Params.getParams();
        
        % ...and resize as needed
        if numel(lambda) == 1, lambda = lambda*ones(N,1); end
        if numel(eta) == 1, eta = eta*ones(N,1); end
        if numel(kappa) == 1, kappa = kappa*ones(N,1); end
        if numel(alpha) == 1, alpha = alpha*ones(N,1); end
        if numel(rho) == 1, rho = rho*ones(N,1); end
    end
end  % End of input error checking/unpacking


%% Run the multi-timestep sparse recovery algorithm w/ the parallel
%% recovery message passing schedule

% ********************************************************************************************
% Would be nice to use user-specified parameters to initialize parameters
% at each timestep

% Message matrix declarations and initializations (all are N-by-T+1 dim)
LAMBDA_FWD = lambda*ones(1,T+1);            % Matrix of messages from h(t) to s(t)
LAMBDA_BWD = 0.5*ones(N,T+1);               % Matrix of messages from h(t+1) to s(t)
ETA_FWD = eta*ones(1,T+1);                  % Matrix of means from d(t) to theta(t)
ETA_BWD = zeros(N,T+1);                     % Matrix of means from d(t+1) to theta(t)
KAPPA_FWD = kappa*ones(1,T+1);              % Matrix of vars from d(t) to theta(t)
KAPPA_BWD = inf*ones(N,T+1);                % Matrix of vars from d(t+1) to theta(t)
PI_IN = NaN*ones(N,T+1);                    % Matrix of messages from s(t) to f(t)
PI_OUT = NaN*ones(N,T+1);                   % Matrix of messages from f(t) to s(t)
XI_IN = NaN*ones(N,T+1);                    % Matrix of means from theta(t) to f(t)
XI_OUT = NaN*ones(N,T+1);                   % Matrix of means from f(t) to theta(t)
PSI_IN = NaN*ones(N,T+1);                   % Matrix of vars from theta(t) to f(t)
PSI_OUT = NaN*ones(N,T+1);                  % Matrix of vars from f(t) to theta(t)

% Create space to store the state of the AMP variables between iterations
if ~isempty(StateIn)
    C_STATE = StateIn.C_STATE;
    Z_State = StateIn.Z_State;
    MU_STATE = StateIn.MU_STATE;
else
    C_STATE = 100*mean(kappa)*ones(1,T+1);    % Initialize to large c values
    Z_State = y;                    % Initialize residual to the measurements
    MU_STATE = zeros(N,T+1);
end

% Make room for some variables that will be filled in later
x_hat = cell(1,T+1);                        % Placeholder for MMSE estimates
v_hat = cell(1,T+1);                        % Placeholder for MMSE variance
lambda_hat = cell(1,T+1);                   % Placeholder for support posteriors
V_STATE = NaN*ones(N,T+1);                  % Placeholder for x variances
NMSEavg_dB = NaN*ones(1,smooth_iter);       % Placeholder for alg. avg. NMSE (dB)

% Declare constants
Eq_Params.eps = eps;        % Gaussian "squelch" parameter on p(x_n)
Eq_Params.sig2e = sig2e;    % Circular, complex, additive, Gaussian noise variance
Eq_Options.iter = eq_iter;  % Number of BP/AMP iterations per equalization run
Eq_Options.alg = alg;       % Specifies BP- or AMP-driven inference
Support.p01 = p01;          % Pr{s_n(t+1) = 0 | s_n(t) = 1}
Support.p10 = p10;          % Pr{s_n(t+1) = 1 | s_n(t) = 0}
Amplitude.alpha = alpha;    % Amplitude Gauss-Markov process innovation rate
Amplitude.rho = rho;        % Amplitude Gauss-Markov process driving noise

last_iter = 0;          % Clear flag
start_time = tic;       % Start stopwatch

% Execute the message passing routine
for k = 1:smooth_iter       % Iteration of the forwards/backwards message pass
    
    % Begin by computing the messages moving into each frame.  Then obtain
    % recoveries of x(t) for every timestep "in parallel"
    for t = 1:T+1           % Current timestep index (ahead by 1)
        % At the current timestep, calculate messages going from s(t) and
        % theta(t) to f(t)
        if t < T+1      % Not the terminal timestep, so msgs are multiplied
            [PI_IN(:,t), XI_IN(:,t), PSI_IN(:,t)] = ...
                sp_msg_mult_fxn(LAMBDA_FWD(:,t), LAMBDA_BWD(:,t), ...
                ETA_FWD(:,t), KAPPA_FWD(:,t), ETA_BWD(:,t), KAPPA_BWD(:,t));
        else            % Terminal timestep, thus just pass the lone quantities
            PI_IN(:,t) = LAMBDA_FWD(:,t);
            XI_IN(:,t) = ETA_FWD(:,t);
            PSI_IN(:,t) = KAPPA_FWD(:,t);
        end
        
        % Perform equalization to obtain an estimate of x(t) using the
        % priors specified by the current values of PI_IN(:,t), XI_IN(:,t),
        % and PSI_IN(:,t).  Update the outoing quantities, and save current
        % best estimate of x(t)
        Eq_ParamsB = Eq_Params;
        Eq_ParamsB.pi = PI_IN(:,t); Eq_ParamsB.xi = XI_IN(:,t); 
        Eq_ParamsB.psi = PSI_IN(:,t);
        StateIn = struct();
        StateIn.junk = 0;
        StateIn.c = C_STATE(t); StateIn.z = Z_State{t};
        StateIn.mu = MU_STATE(:,t);
        % ****************************************************************
        [x_hat{t}, OutMessages, StateOutB] = sp_frame_fxn(y{t}, A{t}, ...
            Eq_ParamsB, Eq_Options, StateIn);
        % ****************************************************************
        PI_OUT(:,t) = OutMessages.pi; XI_OUT(:,t) = OutMessages.xi;
        PSI_OUT(:,t) = OutMessages.psi;     % Updated message parameters
        C_STATE(t) = StateOutB.c; Z_State{t} = StateOutB.z;
        MU_STATE(:,t) = StateOutB.mu;    % Save state of AMP for warm start
        V_STATE(:,t) = StateOutB.v;
    end
    
    % Now use the outgoing messages from each frame to conduct a
    % forward/backward pass to update beliefs about the values of each s(t)
    % and theta(t).  Start with the forward portion of the pass
    for t = 1:T+1       % Current timestep index (ahead by 1)
        % Use the outgoing messages from f(t) to s(t) and
        % theta(t) to update the priors on s(t+1) and theta(t+1)
        if t < T+1      % We aren't at the terminal timestep, so update fwd quantities
            Support.lambda = LAMBDA_FWD(:,t);   % Msg. h(t) to s(t)
            Support.pi = PI_OUT(:,t);           % Msg. f(t) to s(t)
            Amplitude.eta = ETA_FWD(:,t);       % Msg. d(t) to theta(t)
            Amplitude.kappa = KAPPA_FWD(:,t);   % Msg. d(t) to theta(t)
            Amplitude.xi = XI_OUT(:,t);         % Msg. f(t) to theta(t)
            Amplitude.psi = PSI_OUT(:,t);       % Msg. f(t) to theta(t)
            Amplitude.eta_0 = ETA_FWD(:,1);     % Mean of the amplitude RP
            Msg.direction = 'forward';          % Indicate forward propagation
            Msg.terminal = 0;                   % Non-terminal update
            
            % Compute the updates
            [LAMBDA_FWD(:,t+1), ETA_FWD(:,t+1), KAPPA_FWD(:,t+1)] = ...
                sp_timestep_fxn(Support, Amplitude, Msg);
        else            % Terminal timestep, thus update backwards priors
            Support.lambda = NaN*ones(N,1);     % Msg. D.N.E.
            Support.pi = PI_OUT(:,t);           % Msg. f(t) to s(t)
            Amplitude.eta = NaN*ones(N,1);    	% Msg. D.N.E.
            Amplitude.kappa = NaN*ones(N,1);    % Msg. D.N.E.
            Amplitude.xi = XI_OUT(:,t);         % Msg. f(t) to theta(t)
            Amplitude.psi = PSI_OUT(:,t);       % Msg. f(t) to theta(t)
            Amplitude.eta_0 = ETA_FWD(:,1);     % Mean of the amplitude RP
            Msg.direction = 'backward';      	% Indicate forward propagation
            Msg.terminal = 1;                   % Terminal update            
            
            % Compute the updates
            [LAMBDA_BWD(:,t-1), ETA_BWD(:,t-1), KAPPA_BWD(:,t-1)] = ...
                sp_timestep_fxn(Support, Amplitude, Msg);
        end
        
        % On final pass, save the estimates of the coefficient variances
        if (k == smooth_iter)
            if t == T+1
                v_hat{t} = V_STATE(:,t);        % var{x_n(t) | y(t)}
                lambda_hat{t} = PI_OUT(:,t).*LAMBDA_FWD(:,t) ./ ...
                    ((1 - PI_OUT(:,t)).*(1 - LAMBDA_FWD(:,t)) ...
                    + PI_OUT(:,t).*LAMBDA_FWD(:,t));
                if filter, return; end; % Early break for filter mode
            elseif filter && (t < T+1)
                % Finalize quantities for filter mode output
                v_hat{t} = V_STATE(:,t);        % var{x_n(t) | y(t)}
                % Also compute Pr{s_n(t) = 1 | y(t)}
                lambda_hat{t} = PI_OUT(:,t).*LAMBDA_FWD(:,t).*LAMBDA_BWD(:,t) ./ ...
                    ((1 - PI_OUT(:,t)).*(1 - LAMBDA_FWD(:,t)).*(1 - LAMBDA_BWD(:,t)) ...
                    + PI_OUT(:,t).*LAMBDA_FWD(:,t).*LAMBDA_BWD(:,t));
            end
        end
    end
    
    % Now execute the backwards portion of the message passing for all the 
    % s(t) and theta(t) variables (t = 0, ..., T-1)
    for t = T:-1:1           % Descend from 2nd-to-last timestep to the first
        % Use the outgoing messages from f(t) to s(t) and
        % theta(t) to update the priors on s(t+1) and theta(t+1)
        if t > 1        % We aren't at the first timestep, so update bwd quantities
            Support.lambda = LAMBDA_BWD(:,t);   % Msg. h(t) to s(t)
            Support.pi = PI_OUT(:,t);           % Msg. f(t) to s(t)
            Amplitude.eta = ETA_BWD(:,t);       % Msg. d(t) to theta(t)
            Amplitude.kappa = KAPPA_BWD(:,t);   % Msg. d(t) to theta(t)
            Amplitude.xi = XI_OUT(:,t);         % Msg. f(t) to theta(t)
            Amplitude.psi = PSI_OUT(:,t);       % Msg. f(t) to theta(t)
            Amplitude.eta_0 = ETA_FWD(:,1);     % Mean of the amplitude RP
            Msg.direction = 'backward';      	% Indicate backward propagation
            Msg.terminal = 0;                   % Non-terminal update
            
            % Compute the updates
            [LAMBDA_BWD(:,t-1), ETA_BWD(:,t-1), KAPPA_BWD(:,t-1)] = ...
                sp_timestep_fxn(Support, Amplitude, Msg);
        else            % Initial timestep, thus there is nothing to update
            % Nothing to do in here since lambda, eta, and kappa do
            % not change
        end
        
        % On final pass, save the estimates of the coefficient variances
        if k == smooth_iter || last_iter
            v_hat{t} = V_STATE(:,t);              % var{x_n(t) | y(t)}
            % Also compute Pr{s_n(t) = 1 | y(t)}
            lambda_hat{t} = PI_OUT(:,t).*LAMBDA_FWD(:,t).*LAMBDA_BWD(:,t) ./ ...
                ((1 - PI_OUT(:,t)).*(1 - LAMBDA_FWD(:,t)).*(1 - LAMBDA_BWD(:,t)) ...
                + PI_OUT(:,t).*LAMBDA_FWD(:,t).*LAMBDA_BWD(:,t));
        end
    end

    
    % Compute the time-averaged residual energy
    avg_resid_energy = 0;
    for t = 1:T+1
        if explicitA
            avg_resid_energy = avg_resid_energy + ...
                norm(y{t} - A{t}*x_hat{t})^2 / (T + 1);
        else
            avg_resid_energy = avg_resid_energy + ...
                norm(y{t} - A{t}(x_hat{t}, 1))^2 / (T + 1);
        end
    end
    
    % Display execution info, if requested
    if verbose
        fprintf('SP_PARALLEL_FRAME_FXN: Completed %d iterations\n', k);
        fprintf('Total elapsed time: %f s\n', toc(start_time));
        fprintf('Time-averaged residual energy: %f\n', avg_resid_energy);
        fprintf('-----------------------------------------------------\n');
    end
    
    % ********************************************************************
    % Having completed a forward\backward pass, update the hyperparameters
    % using the current recovery, if the user wishes
    % ********************************************************************
    if update
        % Pass current state of factor graph to the parameter update
        % function through this structure
        State = struct('Mu_x', []);     % Clear contents of structure
        State.Mu_x = MU_STATE;      State.V_x = V_STATE;
        State.Eta_fwd = ETA_FWD;    State.Eta_bwd = ETA_BWD;
        State.Xi_out = XI_OUT;      State.Kap_fwd = KAPPA_FWD;
        State.Kap_bwd = KAPPA_BWD;  State.Psi_out = PSI_OUT;
        State.Lam_fwd = LAMBDA_FWD; State.Lam_bwd = LAMBDA_BWD;
        State.Pi_out = PI_OUT;      State.iter = k;
        
        % Also pass the current values of the parameters
        UpdParams = struct('eta', []);    % Clear contents of structure
        UpdParams.eta = ETA_FWD(:,1);     
        UpdParams.kappa = KAPPA_FWD(:,1);
        UpdParams.alpha = Amplitude.alpha;
        UpdParams.rho = Amplitude.rho;
        UpdParams.lambda = LAMBDA_FWD(:,1);
        UpdParams.p01 = Support.p01;
        UpdParams.p10 = Support.p10;
        UpdParams.sig2e = Eq_Params.sig2e;
        
        %Now compute the updates
        Updates = parameter_update_fxn(y, A, State, UpdParams, upd_groups);
        
        % Store the updated hyperparameters in needed locations
        for g = 1:length(upd_groups)
            if any(strcmpi(update_list, 'lambda'))
                LAMBDA_FWD(upd_groups{g},1) = Updates.lambda(g);
            end
            if any(strcmpi(update_list, 'p01'))
                Support.p01(upd_groups{g},1) = Updates.p01(g);
            end
            if any(strcmpi(update_list, 'p01'))
                Support.p10(upd_groups{g},1) = Updates.p10(g);
            end
            if k ~= 1 && any(strcmpi(update_list, 'kappa'))
                KAPPA_FWD(upd_groups{g},1) = Updates.kappa(g);
            end
            if any(strcmpi(update_list, 'eta'))
                ETA_FWD(upd_groups{g},1) = Updates.eta(g);
            end
            if any(strcmpi(update_list, 'alpha'))
                Amplitude.alpha(upd_groups{g}) = .01*Amplitude.alpha(upd_groups{g}) + ...
                    .99*Updates.alpha(g)*ones(length(upd_groups{g}),1);
            end
            if k ~= 1 && any(strcmpi(update_list, 'kappa') | ...
                    strcmpi(update_list, 'alpha'))
                Amplitude.rho(upd_groups{g},1) = Updates.rho(g);
            end
        end
        if k ~= 1 && any(strcmpi(update_list, 'sig2e'))
            Eq_Params.sig2e = .01*Eq_Params.sig2e + .99*Updates.sig2e; 
        end
        
        % Depending on the updated value of p01, we will switch between a
        % Taylor approximation of the outgoing (from AMP) amplitude
        % messages and a simple thresholding operation based on the
        % incoming (to AMP) prior activity probabilities
        if any(Support.p01 > 0.025)
            Eq_Params.tau = RunOptions.tau; 	% Use tau-thresholding
        else
            Eq_Params.tau = -1;             % Use Taylor approx
        end
        
        if verbose
            for g = 1:length(upd_groups)
                if any(strcmpi(update_list, 'lambda'))
                    fprintf('Updated value of ''lambda'' for Group %d: %f\n', ...
                        g, Updates.lambda(g));
                end
                if any(strcmpi(update_list, 'p01'))
                    fprintf('Updated value of ''p01'' for Group %d: %f\n', ...
                        g, Updates.p01(g));
                end
                if any(strcmpi(update_list, 'alpha'))
                    fprintf('Updated value of ''alpha'' for Group %d: %f\n', ...
                        g, Amplitude.alpha(upd_groups{g}(1)));
                end
                if any(strcmpi(update_list, 'eta'))
                    fprintf('Updated value of ''eta'' for Group %d: %s\n', ...
                        g, num2str(Updates.eta(g)));
                end
                if k ~= 1 && any(strcmpi(update_list, 'kappa'))
                    fprintf('Updated value of ''kappa'' for Group %d: %f\n', ...
                        g, Updates.kappa(g));
                end
            end
            if k ~= 1 && any(strcmpi(update_list, 'sig2e'))
                fprintf('Updated value of ''sig2e'': %f\n', Eq_Params.sig2e);
            end
            disp('---------------------------------------------------------');
        end
    end
    
   	% If this is a default test case, then plot the result
    if nargin == 0
        for l = 1,      % Plot various things
%             % First plot the current recovery
%             figure(2); imagesc(abs([x_hat{:}])); colorbar; 
%             title(['MMSE estimate | Fwd./Bwd. iters: ' num2str(k) '/' num2str(k)]);
%             xlabel('Timestep (t)'); ylabel('|x_{mmse}(t)|');
%             K_handle = line(1:T+1, NNZ); 
%             set(K_handle, 'Color', 'Cyan', 'LineStyle', '-.'); 
%             M_handle = line(1:T+1, M); 
%             set(M_handle, 'Color', 'White', 'LineStyle', '--'); 
%             legend_handle = legend('K(t)', 'M(t)');
%             set(legend_handle, 'Color', [.392 .475 .635])
            
            % Next plot the NMSEs of the different recovery methods
            NMSE = NaN(1,T+1);
            for t = 1:T+1; NMSE(t) = (norm(x_true{t} - ...
                    x_hat{t})/norm(x_true{t}))^2; end
            NMSEavg_dB(k) = 10*log10(sum(NMSE)/(T + 1));    % Alg. avg NMSE (dB)
            figure(3); subplot(211)
            plot(1:1:smooth_iter, NMSEavg_dB); hold on
            genie_line = line([0, smooth_iter], [NMSEavg_dB_genie, NMSEavg_dB_genie]);
            indgenie_line = line([0, smooth_iter], [NMSEavg_dB_indgenie, NMSEavg_dB_indgenie]);
            bp_line = line([0, smooth_iter], [NMSEavg_dB_bp, NMSEavg_dB_bp]);
            damp_line = line([0, smooth_iter], [NMSEavg_dB_damp, NMSEavg_dB_damp]);
            set(genie_line, 'Color', 'Green'); set(indgenie_line, 'Color', 'Black');
            set(bp_line, 'Color', 'Red'); set(damp_line, 'Color', 'Cyan'); hold off
            legend('Alg. NMSE', 'Smoothed genie NMSE', 'Indep. genie NMSE', ...
                'Naive BP NMSE', 'Alt. Mesg. Sched.')
            xlabel('Iteration'); ylabel('Avg. NMSE [dB]');
            title(['Avg. NMSEs | Iterations: ' num2str(k)]);
            subplot(212)
            plot(0:T, SNRdB_emp);
            xlabel('Timestep (t)'); ylabel('Empirical SNR [dB]')
            title('Empirical SNR in dB at each timestep')
        end
    end
    
    % Check for early termination this round
    if last_iter, return; end
    
    % Check for early termination for next round
    if k > min_iters && avg_resid_energy < sum(M)*Eq_Params.sig2e/(T+1) && ...
            avg_resid_energy > 1e-2*sum(M)*Eq_Params.sig2e/(T+1)
        last_iter = 1;      % Set the flag for last iteration
    end
    
    % Report to user if maximum number of smoothing iterations was exceeded
    if k == smooth_iter
        fprintf('sp_parallel_frame_fxn: Max # of smoothing iterations reached\n')
    end
end

% Save AMP states
StateOut = struct();
StateOut.C_STATE = C_STATE;
StateOut.Z_State = Z_State;
StateOut.MU_STATE = MU_STATE;