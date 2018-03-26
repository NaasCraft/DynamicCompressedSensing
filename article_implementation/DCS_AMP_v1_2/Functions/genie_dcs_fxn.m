% GENIE_DCS_FXN  	Function for performing soft-input, soft-output 
% sparse reconstruction of a time-series of complex-valued signals, {x(t)}, 
% t = 1, ..., T, with each x(t) being N-dimensional, using a 
% support-aware Kalman smoother (SKS) to perform the belief propagation.  
% Complex- (real-) valued observations, {y(t)}, are obtained through a 
% noisy linear combination, y(t) = A(t)x(t) + e(t), with y(t) being an 
% M(t)-dimensional vector of observations, A(t) being an M(t)-by-N 
% complex- (real-)valued measurement matrix, and e(t) being circular, 
% complex (real), additive Gaussian noise with covariance matrix 
% sig2e*eye(M(t)).
% 
% At any given time, t, it is assumed that the coefficients of x(t),
% x_n(t), can be represented by the product of a discrete variable, s_n,
% and a continuous-valued variable, theta_n(t), that is, x_n(t) =
% s_n(t)*theta_n(t).  s_n(t) either takes the value 0 or 1, and is 
% marginally a Bernoulli distribution with Pr{s_n(t) = 1} = lambda(n).  
% theta_n(t) is marginally a complex (real) Gaussian distribution with mean 
% eta(n) and variance kappa(n).  At timesteps t > 1, theta_n(t) evolves 
% according to a Gauss-Markov process, i.e. 
%       theta_n(t) = (1 - alpha)*theta_n(t-1) + alpha*w_n(t), 
% where alpha is a scalar between 0 and 1, and w_n(t) is complex (real) 
% Gaussian driving noise with zero mean and variance rho chosen to maintain 
% a steady-state variance of kappa. Note that the above Gauss-Markov 
% evolution expression assumes eta(n) = 0, and is modified for non-zero 
% eta(n)'s.  s_n(t) evolves according to a first-order Markov chain with
% steady-state activity probability lambda, and active-to-inactive 
% transition probability p01 = Pr{s_n(t) = 0 | s_n(t-1) = 1}.
%
% Inference is performed using a belief propagation algorithm.  (To speed
% things up, initial message passing within frames may be done using AMP,
% and then for the final iterations, it will switch to the slower, but
% exact, belief propagation message passing procedure.)
%
% SYNTAX:
% [x_genie, v_genie] = genie_dcs_fxn(y, A, support, Params, RunOptions)
%
% INPUTS:
% y                 A 1-by-T cell array of observation vectors.  Each
%                   complex-valued observation vector is defined to be of
%                   length M(t), t = 1, ..., T
% A                 A 1-by-T cell array of complex-valued measurement
%                   matrices, with each matrix of dimension M(t)-by-N.
% support           An array of signal support indices, i.e., a vector of 
%                   indices for the coefficients that are non-zero
% Params            An object of the DCSModelParams class (see 
%                   DCSModelParams.m in the folder ClassDefs) which 
%                   contains the parameters needed to specify the 
%                   signal/measurement model
% RunOptions        An [optional] object of the Options class (see
%                   Options.m in the ClassDefs folder) that contains
%                   various runtime configuration options of GENIE_DCS_FXN.
%                   If not provided, a default configuration is used.  Note 
%                   that the options min_iters, update, upd_groups, and 
%                   verbose are not used in this function.
%
% OUTPUTS:
% x_genie           A 1-by-T cell array of length-N complex-valued
%                   support-aware genie estimates of {x(t)}
% v_genie           A 1-by-T cell array of length-N estimates of the
%                   variances of the coefficient estimates in x_genie
%

%
% Coded by: Justin Ziniel, The Ohio State Univ.
% E-mail: ziniel.1@osu.edu
% Last change: 12/05/12
% Change summary: 
%		- Created from genie_mmv_fxn v1.1 (02/08/12; JAZ)
%       - Updated naming conventions and use of DCSModelParams class
%         argument (12/05/12; JAZ; v1.1)
% Version 1.2
%

function [x_genie, v_genie] = genie_dcs_fxn(y, A, support, Params, RunOptions)

%% Begin by creating a default test case matched to the signal model

if nargin == 0      % No user-provided variables, create default test
    
    % *********************************************************************
    % First generate a signal, measurements, and matrices by using the
    % default configuration specified in SigGenParams.m
    SignalParams = SigGenParams('p01', 0.05);   % Create DCS model
    SignalParams.print();   % Print configuration to command window
    
    % Pass SigGenParams object to the signal generation function
    [x_true, y, A, support, K, sig2e] = dcs_signal_gen_fxn(SignalParams);
    
    % Plot true signal
    figure(1); clf; imagesc(abs([x_true{:}])); xlabel('Timestep (t)');
    ylabel('|x_n^{(t)}|'); title('True signal magnitude trajectories');
    colorbar;
    % *********************************************************************
    
    % *********************************************************************
    % Now use the exact same default model parameters to create the
    % required Params input to this function, (see ModelParams.m)
    Params = DCSModelParams(SignalParams, sig2e);  % Use secondary constructor
    Params.print();     % Print model parameters to command window
    % *********************************************************************
    
end


%% If user is providing variables, do some basic error checking

if nargin < 4 && nargin > 0
    error('genie_dcs_fxn: Insufficient number of input arguments')
else     	% Correct number of input arguments
    
    % Verify dimensionality agreement between y(t) and A(t)
    T = length(y);      % Total # of timesteps
    if isa(A{1}, 'function_handle')
        explicitA = 0;      % Matrix A is not given explicitly
        try
            N = length(A{1}(y{1},2));
        catch
            error('sp_mmv_fxn: Improperly designed function handle')
        end
        for t = 1:T
            M(t) = length(y{t});
        end
    else    % Matrix is provided explicitly
        for t = 1:T
            M(t) = length(y{t});
        end
        explicitA = 1;
        N = size(A{1}, 2);
    end

    % Declare/initialize certain variables
    tol = 1e-1;         % Tolerance for early termination
    last_iter = 0;      % Flag for early termination
    
    if nargin == 5  % Options object passed
        % Grab the runtime options from the RunOptions input
        if isa(RunOptions, 'Options')
            [smooth_iters, ~, inner_iters, alg, ~, ~, ~, ~, eps, tau] = ...
                RunOptions.getOptions();
        else
            error('Input to RunOptions must be object of Options class')
        end
    else
        % No Options object passed to function, thus use default options
        Opt = Options();
        Opt.print();    % Print runtime options to command window
        [smooth_iters, ~, inner_iters, alg, ~, ~, ~, ~, eps, tau] = ...
            Opt.getOptions();
    end

    % Do a little conditioning on the inputs from the RunOptions
    if smooth_iters == -1, smooth_iters = 1; filter = 1; end

    % Unpack 'Params' structure
    if isa(Params, 'DCSModelParams')
        [lambda, ~, eta, kappa, alpha, sig2e, ~, rho] = Params.getParams();
    else
        error('Input to Params must be object of ModelParams class');
    end
    
    % Check to make sure dimensions of the parameters are okay, vectorize
    % if necessary
    if numel(lambda) == 1, lambda = lambda*ones(N,1);
    elseif size(lambda,1) ~= N, error('Incorrect dimension: lambda'); end
    if numel(eta) == 1, eta = eta*ones(N,1);
    elseif size(eta,1) ~= N, error('Incorrect dimension: eta'); end
    if numel(kappa) == 1, kappa = kappa*ones(N,1);
    elseif size(kappa,1) ~= N, error('Incorrect dimension: kappa'); end
    if numel(alpha) == 1, alpha = alpha*ones(N,1);
    elseif size(alpha,1) ~= N, error('Incorrect dimension: alpha'); end
    if numel(rho) == 1, rho = rho*ones(N,1);
    elseif size(rho,1) ~= N, error('Incorrect dimension: rho'); end
    if numel(sig2e) ~= 1, error('Incorrect dimension: sig2e'); end

end  % End of input error checking/unpacking


%% Run the multi-timestep sparse recovery algorithm

% Message matrix declarations and initializations (all are N-by-T dim)
LAMBDA_FWD = [lambda, NaN*ones(N,T-1)];     % Matrix of messages from h(t) to s(t)
LAMBDA_BWD = 0.5*ones(N,T);                 % Matrix of messages from h(t+1) to s(t)
ETA_FWD = [eta, NaN*ones(N,T-1)];           % Matrix of means from d(t) to theta(t)
ETA_BWD = zeros(N,T);                       % Matrix of means from d(t+1) to theta(t)
KAPPA_FWD = [kappa, NaN*ones(N,T-1)];     	% Matrix of vars from d(t) to theta(t)
KAPPA_BWD = inf*ones(N,T);                  % Matrix of vars from d(t+1) to theta(t)
PI_IN = NaN*ones(N,T);                      % Matrix of messages from s(t) to f(t)
PI_OUT = NaN*ones(N,T);                     % Matrix of messages from f(t) to s(t)
XI_IN = NaN*ones(N,T);                      % Matrix of means from theta(t) to f(t)
XI_OUT = NaN*ones(N,T);                     % Matrix of means from f(t) to theta(t)
PSI_IN = NaN*ones(N,T);                     % Matrix of vars from theta(t) to f(t)
PSI_OUT = NaN*ones(N,T);                    % Matrix of vars from f(t) to theta(t)
C_STATE = 100*mean(kappa)*ones(1,T);        % Initialize to large c values
Z_State = y;                                % Initialize residual to the measurements
MU_STATE = zeros(N,T);

% Make room for some variables that will be filled in later
x_genie = cell(1,T);                        % Placeholder for MMSE estimates
x_old = inf*ones(N,T);                      % Holds prev. estimate
NMSEavg_dB = NaN*ones(1,2*smooth_iters);     % Placeholder for alg. avg. NMSE (dB)

% Declare constants
Eq_Params.eps = eps;        % Gaussian "squelch" parameter on p(x_n)
Eq_Params.tau = tau;        % Type of approx to use for f-to-theta msgs
Eq_Params.sig2e = sig2e;    % Circular, complex, additive, Gaussian noise variance
Eq_Options.iter = inner_iters;  % Number of BP/AMP iterations per equalization run
Eq_Options.alg = alg;       % Specifies BP- or AMP-driven inference
Support.p01 = 0;            % Irrelevant parameter for SKS
Support.p10 = 0;            % Irrelevant parameter for SKS
Amplitude.alpha = alpha;    % Amplitude Gauss-Markov process innovation rate
Amplitude.rho = rho;        % Amplitude Gauss-Markov process driving noise


% Execute the message passing routine
for k = 1:smooth_iters     	% Iteration of the forwards/backwards message pass
    
    % Begin with the forward message pass
    for t = 1:T           % Current timestep index
        % At the current timestep, calculate messages going from s(t) and
        % theta(t) to f(t)
        if t < T      % Not the terminal timestep, so msgs are multiplied
            [~, XI_IN(:,t), PSI_IN(:,t)] = ...
                sp_msg_mult_fxn(LAMBDA_FWD(:,t), LAMBDA_BWD(:,t), ...
                ETA_FWD(:,t), KAPPA_FWD(:,t), ETA_BWD(:,t), KAPPA_BWD(:,t));
            % The genie knows the activity priors perfectly
            PI_IN(:,t) = zeros(N,1);
            PI_IN(support{t},t) = 1;
        else            % Terminal timestep, thus just pass the lone quantities
            XI_IN(:,t) = ETA_FWD(:,t);
            PSI_IN(:,t) = KAPPA_FWD(:,t);
            % The SKS knows the activity priors perfectly
            PI_IN(:,t) = zeros(N,1);
            PI_IN(support{t},t) = 1;
        end
        
        % Perform equalization to obtain an estimate of x(t) using the
        % priors specified by the current values of PI_IN(:,t), XI_IN(:,t),
        % and PSI_IN(:,t).  Update the outoing quantities, and save current
        % best estimate of x(t)
        Eq_Params.pi = PI_IN(:,t); Eq_Params.xi = XI_IN(:,t); 
        Eq_Params.psi = PSI_IN(:,t);
        % ****************************************************************
        [x_genie{t}, OutMessages, StateOut] = sp_frame_fxn(y{t}, A{t}, ...
            Eq_Params, Eq_Options);
        % ****************************************************************
        PI_OUT(:,t) = OutMessages.pi; XI_OUT(:,t) = OutMessages.xi;
        PSI_OUT(:,t) = OutMessages.psi;     % Updated message parameters
        C_STATE(t) = StateOut.c; Z_State{t} = StateOut.z;
        MU_STATE(:,t) = StateOut.mu;    % Save state of AMP for warm start
        V_STATE(:,t) = StateOut.v;
        
        % Use the resulting outgoing messages from f(t) to s(t) and
        % theta(t) to update the priors on s(t+1) and theta(t+1)
        if t < T      % We aren't at the terminal timestep, so update fwd quantities
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
        if ((k == smooth_iters) || last_iter) && (t == T)
            v_genie{t} = StateOut.v;        	% var{x_n(t) | y(t)}
            lambda_hat{t} = PI_OUT(:,t).*LAMBDA_FWD(:,t) ./ ...
                ((1 - PI_OUT(:,t)).*(1 - LAMBDA_FWD(:,t)) ...
                + PI_OUT(:,t).*LAMBDA_FWD(:,t));
        end
    end
    
    % If this is a default test case, then plot various things
    if nargin == 0
        for l = 1,      % Plot various things
            % First plot the current recovery
            figure(2); imagesc(abs([x_genie{:}])); colorbar; 
            title(['Support-aware genie MMSE estimate | Fwd./Bwd. iters: ' ...
                num2str(k) '/' num2str(k-1)]);
            xlabel('Timestep (t)'); ylabel('|x_{genie}(t)|');
            K_handle = line(1:T, K); 
            set(K_handle, 'Color', 'Cyan', 'LineStyle', '-.'); 
            M_handle = line(1:T, M); 
            set(M_handle, 'Color', 'White', 'LineStyle', '--'); 
            legend_handle = legend('K(t)', 'M(t)');
            set(legend_handle, 'Color', [.392 .475 .635])
            
            % Next plot the NMSEs of the different recovery methods
            for t = 1:T; NMSE(t) = (norm(x_true{t} - ...
                    x_genie{t})/norm(x_true{t}))^2; end
            NMSEavg_dB(2*k-1) = 10*log10(sum(NMSE)/T);    % Alg. avg NMSE (dB)
            figure(3);
            plot([0.5 : 0.5 : smooth_iters], NMSEavg_dB); hold on
%             genie_line = line([0, smooth_iters], [NMSEavg_dB_genie, NMSEavg_dB_genie]);
%             bp_line = line([0, smooth_iters], [NMSEavg_dB_bp, NMSEavg_dB_bp]);
%             set(genie_line, 'Color', 'Green'); set(bp_line, 'Color', 'Red'); hold off
%             legend('Genie NMSE', 'Timestep-independent genie NMSE', 'Naive BP NMSE')
            xlabel('Fwd/Bwd Iteration'); ylabel('Avg. NMSE [dB]');
            title(['Avg. NMSEs | Fwd./Bwd. iters: ' num2str(k) '/' num2str(k-1)]);
        end
    end
    
    % Now execute the backwards message pass
    for t = T-1:-1:1           % Descend from 2nd-to-last timestep to the first
        % At the current timestep, calculate messages going from s(t) and
        % theta(t) to f(t)
        [~, XI_IN(:,t), PSI_IN(:,t)] = ...
            sp_msg_mult_fxn(LAMBDA_FWD(:,t), LAMBDA_BWD(:,t), ...
            ETA_FWD(:,t), KAPPA_FWD(:,t), ETA_BWD(:,t), KAPPA_BWD(:,t));
        % The genie knows the activity priors perfectly
        PI_IN(:,t) = zeros(N,1);
        PI_IN(support{t},t) = 1;
        
        % Perform equalization to obtain an estimate of x(t) using the
        % priors specified by the current values of PI_IN(:,t), XI_IN(:,t),
        % and PSI_IN(:,t).  Update the outoing quantities, and save current
        % best estimate of x(t)
        Eq_Params.pi = PI_IN(:,t); Eq_Params.xi = XI_IN(:,t); 
        Eq_Params.psi = PSI_IN(:,t);
        % ****************************************************************
        [x_genie{t}, OutMessages, StateOut] = sp_frame_fxn(y{t}, A{t}, ...
            Eq_Params, Eq_Options);
        % ****************************************************************
        PI_OUT(:,t) = OutMessages.pi; XI_OUT(:,t) = OutMessages.xi;
        PSI_OUT(:,t) = OutMessages.psi;     % Updated message parameters
        C_STATE(t) = StateOut.c; Z_State{t} = StateOut.z;
        MU_STATE(:,t) = StateOut.mu;    % Save state of AMP for warm start
        V_STATE(:,t) = StateOut.v;
        
        % Use the resulting outgoing messages from f(t) to s(t) and
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
            % Nothing to do in here since lambda_n(1), eta_n(1), and 
            % kappa_n(1) do not change
        end
        
        % On final pass, save the estimates of the coefficient variances
        if k == smooth_iters || last_iter
            v_genie{t} = StateOut.v;              % var{x_n(t) | y(t)}
            % Also compute Pr{s_n(t) = 1 | y(t)}
            lambda_hat{t} = PI_OUT(:,t).*LAMBDA_FWD(:,t).*LAMBDA_BWD(:,t) ./ ...
                ((1 - PI_OUT(:,t)).*(1 - LAMBDA_FWD(:,t)).*(1 - LAMBDA_BWD(:,t)) ...
                + PI_OUT(:,t).*LAMBDA_FWD(:,t).*LAMBDA_BWD(:,t));
        end
    end
    
    % Compute the time-averaged residual energy
    avg_resid_energy = 0;
    for t = 1:T
        avg_resid_energy = avg_resid_energy + ...
            norm(y{t} - A{t}*x_genie{t})^2 / T;
    end

   	% If this is a default test case, then plot the result
    if nargin == 0
        for l = 1,      % Plot various things
            % First plot the current recovery
            figure(2); imagesc(abs([x_genie{:}])); colorbar; 
            title(['Support-aware genie MMSE estimate | Fwd./Bwd. iters: ' ...
                num2str(k) '/' num2str(k)]);
            xlabel('Timestep (t)'); ylabel('|x_{genie}(t)|');
            K_handle = line(1:T, K); 
            set(K_handle, 'Color', 'Cyan', 'LineStyle', '-.'); 
            M_handle = line(1:T, M); 
            set(M_handle, 'Color', 'White', 'LineStyle', '--'); 
            legend_handle = legend('K(t)', 'M(t)');
            set(legend_handle, 'Color', [.392 .475 .635])
            
            % Next plot the NMSEs of the different recovery methods
            for t = 1:T; NMSE(t) = (norm(x_true{t} - ...
                    x_genie{t})/norm(x_true{t}))^2; end
            NMSEavg_dB(2*k) = 10*log10(sum(NMSE)/T);    % Alg. avg NMSE (dB)
            figure(3);
            plot([0.5 : 0.5 : smooth_iters], NMSEavg_dB); hold on
%             genie_line = line([0, smooth_iters], [NMSEavg_dB_genie, NMSEavg_dB_genie]);
%             bp_line = line([0, smooth_iters], [NMSEavg_dB_bp, NMSEavg_dB_bp]);
%             set(genie_line, 'Color', 'Green'); set(bp_line, 'Color', 'Red'); hold off
%             legend('Genie NMSE', 'Timestep-independent genie NMSE', 'Naive BP NMSE')
            xlabel('Fwd/Bwd Iteration'); ylabel('Avg. NMSE [dB]');
            title(['Avg. NMSEs | Fwd./Bwd. iters: ' num2str(k) '/' num2str(k)]);
        end
    end
    
    % Check for early termination this round
    if last_iter, return; end
    
    % Check for early termination for next round
    if norm([x_genie{:}] - x_old, 'fro')/norm([x_genie{:}], 'fro') < tol && ...
            avg_resid_energy < 2*sum(M)*sig2e/T
        last_iter = 1;      % Set the flag for last iteration
    else
        x_old = [x_genie{:}];
    end
end