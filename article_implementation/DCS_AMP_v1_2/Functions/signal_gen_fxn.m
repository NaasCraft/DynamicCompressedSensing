%
% ** SIGNAL_GEN_FXN has been deprecated.  Please use DCS_SIGNAL_GEN_FXN **
%

% SIGNAL_GEN_FXN      Function for generating a time-series of 
% complex-valued signals, {x(t)}, t = 0, 1, ..., T, with each x(t) being 
% N-dimensional.  Complex-valued observations, {y(t)}, are then generated
% through a noisy linear combination, y(t) = A(t)x(t) + e(t), with y(t) 
% being an M(t) dimensional vector of observations, A(t) being an 
% M(t)-by-N complex-valued measurement matrix whose form is specified by
% the user, and e(t) being circular, complex, additive Gaussian noise with
% covariance matrix sig2e*eye(M(t)).
%
% At any given time, t, it is assumed that the coefficients of x(t),
% x_n(t), can be represented by the product of a discrete variable, s_n(t),
% and a continuous-valued variable, theta_n(t), that is, x_n(t) =
% s_n(t)*theta_n(t).  s_n(t) either takes the value 0 or 1. s_n(t) follows 
% a Bernoulli distribution with Pr{s_n(t) = 1} = lambda.  theta_n(t) 
% follows a complex Gaussian distribution with mean eta and variance kappa.
%
% Across frames (timesteps), s_n(t) evolves according to a Markov chain,
% characterized by the transition probability, p1z == Pr{s_n(t) = 1 |
% s_n(t-1) = 0}.  theta_n(t) evolves according to a Gauss-Markov process, 
% i.e., theta_n(t) = (1 - alpha)*theta_n(t-1) + alpha*w_n(t), where alpha 
% is a scalar between 0 and 1, and w_n(t) is complex Gaussian driving noise 
% with zero mean and variance chosen to maintain a steady-state amplitude
% variance of kappa.
%
% SYNTAX:
% [x_true, y, A, support, NNZ, sig2e] = signal_gen_fxn(Params)
%
% INPUTS:
% Params            A structure containing signal model parameters
%   .N              The dimension of the true signals, x_true(t)
%   .M              A 1-by-T+1 dimensional vector of the number of
%                   measurements to acquire at each timestep
%   .A_type         Type of random measurement matrix to generate.  (1) for
%                   a normalized complex IID Gaussian matrix, (2) for a
%                   Rademacher matrix, (3) for subsampled DFT matrix, (4)
%                   for a single, common complex IID Gaussian matrix
%   .lambda         Scalar activity probability for s(t), i.e.,
%                   Pr{s_n(t) = 1} = Params.lambda
%   .pz1            Scalar Markov chain transition probability,
%                   Pr{s_n(t) = 0 | s_n(t-1) = 1}
%   .eta        	Complex scalar of theta means, i.e., E[theta_n(t)] = 
%                   Params.eta
%   .kappa          Scalar theta variances, i.e., var{theta_n(t)} = 
%                   Params.kappa
%   .alpha          Scalar Gauss-Markov "innovation rate" parameter, btwn.
%                   0 and 1, (see above)
%   .SNRmdB         The desired per-measurement empirical SNR, in dB, i.e.
%                   the ratio of the average signal power to the average
%                   noise power at each measurement should be SNRmdB
%   .version        Type of data to generate. (1) for complex, (2) for real
%                   [dflt = 1]
%
% OUTPUTS:
% x_true            A 1-by-T+1 cell array of length-N complex-valued true
%                   signals generated according to the signal model
%                   specified
% y                 A 1-by-T+1 cell array of observation vectors.  Each
%                   complex-valued observation vector is defined to be of
%                   length M(t), t = 0, 1, ..., T
% A                 A 1-by-T+1 cell array of complex-valued measurement
%                   matrices, with each matrix of dimension M(t)-by-N.
% support           A 1-by-T+1 cell array of indices of active coefficients
%                   at each timestep, (an active coefficient is one at
%                   which s_n(t) = 1)
% NNZ               A 1-by-T+1 vector of the number of active coefficients
%                   at each timestep
% sig2e             Scalar variance of circular, complex, additive Gaussian
%                   measurement noise that will yield an average
%                   per-measurement SNR of Params.SNRmdB
%

%
% Coded by: Justin Ziniel, The Ohio State Univ.
% E-mail: ziniel.1@osu.edu
% Last change: 12/05/12
% Change summary: 
%		- Created from signal_gen_fxn v0.2 (12/09/10; JAZ)
%       - Modified the generation of the theta random process to allow the
%           process to have a non-zero mean (01/20/11; JAZ)
%       - Added a fourth A_type: common Gaussian MMV matrix (04/20/11; JAZ)
%       - Added option to generate real-valued data (04/21/11; JAZ; v0.3)
%       - Modified some EM computations and naming conventions to reflect
%         changes in signal model to a steady-state dynamic system;
%         deprecated in favor of dcs_signal_gen_fxn (12/05/12; JAZ; v1.1)
% Version 1.2
%

function [x_true, y, A, support, NNZ, sig2e] = signal_gen_fxn(Params)

%% Begin by creating a default test case matched to the signal model

if nargin == 0      % No user-provided variables, create default test
    % Signal generation parameters
    N = 256;                % # parameters                              [dflt=256]
    T = 25;                 % # of timesteps - 1                        [dflt=9]
    M = N/12*ones(1,T+1);	% # observations                            [dflt=N/4]
    lambda = 0.10;        % prior probability of non-zero tap         [dflt=0.04]
    pz1 = 0.05;             % Pr{s_n(t) = eps | s_n(t-1) = 1}           [dflt=0.05]
    A_type = 1;             % 1=iid CN, 2=rademacher, 3=subsampled DFT	[dflt=1]
    eta = 1+1j;            	% Complex mean of active coefficients       [dflt=0]
    kappa = 1;        	% Circular variance of active coefficients	[dflt=1]
    alpha = 0.05;           % Innovation rate of thetas (1 = total)     [dflt=0.10]
    SNRmdB = 15;           	% Per-measurement SNR (in dB)               [dflt=15]
end


%% Check errors in the input

if nargin > 0
    % Make sure all Params fields are present (allowing for deprecated
    % naming convention, for backwards compatibility)
    if ~(   isfield(Params, 'N') && isfield(Params, 'M') ...
            && (isfield(Params, 'lambda') || isfield(Params, 'lambda_0')) ...
            && isfield(Params, 'pz1') && isfield(Params, 'alpha') ...
            && (isfield(Params, 'eta') || isfield(Params, 'eta_0')) ...
            && (isfield(Params, 'kappa') || isfield(Params, 'kappa_0')) ...
            && isfield(Params, 'SNRmdB') && isfield(Params, 'A_type') )
        error('signal_gen_fxn: ''Params'' structure is missing arguments')
    end
    
    % For backwards compatibility, migrate any references to lambda_0,
    % kappa_0, or eta_0 to the appropriate updated values, but warn user
    if isfield(Params, 'lambda_0')
        Params.lambda = Params.lambda_0;
        warning('Parameter ''lambda_0'' has been deprecated. Use ''lambda''.')
    end
    if isfield(Params, 'kappa_0')
        Params.kappa = Params.kappa_0;
        warning('Parameter ''kappa_0'' has been deprecated. Use ''kappa''.')
    end
    if isfield(Params, 'eta_0')
        Params.eta = Params.eta_0;
        warning('Parameter ''eta_0'' has been deprecated. Use ''eta''.')
    end

    % Unpack 'Params' structure
    if numel(Params.N) == 1
        N = Params.N;
    else
        error('signal_gen_fxn: Incorrect number of elements, ''Params.N''')
    end
    M = Params.M;
    T = length(M) - 1;
    if Params.A_type == 1 || Params.A_type == 2 || Params.A_type == 3 || ...
            Params.A_type == 4
        A_type = Params.A_type;
    else
        error('signal_gen_fxn: Invalid input, ''Params.A_type''')
    end
    if numel(Params.lambda) == 1
        lambda = Params.lambda;
    else
        error('signal_gen_fxn: Incorrect number of elements, ''Params.lambda''')
    end
    if numel(Params.pz1) == 1
        pz1 = Params.pz1;
        p1z = lambda*pz1/(1 - lambda);
    else
        error('signal_gen_fxn: Incorrect number of elements, ''Params.pz1''')
    end
    if numel(Params.eta) == 1
        eta = Params.eta;
    else
        error('signal_gen_fxn: Incorrect number of elements, ''Params.eta''')
    end
    if numel(Params.kappa) == 1
        kappa = Params.kappa;
    else
        error('signal_gen_fxn: Incorrect number of elements, ''Params.kappa''')
    end
    if numel(Params.alpha) == 1
        alpha = Params.alpha;
    else
        error('signal_gen_fxn: Incorrect number of elements, ''Params.alpha''')
    end
    rho = (2 - alpha)*kappa/alpha;
    if numel(Params.SNRmdB) == 1
        SNRmdB = Params.SNRmdB;
    else
        error('signal_gen_fxn: Incorrect number of elements, ''Params.SNRmdB''')
    end
    if isfield(Params, 'version')
        version = Params.version;
    else
        version = 1;    % Default to generating complex-valued data
    end
end

%% Create a true signal and measurements based on the signal model

vnc = 1;                % =1 variable # of active coefs, =0 fixed       [dflt=1]

% Create the time series of true supports
if vnc == 1,	% Variable number of active coeffs
    NNZ(1) = binornd(N, lambda);      % Number of active coeffs
else            % Fixed number of active coeffs
    NNZ(1) = round(lambda*N);         % Number of active coeffs
end;
if NNZ(1) == 0, NNZ(1) = 1; end;        % Force at least one active coeff
[~, locs] = sort(rand(N,1));        % Indices of active coeffs
s_true{1} = zeros(N,1);
s_true{1}(locs(1:NNZ(1))) = 1;          % True model vector for t=0
support{1} = locs(1:NNZ(1));         	% Active coeff support for t=0
comp_support{1} = locs(NNZ(1)+1:end); 	% Inactive coeff support for t=0

% Now create subsequent model supports according to Markov chain
% evolution
for t = 2:T+1
    % First evolve active elements
    deact_locs = find(rand(NNZ(t-1),1) < pz1);	% Deactivated taps
    support{t} = support{t-1}(setdiff([1:NNZ(t-1)], deact_locs));
    comp_support{t} = support{t-1}(deact_locs);     % Move indices to inactive

    % Next evolve inactive elements
    act_locs = find(rand(N-NNZ(t-1),1) < p1z);  % Activated taps
    support{t} = union(support{t}, comp_support{t-1}(act_locs));
    comp_support{t} = union(comp_support{t}, ...
        comp_support{t-1}(setdiff([1:N-NNZ(t-1)], act_locs)));

    % Update the total sparsity for this next timestep
    NNZ(t) = length(support{t});

    % Construct model vector from signal support
    s_true{t} = zeros(N,1);
    s_true{t}(support{t}) = 1;      % Locations of active elements
end

if nargin == 0,     % Plot the signal support
    figure(1); clf; imagesc([s_true{:}]); xlabel('Timestep (t)');
    ylabel('s_n^{(t)}'); title('Signal support trajectories'); pause(2)
end

% Create the time series of true amplitudes
% First theta vector
if version == 1     % Complex-valued case
    theta_true{1} = eta*ones(N,1) + sqrt(kappa/2)*randn(N,2)*[1; j];
elseif version == 2
    theta_true{1} = eta*ones(N,1) + sqrt(kappa)*randn(N,1);
else
    error('signal_gen_fxn: Params.version must either be ''1'' or ''2''')
end
for t = 2:T+1       % Subsequent timesteps
    % Gauss-Markov process
    if version == 1     % Compex-valued
        theta_true{t} = ((1 - alpha)*(theta_true{t-1} - eta*ones(N,1)) + ...
            alpha*sqrt(rho/2)*randn(N,2)*[1; j]) + eta*ones(N,1);
    else                % Real-valued
        theta_true{t} = ((1 - alpha)*(theta_true{t-1} - eta*ones(N,1)) + ...
            alpha*sqrt(rho)*randn(N,1)) + eta*ones(N,1);
    end
end

if nargin == 0,     % Plot the un-squelched signal amplitudes
    figure(1); clf; imagesc(abs([theta_true{:}])); xlabel('Timestep (t)');
    ylabel('|\theta_n^{(t)}|'); title('Un-squelched amplitude trajectories');
    colorbar; pause(2)
end

% Create the time series of true signals, and measurement matrices
total_signal_power = 0;     % Will give empirical overall signal power
for t = 1:T+1
    % Create true signal
    x_true{t} = s_true{t}.*theta_true{t};

    % Create measurement matrices
    if A_type == 1,             % iid gaussian
        if version == 1
            A{t} = (randn(M(t),N) + j*randn(M(t),N))/sqrt(2*M(t));
            for n=1:N, A{t}(:,n) = A{t}(:,n)/norm(A{t}(:,n)); end;
        else
            A{t} = randn(M(t),N)/sqrt(M(t));
            for n=1:N, A{t}(:,n) = A{t}(:,n)/norm(A{t}(:,n)); end;
        end
    elseif A_type == 2,			% rademacher
        A{t} = sign(randn(M(t),N))/sqrt(M(t));     
    elseif A_type == 3, 		% subsampled DFT
        if version == 2, error('Inappropriate A_type for version'); end;
        mm = zeros(N,1); while sum(mm)<M(t), mm(ceil(rand(1)*N))=1; end; 
        A{t} = dftmtx(N); A{t} = A{t}(find(mm==1),:)/sqrt(M(t));
    elseif A_type == 4,         % common Gaussian MMV matrix
        if t == 1
            if version == 1
                A{t} = (randn(M(t),N) + j*randn(M(t),N))/sqrt(2*M(t));
                for n=1:N, A{t}(:,n) = A{t}(:,n)/norm(A{t}(:,n)); end;
            else
                A{t} = randn(M(t),N)/sqrt(M(t));
                for n=1:N, A{t}(:,n) = A{t}(:,n)/norm(A{t}(:,n)); end;
            end
        else
            A{t} = A{1};
        end
    end;

    total_signal_power = total_signal_power + norm(A{t}*x_true{t})^2;
end

if nargin == 0,     % Plot the true signal
    figure(1); clf; imagesc(abs([x_true{:}])); xlabel('Timestep (t)');
    ylabel('|x_n^{(t)}|'); title('True signal magnitude trajectories');
    colorbar;
end

% Determine the appropriate CAWGN variance to yield a
% per-measurement SNR of SNRmdB
sig2e = (total_signal_power/sum(M))*10^(-SNRmdB/10);

% Construct noisy measurements
for t = 1:T+1
    if version == 1
        e{t} = sqrt(sig2e/2)*randn(M(t),2)*[1; j];  % CAWGN
    else
        e{t} = sqrt(sig2e)*randn(M(t),1);   % AWGN
    end
    
    y{t} = A{t}*x_true{t} + e{t};

    % Store the empirical SNR at this timestep
    SNRdB_emp(t) = 20*log10(norm(A{t}*x_true{t})/norm(e{t}));
end
1+1;
