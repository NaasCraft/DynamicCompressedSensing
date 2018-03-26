% DCS_SIGNAL_GEN_FXN      Function for generating a time-series of 
% complex- (or real-)valued signals, {x(t)}, t = 1, ..., T, with each x(t)  
% being N-dimensional.  Complex-valued observations, {y(t)}, are then
% obtained through a noisy linear combination, y(t) = A*x(t) + e(t), with 
% y(t) being an M-dimensional vector of observations, A being an 
% M-by-N complex- (or real-)valued measurement matrix whose form is
% specified by the user, and e(t) being circular, complex (real), additive 
% Gaussian noise with iid elements of zero mean and variance sig2e.
%
% At any given time, t, it is assumed that the coefficients of x(t),
% x_n(t), can be represented by the product of a discrete variable, s_n(t),
% and a continuous-valued variable, theta_n(t), that is, 
% x_n(t) = s_n(t) * theta_n(t).  s_n(t) either takes the value 0 or 1, and 
% follows a Bernoulli distribution with Pr{s_n(t) = 1} = lambda.  theta_n(t) 
% follows a complex (real) Gaussian distribution with apriori mean zeta and
% variance sigma2. At timesteps t > 1, theta_n(t) evolves according to a 
% Gauss-Markov process, i.e. theta_n(t) = (1 - alpha)*theta_n(t-1) + 
% alpha*w_n(t), where alpha is a scalar between 0 and 1, and w_n(t) is 
% complex (real) Gaussian driving noise with zero mean and a variance
% chosen to yield var{theta_n(t)} = sigma2.  s_n(t) evolves according to a
% first-order Markov chain with steady-state activity probability lambda,
% and active-to-inactive transition probability p01 = Pr{s_n(t) = 0 |
% s_n(t-1) = 1}.
%
% SYNTAX:
% [x_true, y, A, support, K, sig2e] = dcs_signal_gen_fxn(SignalParams)
%
% INPUTS:
% SignalParams      An object of the class SigGenParams (see SigGenParams.m
%                   in the folder ClassDefs), which will contain all of the
%                   model parameters needed to specify the distributions
%                   described above.
%
% OUTPUTS:
% x_true            A 1-by-T cell array of length-N complex- (real-)valued 
%                   signals generated according to the specified
%                   Bernoulli-Gaussian signal model
% y                 A 1-by-T cell array of measurement vectors.  Each
%                   complex- (real-)valued observation vector is defined to
%                   be of length M, t = 1, ..., T
% A                 A 1-by-T cell array of M-by-N measurement matrices.
% support           A 1-by-T cell array of indices of the non-zero elements
%                   of s(t), i.e., support{1} = {n | s_n(t) = 1}
% K                 The total number of non-zero elements of s(t), i.e.,
%                   K(t) = cardinality(support{t})
% sig2e             Scalar variance of circular, additive Gaussian
%                   measurement noise that will yield an average
%                   per-measurement SNR of SigGenParams.SNRmdB
%

%
% Coded by: Justin Ziniel, The Ohio State Univ.
% E-mail: zinielj@ece.osu.edu
% Last change: 12/07/15
% Change summary: 
%		- Created from mmv_signal_gen_fxn v1.1 (02/08/12; JAZ)
%       - Changed generation of A matrices (12/05/12; JAZ)
%       - Fixed bugs introduced in v1.1 due to migration to SigGenParams
%           and DCSModelParams objects (12/07/15; JAZ)
% Version 1.2
%

function [x_true, y, A, support, K, sig2e] = dcs_signal_gen_fxn(SignalParams)

%% Begin by creating a default test case matched to the signal model

if nargin == 0      % No user-provided variables, create default test
    % Use the default constructor of SigGenParams (note that the folder
    % ClassDefs must be included in MATLAB's path for this to work)
    try
        SignalParams = SigGenParams();  % Call default constructor
        print(SignalParams);            % Print configuration to screen
    catch
        error('Please include the folder ClassDefs in MATLAB''s path')
    end
end


%% Unpack the SigGenParams structure

[N, M, T, A_type, lambda, zeta, sigma2, alpha, SNRmdB, version, rho, ...
    p01] = getParams(SignalParams);
p10 = lambda*p01/(1 - lambda);  	% This transition prob. ensures
                                    % that E[K(t)] = E[K(0)] for all t

%% Create a true signal and measurements based on the signal model

vnc = 1;                % = 1 variable # of active coefs, = 0 fixed [dflt=1]

% Create the true support
s_true = cell(1,T);
[s_true{:}] = deal(zeros(N,1));
support = cell(1,T);
K = NaN(1,T);
if vnc == 1,	% Variable number of active coeffs
    K(1) = binornd(N, lambda);   	% Number of active coeffs
else            % Fixed number of active coeffs
    K(1) = round(lambda*N);      	% Number of active coeffs
end;
if K == 0, K = 1; end;          % Force at least one active coeff
[~, locs] = sort(rand(N,1));   	% Indices of active coeffs
s_true{1}(locs(1:K(1))) = 1;          % True support vector
support{1} = locs(1:K(1));         	% Active coeff support
for t = 2:T
    Inacts = find(s_true{t-1} == 0);
    s_true{t}(Inacts) = rand(numel(Inacts),1) < p10;
    Acts = find(s_true{t-1} == 1);
    s_true{t}(Acts) = 1 - (rand(numel(Acts),1) < p01);
    support{t} = find(s_true{t} == 1);
    K(t) = numel(support{t});
end

if nargin == 0,     % Plot the signal support
    figure(1); clf; stem([s_true{:}]); xlabel('Coefficient index [n]');
    ylabel('s[n]'); title('Signal support'); pause(2)
end

% Create the time series of true amplitudes
theta_true = cell(1,T);
% First theta vector
if version == 1     % Complex-valued case
    theta_true{1} = zeta*ones(N,1) + sqrt(sigma2/2)*randn(N,2)*[1; 1j];
elseif version == 2
    theta_true{1} = zeta*ones(N,1) + sqrt(sigma2)*randn(N,1);
end
for t = 2:T         % Subsequent timesteps
    % Gauss-Markov process
    if version == 1     % Compex-valued
        theta_true{t} = ((1 - alpha)*(theta_true{t-1} - zeta*ones(N,1)) + ...
            alpha*sqrt(rho/2)*randn(N,2)*[1; 1j]) + zeta*ones(N,1);
    else                % Real-valued
        theta_true{t} = ((1 - alpha)*(theta_true{t-1} - zeta*ones(N,1)) + ...
            alpha*sqrt(rho)*randn(N,1)) + zeta*ones(N,1);
    end
end

if nargin == 0,     % Plot the un-squelched signal amplitudes
    figure(1); clf; imagesc(abs([theta_true{:}])); xlabel('Timestep (t)');
    ylabel('|\theta_n^{(t)}|'); title('Un-squelched amplitude trajectories');
    colorbar; pause(2)
end

% Create the time series of true signals, and measurement matrices
total_signal_power = 0;     % Will give empirical overall signal power
for t = 1:T
    % Create true signal
    x_true{t} = s_true{t} .* theta_true{t};

    % Create measurement matrices

    if A_type == 1,             % iid gaussian w/ unit-norm columns
        if version == 1     % Complex
            A{t} = (randn(M,N) + 1j*randn(M,N))/sqrt(2*M);
            for n=1:N, A{t}(:,n) = A{t}(:,n)/norm(A{t}(:,n)); end;
        else
            A{t} = randn(M,N)/sqrt(M);
            for n=1:N, A{t}(:,n) = A{t}(:,n)/norm(A{t}(:,n)); end;
        end
    elseif A_type == 2,			% rademacher
        A{t} = sign(randn(M,N))/sqrt(M);     
    elseif A_type == 3, 		% subsampled DFT
        if version == 2, error('Inappropriate A_type for version'); end;
        mm = zeros(N,1); while sum(mm)<M, mm(ceil(rand(1)*N))=1; end; 
        A{t} = dftmtx(N); A{t} = A{t}(mm==1,:)/sqrt(M);
    elseif A_type == 4      % Single, common iid Gaussian matrix
        if t == 1
            if version == 1     % Complex
                A{t} = (randn(M,N) + 1j*randn(M,N))/sqrt(2*M);
                for n=1:N, A{t}(:,n) = A{t}(:,n)/norm(A{t}(:,n)); end;
            else
                A{t} = randn(M,N)/sqrt(M);
                for n=1:N, A{t}(:,n) = A{t}(:,n)/norm(A{t}(:,n)); end;
            end
        else
            A{t} = A{1};
        end
    end

    total_signal_power = total_signal_power + norm(A{t}*x_true{t})^2;
end

if nargin == 0,     % Plot the true signal
    figure(1); clf; imagesc(abs([x_true{:}])); xlabel('Timestep (t)');
    ylabel('|x_n^{(t)}|'); title('True signal magnitude trajectories');
    colorbar;
end

% Determine the appropriate CAWGN variance to yield a
% per-measurement SNR of SNRmdB
sig2e = (total_signal_power/M/T)*10^(-SNRmdB/10);

% Construct noisy measurements
for t = 1:T
    if version == 1
        e{t} = sqrt(sig2e/2)*randn(M,2)*[1; 1j];  % CAWGN
    else
        e{t} = sqrt(sig2e)*randn(M,1);   % AWGN
    end
    
    y{t} = A{t}*x_true{t} + e{t};

    % Store the empirical SNR at this timestep
    SNRdB_emp(t) = 20*log10(norm(A{t}*x_true{t})/norm(e{t}));
end

