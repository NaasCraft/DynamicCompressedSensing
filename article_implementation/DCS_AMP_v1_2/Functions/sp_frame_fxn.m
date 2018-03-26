% SP_FRAME_FXN      Function for performing soft-input, soft-output sparse
% reconstruction of a signal, x, observed through a noisy linear 
% combination, y = Ax + e, where the prior distribution of coefficient x_n
% is given by the following mixture-of-Gaussians:
%       p(x_n) = (1 - pi_n)*delta(x_n) + pi_n*CN(xi_n, psi_n),
%                               OR
%       p(x_n) = (1 - pi_n)*delta(x_n) + pi_n*N(xi_n, psi_n),
% where delta(x_n) is a point-mass prior at x_n = 0. The parameters pi_n, 
% xi_n, and psi_n are all user-specified.  If the signal is complex-valued,
% then the circular Gaussian component, CN(xi_n, psi_n), is referred to as
% the "active" component.  If the signal is real-valued, then 
% N(xi_n, psi_n) is the active Gaussian component. pi_n is referred as to
% the "activity prior", or "active Gaussian mixture weighting", and is a 
% real-valued scalar in [0,1].  Inference is performed using either 
% approximate belief propagation (ABP) or approximate message passing 
% (AMP).  Whether to use the complex-valued version, or the real-valued
% version of the algorithm is determined automatically during execution of
% this function.
%
% SYNTAX:
% [x_hat, OutMessages, StateOut] = sp_frame_fxn(y, A, Params, Options, ...
%                                               StateIn)
%
% INPUTS:
% y                 An M-by-1 vector of complex- or real-valued noisy 
%                   measurements
% A                 An M-by-N complex- or real-valued measurement matrix, 
%                   or a function handle that returns a matrix-vector 
%                   product.  See below for the form of this function 
%                   handle
% Params            A structure containing signal distribution parameters
%   .pi             N-by-1 vector of "active" Gaussian component 
%                   weightings, each with values between 0 and 1
%   .xi             N-by-1 complex- or real-valued vector of "active" 
%                   Gaussian means
%   .psi            N-by-1 vector of "active" Gaussian variances
%   .eps            Real-valued scalar, (<< 1), required for producing an
%                   approximation of the outbound BP messages of xi and psi
%                   [dflt: 1e-5]
%   .sig2e          Scalar noise variance.  In the complex case, this noise
%                   is circular, complex, additive Gaussian, and in the
%                   real case, is additive white Gaussian
%   .tau            Optional parameter.  If tau = -1, a Taylor series
%                   approximation is used to determine the outgoing
%                   messages OutMessages.xi and OutMessages.psi.  If a
%                   positive value between zero and one is passed for tau, 
%                   e.g., tau = 1 - 1e-4, it will be used as a threshold
%                   on the incoming activity probabilities, Params.pi, to
%                   determine whether to pass an informative or 
%                   non-informative message via OutMessages.xi and
%                   OutMessages.psi [dflt: -1]
% Options           An *optional* structure of runtime configurations
%   .iter           Max number of BP message update iterations [dflt: 50]
%   .alg            Type of BP algorithm to use: (1) for Gaussian BP, (2)
%                   for AMP [dflt: 2]
% StateIn           An *optional* structure of BP message initializations
%   .c              Variance of residual [dflt: 100*Params.psi]
%   .z              Mean of residual [dflt: y]
%   .mu             Coefficient-to-measurement msg means [dflt: zeros(N,1)]
%
% OUTPUTS:
% x_hat             N-by-1 vector MMSE estimate of x
% OutMessages       Structure containing outbound BP message parameters
%   .pi             N-by-1 vector of outbound activity priors
%   .xi             N-by-1 vector of "active" Gaussian means
%   .psi            N-by-1 vector of "active" Gaussian variances
% StateOut          Structure of terminal states of BP message parameters
%   .c              Variance of residual
%   .z              Mean of residual
%   .mu             Coefficient-to-measurement msg means
%   .v              N-by-1 vector of conditional variances, var{x_n|y}
%
%
% *** If input A is a function handle, it should be of the form @(x,mode).
%   If mode == 1, then the function handle returns an M-by-1 vector that is
%   the result of A*x.  If mode == 2, then the function handle returns an
%   N-by-1 vector that is the result of A'*x. ***
%

%
% Coded by: Justin Ziniel, The Ohio State Univ.
% E-mail: zinielj@ece.osu.edu
% Last change: 12/29/11
% Change summary: 
%		- Created from sp_frame_fxn v1.0 (12/29/11; JAZ)
%       - Corrected mistake in how outgoing Taylor approximation messages
%         were computed for the real-valued signal case, improved numerical
%         precision handling for exponentiated quantities 
%         (12/29/11; JAZ; v1.1)
% Version 1.2
%

function [x_hat, OutMessages, StateOut] = sp_frame_fxn(y, A, Params, Options, StateIn)

%% Initialize algorithm, declare variables

if nargin == 0      % Create default test-case, absent user-provided variables
    % Signal generation parameters
    N = 400;   			% # parameters                      [dflt=256]
    M = N/4;			% # observations                    [dflt=N/4]
    pi = 0.05;			% prior probability of non-zero tap	[dflt=0.04]
    vnc = 1;			% =1 variable # of active coefs, =0 fixed	[dflt=1]
    SNRdB = 25;			% SNR in dB                         [dflt=15]
    A_type = 1;			% 1=iid CN, 2=rademacher, 3=subsampled DFT	[dflt=1]
    xi = 0;             % mean of active coefficients       [dflt=0]
    psi = 1;            % variance of active coefficients	[dflt=1]

    % Algorithmic parameters
    iter = 100;			% number of BP iterations			[dflt=10]
    alg = 2;			% type of BP approximation (1=ABP, 2=AMP)	[dflt=2]

    % Debugging parameters
    plot_bp = 2;			% 0=no plot, 1=plot at end, 2=plot each iteration
    plot_fbmp = 0;			% 0=no plot, 1=plot at end
    
    for t = 1

%         % set randn and rand seeds
%         reset(RandStream.getDefaultStream);	% uncomment to start at reference
%         %defaultStream.State = savedState;	% uncomment to use previous state
%         defaultStream = RandStream.getDefaultStream;		% leave uncommented!
%         savedState = defaultStream.State;		% leave uncommented!

        % Generate true sparsity pattern 
        if vnc == 1,	% Variable number of active coeffs
        	NNZ = binornd(N, pi);       	% Number of active coeffs
        else            % Fixed number of active coeffs
            NNZ = round(pi*N);              % Number of active coeffs
        end;
        [dummy, locs] = sort(rand(N,1));    % Indices of active coeffs
        s_true = zeros(N,1);
        s_true(locs(1:NNZ)) = 1;         	% True model vector
        support = locs(1:NNZ);              % Active coeff support
        comp_support = locs(NNZ+1:end);     % Inactive coeff support
        if NNZ == 0, error('No active taps!'); end;

        % Generate observation
        if A_type == 1,         % IID complex Gaussian
            A = (randn(M,N) + j*randn(M,N))/sqrt(2*M);     % approx unit-norm columns
            for n=1:N, A(:,n) = A(:,n)/norm(A(:,n)); end;
        elseif A_type == 2,     % Rademacher
            A = sign(randn(M,N))/sqrt(M);     
        elseif A_type == 3,     % Subsampled DFT
            mm = zeros(N,1); while sum(mm)<M, mm(ceil(rand(1)*N))=1; end; 
            A = dftmtx(N); A = A(find(mm==1),:)/sqrt(M);
        end;
        %scale=1; A = A*scale; var_theta = var_theta/scale^2; % only for testing!
        theta_true = xi*ones(N,1)+sqrt(psi/2)*randn(N,2)*[1;j];	% Non-sparse coeffs
        x_true = s_true.*theta_true; 		% Sparse coeffs
        sig2e = norm(A*x_true)^2/M*10^(-SNRdB/10);              % Noise variance 
        w = sqrt(sig2e/2)*randn(M,2)*[1;j];	% CAWGN samples
        y = A*x_true + w;                   % Observations
        
        % Pack structures
        Params.pi = pi; Params.xi = xi; Params.psi = psi;
        Params.sig2e = sig2e; Options.iter = iter; Options.alg = alg;

        % Genie MSE performance (for mean_theta = 0)
        s_mod = ones(N,1); s_mod(support) = 2;
        mean_tmp = [0; xi];             % MoG means
        var_tmp = [0; psi];             % MoG variances
        R = A*diag(var_tmp(s_mod))*A' + sig2e*eye(M);
        x_mmsegenie = mean_tmp(s_mod) + diag(var_tmp(s_mod))*A'*(R\(y - ...
            A*mean_tmp(s_mod)));
        NMSEdB_genie = 20*log10(norm(x_true - x_mmsegenie)/norm(x_true));
        
        % Plot flag
        plot_bp = 2; 	% 0=no plots, 1=plot at end, 2=plot each iteration 
    end
end


%% Error checking

if (nargin < 3) && (nargin > 0)
    error('sp_frame_fxn: Function requires at least 3 arguments');
else
    % Fill-in missing variables
    for t = 1
        if isa(A, 'function_handle')
            explicitA = 0;      % Matrix A is not given explicitly
            M = length(y);
            N = length(A(y,2));
        else
            [M, N] = size(A);
            explicitA = 1;
        end
        
        % Declare important algorithmic constants
        maxprob = 1 - 1e-15; 	% OutMessages.pi(n) cannot exceed this
        minprob = 1 - maxprob;  % OutMessages.pi(n) cannot fall below this
        tol = 1e-5;             % Early termination tolerance

        % Unpack 'params' structure
        if ~isfield(Params, 'pi') || ~isfield(Params, 'xi') || ...
                ~isfield(Params, 'psi') || ~isfield(Params, 'sig2e')
            error('sp_frame_fxn: ''Params'' is missing req''d fields')
        end
        if ~isfield(Params, 'eps')
            Params.eps = 1e-5;      % Parameter 'eps' not defined, use dflt
        end
        if ~isfield(Params, 'tau')
            Params.tau = -1;        % Default is to use Taylor approx.
        end
        if numel(Params.pi) == 1
            pi = Params.pi*ones(N,1);
        elseif numel(Params.pi) == N;
            pi = Params.pi;
        else
            error('sp_frame_fxn: Incorrect number of elements, ''Params.pi''')
        end
        if numel(Params.xi) == 1
            xi = Params.xi*ones(N,1);
        elseif numel(Params.xi) == N
            xi = Params.xi;
        else
            error('sp_frame_fxn: Incorrect number of elements, ''Params.xi''')
        end
        if numel(Params.psi) == 1
            psi = Params.psi*ones(N,1);
        elseif numel(Params.psi) == N
            psi = Params.psi;
        else
            error('sp_frame_fxn: Incorrect number of elements, ''Params.psi''')
        end
        if numel(Params.eps) == 1
            eps = Params.eps;
        else
            error('sp_frame_fxn: Incorrect number of elements, ''Params.eps''')
        end
        if numel(Params.sig2e) == 1
            sig2e = Params.sig2e;
        else
            error('sp_frame_fxn: Incorrect number of elements, ''Params.sig2e''')
        end
        if (Params.tau < 0 && Params.tau ~= -1) || Params.tau > 1
            error(['sp_frame_fxn: Threshold tau should be either be -1' ...
                ' to use the Taylor series approximation, or else be' ...
                'between 0 and 1'])
        else
            tau = Params.tau;
        end

        % Unpack 'Options' structure
        iter = 50;      % Default number of BP iterations
        alg =  2;       % Default to AMP version
        if (nargin >= 4) || (nargin == 0)
            if isfield(Options, 'iter')
                iter = Options.iter;     
            end
            if isfield(Options, 'alg')
                alg = Options.alg;
                if ~explicitA && alg == 1
                    error(['sp_frame_fxn cannot operate in standard BP' ...
                        ' mode when input A is a function handle'])
                end
            end
        end

        % Unpack 'StateIn' structure
        if alg == 1
            c = 100*psi;      	% Default residual variance
            Z = y*ones(1,N);    % Default residual mean (std. BP case)
            MU = inf*ones(size(Z));
        elseif alg == 2
            c = 100*mean(psi);	% Default residual variance
            z = y;              % Default residual mean (AMP case)
            Mu = zeros(N,1);    % Default coefficient-to-msg mean
        else
            error('sp_frame_fxn: Unknown algorithm (''Options.alg'')')
        end
        if nargin >= 5
            if isfield(StateIn, 'c')
                if alg == 1
                    c = StateIn.c*ones(N,1);
                else
                    c = StateIn.c;
                end
            end
            if isfield(StateIn, 'z')
                if alg == 1
                    Z = StateIn.z;
                else
                    z = StateIn.z;
                end
            end
            if isfield(StateIn, 'mu')
                Mu = StateIn.mu;
            end
        end
        
        % Determine if algorithm should be using complex-valued version (1)
        % or real-valued version (2)
        if norm(imag(y)) == 0
            version = 2;    % Real-valued version
            e1 = zeros(N,1); e1(1) = 1;
            if explicitA
                if norm(imag(A*e1)) > 1e-2 || norm(imag(xi)) > 1e-2
                    version = 1;    % Complex-valued version
                end
            else
                if norm(imag(A(e1,1))) ~= 0 || norm(imag(xi)) ~= 0
                    version = 1;    % Complex-valued version
                end
            end
        else
            version = 1;    % Complex-valued version
        end
        
        tol_flag = 0;   % Initialize flag for early termination
    end
end


%% Execute std. BP or AMP algorithm

% Declare static runtime variables
gam_A = ((1 - pi)./pi);             % Constant "A" in gamma definition
if sum(gam_A == inf | gam_A == 0) == N
    genie_version = 1;      % We are running genie_multi_frame_fxn thus
else                        % we must use a modified method of computing
    genie_version = 0;      % BP or AMP variances (to avoid NaNs)
end
oneN = ones(N,1);
oneM = ones(M,1);
if alg == 1
    GAM_A = oneM*gam_A.';         	% Tile the vector
    A2 = abs(A).^2;                 % Matrix modulus-squared
end
x_hat = NaN*ones(N,1);              % Initialize x_hat

for k = 1:iter      % Number of BP updates to perform
    
    % Update x-to-g messages
    if alg == 1     % Std. BP
        AhZ = conj(A).*Z;
        PHI = oneM*sum(AhZ, 1) - AhZ;
        % Setup variables to compute gamma
        GAM_B = oneM*(c + psi).';
        GAM_C = oneM*(c./psi).';
        % Calculate gamma
        if version == 2     % Real-valued version
            GAM_EXP = exp(-(1/2)*[(abs(PHI + GAM_C.*(oneM*xi.')).^2 - ...
                GAM_C.*(1 + GAM_C).*(oneM*(abs(xi).^2).')) ...
                ./ (GAM_C.*GAM_B)]);
            % Clip the Gamma exponent values that are too large or small
            % due to numerical precision issues
            GAM_EXP(GAM_EXP == 0) = 1/realmax;
            GAM_EXP(GAM_EXP == inf) = realmax;
            GAMMA = GAM_A.*sqrt(GAM_B./(oneM*c.')).*GAM_EXP;  % Eqn. xx
        else    % Complex-valued version
            GAM_EXP = exp(-((abs(PHI + GAM_C.*(oneM*xi.')).^2 - ...
                GAM_C.*(1 + GAM_C).*(oneM*(abs(xi).^2).')) ...
                ./ (GAM_C.*GAM_B)));
            % Clip the Gamma exponent values that are too large or small
            % due to numerical precision issues
            GAM_EXP(GAM_EXP == 0) = 1/realmax;
            GAM_EXP(GAM_EXP == inf) = realmax;
            GAMMA = GAM_A.*(GAM_B./(oneM*c.')).*GAM_EXP;  % Eqn. xx
        end
        % Compute means and variances of x-to-g messages
        GAMMA_REDUX = 1./(1 + GAMMA);   % This will get used several times
        MU_OLD = MU;    % Move mu matrix to old slot
        MU = GAMMA_REDUX.*((PHI.*(oneM*psi.') + oneM*(c.*xi).') ./ GAM_B);
        if genie_version        % Must distinguish here to avoid NaNs
            V = GAMMA_REDUX.*(oneM*(c.*psi).')./GAM_B;
        else
            V = GAMMA_REDUX.*(oneM*(c.*psi).')./GAM_B + GAMMA.*abs(MU).^2;
        end
        
        % Check for early termination
        if norm(sum(MU_OLD - MU, 2))^2/N < tol && k > 1
            tol_flag = 1;
        end
        
    else            % AMP version
        if explicitA
            Phi = A'*z + Mu;            % AMP method for calculating Phi (Eqn. (A4))
        else
            Phi = A(z,2) + Mu;          % AMP method w/ implicit A' mult.
        end
        % Set-up variables to compute gamma
        gam_B = (c*oneN + psi);
        gam_C = c./psi;
    end
    
    
    % Estimate x on final iteration, (or every iteration for debugging)
    if (k == iter) || (nargin == 0) || (tol_flag == 1)
        if alg == 1     % Std. BP method
            phi = sum(AhZ, 1).';
            % Un-tile most gamma sub-variables
            gam_B = GAM_B(1,:).'; gam_C = GAM_C(1,:).';
        else            % AMP method
            phi = Phi;
            c = c*oneN;     % Vectorize the scalar c
        end
        % Compute gamma vector
        if version == 2     % Real-valued version
            gam_exp = exp(-(1/2)*[(abs(phi + gam_C.*xi).^2 - ...
                gam_C.*(1 + gam_C).*(abs(xi).^2))./(gam_C.*gam_B)]);
            % Clip the Gamma exponent values that are too large or small
            % due to numerical precision issues
            gam_exp(gam_exp == 0) = 1/realmax;
            gam_exp(gam_exp == inf) = realmax;
            gamma = gam_A.*sqrt(gam_B./c).*gam_exp;
        else        % Complex-valued version
            gam_exp = exp(-((abs(phi + gam_C.*xi).^2 - ...
                gam_C.*(1 + gam_C).*(abs(xi).^2))./(gam_C.*gam_B)));
            % Clip the Gamma exponent values that are too large or small
            % due to numerical precision issues
            gam_exp(gam_exp == 0) = 1/realmax;
            gam_exp(gam_exp == inf) = realmax;
            gamma = gam_A.*(gam_B./c).*gam_exp;  % Eqn. (D4)
        end
        % Compute mu, x_hat, and v
        gamma_redux = 1./(1 + gamma);
        mu = gamma_redux.*((phi.*psi + c.*xi) ./ gam_B);
        x_hat = mu;
        
        % Compute outgoing messages
        if genie_version    % Must distinguish here to avoid NaNs
            v = gamma_redux.*(c.*psi)./gam_B;
            % Calculate outgoing pi
            if version == 2     % Real-valued version
                OutMessages.pi = 1./(1 + sqrt(gam_B./c).*...
                    exp(-(1/2)*[(abs(phi + gam_C.*xi).^2 - ...
                    gam_C.*(1 + gam_C).*(abs(xi).^2))./(gam_C.*gam_B)]));
            else        % Complex-valued version
                OutMessages.pi = 1./(1 + (gam_B./c).*...
                    exp(-[(abs(phi - gam_C.*xi).^2 - ...
                    gam_C.*(1 + gam_C).*(abs(xi).^2))./(gam_C.*gam_B)]));
            end
            OutMessages.xi(pi == 0) = xi(pi == 0);
            OutMessages.xi(pi == 1) = phi(pi == 1);
            OutMessages.psi(pi == 0) = inf;
            OutMessages.psi(pi == 1) = c(pi == 1);
            OutMessages.xi = OutMessages.xi.';
            OutMessages.psi = OutMessages.psi.';
        else
            v = gamma_redux.*(c.*psi)./gam_B + gamma.*abs(mu).^2;
            % Correct Inf*0 numerical precision problem
            v(gamma == inf) = 0;
            % Calculate outgoing pi
            if version == 2     % Real-valued version
                OutMessages.pi = max(min(1./(1 + sqrt((gam_B./c)).*gam_exp), ...
                    maxprob*ones(N,1)), minprob*ones(N,1));
            else    % Complex-valued version
                OutMessages.pi = max(min(1./(1 + (gam_B./c).*gam_exp), ...
                    maxprob*ones(N,1)), minprob*ones(N,1));
            end
            % Outgoing messages for theta are computed using the sub-function
            % defined at the bottom of this file
            [OutMessages.xi, OutMessages.psi] = OutgoingMessages(pi, ...
                phi, c, eps, version, tau, alg);
        end
        
        % Next, store outgoing conditional variances
        StateOut.v = v;
        
        % Break from for loop if tol_flag == 1
        if tol_flag == 1, break; end
    end
    
    
    % Update g-to-x messages
    if alg == 1     % Std. BP
        AMU = A.*MU;
        Z = y*oneN' - sum(AMU, 2)*oneN' + AMU;
        A2V = sum(A2.*V, 1).';
        c = sig2e + (1/M)*sum(A2V)*oneN - (1/M)*A2V;
        
    else            % AMP version
        % Calculate gamma
        if version == 2     % Real-valued version
            Gam_exp = exp(-(1/2)*[(abs(Phi + gam_C.*xi).^2 - ...
                gam_C.*(1 + gam_C).*(abs(xi).^2))./(gam_C.*gam_B)]);  % Eqn. xx
            % Clip the Gamma exponent values that are too large or small
            % due to numerical precision issues
            Gam_exp(Gam_exp == 0) = 1/realmax;
            Gam_exp(Gam_exp == inf) = realmax;
            Gamma = gam_A.*sqrt(gam_B./c).*Gam_exp;
        else        % Complex-valued version
            Gam_exp = exp(-((abs(Phi + gam_C.*xi).^2 - ...
                gam_C.*(1 + gam_C).*(abs(xi).^2))./(gam_C.*gam_B)));
            % Clip the Gamma exponent values that are too large or small
            % due to numerical precision issues
            Gam_exp(Gam_exp == 0) = 1/realmax;
            Gam_exp(Gam_exp == inf) = realmax;
            Gamma = gam_A.*(gam_B./c).*Gam_exp;  % Eqn. (D4)
        end
        % Compute means and variances of x-to-g messages
        Gamma_redux = 1./(1 + Gamma);
        Mu_old = Mu;    % Move mu vector to old slot
        Mu = Gamma_redux.*((Phi.*psi + c.*xi) ./ gam_B);    % Eqn. (A5)
        if genie_version    % Must distinguish here to avoid NaNs
            V = Gamma_redux.*(c.*psi)./gam_B;
        else
            V = Gamma_redux.*(c.*psi)./gam_B + Gamma.*abs(Mu).^2;    % Eqn. (A6)
            % Correct numerical precision error caused by Inf*0 when Gamma
            % == inf (and Mu == 0)
            V(Gamma == inf) = 0;
        end
        
        % Calculate F', c, and z
        F_prime = V./c;
        c = sig2e + (1/M)*V.'*oneN;     % Eqn. (A7)
        if explicitA
            z = y - A*Mu + (1/M)*sum(F_prime)*z;        % Eqn. (A8)
        else
            z = y - A(Mu,1) + (1/M)*sum(F_prime)*z;     % Implicit A mult.
        end
        
        % Check for early termination
        if norm(Mu_old - Mu)^2/N < tol && k > 1
            tol_flag = 1;
            if genie_version    % Switch now to the slower BP method
                tol_flag = 0;
                tol = 1e-3;
                alg = 1;        % Run BP now, not AMP
                % Initialize the BP messages using the final AMP messages
                A2 = abs(A).^2;                 % Matrix modulus-squared
                GAM_A = oneM*gam_A.';         	% Tile the gam_a vector
                MU = oneM * Mu';
                AMU = A.*MU;
                Z = y*oneN' - sum(AMU, 2)*oneN' + AMU;
                V = oneM * V.';
                A2V = sum(A2.*V, 1).';
                c = sig2e + (1/M)*sum(A2V)*oneN - (1/M)*A2V;
            end
        end
    end
    
    % Plot results
    if nargin==0,
        for t = 1
            BER_(k) = sum((s_true > 2*eps) ~= (OutMessages.pi > 1/2))/N;
            NMSEdB_(k) = 20*log10(norm(x_true - x_hat)/norm(x_true));
            if (plot_bp==2) || ((plot_bp==1) && (k==iter))
                title_str = ['BP-equalizer: iteration ',num2str(k)];
                figure(1)
                clf;
                v0 = NaN*ones(N,1); v0(comp_support) = v(comp_support);
                v1 = NaN*ones(N,1); v1(support) = v(support);
                subplot(411)
                mr0 = NaN*ones(N,1); mr0(comp_support) = real(x_hat(comp_support));
                mr1 = NaN*ones(N,1); mr1(support) = real(x_hat(support));
                x1 = NaN*ones(N,1); x1(support) = x_true(support);
                plot(real(x_mmsegenie),'k.')
                grid on;
                title(title_str);
                hold on;
                errorbar(mr0,sqrt(v0/2),'+');
                errorbar(mr1,sqrt(v1/2),'r+');
                hold off;
                axis([0,N,1.5*min(real(x1)),1.5*max(real(x1))])
                ylabel('real(x-mmse)')
                subplot(412)
                mi0 = NaN*ones(N,1); mi0(comp_support) = imag(x_hat(comp_support));
                mi1 = NaN*ones(N,1); mi1(support) = imag(x_hat(support));
                x1 = NaN*ones(N,1); x1(support) = x_true(support);
                plot(imag(x_mmsegenie),'k.')
                grid on;
                title(['NMSE: ',num2str(NMSEdB_(k)),'dB']);
                hold on;
                errorbar(mi0,sqrt(v0/2),'+');
                errorbar(mi1,sqrt(v1/2),'r+');
                hold off;
%                 axis([0,N,1.5*min(imag(x1)),1.5*max(imag(x1))])
                ylabel('imag(x-mmse)')
                if alg == 1
                    subplot(413)
                    c0 = NaN*ones(N,1); c0(comp_support) = c(comp_support);
                    c1 = NaN*ones(N,1); c1(support) = c(support);
                    stem(10*log10(c0),'.'); hold on; stem(10*log10(c1),'r.'); hold off;
                    axis([1,N,10*log10(sig2e),0])
                    ylabel('c [dB]')
                    title(['NMSE-genie: ',num2str(NMSEdB_genie),'dB']);
                else
                    subplot(413)
                    c0 = NaN*ones(N,1); c0(comp_support) = c;
                    c1 = NaN*ones(N,1); c1(support) = c;
                    stem(10*log10(c0),'.'); hold on; stem(10*log10(c1),'r.'); hold off;
                    axis([1,N,10*log10(sig2e),0])
                    ylabel('c [dB]')
                    title(['NMSE-genie: ',num2str(NMSEdB_genie),'dB']);
                end
                subplot(414)
                p0 = NaN*ones(N,1); 
                p0(comp_support) = OutMessages.pi(comp_support);
                p1 = NaN*ones(N,1); 
                p1(support) = OutMessages.pi(support);
                stem(p0,'.'); 
                hold on; 
                stem(p1,'r.'); 
                hold off;
                ylabel('Outgoing \pi')
                axis([1,N,0,1])
                grid on;
                title(['BER: ',num2str(BER_(k))]);
                %if (k~=iter), pause; end;
            end;% if plot_bp
        end
    end;%if nargin==0
end

% Save final states of AMP variables
StateOut.c = mean(c);
if alg == 1 && ~genie_version    % Std. BP
    StateOut.z = Z;
    StateOut.mu = NaN;  % Mu not req'd for initialization
else            % AMP
    try
        StateOut.z = z;
        StateOut.mu = Mu;
    catch
        StateOut.z = Z;
        StateOut.mu = NaN;  % Mu not req'd for initialization
    end
end

end     % End of main AMP function


%% Sub-function used to compute the outgoing messages from AMP

% INPUTS
%  pi       Incoming activity probabilities, (see SP_FRAME_FXN)
%  phi      AMP variable (see SP_FRAME_FXN)
%  c        AMP variable (see SP_FRAME_FXN)
%  eps      Small positive scalar for approximation purposes
%  version  Use complex-valued update (1) or real-valued update (2)
%  tau      (See Params.tau, SP_FRAME_FXN)
%  alg      Using ABP (1) or AMP (1) mode of message passing
%
% OUTPUTS
%  xi_out   Outbound message of mean of theta
%  psi_out  Outbound message of variance of theta

function [xi_out, psi_out] = OutgoingMessages(pi, phi, c, eps, ...
    version, tau, alg)

if tau > 0  % Use the threshold method of producing messages
    
    Omega = (pi >= tau);
    xi_out = (Omega <= 1/2).*((1/eps)*phi) + (Omega > 1/2).*phi;
    psi_out = (Omega <= 1/2).*((1/eps^2)*c) + ...
        (Omega > 1/2).*c;
        
elseif tau == -1    % Use the Taylor series approx. method
    
    % Taylor series approximation about theta_0 is used to derive means
    % and variances
    theta_0 = phi;      % Taylor series expansion point
    
    if version == 1,    % Complex-valued version
        
        % Compute the various factors
        Omega = (eps^2*pi)./((1 - pi) + eps^2*pi);
        Omega(Omega < sqrt(realmin)) = sqrt(realmin);
        alpha = eps^2 * (1 - Omega);
        alpha_bar = Omega;
        beta = (eps^2./c) .* abs(theta_0 - phi/eps).^2;
        maxbeta = max(beta);
        beta_bar = (1./c) .* abs(theta_0 - phi).^2;
        delta = -(eps^2./c) .* (2*real(theta_0) - 2*real(phi)/eps);
        delta_bar = -(1./c) .* (2*real(theta_0) - 2*real(phi));
        gamma = -eps^2./c .* (2*imag(theta_0) - 2*imag(phi)/eps);
        gamma_bar = -1./c .* (2*imag(theta_0) - 2*imag(phi));

        % Use factors to compute outgoing variance.  Perform computation
        % slightly differently for coefficients with large abs(beta)'s for
        % numerical precision reasons
        LB = find(abs(beta) > 10);      % Indices with large betas
        SB = find(abs(beta) <= 10);     % Indices with small betas
        numer = NaN*ones(size(phi));
        denom = NaN*ones(size(phi));
        numer(SB) = alpha(SB).^2 .* exp(-(beta(SB))) + ...
            alpha(SB) .* alpha_bar(SB) + alpha_bar(SB).^2 .* exp(beta(SB));
        denom(SB) = (eps^2./c(SB)).*alpha(SB).^2 .* exp(-(beta(SB))) + ...
            ((eps^2 + 1)./c(SB) - (1/2)*delta(SB).^2).*alpha(SB) .* ...
            alpha_bar(SB) + (1./c(SB)).*alpha_bar(SB).^2 .* exp(beta(SB));
        numer(LB) = alpha(LB).^2 .* exp(-2*beta(LB)) + ...
            alpha(LB) .* alpha_bar(LB) .* ...
            exp(-beta(LB)) + alpha_bar(LB).^2;
        denom(LB) = (eps^2./c(LB)).*alpha(LB).^2 .* exp(-2*beta(LB)) + ...
            ((eps^2 + 1)./c(LB) - (1/2)*delta(LB).^2) .* alpha(LB) .* ...
            alpha_bar(LB) .* exp(-beta(LB)) + (1./c(LB)).*alpha_bar(LB).^2;
        psi_out = numer ./ denom;
        % Use factors to compute compute outgoing mean
        xi_real = real(phi) - (1/2)*psi_out.*(-alpha.*delta.*...
            exp(-beta))./(alpha.*exp(-beta) + alpha_bar);
        xi_imag = imag(phi) - (1/2)*psi_out.*(-alpha.*gamma.*...
            exp(-beta))./(alpha.*exp(-beta) + alpha_bar);
        xi_out = xi_real + 1j*xi_imag;
        
    elseif version == 2,    % Real-valued version

        Omega = (eps*pi)./((1 - pi) + eps*pi);
        alpha = 1 - Omega;
        alpha_bar = Omega;
        beta = eps^2./c;
        beta_bar = 1./c;
        delta = theta_0 - phi/eps;
        delta_bar = theta_0 - phi;
        
        G = alpha .* beta.^(1/2) .* exp(-(1/2)*beta.*delta.^2) + ...
            alpha_bar .* beta_bar.^(1/2) .* ...
            exp(-(1/2)*beta_bar.*delta_bar.^2);
        
        dG = -alpha .* beta.^(3/2) .* exp(-(1/2)*beta.*delta.^2) .* ...
            delta - alpha_bar .* beta_bar.^(3/2) .* ...
            exp(-(1/2)*beta_bar.*delta_bar.^2).*delta_bar;
        
        d2G = alpha .* beta.^(3/2) .* exp(-(1/2)*beta.*delta.^2) .* ...
            (beta .* delta.^2 - 1) + alpha_bar .* beta_bar.^(3/2) .* ...
            exp(-(1/2)*beta_bar.*delta_bar.^2) .* ...
            (beta_bar .* delta_bar.^2 - 1);
        
        if any(G == Inf) || any(G == 0)
            fprintf('Taylor approx numerical precision probs: G\n');
            G(G == 0) = realmin;
            dG(dG == 0) = realmin;
            d2G(d2G == 0) = realmin;
        end
                
        % Use G and its derivatives to compute the derivatives of
        % F = -log(G) 
        dF = -dG ./ G;

%         d2F = (dG.^2 - d2G.*G) ./ G.^2;
        % Improved numerical stability
        d2F = (dG.^2./G - d2G) ./ G;

        if any(isnan(dF)) || any(isnan(d2F))
            fprintf('Taylor approx numerical precision probs: F\n');
        end

        % Compute outgoing (from GAMP) message variables...
        psi_out = d2F.^(-1);
        xi_out = theta_0 - psi_out.*dF;
    end
    
    % Numerical precision adjustments
    psi_out(psi_out == Inf) = realmax;
end

end     % End of subfunction