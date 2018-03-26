% PARAMETER_UPDATE_FXN      This function will compute updates to the
% hyperparameters of the time-varying signal model described in
% SP_MULTI_FRAME_FXN using information obtained from the most recent
% forward/backward pass of SP_MULTI_FRAME_FXN.  In other words, while our
% belief propagation-based algorithm is recovering the signal, it will also
% refine its estimates of the signal model's hyperparameters according to
% the data and the current recovery.  The updates will be based on EM
% estimators of the individual parameters.  Only subsets of all the
% parameters will be updated on each iteration, to improve stability in the
% EM procedure.
%
% SYNTAX:
% Updates = parameter_update_fxn(y, A, State, Params, Upd_Groups)
%
% INPUTS:
% y                 A 1-by-T+1 cell array of observation vectors.  Each
%                   complex- or real-valued observation vector is defined 
%                   to be of length M(t), t = 0, 1, ..., T
% A                 A 1-by-T+1 cell array of complex- or real-valued 
%                   measurement matrices, with each matrix of dimension 
%                   M(t)-by-N, or a 1-by-T+1 cell array of function handles
%                   which implement matrix-vector multiplications.  See 
%                   below for the form of these function handles
% State             A structure containing needed state variables from the
%                   calling function, e.g., sp_multi_frame_fxn
%   .Mu_x           The N-by-T+1 matrix MU_STATE from calling function
%   .V_x            The N-by-T+1 matrix V_STATE from calling function
%   .Eta_fwd        The N-by-T+1 matrix ETA_FWD from calling function
%   .Eta_bwd        The N-by-T+1 matrix ETA_BWD from calling function
%   .Xi_out         The N-by-T+1 matrix XI_OUT from calling function
%   .Kap_fwd        The N-by-T+1 matrix KAPPA_FWD from calling function
%   .Kap_bwd        The N-by-T+1 matrix KAPPA_BWD from calling function
%   .Psi_out        The N-by-T+1 matrix PSI_OUT from calling function
%   .Lam_fwd        The N-by-T+1 matrix LAMBDA_FWD from calling function
%   .Lam_bwd        The N-by-T+1 matrix LAMBDA_BWD from calling function
%   .Pi_out         The N-by-T+1 matrix PI_OUT from calling function
%   .iter           Current smoothing iteration of the calling function,
%                   needed for the update procedure to identify what subset
%                   of parameters to update on this smoothing iteration
% Params            A structure containing current best parameter estimates
%   .lambda         N-by-1 vector of activity probabilities
%   .p01            N-by-1 vector of active-to-inactive transition
%                   probabilities
%   .p10            N-by-1 vector of inactive-to-active transition
%                   probabilities
%   .eta        	N-by-1 complex vector of the means of the amplitude
%                   evolution process, i.e., E[theta(n,t)] = eta(n)
%   .kappa           N-by-1 vector of theta variances, i.e.
%                   var{theta_n(t)} = Params.kappa(n)
%   .alpha          Scalar Gauss-Markov "innovation rate" parameter, btwn.
%                   0 and 1, (see above). Can also be N-by-1 if coefficents
%                   are being segregated into groups that each have
%                   different alpha values (see Options.upd_groups)
%   .rho            Variance of the Gauss-Markov process driving noise,
%                   (see above). Can be N-by-1 if coefficents
%                   are being segregated into groups that each have
%                   different alpha values (see Options.upd_groups)
%   .sig2e          Scalar variance of circular, complex, additive Gaussian
%                   measurement noise
% Upd_Groups        An optional cell array that will allow 
%                   parameter_update_fxn to perform separate
%                   hyperparameter updates for different subsets of
%                   coefficients.  Each cell in the array should contain
%                   the indices of the coefficients that belong exclusively
%                   to that group, and the union of all cells should
%                   contain the indices 1:N.  The number of cells dictates
%                   the number of unique groups, N_groups.  If this field
%                   is provided, all output parameters will become of
%                   dimension 1-by-N_groups, with the exception of sig2e.
%                   The ordering of values in the outputs will be the same
%                   as the ordering of the groups in Upd_Groups.
%
% OUTPUTS:
% Updates           A structure containing the updated parameter values
%   .lambda
%   .p01
%   .p10
%   .eta
%   .kappa
%   .alpha
%   .rho
%   .sig2e
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
% Last change: 12/04/12
% Change summary: 
%		- Created (12/09/10; JAZ)
%       - Modified computation of rho by dividing everything by the newly
%           updated value of alpha (03/02/11; JAZ)
%       - Changed all of the estimators to be EM estimators (05/14/11; JAZ)
%       - Changed procedure to only update subsets of all the parameters on
%           a given smoothing iteration, to improve stability in the
%           estimates by reducing coupling btwn. estimates (06/16/11; JAZ)
%       - Modified some EM computations and naming conventions to reflect
%         changes in signal model to a steady-state dynamic system
%         (12/04/12; JAZ; v1.1)
% Version 1.2
%

function Updates = parameter_update_fxn(y, A, State, Params, Upd_Groups)

%% Check for errors and unpack inputs

for l = 1       % Enable code folding for the error checking section
    if nargin < 4
        error('parameter_update_fxn: Insufficient input arguments')
    end

    if ~isfield(State, 'Mu_x') || ~isfield(State, 'V_x') ...
            || ~isfield(State, 'Eta_fwd') || ~isfield(State, 'Eta_bwd') ...
            || ~isfield(State, 'Xi_out') || ~isfield(State, 'Kap_fwd') ...
            || ~isfield(State, 'Kap_bwd') || ~isfield(State, 'Psi_out') ...
            || ~isfield(State, 'Lam_fwd') || ~isfield(State, 'Lam_bwd') ...
            || ~isfield(State, 'Pi_out') || ~isfield(State, 'iter')
        error(['parameter_update_fxn: Input ''State'' is missing ' ...
            'required fields'])
    end
    
    if ~isfield(Params, 'eta') || ~isfield(Params, 'kappa') ...
            || ~isfield(Params, 'alpha') || ~isfield(Params, 'rho') ...
            || ~isfield(Params, 'sig2e') || ~isfield(Params, 'p01') ...
            || ~isfield(Params, 'p10')
        error(['parameter_update_fxn: Input ''Params'' is missing ' ...
            'required fields'])
    end

    T = length(y) - 1;
    N = size(State.Mu_x,1);
    if isa(A{1}, 'function_handle')
        explicitA = 0;  % Implicit operators
    else
        explicitA = 1;
    end
    
    % Unpack State structure
    MU_STATE = State.Mu_x;
    V_STATE = State.V_x;
    ETA_FWD = State.Eta_fwd;
    ETA_BWD = State.Eta_bwd;
    XI_OUT = State.Xi_out;
    KAPPA_FWD = State.Kap_fwd;
    KAPPA_BWD = State.Kap_bwd;
    PSI_OUT = State.Psi_out;
    LAMBDA_FWD = State.Lam_fwd;
    LAMBDA_BWD = State.Lam_bwd;
    PI_OUT = State.Pi_out;
    
    % Unpack Params structure
    eta = Params.eta;
    kappa = Params.kappa;
    if numel(Params.alpha) == 1, alpha = Params.alpha*ones(N,1);
    elseif numel(Params.alpha) == N, alpha = Params.alpha;
    else error('parameter_update_fxn: Incorrect argument in Params.alpha')
    end
    if numel(Params.rho) == 1, rho = Params.rho*ones(N,1);
    elseif numel(Params.rho) == N, rho = Params.rho;
    else error('parameter_update_fxn: Incorrect argument in Params.rho')
    end
    if numel(Params.p01) == 1, p01 = Params.p01*ones(N,1);
    elseif numel(Params.p01) == N, p01 = Params.p01;
    else error('parameter_update_fxn: Incorrect argument in Params.p01')
    end
    if numel(Params.p10) == 1, p10 = Params.p10*ones(N,1);
    elseif numel(Params.p10) == N, p10 = Params.p10;
    else error('parameter_update_fxn: Incorrect argument in Params.p10')
    end
    sig2e = Params.sig2e;

    if nargin == 5
        N_groups = length(Upd_Groups);
        if numel(setdiff(1:N, cat(1, Upd_Groups{:}))) ~= 0
            error(['parameter_update_fxn: ''Upd_Groups'' ' ...
                'does not partition every coefficient into a ' ...
                'unique group'])
        end
    else
        N_groups = 1;
        Upd_Groups = {1:N};
    end

    % Determine whether to use real- or complex-updates
    version = (norm(imag(y{1})) == 0) + 1;
end         % End code folding


%% Start estimating the updated parameters

% Begin by computing the means and variances of p(theta(n,t)|{y}) and the
% means of p(s(n,t)|{y})
V_THETA = (1./KAPPA_FWD + 1./KAPPA_BWD + 1./PSI_OUT).^-1;
MU_THETA = V_THETA.*(ETA_FWD./KAPPA_FWD + ETA_BWD./KAPPA_BWD + ...
    XI_OUT./PSI_OUT);
MU_S = LAMBDA_FWD.*PI_OUT.*LAMBDA_BWD ./ ...
    (LAMBDA_FWD.*PI_OUT.*LAMBDA_BWD + ...
    (1 - LAMBDA_FWD).*(1 - PI_OUT).*(1 - LAMBDA_BWD));

% Initial values for updated variables
for g = 1:N_groups
    Updates.lambda(g) = Params.lambda(Upd_Groups{g}(1));
    Updates.p01(g) = Params.p01(Upd_Groups{g}(1));
    Updates.p10(g) = Params.p10(Upd_Groups{g}(1));
    Updates.eta(g) = Params.eta(Upd_Groups{g}(1));
    Updates.kappa(g) = Params.kappa(Upd_Groups{g}(1));
    Updates.alpha(g) = Params.alpha(Upd_Groups{g}(1));
    Updates.rho(g) = Params.rho(Upd_Groups{g}(1));
end
Updates.sig2e = Params.sig2e;

% -----------------------
% Updated noise variance
% -----------------------
sig2e_sum = 0; M = NaN*ones(1,length(y));
for t = 1:T+1
    if explicitA
        sig2e_sum = sig2e_sum + norm(y{t} - A{t}*MU_STATE(:,t))^2 + ...
            sum(V_STATE(:,t));
    else    % Implicit A function handle
        sig2e_sum = sig2e_sum + norm(y{t} - A{t}(MU_STATE(:,t),1))^2 + ...
            sum(V_STATE(:,t));
    end
    M(t) = length(y{t});
end
Updates.sig2e = (sig2e_sum)/sum(M);


% Carry out the remaining estimates by group
for g = 1:N_groups
    N_g = numel(Upd_Groups{g});     % # of members of this group
    % Grab the indices of the coefficients included in this group
    g_ind = reshape(Upd_Groups{g}, 1, numel(Upd_Groups{g}));
    
    if mod(State.iter, 3) == 0
        % Update the first subset of parameters
        
        % ------------------------------
        % Updated activity prob., lambda
        % ------------------------------
        Updates.lambda(g) = sum(sum(MU_S(g_ind,:))) / N_g / (T + 1);


        % ------------------------------------------------
        % Updated active-to-inactive transition prob., p01
        % ------------------------------------------------
        % First compute E[s(n,t)*s(n,t-1)|{y}]
        PS0S0 = (1 - p10(g_ind))*ones(1,T) .* ((1 - LAMBDA_FWD(g_ind,1:T)).*...
            (1 - PI_OUT(g_ind,1:T))) .* ((1 - LAMBDA_BWD(g_ind,2:T+1)).*...
            (1 - PI_OUT(g_ind,2:T+1)));
        PS1S0 = (p10(g_ind)*ones(1,T)) .* ((1 - LAMBDA_FWD(g_ind,1:T)).*...
            (1 - PI_OUT(g_ind,1:T))) .* ((LAMBDA_BWD(g_ind,2:T+1)).*...
            (PI_OUT(g_ind,2:T+1)));
        PS0S1 = (p01(g_ind)*ones(1,T)) .* ((LAMBDA_FWD(g_ind,1:T)).*...
            (PI_OUT(g_ind,1:T))).* ((1 - LAMBDA_BWD(g_ind,2:T+1)).*...
            (1 - PI_OUT(g_ind,2:T+1)));
        PS1S1 = (1 - p01(g_ind))*ones(1,T) .* ((LAMBDA_FWD(g_ind,1:T)).*...
            (PI_OUT(g_ind,1:T))).* ((LAMBDA_BWD(g_ind,2:T+1)).*...
            (PI_OUT(g_ind,2:T+1)));
        S_CORR = PS1S1 ./ (PS0S0 + PS0S1 + PS1S0 + PS1S1);

        pz1_top_sum = sum(sum(MU_S(g_ind,1:T) - S_CORR));
        pz1_bottom_sum = sum(sum(MU_S(g_ind,1:T)));

        if pz1_bottom_sum ~= 0
            Updates.p01(g) = max(pz1_top_sum, 0) / pz1_bottom_sum;
        end

        % Now, update "current" value of p01 for subsequent EM estimators
        p01(g_ind) = Updates.p01(g);


        % ------------------------------------------------
        % Updated inactive-to-active transition prob., p10
        % ------------------------------------------------
        % Now, update "current" value of p10 for subsequent EM estimators
        Updates.p10(g) = Updates.p01(g) * Updates.lambda(g) / ...
            (1 - Updates.lambda(g));    % Maintain steady-state sparsity
        p10(g_ind) = Updates.p10(g);
    
        
    elseif mod(State.iter, 3) == 1
        % Update the second subset of parameters
        
        % -----------------------------------------------------
        % Updated amplitude variance, kappa (sigma^2)
        % -----------------------------------------------------
        eta_rep = repmat(eta(g_ind), 1, T+1);
        QTY = V_THETA(g_ind,:) + abs(MU_THETA(g_ind,:)).^2 - ...
            2*real(conj(eta_rep).*MU_THETA(g_ind,:)) + abs(eta_rep).^2;
        Updates.kappa(g) = (1/N_g/(T+1))*sum(QTY(:));

        % Now, update "current" value of kappa for subsequent EM estimators
        kappa(g_ind) = Updates.kappa(g);


        % --------------------------
        % Updated correlation, alpha
        % --------------------------
        % Start by computing E[theta(n,t)'*theta(n,t-1)|{y}]
        Q = (1./PSI_OUT(g_ind,2:T+1) + 1./KAPPA_BWD(g_ind,2:T+1)).^(-1);
        R = (XI_OUT(g_ind,2:T+1)./PSI_OUT(g_ind,2:T+1) + ...
            ETA_BWD(g_ind,2:T+1)./KAPPA_BWD(g_ind,2:T+1));
        Q_BAR = (1./PSI_OUT(g_ind,1:T) + 1./KAPPA_FWD(g_ind,1:T)).^(-1);     
        R_BAR = (XI_OUT(g_ind,1:T)./PSI_OUT(g_ind,1:T) + ...
            ETA_FWD(g_ind,1:T)./KAPPA_FWD(g_ind,1:T));        
        Q_TIL = (1./Q_BAR + ((1-alpha(g_ind)).^2*ones(1,T))./(Q + ...
            (alpha(g_ind).^2.*rho(g_ind))*ones(1,T))).^(-1);        
        M_BAR = ((1 - alpha(g_ind))*ones(1,T)).*(Q.*R - ...
            (alpha(g_ind).*eta(g_ind))*ones(1,T)) ./ (Q + ...
            (alpha(g_ind).^2.*rho(g_ind))*ones(1,T)) + R_BAR;        
        THETA_CORR = (Q./(Q + (alpha(g_ind).^2.*rho(g_ind))*ones(1,T))) .* ...
            (((1-alpha(g_ind))*ones(1,T)).*(Q_TIL + abs(Q_TIL.*M_BAR).^2) + ...
            ((alpha(g_ind).*eta(g_ind))*ones(1,T)).*Q_TIL.*conj(M_BAR) + ...
            ((alpha(g_ind).^2.*rho(g_ind))*ones(1,T)).*Q_TIL.*conj(M_BAR).*R);

        % Compute the values of the coefficients of the quadratic polynomial
        % whose solution gives the appropriate value of alpha
        mult_b = (rho(g_ind)*ones(1,T)).^(-1) .* (2*real(THETA_CORR) - ...
            2*real(conj(eta(g_ind)*ones(1,T)).*MU_THETA(g_ind,2:T+1)) - ...
            2*(V_THETA(g_ind,1:T) + abs(MU_THETA(g_ind,1:T)).^2) + ...
            2*real(conj(eta(g_ind)*ones(1,T)).*MU_THETA(g_ind,1:T)));
        mult_b = sum(sum(mult_b));
        mult_c = (rho(g_ind)*ones(1,T)).^(-1) .* ((V_THETA(g_ind,2:T+1) + ...
            abs(MU_THETA(g_ind,2:T+1)).^2) - 2*real(THETA_CORR) + ...
            (V_THETA(g_ind,1:T) + abs(MU_THETA(g_ind,1:T)).^2));
        mult_c = sum(sum(mult_c));

        % Now find the roots of the quadratic polynomial of alpha
        try
            if version == 1,        % Complex-valued update
                alpha_roots = roots([-2*N_g*T, mult_b, 2*mult_c]);
            elseif version == 2,    % Real-valued update
                alpha_roots = roots([-N_g*T, mult_b/2, mult_c]);
            end
            if norm(imag(alpha_roots)) > 0
                warning(['EM update procedure found non-zero imaginary '...
                    'component of ''alpha''. Returning real part only.'])
                alpha_roots = real(alpha_roots);
            end
            if isempty(alpha_roots(alpha_roots > 0))
                warning(['Unable to find positive root in computing alpha for ' ...
                    'group ' num2str(g) ', thus returning previous estimate'])
                Updates.alpha(g) = alpha(g_ind(1));    % Default when estimation failed
            else
                % Clip allowable range for alpha
                Updates.alpha(g) = max(min(max(alpha_roots), 0.99), 0.001);
            end
        catch
            % Either NaN or inf arguments were passed to roots fxn,
            % suggesting that the EM procedure is diverging.  We can try to
            % salvage it by just returning the previous estimate of alpha,
            % but no guarantees here...
            warning(['NaN or inf arguments encountered during alpha update' ...
                ' for group ' num2str(g) ', thus returning previous estimate'])
        end

        % Now, update our "current" value of alpha for subsequent EM estimators
        alpha(g_ind) = Updates.alpha(g);
        
        
        % ---------------------------------------------
        % Updated amplitude driving noise variance, rho
        % ---------------------------------------------
        Updates.rho(g) = (2 - Updates.alpha(g)) * Updates.kappa(g) / ...
            Updates.alpha(g);
    
        
    else
        % Update the final subset of parameters
        
        % ------------------------------------
        % Updated amplitude mean, eta (zeta)
        % ------------------------------------
        eta_top_sum = sum(sum((1./((alpha(g_ind).*rho(g_ind))*ones(1,T))).*...
            (MU_THETA(g_ind,2:T+1) - ((1 - alpha(g_ind))*ones(1,T)) .* ...
            MU_THETA(g_ind,1:T)))) + sum(MU_THETA(g_ind,1)./kappa(g_ind));
        eta_bottom_sum = T*sum(1./rho(g_ind)) + sum(1./kappa(g_ind));
        if version == 1     % Complex-valued case
            Updates.eta(g) = eta_top_sum / eta_bottom_sum;
        elseif version == 2 % Real-valued case
            Updates.eta(g) = eta_top_sum / eta_bottom_sum;
        end

        % Now, update "current" value of eta for subsequent EM estimators
        eta(g_ind) = Updates.eta(g);
    
    end     % End of subset parameter update
    
    % --------------------------------------------------
    % If we had trouble estimating alpha, we can try for
    % joint estimation of both alpha and rho
    % --------------------------------------------------
%     if isempty(alpha_roots(alpha_roots > 0))     % alpha estimation failed
%         % Alternative joint estimator for alpha and rho
%         v_bar = (1./KAPPA_FWD + 1./KAPPA_BWD + 1./PSI_OUT).^-1;
%         mu_bar = v_bar.*(ETA_FWD./KAPPA_FWD + ETA_BWD./KAPPA_BWD + XI_OUT./PSI_OUT);
% 
%         mult_0 = 0; mult_1 = 0; mult_2 = 0; mult_3 = 0; mult_4 = 0; total = 0;
%         for t = 2:T+1
%             for n = g_ind
%                 if t < T+1
%                     q = (PSI_OUT(n,t).*KAPPA_BWD(n,t)./(PSI_OUT(n,t) + KAPPA_BWD(n,t)));
%                 else
%                     q = PSI_OUT(n,t);
%                 end
% 
%                 r = (XI_OUT(n,t)./PSI_OUT(n,t) + ETA_BWD(n,t)./KAPPA_BWD(n,t));
%                 q_bar = (PSI_OUT(n,t-1).*KAPPA_FWD(n,t-1)./(PSI_OUT(n,t-1) + KAPPA_FWD(n,t-1)));
%                 r_bar = (XI_OUT(n,t-1)./PSI_OUT(n,t-1) + ETA_FWD(n,t-1)./KAPPA_FWD(n,t-1));
%                 q_til = (1./q_bar + (1-alpha(n))^2./(q + alpha(n)^2*rho(n))).^-1;
%                 m_bar = (1-alpha(n))*(q.*r - alpha(n)*eta(n))./(q + alpha(n)^2*rho(n)) + r_bar;
% 
%                 theta_corr(total+1) = (q./(q + alpha(n)^2*rho(n)))...
%                     .*((1-alpha(n))*(q_til + abs(q_til.*m_bar).^2) + ...
%                     alpha(n)*eta(n)*q_til.*conj(m_bar) + ...
%                     alpha(n)^2*rho(n)*q_til.*conj(m_bar).*r);
% 
%                 % Add to quantities needed for estimating alpha
%                 mult_0 = mult_0 + (v_bar(n,t) + abs(mu_bar(n,t))^2) - ...
%                     2*real(theta_corr(total+1)) + (v_bar(n,t-1) + abs(mu_bar(n,t-1))^2);
%                 mult_1 = mult_1 + 2*real(theta_corr(total+1)) - 2*real(conj(eta(n))*mu_bar(n,t)) - ...
%                     2*(v_bar(n,t-1) + abs(mu_bar(n,t-1))^2) + 2*real(conj(eta(n))*mu_bar(n,t-1));
%                 mult_2 = mult_2 + v_bar(n,t) + abs(mu_bar(n,t))^2 - 2*real(theta_corr(total+1)) + ...
%                     v_bar(n,t-1) + abs(mu_bar(n,t-1))^2;
%                 mult_3 = mult_3 + 2*real(theta_corr(total+1)) - 2*real(eta(n)*mu_bar(n,t)) - ...
%                     2*(v_bar(n,t-1) + abs(mu_bar(n,t-1))^2) + 2*real(mu_bar(n,t-1)*eta(n));
%                 mult_4 = mult_4 + v_bar(n,t-1) + abs(mu_bar(n,t-1))^2 - 2*real(mu_bar(n,t-1)'*eta(n)) + ...
%                     abs(eta(n))^2;
% 
%                 % Update the count of sample points we are including
%                 total = total + 1;
%             end
%         end
% 
%         alpha_roots = roots([-2*mult_4, -2*mult_3 + mult_1, -2*mult_2 + 2*mult_0]);
%         Updates.alpha(g) = max(alpha_roots);
%         Updates.rho(g) = (Updates.alpha(g)*mult_1 + 2*mult_0)/(2*total*Updates.alpha(g)^2);
%         if Updates.rho(g) < 0   % Something must've gone wrong, probably with alpha
%             % Resort to a default setting of rho, based on alpha, that will
%             % maintain a uniform amplitude variance across time
%             Updates.rho(g) = (2 - Updates.alpha(g))*Updates.kappa(g)/...
%                 Updates.alpha(g);
%         end
%     end
    if Updates.rho(g) < 0
        pause(1);   % This indicates major problems in EM estimator
    end
end
