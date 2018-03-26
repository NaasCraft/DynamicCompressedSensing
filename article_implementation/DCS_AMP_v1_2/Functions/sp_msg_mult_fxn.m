% SP_MSG_MULT_FXN 	Function for taking the two incoming messages to s(t)
% and theta(t), t = 0, 1, ..., T-1, and producing the messages passing to
% f(t), (i.e. the messages from s_n(t) to f_n(t), and those from theta_n(t)
% to f_n(t)).  It is unnecessary to call this function to produce messages
% at timestep t = T.
%
% SYNTAX:
% [pi_in, xi_in, psi_in] = sp_msg_mult_fxn(lambda_fwd, lambda_bwd, ...
%       eta_fwd, kappa_fwd, eta_bwd, kappa_bwd)
%
% INPUTS:
% lambda_fwd        The N-by-1 "forward-propagating" prior activity 
%                   probabilities, i.e. the messages from h_n(t) to s_n(t)
% lambda_bwd        The N-by-1 "backwards-propagating" prior activity 
%                   probabilities, i.e. the messages from h_n(t+1) to s_n(t)
% eta_fwd           The N-by-1 "forward-propagating" amplitude means, i.e.
%                   the means of the Gaussian msgs from d_n(t) to theta_n(t)
% kappa_fwd         The N-by-1 "forward-propagating" amplitude variances, 
%                   i.e. the variances of the Gaussian msgs from d_n(t) to
%                   theta_n(t)
% eta_bwd           The N-by-1 "backwards-propagating" amplitude means, i.e.
%                   the means of the Gaussian msgs from d_n(t+1) to theta_n(t)
% kappa_bwd         The N-by-1 "backwards-propagating" amplitude variances, 
%                   i.e. the variances of the Gaussian msgs from d_n(t+1) to
%                   theta_n(t)
%
% OUTPUTS:
% pi_in             N-by-1 vector of activity probabilities for
%                   messages from s_n(t) to f_n(t)
% xi_in             N-by-1 vector of amplitude means for messages from 
%                   theta_n(t) to f_n(t)
% psi_in            N-by-1 vector of amplitude variances for messages from
%                   theta_n(t) to f_n(t)
%

%
% Coded by: Justin Ziniel, The Ohio State Univ.
% E-mail: zinielj@ece.osu.edu
% Last change: 12/29/11
% Change summary: 
%		- Created from sp_msg_mult_fxn v1.0 (12/29/11; JAZ)
% Version 1.2
%

function [pi_in, xi_in, psi_in] = sp_msg_mult_fxn(lambda_fwd, ...
    lambda_bwd, eta_fwd, kappa_fwd, eta_bwd, kappa_bwd)

%% Compute the normalized messages

% First compute pi_in
pi_in = min(max((lambda_fwd.*lambda_bwd)./(1 - lambda_fwd - lambda_bwd + ...
    2*lambda_fwd.*lambda_bwd), 0), 1);      % Eqn. xx

% Next compute psi_in
psi_in = (kappa_fwd.^(-1) + kappa_bwd.^(-1)).^(-1);     % Eqn. (A2)

% Finally, compute xi_in
xi_in = (eta_fwd./kappa_fwd + eta_bwd./kappa_bwd);
xi_in = psi_in.*xi_in;              % Eqn. (A3)