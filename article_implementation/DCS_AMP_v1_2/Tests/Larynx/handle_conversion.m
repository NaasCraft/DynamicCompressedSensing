% HANDLE_CONVERSION         This script is used to convert the form of the
% larynx measurement matrix function handles into a form that is matched to
% the one expected by SP_MULTI_FRAME_FXN
%
% SYNTAX:
% y = handle_conversion(x, A, AT, mode)
%
% INPUTS:
% x                 Either an N-by-1 vector, or an M(t)-by-1 vector which
%                   is to be multiplied by the measurement matrix A or A',
%                   respectively.
% A                 A function handle that takes a single argument, x, and
%                   returns A*x
% AT                A function handle that takes a single argument, x, and
%                   returns A'*x
% mode              Indicator as to which matrix-vector multiplication to
%                   perform, (1) for A*x, (2) for A'*x
%
% OUTPUTS:
% y                 Either an M(t)-by-1 vector, or an N-by-1 vector which
%                   is the result of multiplication by the measurement 
%                   matrix A or A', respectively.
%

% 
% Coded by: Justin Ziniel, The Ohio State Univ.
% E-mail: ziniel.1@osu.edu
% Last change: 12/11/10
% Change summary: 
%		- Created (12/11/10; JAZ)
% Version 1.2
%

function y = handle_conversion(x, A, AT, mode)

%% Perform the matrix-vector multiplication

if mode == 1
    y = A(x);
elseif mode == 2
    y = AT(x);
else
    error('handle_conversion: Invalid argument for ''mode''')
end