% NSER 	Compute the normalized support error rate (NSER) of a recovered
% support
%
%
% SYNTAX:
% rate = nser(TrueSupp, EstSupp)
%
% INPUTS:
%   TrueSupp    Indices of the true support
%   EstSupp     Indices of the estimated support
%   
%
% OUTPUTS:
%   rate        The NSER of the estimated support
%

%
% Coded by: Justin Ziniel, The Ohio State Univ.
% E-mail: ziniel.1@osu.edu
% Last change: 12/29/11
% Change summary: 
%		- Created from nser v1.0 (12/29/11; JAZ)
% Version 1.2


function rate = nser(TrueSupp, EstSupp)

%% Compute the NSER

rate = ( numel(setdiff(TrueSupp, EstSupp)) + ...
    numel(setdiff(EstSupp, TrueSupp)) ) / numel(TrueSupp);