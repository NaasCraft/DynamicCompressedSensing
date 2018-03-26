% CLASS: DCSModelParams
% 
% HIERARCHY (Enumeration of the various super- and subclasses)
%   Superclasses: hgsetget (MATLAB handle class)
%   Subclasses: N/A
% 
% TYPE (Abstract or Concrete)
%   Concrete
%
% DESCRIPTION (High-level overview of the class)
%   This class is used by the main DCS-AMP inference functions, (see
%   SP_MULTI_FRAME_FXN or SP_PARALLEL_FRAME_FXN), in order to specify the 
%   parameters (or initialization of the parameters if performing EM 
%   parameter learning) that make up the signal/measurement model.  The 
%   parameters that are needed to define the model are described briefly 
%   below (see PROPERTIES), and a full description of the
%   signal/measurement model can be found in either SP_MULTI_FRAME_FXN or 
%   SP_PARALLEL_FRAME_FXN.
%
%   To create a DCSModelParams object, there are two constructors to choose
%   from (see METHODS section below).  The first constructor simply accepts
%   an input for each of the required model parameters, i.e.,
%   ModelParams(lambda, p01, eta, kappa, alpha, sig2e).  The second
%   constructor accepts a SigGenParams object, along with sig2e, and copies
%   over the required parameter values, i.e., ModelParams(SigGenParams,
%   sig2e).
%
% PROPERTIES (Model Parameters)
%   lambda          Scalar activity probability for s_n(t), i.e., 
%                   Pr{s_n(t) = 1} = lambda
%   p01             Scalar active-to-inactive Markov transition
%                   probability, i.e., p01 = Pr{s_n(t) = 0 | s_n(t-1) = 1}
%   eta             (Complex) scalar of theta means, i.e.,
%                   E[theta_n(t)] = eta
%   kappa           Scalar steady-state theta variances, i.e.
%                   var{theta_n(t)} = sigma2
%   alpha           Scalar Gauss-Markov "innovation rate" parameter, btwn.
%                   0 and 1 (Note: Correlation coeff. = 1 - alpha)
%   sig2e           Variance of the additive Gaussian measurement noise
%
%   ***Note: The parameters lambda, eta, kappa, and alpha can, in
%            addition to scalar inputs, accept N-by-1 vector inputs if you
%            wish to assign model parameters on a per-coefficient basis.
%
% METHODS (Subroutines/functions)
%   DCSModelParams(lambda, p01, eta, kappa, alpha, sig2e)
%       - Primary constructor for creating a DCSModelParams object
%   DCSModelParams(SigGenParams, sig2e)
%       - Secondary constructor that uses the SigGenParams object that
%         generated the data, in addition to sig2e, to fill in the values
%         of the model parameters
%   set(obj, 'ParameterName', Value)
%       - Method that can be used to set the value of a parameter once the
%         object of class ModelParams, obj, has been constructed
%   print()
%       - Print the current value of each property to the command window
%                   
%

%
% Coded by: Justin Ziniel, The Ohio State Univ.
% E-mail: zinielj@ece.osu.edu
% Last change: 12/07/15
% Change summary: 
%       - Created from ModelParams v1.1 (12/05/12; JAZ)
%       - Small bug fixes (12/07/15; JAZ)
% Version 1.2
%

classdef DCSModelParams < hgsetget

    properties
        lambda;  	% Pr{s_n = 1}
        p01;        % Pr{s_n(t) = 0 | s_n(t-1) = 1}
        eta;     	% E[theta_n(t)]
        kappa;   	% var{theta_n(t)}
        alpha;    	% Amplitude process innovation rate
        sig2e;      % AWGN variance
    end % properties
    
    properties (Dependent)
        rho;      	% The variance of the amplitude driving noise
        p10;        % Inactive-to-active transition probability
    end
    
    methods
        % Constructor
        function obj = DCSModelParams(varargin)
            if nargin == 6
                obj.lambda = varargin{1};
                obj.p01 = varargin{2};
                obj.eta = varargin{3};
                obj.kappa = varargin{4};
                obj.alpha = varargin{5};
                obj.sig2e = varargin{6};
            elseif nargin == 2
                if isa(varargin{1}, 'SigGenParams')
                    obj.lambda = varargin{1}.lambda;
                    obj.p01 = varargin{1}.p01;
                    obj.eta = varargin{1}.zeta;
                    obj.kappa = varargin{1}.sigma2;
                    obj.alpha = varargin{1}.alpha;
                    obj.sig2e = varargin{2};
                else
                    error(['First argument to DCSModelParams constructor is ' ...
                        'not an object of the SigGenParams class'])
                end
            elseif nargin == 0
                % User will populate values using set/get syntax
            else
                error('Incorrect number of arguments to DCSModelParams constructor')
            end
        end
        
        % Set method for lambda
        function obj = set.lambda(obj, lambda)
           if all(lambda >= 0 & lambda <= 1)
               obj.lambda = lambda;
           else
              error('Invalid assignment: lambda')
           end
        end
        
        % Set method for p01
        function obj = set.p01(obj, p01)
           if all(p01 >= 0) && all(p01 <= 1)
               obj.p01 = p01;
           else
              error('Invalid assignment: p01')
           end
        end
        
        % Set method for eta
        function obj = set.eta(obj, eta)
            obj.eta = eta;
        end
        
        % Set method for kappa
        function obj = set.kappa(obj, kappa)
           if all(kappa > 0)
               obj.kappa = kappa;
           else
              error('Invalid assignment: kappa')
           end
        end
        
        % Set method for alpha
        function obj = set.alpha(obj, alpha)
           if all(alpha >= 0 & alpha < 1)
               obj.alpha = alpha;
           else
              error('Invalid assignment: alpha')
           end
        end
        
        % Set method for sig2e
        function obj = set.sig2e(obj, sig2e)
           if isscalar(sig2e) && sig2e >= 0
               obj.sig2e = sig2e;
           else
              error('Invalid assignment: sig2e')
           end
        end
        
        % Get method for dependent property p10 (which is set based on
        % lambda and p01 in order to maintain a steady-state sparsity rate
        % of lambda across timesteps)
        function p10 = get.p10(obj)
            p10 = obj.p01 .* obj.lambda ./ (1 - obj.lambda);
        end
        
        % Get method for dependent property rho (which is set based on the
        % values of alpha and kappa to ensure a stationary amplitude
        % variance over time
        function rho = get.rho(obj)
            try
                rho = (2 - obj.alpha) .* obj.kappa ./ obj.alpha;
            catch ME
                error('Dimension mismatch between alpha and kappa (%s)', ...
                    ME.message)
            end
        end
        
        % Get method for grabbing all the properties at once
        function [lambda, p01, eta, kappa, alpha, sig2e, p10, rho] = getParams(obj)
          	lambda = obj.lambda;
            p01 = obj.p01;
            eta = obj.eta;
            kappa = obj.kappa;
            alpha = obj.alpha;
            sig2e = obj.sig2e;
            p10 = obj.p10;
            rho = obj.rho;
        end
        
        % Print the current configuration to the command window
        function print(obj)
            fprintf('****************************************\n')
            fprintf('        Signal Model Parameters\n')
            fprintf('****************************************\n')
            fprintf(' Activity probability (lambda): %s\n', form(obj, obj.lambda))
            fprintf('Active-to-inactive prob. (p01): %s\n', form(obj, obj.p01))
            fprintf('             Active mean (eta): %s\n', form(obj, obj.eta))
            fprintf('       Active variance (kappa): %s\n', form(obj, obj.kappa))
            fprintf('       Innovation rate (alpha): %s\n', form(obj, obj.alpha))
        	fprintf('         AWGN variance (sig2e): %s\n', form(obj, obj.sig2e))
        end
          
    end % methods
    
    methods (Access = private)
        % This method is called by the print method in order to format how
        % properties are printed.  Properties that are scalars will have
        % their values printed to the command window, while arrays will
        % have their dimensions, minimal, and maximal values printed to the
        % screen
        function string = form(obj, prop)
            if numel(prop) == 1
                string = num2str(prop);
            else
                string = sprintf('%d-by-%d array (Min: %g, Max: %g)', ...
                    size(prop, 1), size(prop, 2), min(min(prop)), ...
                    max(max(prop)));
            end
        end
    end % Private methods
   
end % classdef