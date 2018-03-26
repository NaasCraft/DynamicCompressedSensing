% CLASS: SigGenParams
% 
% HIERARCHY (Enumeration of the various super- and subclasses)
%   Superclasses: hgsetget (MATLAB handle class)
%   Subclasses: N/A
% 
% TYPE (Abstract or Concrete)
%   Concrete
%
% DESCRIPTION (High-level overview of the class)
%   This class is used by DCS_SIGNAL_GEN_FXN (see dcs_signal_gen_fxn.m) in 
%   order to generate a signal according to a desired apriori distribution,
%   along with a randomly generated measurement matrix, and a collection of 
%   noisy measurement vectors.  A set of parameters are needed in order to 
%   fully specify a particular setup (see PROPERTIES below).
%
%   To create a SigGenParams object, there are two constructors to choose
%   from (see METHODS section below).  The default constructor, 
%   SigGenParams(), will create a SigGenParams object initialized with all 
%   default values for each parameter/property.  The alternative 
%   constructor allows the user to initialize any subset of the parameters,
%   with the remaining parameters initialized to their default values, by
%   using MATLAB's property/value string pairs convention, e.g.,
%   SigGenParams('N', 512, 'M', 128, 'T', 5) will construct a SigGenParams
%   object in which the signal dimension is set to 512, the number of 
%   measurements at each timestep is set to 128, and 5 timesteps will be 
%   generated.  Any parameters not explicitly set in the constructor will 
%   be set to their default values (but can be later modified, if desired).
%
% PROPERTIES (Model Parameters)
%   N               The dimension of the true signals, x_true(t) 
%                   [dflt: 1024]
%   M               The dimension of each measurement vector, y(t)
%                   [dflt: 256]
%   T               The total number of timesteps  [dflt: 4]
%   A_type          Type of random measurement matrix to generate.  (1) for
%                   a normalized (complex) IID Gaussian matrix, (2) for a
%                   Rademacher matrix, or (3) for subsampled DFT matrix
%                   [dflt: 1]
%   lambda          Scalar activity probability for s_n or s_n(t), i.e., 
%                   Pr{s_n(t) = 1} = lambda  [dflt: 0.08]
%   zeta        	(Complex) scalar of theta means, i.e.,
%                   E[theta_n(t)] = zeta  [dflt: 0]
%   sigma2          Scalar steady-state theta variances, i.e.
%                   var{theta_n(t)} = sigma2  [dflt: 1]
%   alpha           Scalar Gauss-Markov "innovation rate" parameter, btwn.
%                   0 and 1 (Note: Correlation coeff. = 1 - alpha)
%                   [dflt: 0.10]
%   p01             Active-to-inactive transition probability, 
%                   Pr{s_n(t) = 0 | s_n(t-1) = 1}.  This property is needed
%                   in the dynamic CS (DCS) signal model, but should be
%                   left set to zero for the MMV model [dflt: 0]
%   SNRmdB          The desired per-measurement empirical SNR, in dB, i.e.
%                   the ratio of the average signal power to the average
%                   noise power at each measurement should be SNRmdB
%                   [dflt: 25]
%   version         Type of data to generate: ('C') for complex, ('R') for
%                   real [dflt = 'C']
%
%   ***Note: The parameters lambda, zeta, sigma2, and alpha can, in
%            addition to scalar inputs, accept N-by-1 vector inputs if you
%            wish to assign model parameters on a per-coefficient basis.
%
% METHODS (Subroutines/functions)
%   SigGenParams()
%       - Default constructor.  Assigns all properties to their default 
%         values.
%   SigGenParams('ParameterName1', Value1, 'ParameterName2', Value2, ...)
%       - Custom constructor.  Can be used to set any subset of the
%         properties, with remaining properties set to their defaults
%   set(obj, 'ParameterName', Value)
%       - Method that can be used to set the value of a parameter once the
%         object of class SigGenParams, obj, has been constructed
%   print()
%       - Print the current value of each property to the command window
%                   
%

%
% Coded by: Justin Ziniel, The Ohio State Univ.
% E-mail: zinielj@ece.osu.edu
% Last change: 12/07/15
% Change summary: 
%       - Created from SigGenParams v1.0 (12/29/11; JAZ)
%       - Made suitable for both MMV and DCS signal models (02/08/12; JAZ)
%       - Minor bug fixes (12/07/15; JAZ)
% Version 1.2
%

classdef SigGenParams < hgsetget

    properties
        % Set all parameters to their default values
        N = 1024;           % Signal dimension
        M = 256;            % Measurement dimension
        T = 4;              % # of timesteps
        A_type = 1;         % IID Gaussian A matrix
        lambda = 0.08;      % Pr{s_n = 1}
        zeta = 0;           % E[theta_n(t)]
        sigma2 = 1;         % var{theta_n(t)}
        alpha = 0.10;       % Amplitude process innovation rate
        p01 = 0;            % Pr{s_n(t) = 0 | s_n(t-1) = 1}
        SNRmdB = 25;        % Per-measurement empirical SNR (in dB)
        version = 1;        % Default to complex-valued quantities
    end % properties
    
    properties (Dependent)
        rho;                % The variance of the amplitude driving noise
        mmv;                % = true if MMV model, = false if DCS model
    end
    
    methods
        % Constructor
        function obj = SigGenParams(varargin)
            if nargin == 1 || mod(nargin, 2) ~= 0
                error('Improper constructor call')
            else
                for i = 1 : 2 : nargin - 1
                    obj.set(varargin{i}, varargin{i+1});
                end
            end
        end                    
        
        % Set method for N
        function obj = set.N(obj, N)
           if N > 0 && numel(N) == 1
               obj.N = N;
           else
              error('Invalid assignment: N')
           end
        end
        
        % Set method for M
        function obj = set.M(obj, M)
           if all(M > 0) && numel(M) == 1
               obj.M = M;
           else
              error('Invalid assignment: M')
           end
        end
        
        % Set method for T
        function obj = set.T(obj, T)
           if T > 0 && numel(T) == 1
               obj.T = T;
           else
              error('Invalid assignment: T')
           end
        end
        
        % Set method for type of random A matrix
        function obj = set.A_type(obj, A_type)
            if A_type == 1 || A_type == 2 || A_type == 3
                obj.A_type = A_type;
            else
                error('Invalid assignment: A_type')
            end
        end
        
        % Set method for lambda
        function obj = set.lambda(obj, lambda)
           if lambda >= 0 && lambda <= 1
               obj.lambda = lambda;
           else
              error('Invalid assignment: lambda')
           end
        end
        
        % Set method for zeta
        function obj = set.zeta(obj, zeta)
            obj.zeta = zeta;
           if ~isreal(zeta)
               fprintf('Complex mean detected - Setting version = ''C''\n')
               obj.version = 'C';
           end
        end
        
        % Set method for sigma2
        function obj = set.sigma2(obj, sigma2)
           if sigma2 > 0
               obj.sigma2 = sigma2;
           else
              error('Invalid assignment: sigma2')
           end
        end
        
        % Set method for alpha
        function obj = set.alpha(obj, alpha)
           if alpha >= 0 && alpha < 1
               obj.alpha = alpha;
           else
              error('Invalid assignment: alpha')
           end
        end
        
        % Set method for p01
        function obj = set.p01(obj, p01)
           if p01 >= 0 && p01 <= 1
               obj.p01 = p01;
           else
              error('Invalid assignment: p01')
           end
        end
        
        % Set method for SNRmdB
        function obj = set.SNRmdB(obj, SNRmdB)
           if numel(SNRmdB) == 1
               obj.SNRmdB = SNRmdB;
           else
              error('Invalid assignment: SNRmdB')
           end
        end
        
        % Set method for type of data to generate
        function obj = set.version(obj, version)
           if ~ischar(version)
               error('Assignments to version must be characters')
           else
               switch upper(version)
                   case 'R'
                       if ~isreal(obj.zeta)
                           error('zeta is complex-valued')
                       else
                           obj.version = 2;	% Numeric code for real case
                       end
                   case 'C'
                       obj.version = 1;     % Numeric code for complex case
                   otherwise
                       error(['Assignment to version must be either ' ...
                           '''C'' or ''R'''])
               end
           end
        end
        
        % Get method for dependent property rho (which is set based on the
        % values of alpha and sigma2 to ensure a stationary amplitude
        % variance over time
        function rho = get.rho(obj)
            if numel(obj.alpha) == numel(obj.sigma2)
                rho = (2 - obj.alpha) .* obj.sigma2 ./ obj.alpha;
            elseif numel(obj.alpha) > 1 && numel(obj.sigma2) == 1
                rho = obj.sigma2 * (2 - obj.alpha) ./ obj.alpha;
            elseif numel(obj.sigma2) > 1 && numel(obj.alpha) ==  1
                rho = ((2 - obj.alpha)/obj.alpha)*obj.sigma2;
            else
                error('Dimension mismatch between alpha and sigma2')
            end
        end
        
         % Get method for dependent property mmv
        function mmv = get.mmv(obj)
            if obj.p01 == 0
                mmv = true;
            else
                mmv = false;
            end
        end
        
        % Get method for grabbing all the properties at once
        function [N, M, T, A_type, lambda, zeta, sigma2, alpha, SNRmdB, ...
                version, rho, p01, mmv] = getParams(obj)
            N = obj.N;
            M = obj.M;
            T = obj.T;
            A_type = obj.A_type;
            lambda = obj.lambda;
            zeta = obj.zeta;
            sigma2 = obj.sigma2;
            alpha = obj.alpha;
            SNRmdB = obj.SNRmdB;
            version = obj.version;
            rho = obj.rho;
            p01 = obj.p01;
            mmv = obj.mmv;
        end
        
        % Print the current configuration to the command window
        function print(obj)
            fprintf('*********************************************\n')
            fprintf('        Signal Generation Parameters\n')
            fprintf('*********************************************\n')
            fprintf('         Signal dimension (N): %s\n', form(obj, obj.N))
            fprintf('    Measurement dimension (M): %s\n', form(obj, obj.M))
            fprintf('           # of timesteps (T): %s\n', form(obj, obj.T))
            switch obj.A_type
                case 1
                    fprintf('      Measurement matrix type: IID Gaussian\n')
                case 2
                    fprintf(      'Measurement matrix type: Rademacher\n')
                case 3
                    fprintf(      'Measurement matrix type: Subsampled DFT\n')
            end
            fprintf('Activity probability (lambda): %s\n', form(obj, obj.lambda))
            fprintf('           Active mean (zeta): %s\n', form(obj, obj.zeta))
            fprintf('     Active variance (sigma2): %s\n', form(obj, obj.sigma2))
            fprintf('      Innovation rate (alpha): %s\n', form(obj, obj.alpha))
            switch obj.mmv
                case true
                    % Do not print p01 for MMV model
                case false
                    fprintf('Pr{s_n(t) = 0 | s_n(t-1) = 1}: %s\n', form(obj, obj.p01))
            end
        	fprintf('       Empirical SNR (SNRmdB): %s dB\n', form(obj, obj.SNRmdB))
            switch obj.version
                case 2
                    fprintf('                    Data type: Real-valued\n')
                case 1
                    fprintf('                    Data type: Complex-valued\n')
            end
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
