% CLASS: Options
% 
% HIERARCHY (Enumeration of the various super- and subclasses)
%   Superclasses: hgsetget (MATLAB handle class)
%   Subclasses: N/A
% 
% TYPE (Abstract or Concrete)
%   Concrete
%
% DESCRIPTION (High-level overview of the class)
%   This class is used to specify various runtime options of the DCS-AMP or
%   AMP-MMV algorithm.  These options include numbers of iterations to
%   execute, whether to perform EM model parameter learning, and whether to
%   work silently or verbosely.
%
%   To create an Options object, there are two constructors to choose
%   from (see METHODS section below).  The default constructor, 
%   Options(), will create an Options object initialized with all default
%   values for each parameter/property.  The alternative constructor 
%   allows the user to initialize any subset of the parameters, with the
%   remaining parameters initialized to their default values, by using 
%   MATLAB's property/value string pairs convention, e.g.,
%   Options('smooth_iters', 10, 'update', 'true') will
%   construct an Options object in which the maximum number of smoothing
%   iterations is set to 10, and all model parameters will be updated using
%   the EM learning procedure.  Any parameters not explicitly set in the 
%   constructor will be set to their default values (but can be later
%   modified, if desired).  
%
% PROPERTIES (DCS-AMP / AMP-MMV Configuration Options)
%   smooth_iters    Maximum number of smoothing iterations to
%                   perform.  If one wishes to perform filtering, i.e.,
%                   causal message passing only, then set this field to
%                   -1 [dflt: 5]
%   min_iters       Minimum number of smoothing iterations to perform
%                   [dflt: 5]
%   inner_iters     Maximum number of inner AMP/BP iterations to perform, 
%                   per forward/backward pass, in each frame (timestep)
%                   [dflt: 15]
%   alg             Type of algorithm to use for message passing within
%                   each frame (timestep): 'BP' for Gaussian belief 
%                   propagation (slow), 'AMP' for Approximate Message 
%                   Passing (AMP) [dflt: 'AMP']
%   update          Treat model parameters as fixed (false), or attempt
%                   to learn them from the data using an EM learning
%                   algorithm (true) [dflt: true]
%   update_list     (DCS-AMP only) A cell array of strings indicating which 
%                   parameters (in the DCSModelParams structure) should be 
%                   updated, if Options.update = 1.  By default, all 
%                   parameters are updated, i.e., Options.update_list = 
%                   {'lambda', 'p01', 'eta', 'kappa', 'alpha', 'sig2e'}.
%                   Removing any parameters from the cell array will keep 
%                   them fixed at their initial value
%   upd_groups      This field can be used when updating model parameters 
%                   to force the algorithm to estimate different values of 
%                   the model parameters for different subsets of 
%                   coefficients.  If specified, Options.upd_groups must be
%                   a cell array in which each cell contains indices of
%                   coefficients whose parameters are to be estimated
%                   simultaneously.  The total number of cells determines
%                   the total number of unique subsets, and hence, unique
%                   parameter estimates. [dflt: {1:N}, (i.e. single group)]
%   verbose         Output information during execution (true), or display
%                   nothing (false) [dflt: false]
%   StateIn         An optional structure that contains variables needed to
%                   warm-start DCS-AMP or AMP-MMV.  Can be obtained 
%                   initially from the 'StateOut' output of either 
%                   SP_MULTI_FRAME_FXN, SP_PARALLEL_FRAME_FXN, SP_MMV_FXN 
%                   or SP_MMV_SERIAL_FXN.
%   eps             Real-valued scalar, (<< 1), needed to provide
%                   approximations of outgoing BP messages of active
%                   coefficient means (xi's) and variances (psi's) [dflt:
%                   1e-6]
%   tau             Optional parameter.  If tau = -1, a Taylor series
%                   approximation is used to determine the outgoing AMP
%                   messages OutMessages.xi and OutMessages.psi.  If a
%                   positive value between zero and one is passed for tau, 
%                   e.g., tau = 1 - 1e-3, it will be used as a threshold
%                   on the incoming AMP activity probabilities to
%                   determine whether to pass an informative or 
%                   non-informative message via OutMessages.xi and
%                   OutMessages.psi [dflt: 1 - 1e-3]
%
% METHODS (Subroutines/functions)
%   Options()
%       - Default constructor.  Assigns all properties to their default 
%         values.
%   Options('ParameterName1', Value1, 'ParameterName2', Value2, ...)
%       - Custom constructor.  Can be used to set any subset of the
%         properties, with remaining properties set to their defaults
%   set(obj, 'ParameterName', Value)
%       - Method that can be used to set the value of a parameter once the
%         object of class Options, obj, has been constructed
%   print()
%       - Print the current value of each property to the command window
%                   
%

%
% Coded by: Justin Ziniel, The Ohio State Univ.
% E-mail: zinielj@ece.osu.edu
% Last change: 12/29/11
% Change summary: 
%       - Created from Options v1.0 (12/29/11; JAZ)
% Version 1.2
%

classdef Options < hgsetget

    properties
        % Set all parameters to their default values
        smooth_iters = 5;       % Max # of smoothing iterations
        min_iters = 5;          % Min # of smoothing iterations
        inner_iters = 15;       % Max # of inner AMP/BP iterations
        alg = 2;                % Use AMP (and not Gaussian BP)
        update = true;          % Use EM learning on model parameters
        update_list = {'lambda', 'p01', 'eta', 'kappa', 'alpha', 'sig2e'};
        upd_groups;             % All coefficients belong to single group
        verbose = false;        % Work silently
        StateIn;                % Default is for no warm-starting
        eps = 1e-6;             % Small positive scalar for msg approx.
        tau = 1-1e-3;         	% Threshold-based message computation
    end % properties
    
    properties (Hidden)
        % These parameters exist to maintain backwards-compatibility with
        % previous naming conventions
        smooth_iter;    % smooth_iter -> smooth_iters
        eq_iter;        % eq_iter -> inner_iters
    end
    
    methods
        % Constructor
        function obj = Options(varargin)
            if nargin == 1 || mod(nargin, 2) ~= 0
                error('Improper constructor call')
            else
                for i = 1 : 2 : nargin - 1
                    obj.set(lower(varargin{i}), varargin{i+1});
                end
            end
        end                    
        
        % Set method for max # of smoothing iterations
        function obj = set.smooth_iters(obj, smooth_iters)
           if smooth_iters > 0 || smooth_iters == -1
               obj.smooth_iters = smooth_iters;
               obj.min_iters = min(obj.smooth_iters, obj.min_iters);
           else
              error('Invalid assignment: smooth_iters')
           end
        end
        
        % Set method for min # of smoothing iterations
        function obj = set.min_iters(obj, min_iters)
           if min_iters <= obj.smooth_iters
               obj.min_iters = min_iters;
           else
              obj.min_iters = obj.smooth_iters;
              fprintf(['min_iters exceeded smooth_iters, thus setting ' ...
                  'min_iters equal to smooth_iters\n']);
           end
        end
        
        % Set method for max # of inner AMP/BP iterations
        function obj = set.inner_iters(obj, inner_iters)
           if inner_iters > 0
               obj.inner_iters = inner_iters;
           else
              error('Invalid assignment: inner_iters')
           end
        end
        
        % Set method for type of inner frame algorithm (AMP or Gaussian BP)
        function obj = set.alg(obj, alg)
           if ~ischar(alg) && (alg ~= 1 && alg ~= 2)
               error('Assignments to alg must be character strings')
           elseif ischar(alg)
               switch upper(alg)
                   case 'AMP'
                       obj.alg = 2;     % Numeric code for AMP ver.
                   case 'BP'
                       obj.alg = 1;     % Numeric code for Gaussian BP ver.
                   otherwise
                       error(['Assignment to alg must be either ' ...
                           '''AMP'' or ''BP'''])
               end
           else
               obg.alg = alg;       % Numeric code used
           end
        end
        
        % Set method for whether or not to update params via EM learning
        function obj = set.update(obj, update)
            if islogical(update) || update == 0 || update == 1
                obj.update = logical(update);
            else
                error('Invalid assignment: update (must be logical)')
            end
        end
        
        % Set method for list of parameters to update via EM
        function obj = set.update_list(obj, update_list)
            if iscell(update_list)
                % Good enough; user probably knows what they're doing
                obj.update_list = update_list;
            else
                error('Invalid assignment: update_list (must be cell array)')
            end
        end
        
        % Set method for the update groups
        function obj = set.upd_groups(obj, groups)
            if iscell(groups)
                % Input is at least a cell array.  Defer further error
                % checking to DCS-AMP / AMP-MMV functions
                obj.upd_groups = groups;
            else
                error('Invalid assignment: upd_groups (must be cell array)')
            end
        end
        
        % Set method for the verbosity
        function obj = set.verbose(obj, verbose)
            if islogical(verbose) || verbose == 0 || verbose == 1
                obj.verbose = logical(verbose);
            else
                error('Invalid assignment: verbose (must be logical)')
            end
        end
        
        % Set method for StateIn (for warm-starting)
        function obj = set.StateIn(obj, State)
            if ~strcmp('AMPState', class(State)) && ~isempty(State)
                error('Invalid assignment: StateIn (must be AMPState object)')
            else
                obj.StateIn = State;
            end
        end
        
        % Set method for msg. approximation param, epsilon
        function obj = set.eps(obj, eps)
           if numel(eps) == 1 && eps > 0 && eps <= 1e-3
               obj.eps = eps;
           else
              error('Please choose a small (<< 1) scalar for eps')
           end
        end
        
        % Set method for tau
        function obj = set.tau(obj, tau)
           if (tau >= 0 && tau <= 1) || tau == -1
               obj.tau = tau;
           else
              error('Invalid assignment: tau')
           end
        end
        
        % Set method for max # of smoothing iterations (old name)
        function obj = set.smooth_iter(obj, smooth_iters)
           if smooth_iters > 0 || smooth_iters == -1
               obj.smooth_iters = smooth_iters;
               obj.min_iters = min(obj.smooth_iters, obj.min_iters);
           else
              error('Invalid assignment: smooth_iters (smooth_iter)')
           end
        end
        
        % Set method for max # of inner AMP/BP iterations
        function obj = set.eq_iter(obj, inner_iters)
           if inner_iters > 0
               obj.inner_iters = inner_iters;
           else
              error('Invalid assignment: inner_iters (eq_iter)')
           end
        end
        
        % Get method for grabbing all the properties at once
        function [smooth_iters, min_iters, inner_iters, alg, update, ...
                upd_groups, verbose, StateIn, eps, tau, update_list] = ...
                getOptions(obj)
            smooth_iters = obj.smooth_iters;
            min_iters = obj.min_iters;
            inner_iters = obj.inner_iters;
            alg = obj.alg;
            update = obj.update;
            upd_groups = obj.upd_groups;
            verbose = obj.verbose;
            StateIn = obj.StateIn;
            eps = obj.eps;
            tau = obj.tau;
            update_list = obj.update_list;
        end
        
        % Print the current configuration to the command window
        function print(obj)
            fprintf('***********************************\n')
            fprintf('      AMP-MMV Runtime Options\n')
            fprintf('***********************************\n')
            fprintf('Max. smoothing iterations: %d\n', obj.smooth_iters)
            fprintf('Min. smoothing iterations: %d\n', obj.min_iters)
            fprintf('   Max. AMP/BP iterations: %d\n', obj.inner_iters)
            switch obj.alg
                case 1
                    fprintf('    Intra-frame algorithm: Gaussian BP\n')
                case 2
                    fprintf('    Intra-frame algorithm: AMP\n')
            end
            switch obj.update
                case true
                    fprintf('    EM parameter learning: Yes\n')
                case false
                    fprintf('    EM parameter learning: No\n')
            end
            switch isempty(obj.upd_groups)
                case true
                    fprintf('            Update groups: [Default]\n')
                case false
                    fprintf('            Update groups: [User-Specified]\n')
            end
            switch obj.verbose
                case true
                    fprintf('                Verbosity: Verbose\n')
                case false
                    fprintf('                Verbosity: Silent\n')
            end
            switch isempty(obj.StateIn)
                case true
                    fprintf('               Warm-Start: No\n')
                case false
                    fprintf('               Warm-Start: Yes\n')
            end
            fprintf('                  epsilon: %g\n', obj.eps)
            if obj.tau == -1
                fprintf('  f-to-theta approx (tau): Taylor series approx\n')
            else
                fprintf('  f-to-theta approx (tau): %g\n', obj.tau)
            end
        end
          
    end % methods
   
end % classdef