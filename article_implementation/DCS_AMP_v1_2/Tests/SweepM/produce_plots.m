% PRODUCE_PLOTS     Produce plots for the sweep-M test
%

%
% Coded by: Justin Ziniel, The Ohio State Univ.
% E-mail: ziniel.1@osu.edu
% Last change: 12/09/10
% Change summary: 
%		- Created from produce_plots v0.2, cleaned up title (12/09/10; JAZ)
% Version 1.2
%

clear; clc

%% Load a results file

uiload

%% Produce the plots

algs = {'Support-aware Kalman Smoother', 'Proposed Approach', ...
    'Independent BPDN', 'Independent BP', 'Independent MMSE'};    % Algorithms
algcodes = {'smooth', 'frame', 'bpdn', 'bp', 'iMMSE'};
marker = {'bo-', 'g^--', 'rv-.', 'ks-', 'c*--'}; 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Plot of the NMSE for each approach
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Most likely some of the NMSE's remain as NaNs, because all trials were
% not completed, thus average only over the n completed trials
mean_NMSE_smooth_dB = 10*log10(mean(NMSE_smooth(1:n,:), 1));
mean_NMSE_frame_dB = 10*log10(mean(NMSE_frame(1:n,:), 1));
mean_NMSE_bpdn_dB = 10*log10(mean(NMSE_bpdn(1:n,:), 1));
mean_NMSE_bp_dB = 10*log10(mean(NMSE_bp(1:n,:), 1));
mean_NMSE_iMMSE_dB = 10*log10(mean(NMSE_iMMSE(1:n,:), 1));
mean_SER_frame = mean(SER_frame(1:n,:), 1);
mean_SER_bpdn = mean(SER_bpdn(1:n,:), 1);
mean_SER_bp = mean(SER_bp(1:n,:), 1);

% Clear existing figure
figure(1); clf;

% Plot the mean time-averaged NMSE for all completed trials on a dB scale
for i = [3 4 5 2 1]
    plot(delta, eval(['mean_NMSE_' algcodes{i} '_dB']), marker{i}); hold on
end

% Add title and labeling
axis tight
ylabel('NMSE [dB]');
xlabel('\delta (M/N)    (More measurements)   \rightarrow')
title_string = ['N = ' num2str(N) ', T = ' num2str(T) ...
    ', \lambda = ' num2str(lambda) ', p_{01} = ' num2str(pz1) ...
    ', \alpha = ' num2str(alpha) ', SNR_m = ' num2str(SNRmdB) ...
    'dB, N_{trials} = ' num2str(n)];
title(['NMSE vs. # of Measurements | ' title_string])
legend(algs{[3 4 5 2 1]}, 'Location', 'Best')
