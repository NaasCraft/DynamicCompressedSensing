% PRODUCE_PLOTS     Produce plots for the SNR test
%

%
% Coded by: Justin Ziniel, The Ohio State Univ.
% E-mail: ziniel.1@osu.edu
% Last change: 03/22/11
% Change summary: 
%		- Created (03/23/11; JAZ)
% Version 1.2
%

clear; clc

%% Load a results file

uiload

%% Produce the plots

algs = {'Proposed Approach', 'Support-aware Kalman smoother', ...
    'Independent MMSE', 'Independent AMP'};    % Algorithms
suffix = {'frame', 'smooth', 'iMMSE', 'naive'};
marker = {'-b', '-g', '--k', '--r'};
tile = [2, 2];      % Tiling of subplots, (total must equal N_confs)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Plot of the NMSE for each approach
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Most likely some of the NMSE's remain as NaNs, because all trials were
% not completed, thus average only over the n completed trials
for c = 1:N_confs
    for s = 1:N_snr
        mean_TNMSE_frame_dB(c,s) = 10*log10(mean(TNMSE_frame(c,s,1:n), 3));
        mean_TNMSE_smooth_dB(c,s) = 10*log10(mean(TNMSE_smooth(c,s,1:n), 3));
        mean_TNMSE_iMMSE_dB(c,s) = 10*log10(mean(TNMSE_iMMSE(c,s,1:n), 3));
        mean_TNMSE_naive_dB(c,s) = 10*log10(mean(TNMSE_naive(c,s,1:n), 3));
        mean_TBER_frame(c,s) = mean(TBER_frame(c,s,1:n), 3);
        mean_TBER_naive(c,s) = mean(TBER_naive(c,s,1:n), 3);
    end
end

% Clear existing figure
figure(1); clf;

% Plot the TNMSE of each algorithm for each configuration
for c= 1:N_confs
    subplot(tile(1), tile(2), c); hold on
    for a = 1:numel(algs)
        plot(SNRmdB, eval(['mean_TNMSE_' suffix{a} '_dB(c,:)']), marker{a})
    end
    hold off; grid on;
    title(sprintf(['N = %d, M = %d, \\lambda = %1.2g, \\sigma^2 = %1.1g,' ...
        ' p_{01} = %1.2g, \\alpha = %1.2g'], N, M(1), lambda_0, kappa_0, ...
        pz1(c), alpha(c)))
    xlabel('SNR (dB)'); ylabel('TNMSE (dB)')
    legend(algs, 'Location', 'Best')
end
        


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Plot of Support Error Rate (SER)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Clear existing figure
figure(2); clf;

% Plot the TBER of each algorithm for each configuration
for c= 1:N_confs
    subplot(tile(1), tile(2), c); hold on
    for a = [1, 4]      % Only relevant for frame and naive algs
        semilogy(SNRmdB, eval(['mean_TBER_' suffix{a} '(c,:)']), marker{a})
    end
    hold off; grid on;
    title(sprintf(['N = %d, M = %d, \\lambda = %1.2g, \\sigma^2 = %1.1g,' ...
        ' p_{01} = %1.2g, \\alpha = %1.2g'], N, M(1), lambda_0, kappa_0, ...
        pz1(c), alpha(c)))
    xlabel('SNR (dB)'); ylabel('TBER')
    legend(algs{[1, 4]}, 'Location', 'Best')
end