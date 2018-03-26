% PRODUCE_PLOTS     Produce plots for the ell-1 phase test problem
%

%
% Coded by: Justin Ziniel, The Ohio State Univ.
% E-mail: ziniel.1@osu.edu
% Last change: 12/09/10
% Change summary: 
%		- Created from produce_plots v0.2 (12/09/10; JAZ)
% Version 1.2
%

clear; clc

%% Load a results file

uiload

%% Produce the plots

clim = [-27, 0];    % Range for the colormap
algs = {'Support-aware Kalman smoother', ...
    'Multi-timestep BP', 'BPDN', 'Independent MMSE'};    % Algorithms

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Plot of the NMSE for each approach
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% % Clear existing figure
% figure(1); clf;
% 
% % Plot the mean time-averaged NMSE for all completed trials on a dB scale
% figure(1); imagesc(delta, fliplr(beta), 10*log10(mean(NMSE_smooth(:,:,1:n), 3)), clim);
% colorbar; set(gca, 'YTickLabel', flipud(get(gca, 'YTickLabel')));   % Flip beta labels
% figure(2); imagesc(delta, fliplr(beta), 10*log10(mean(NMSE_frame(:,:,1:n), 3)), clim);
% colorbar; set(gca, 'YTickLabel', flipud(get(gca, 'YTickLabel')));   % Flip beta labels
% figure(3); imagesc(delta, fliplr(beta), 10*log10(mean(NMSE_bpdn(:,:,1:n), 3)), clim);
% colorbar; set(gca, 'YTickLabel', flipud(get(gca, 'YTickLabel')));   % Flip beta labels
%
% % Add x- and y-axis labels and a title string
% for i = 1:3
%     figure(i)
%     ylabel('\beta (E[K]/M)     (More active coefficients) \rightarrow');
%     xlabel('\delta (M/N)    (More measurements) \rightarrow')
%     title_string = [algs{i} ' NMSE [dB] | N = ' num2str(N) ', T = ' num2str(T) ...
%         ', p_{\epsilon 1} = ' num2str(pz1) ', \alpha = ' num2str(alpha) ...
%         ', SNR_m = ' num2str(SNRmdB) 'dB, N_{trials} = ' num2str(n)];
%     title(title_string)
% end

% Most likely some of the NMSE's remain as NaNs, because all trials were
% not completed, thus average only over the n completed trials
for i = 1:Q
    for j = 1:Q
        mean_NMSE_smooth_dB(i,j) = 10*log10(mean(NMSE_smooth(i,j,1:n), 3));
        mean_NMSE_frame_dB(i,j) = 10*log10(mean(NMSE_frame(i,j,1:n), 3));
        mean_NMSE_bpdn_dB(i,j) = 10*log10(mean(NMSE_bpdn(i,j,1:n), 3));
        mean_NMSE_iMMSE_dB(i,j) = 10*log10(mean(NMSE_iMMSE(i,j,1:n), 3));
        mean_BER_frame(i,j) = mean(BER_frame(i,j,1:n), 3);
        mean_BER_bpdn(i,j) = mean(BER_bpdn(i,j,1:n), 3);
    end
end

% Clear existing figure
figure(1); clf;

% Plot the mean time-averaged NMSE for all completed trials on a dB scale
subplot(131); imagesc(delta, fliplr(beta), mean_NMSE_smooth_dB, clim);
colorbar; set(gca, 'YTickLabel', flipud(get(gca, 'YTickLabel')));   % Flip beta labels
subplot(132); imagesc(delta, fliplr(beta), mean_NMSE_frame_dB, clim);
colorbar; set(gca, 'YTickLabel', flipud(get(gca, 'YTickLabel')));   % Flip beta labels
subplot(133); imagesc(delta, fliplr(beta), mean_NMSE_bpdn_dB, clim);
colorbar; set(gca, 'YTickLabel', flipud(get(gca, 'YTickLabel')));   % Flip beta labels
% 
% % Plot the mean time-averaged NMSE for all completed trials on a dB scale
% subplot(131); [C1, H1] = contour(delta, fliplr(beta), flipud(mean_NMSE_smooth_dB));
% clabel(C1, H1)
% subplot(132); [C2, H2] = contour(delta, fliplr(beta), flipud(mean_NMSE_frame_dB));
% clabel(C2, H2)
% subplot(133); [C3, H3] = contour(delta, fliplr(beta), flipud(mean_NMSE_bpdn_dB));
% clabel(C3, H3)

% Add x- and y-axis labels and a title string
for i = 1:3
    subplot(1,3,i)
    ylabel('\beta (E[K]/M)     (More active coefficients)   \rightarrow');
    xlabel('\delta (M/N)    (More measurements)   \rightarrow')
    title_string = [algs{i} ' NMSE [dB]'];
    title(title_string)
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Plot of the proximity to genie smoother NMSE
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Clear existing figure
figure(2); clf;

% Threshold for declaring NMSE as being "close" to the genie smoother
NMSE_thresh = 3;    % [dB]

% Compute the 'proximity masks' for both the multi-frame and BPDN
% methods
frame_mask = abs(mean_NMSE_smooth_dB - mean_NMSE_frame_dB) <= NMSE_thresh;
bpdn_mask = abs(mean_NMSE_smooth_dB - mean_NMSE_bpdn_dB) <= NMSE_thresh;
% iMMSE_mask = abs(mean_NMSE_smooth_dB - mean_NMSE_iMMSE_dB) <= NMSE_thresh;

% Plot the resulting proximity masks for multi-frame and bpdn BP
figure(2);
% subplot(121); contour(delta, fliplr(beta), flipud(frame_mask), 1);
% subplot(122); CMAP = contour(delta, fliplr(beta), flipud(bpdn_mask), 1);
contour(delta, fliplr(beta), flipud(frame_mask), 1, 'k-'); hold on
contour(delta, fliplr(beta), flipud(bpdn_mask), 1, 'k:'); hold on
% contour(delta, fliplr(beta), flipud(iMMSE_mask), 1, 'k-.'); hold off

legend(algs{2}, algs{3}, 'Location', 'Best')
% legend(algs{2}, algs{3}, algs{4}, 'Location', 'Best')
ylabel('\beta (E[K]/M)     (More active coefficients)   \rightarrow');
xlabel('\delta (M/N)    (More measurements)   \rightarrow')
title(['Region in which algorithm is within ' num2str(NMSE_thresh) ...
    ' dB of the support-aware Kalman smoother'])
% gtext('\uparrow Failure'); gtext('\downarrow Success')
% gtext('\uparrow Failure'); gtext('\downarrow Success')


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Report average TNMSE (dB) within these regions
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% To calculate the TNMSE for a particular algorithm in a particular region,
% take the mask for that region, replicate it along the third dimension so
% that the mask is stacked n times (n = # of completed trials of
% simulation) in the third dimension.  Pointwise multiply this mask against
% the raw NMSE data for the particular algorithm.  Then the only non-zero
% elements of the result will correspond to TNMSE's for individual trials
% in the region of interest for the algorithm in question.  Sum up all of
% these TNMSE's, and divide by the total number of TNMSE's being summed (=
% sum(sum(mask))*n).
smooth_avg_TNMSE_frame_success_region = ...
    sum(sum(sum(repmat(frame_mask, [1 1 n]) .* NMSE_smooth(:,:,1:n)))) ...
    / (n*sum(sum(frame_mask)));
smooth_avg_TNMSE_frame_success_region_dB = ...
    10*log10(smooth_avg_TNMSE_frame_success_region);
smooth_avg_TNMSE_frame_fail_region = ...
    sum(sum(sum(repmat(not(frame_mask), [1 1 n]) .* NMSE_smooth(:,:,1:n)))) ...
    / (n*sum(sum(not(frame_mask))));
smooth_avg_TNMSE_frame_fail_region_dB = ...
    10*log10(smooth_avg_TNMSE_frame_fail_region);
frame_avg_TNMSE_frame_success_region = ...
    sum(sum(sum(repmat(frame_mask, [1 1 n]) .* NMSE_frame(:,:,1:n)))) ...
    / (n*sum(sum(frame_mask)));
frame_avg_TNMSE_frame_success_region_dB = ...
    10*log10(frame_avg_TNMSE_frame_success_region);
frame_avg_TNMSE_frame_fail_region = ...
    sum(sum(sum(repmat(not(frame_mask), [1 1 n]) .* NMSE_frame(:,:,1:n)))) ...
    / (n*sum(sum(not(frame_mask))));
frame_avg_TNMSE_frame_fail_region_dB = ...
    10*log10(frame_avg_TNMSE_frame_fail_region);
smooth_avg_TNMSE_bpdn_success_region = ...
    sum(sum(sum(repmat(bpdn_mask, [1 1 n]) .* NMSE_smooth(:,:,1:n)))) ...
    / (n*sum(sum(bpdn_mask)));
smooth_avg_TNMSE_bpdn_success_region_dB = ...
    10*log10(smooth_avg_TNMSE_bpdn_success_region);
smooth_avg_TNMSE_bpdn_fail_region = ...
    sum(sum(sum(repmat(not(bpdn_mask), [1 1 n]) .* NMSE_smooth(:,:,1:n)))) ...
    / (n*sum(sum(not(bpdn_mask))));
smooth_avg_TNMSE_bpdn_fail_region_dB = ...
    10*log10(smooth_avg_TNMSE_bpdn_fail_region);
bpdn_avg_TNMSE_bpdn_success_region = ...
    sum(sum(sum(repmat(bpdn_mask, [1 1 n]) .* NMSE_bpdn(:,:,1:n)))) ...
    / (n*sum(sum(bpdn_mask)));
bpdn_avg_TNMSE_bpdn_success_region_dB = ...
    10*log10(bpdn_avg_TNMSE_bpdn_success_region);
bpdn_avg_TNMSE_bpdn_fail_region = ...
    sum(sum(sum(repmat(not(bpdn_mask), [1 1 n]) .* NMSE_bpdn(:,:,1:n)))) ...
    / (n*sum(sum(not(bpdn_mask))));
bpdn_avg_TNMSE_bpdn_fail_region_dB = ...
    10*log10(bpdn_avg_TNMSE_bpdn_fail_region);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Plot of equal NMSE contours
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Clear existing figure
figure(3); clf;

% The target NMSE contour for each algorithm
NMSE_ref = -10;     % [dB]

% Compute the 'proximity masks' for Kalman smoother, multi-frame and bpdn 
% BP methods
smooth_mask = mean_NMSE_smooth_dB <= NMSE_ref;
frame_mask = mean_NMSE_frame_dB <= NMSE_ref;
bpdn_mask = mean_NMSE_bpdn_dB <= NMSE_ref;

% Plot the resulting proximity masks for the three methods
figure(3);
contour(delta, fliplr(beta), flipud(smooth_mask), 1, 'k--'); hold on
contour(delta, fliplr(beta), flipud(frame_mask), 1, 'k-'); hold on
contour(delta, fliplr(beta), flipud(bpdn_mask), 1, 'k-.'); hold off

legend(algs{1}, algs{2}, algs{3}, 'Location', 'Best')
ylabel('\beta (E[K]/M)     (More active coefficients)   \rightarrow');
xlabel('\delta (M/N)    (More measurements)   \rightarrow')
title(['Region in which algorithm is has an NMSE below ' ...
    num2str(NMSE_ref) ' dB'])
% gtext('\uparrow Failure'); gtext('\downarrow Success')
% gtext('\uparrow Failure'); gtext('\downarrow Success')


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Plot of BER
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Clear existing figure
figure(4); clf;

clim2 = [0 0.40]

subplot(121); imagesc(delta, fliplr(beta), mean_BER_frame, clim2);
colorbar; set(gca, 'YTickLabel', flipud(get(gca, 'YTickLabel')));   % Flip beta labels
subplot(122); imagesc(delta, fliplr(beta), mean_BER_bpdn, clim2);
colorbar; set(gca, 'YTickLabel', flipud(get(gca, 'YTickLabel')));   % Flip beta labels

% Add x- and y-axis labels and a title string
for i = 1:2
    subplot(1,2,i)
    ylabel('\beta (E[K]/M)     (More active coefficients)   \rightarrow');
    xlabel('\delta (M/N)    (More measurements)   \rightarrow')
    title_string = [algs{i+1} ' BER'];
    title(title_string)
end