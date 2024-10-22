% PRODUCE_PLOTS     Produce plots for the phase test problem
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
    'DCS-AMP', 'BG-AMP', 'Independent MMSE'};    % Algorithms

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
% figure(3); imagesc(delta, fliplr(beta), 10*log10(mean(NMSE_naive(:,:,1:n), 3)), clim);
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
        mean_NMSE_naive_dB(i,j) = 10*log10(mean(NMSE_naive(i,j,1:n), 3));
        mean_NMSE_iMMSE_dB(i,j) = 10*log10(mean(NMSE_iMMSE(i,j,1:n), 3));
        mean_diffNMSE_frame_dB(i,j) = 10*log10(mean(NMSE_frame(i,j,1:n) - ...
            NMSE_smooth(i,j,1:n), 3));
        mean_diffNMSE_naive_dB(i,j) = 10*log10(mean(NMSE_naive(i,j,1:n) - ...
            NMSE_smooth(i,j,1:n), 3));
        mean_BER_frame(i,j) = mean(BER_frame(i,j,1:n), 3);
        mean_BER_naive(i,j) = mean(BER_naive(i,j,1:n), 3);
    end
end

% Clear existing figure
figure(1); clf;

% % Plot the mean time-averaged NMSE for all completed trials on a dB scale
% subplot(131); imagesc(delta, fliplr(beta), mean_NMSE_smooth_dB, clim); hold on
% colorbar; set(gca, 'YTickLabel', flipud(get(gca, 'YTickLabel')));   % Flip beta labels
% %  axis square
% subplot(132); imagesc(delta, fliplr(beta), mean_NMSE_frame_dB, clim); hold on
% colorbar; set(gca, 'YTickLabel', flipud(get(gca, 'YTickLabel')));   % Flip beta labels
% %  axis square
% subplot(133); imagesc(delta, fliplr(beta), mean_NMSE_naive_dB, clim); hold on
% colorbar; set(gca, 'YTickLabel', flipud(get(gca, 'YTickLabel')));   % Flip beta labels
% %  axis square

% Plot the mean time-averaged NMSE for all completed trials on a dB scale
subplot(131); [C1, H1] = contour(delta, fliplr(beta), flipud(mean_NMSE_smooth_dB), -28:1:-18);
clabel(C1, H1, 'Color', 'Black', 'fontsize', 18)
subplot(132); [C2, H2] = contour(delta, fliplr(beta), flipud(mean_NMSE_frame_dB), -28:1:-15);
clabel(C2, H2, 'Color', 'Black', 'fontsize', 18)
subplot(133); [C3, H3] = contour(delta, fliplr(beta), flipud(mean_NMSE_naive_dB), -25:2:-3);
clabel(C3, H3, 'Color', 'Black', 'fontsize', 18)

% % Add equal-lambda contours to the plot
% for b = 1:N_beta
%     for d = 1:N_delta
%         M_tmp = round(delta(d)*N);	% Number of measurements per timestep
%         lambda_contours(b,d) = beta(b)*M_tmp/N;	% E[K] for (beta, delta) pair
%     end
% end
% contour_values = [0.10, 0.20, 0.30, 0.50, 0.75];    % Equal lambda contour lines to plot
% subplot(131); 
% [con_out, con_handle] = contour(delta, fliplr(beta), lambda_contours, ...
%     contour_values, 'w-'); 
% clabel(con_out, con_handle, 'Color', 'white'); hold off
% subplot(132); 
% [con_out, con_handle] = contour(delta, fliplr(beta), lambda_contours, ...
%     contour_values, 'w-'); 
% clabel(con_out, con_handle, 'Color', 'white'); hold off
% subplot(133); 
% [con_out, con_handle] = contour(delta, fliplr(beta), lambda_contours, ...
%     contour_values, 'w-'); 
% clabel(con_out, con_handle, 'Color', 'white'); hold off

% Add x- and y-axis labels and a title string
for i = 1:3
    subplot(1,3,i);
    ylabel('\beta (E[K]/M)  (More active coefficients)   \rightarrow');
    xlabel('\delta (M/N)  (More measurements)   \rightarrow')
    title_string = [algs{i} ' TNMSE [dB]'];
    title(title_string)
end
% subplot(132)
% text_string = ['N = ' num2str(N) ', T = ' num2str(T) ...
%     ', p_{01} = ' num2str(pz1) ', \alpha = ' num2str(alpha) ...
%     ', SNR_m = ' num2str(SNRmdB) 'dB, N_{trials} = ' num2str(n)];
% text(0.5, 0.10, text_string, 'Color', 'White', 'HorizontalAlignment', ...
%     'Center', 'FontSize', 5)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Plot of the proximity to genie smoother NMSE
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Clear existing figure
figure(2); clf;

% % Threshold for declaring NMSE as being "close" to the genie smoother
% NMSE_thresh = 3;    % [dB]
% 
% % Compute the 'proximity masks' for both the multi-frame and naive BP
% % methods
% frame_mask = abs(mean_NMSE_smooth_dB - mean_NMSE_frame_dB) <= NMSE_thresh;
% naive_mask = abs(mean_NMSE_smooth_dB - mean_NMSE_naive_dB) <= NMSE_thresh;
% % iMMSE_mask = abs(mean_NMSE_smooth_dB - mean_NMSE_iMMSE_dB) <= NMSE_thresh;
% 
% % Plot the resulting proximity masks for multi-frame and naive BP
% figure(2);
% % subplot(121); contour(delta, fliplr(beta), flipud(frame_mask), 1);
% % subplot(122); CMAP = contour(delta, fliplr(beta), flipud(naive_mask), 1);
% contour(delta, fliplr(beta), flipud(frame_mask), 1, 'k-'); hold on
% contour(delta, fliplr(beta), flipud(naive_mask), 1, 'k:'); hold on
% % contour(delta, fliplr(beta), flipud(iMMSE_mask), 1, 'k-.'); hold off
% 
% legend(algs{2}, algs{3}, 'Location', 'Best')
% % legend(algs{2}, algs{3}, algs{4}, 'Location', 'Best')
% ylabel('\beta (E[K]/M)     (More active coefficients)   \rightarrow');
% xlabel('\delta (M/N)    (More measurements)   \rightarrow')
% title(['Region in which algorithm is within ' num2str(NMSE_thresh) ...
%     ' dB of the support-aware Kalman smoother'])
% % gtext('\uparrow Failure'); gtext('\downarrow Success')
% % gtext('\uparrow Failure'); gtext('\downarrow Success')

% Plot the mean difference in time-averaged NMSE for both the frame method
% and naive BP method, from the Kalman smoother, for all completed trials 
% on a dB scale
% subplot(121); imagesc(delta, fliplr(beta), mean_diffNMSE_frame_dB, clim); hold on
% colorbar; set(gca, 'YTickLabel', flipud(get(gca, 'YTickLabel')));   % Flip beta labels
% %  axis square
% subplot(122); imagesc(delta, fliplr(beta), mean_diffNMSE_naive_dB, clim); hold on
% colorbar; set(gca, 'YTickLabel', flipud(get(gca, 'YTickLabel')));   % Flip beta labels
% %  axis square
% subplot(121); [C2, H2] = contour(delta, fliplr(beta), flipud(mean_diffNMSE_frame_dB));
% clabel(C2, H2); hold on
% subplot(122); [C3, H3] = contour(delta, fliplr(beta), flipud(mean_diffNMSE_naive_dB));
% clabel(C3, H3); hold on
subplot(121); [C2, H2] = contour(delta, fliplr(beta), ...
    flipud(mean_NMSE_frame_dB - mean_NMSE_smooth_dB), [.5:.25:2.5]);
clh1 = clabel(C2, H2); hold on
subplot(122); [C3, H3] = contour(delta, fliplr(beta), ...
    flipud(mean_NMSE_naive_dB - mean_NMSE_smooth_dB), [1:15]);
clh2 = clabel(C3, H3); hold on

% Increase font size of contour labels
for i=1:length(clh1), set(clh1(i),'FontSize',18); end;
for i=1:length(clh2), set(clh2(i),'FontSize',18); end;

% % Add equal-lambda contours to the plot
% subplot(121); 
% [con_out, con_handle] = contour(delta, fliplr(beta), lambda_contours, ...
%     contour_values, 'k--'); 
% clabel(con_out, con_handle, 'Color', 'black'); hold off
% subplot(122); 
% [con_out, con_handle] = contour(delta, fliplr(beta), lambda_contours, ...
%     contour_values, 'k--'); 
% clabel(con_out, con_handle, 'Color', 'black'); hold off

% Add x- and y-axis labels and a title string
for i = 1:2
    subplot(1,2,i);
    ylabel('\beta (E[K]/M)     (More active coefficients)   \rightarrow');
    xlabel('\delta (M/N)    (More measurements)   \rightarrow')
    title_string = [algs{i+1} ' TNMSE [dB] - SKS TNMSE [dB]'];
    title(title_string)
end

% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% % Report average TNMSE (dB) within these regions
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 
% % To calculate the TNMSE for a particular algorithm in a particular region,
% % take the mask for that region, replicate it along the third dimension so
% % that the mask is stacked n times (n = # of completed trials of
% % simulation) in the third dimension.  Pointwise multiply this mask against
% % the raw NMSE data for the particular algorithm.  Then the only non-zero
% % elements of the result will correspond to TNMSE's for individual trials
% % in the region of interest for the algorithm in question.  Sum up all of
% % these TNMSE's, and divide by the total number of TNMSE's being summed (=
% % sum(sum(mask))*n).
% smooth_avg_TNMSE_frame_success_region = ...
%     sum(sum(sum(repmat(frame_mask, [1 1 n]) .* NMSE_smooth(:,:,1:n)))) ...
%     / (n*sum(sum(frame_mask)));
% smooth_avg_TNMSE_frame_success_region_dB = ...
%     10*log10(smooth_avg_TNMSE_frame_success_region);
% smooth_avg_TNMSE_frame_fail_region = ...
%     sum(sum(sum(repmat(not(frame_mask), [1 1 n]) .* NMSE_smooth(:,:,1:n)))) ...
%     / (n*sum(sum(not(frame_mask))));
% smooth_avg_TNMSE_frame_fail_region_dB = ...
%     10*log10(smooth_avg_TNMSE_frame_fail_region);
% frame_avg_TNMSE_frame_success_region = ...
%     sum(sum(sum(repmat(frame_mask, [1 1 n]) .* NMSE_frame(:,:,1:n)))) ...
%     / (n*sum(sum(frame_mask)));
% frame_avg_TNMSE_frame_success_region_dB = ...
%     10*log10(frame_avg_TNMSE_frame_success_region);
% frame_avg_TNMSE_frame_fail_region = ...
%     sum(sum(sum(repmat(not(frame_mask), [1 1 n]) .* NMSE_frame(:,:,1:n)))) ...
%     / (n*sum(sum(not(frame_mask))));
% frame_avg_TNMSE_frame_fail_region_dB = ...
%     10*log10(frame_avg_TNMSE_frame_fail_region);
% smooth_avg_TNMSE_naive_success_region = ...
%     sum(sum(sum(repmat(naive_mask, [1 1 n]) .* NMSE_smooth(:,:,1:n)))) ...
%     / (n*sum(sum(naive_mask)));
% smooth_avg_TNMSE_naive_success_region_dB = ...
%     10*log10(smooth_avg_TNMSE_naive_success_region);
% smooth_avg_TNMSE_naive_fail_region = ...
%     sum(sum(sum(repmat(not(naive_mask), [1 1 n]) .* NMSE_smooth(:,:,1:n)))) ...
%     / (n*sum(sum(not(naive_mask))));
% smooth_avg_TNMSE_naive_fail_region_dB = ...
%     10*log10(smooth_avg_TNMSE_naive_fail_region);
% naive_avg_TNMSE_naive_success_region = ...
%     sum(sum(sum(repmat(naive_mask, [1 1 n]) .* NMSE_naive(:,:,1:n)))) ...
%     / (n*sum(sum(naive_mask)));
% naive_avg_TNMSE_naive_success_region_dB = ...
%     10*log10(naive_avg_TNMSE_naive_success_region);
% naive_avg_TNMSE_naive_fail_region = ...
%     sum(sum(sum(repmat(not(naive_mask), [1 1 n]) .* NMSE_naive(:,:,1:n)))) ...
%     / (n*sum(sum(not(naive_mask))));
% naive_avg_TNMSE_naive_fail_region_dB = ...
%     10*log10(naive_avg_TNMSE_naive_fail_region);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Plot of equal NMSE contours
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Clear existing figure
figure(3); clf;

% The target NMSE contour for each algorithm
NMSE_ref = -10;     % [dB]

% Compute the 'proximity masks' for Kalman smoother, multi-frame and naive 
% BP methods
smooth_mask = mean_NMSE_smooth_dB <= NMSE_ref;
frame_mask = mean_NMSE_frame_dB <= NMSE_ref;
naive_mask = mean_NMSE_naive_dB <= NMSE_ref;

% Plot the resulting proximity masks for the three methods
figure(3);
contour(delta, beta, smooth_mask, 1, 'k--'); hold on
contour(delta, beta, frame_mask, 1, 'k-'); hold on
contour(delta, beta, naive_mask, 1, 'k-.'); hold off

legend(algs{1}, algs{2}, algs{3}, 'Location', 'Best')
ylabel('\beta (E[K]/M)     (More active coefficients)   \rightarrow');
xlabel('\delta (M/N)    (More measurements)   \rightarrow')
title(['Region in which algorithm is has an NMSE below ' ...
    num2str(NMSE_ref) ' dB'])
% gtext('\uparrow Failure'); gtext('\downarrow Success')
% gtext('\uparrow Failure'); gtext('\downarrow Success')


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Plot of Support Error Rate (SER)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Clear existing figure
figure(4); clf;

clim2 = [0 1];

subplot(121); axis square; imagesc(delta, fliplr(beta), mean_BER_frame./lambda_contours, clim2); hold on
colorbar; set(gca, 'YTickLabel', flipud(get(gca, 'YTickLabel')));   % Flip beta labels
axis square
subplot(122); axis square; imagesc(delta, fliplr(beta), mean_BER_naive./lambda_contours, clim2); hold on
colorbar; set(gca, 'YTickLabel', flipud(get(gca, 'YTickLabel')));   % Flip beta labels
axis square

% Add equal-lambda contour lines to the plot
subplot(121); 
[con_out, con_handle] = contour(delta, fliplr(beta), lambda_contours, ...
    contour_values, 'w-'); 
clabel(con_out, con_handle, 'Color', 'white'); hold off
subplot(122); 
[con_out, con_handle] = contour(delta, fliplr(beta), lambda_contours, ...
    contour_values, 'w-'); 
clabel(con_out, con_handle, 'Color', 'white'); hold off

% Add x- and y-axis labels and a title string
for i = 1:2
    subplot(1,2,i)
    ylabel('\beta (E[K]/M)     (More active coefficients)   \rightarrow');
    xlabel('\delta (M/N)    (More measurements)   \rightarrow')
    title_string = [algs{i+1} ' Normalized Support Error Rate [dB]'];
    title(title_string)
%     text_string = ['N = ' num2str(N) ', T = ' num2str(T) ...
%         ', p_{01} = ' num2str(pz1) ', \alpha = ' num2str(alpha) ...
%         ', SNR_m = ' num2str(SNRmdB) 'dB, N_{trials} = ' num2str(n)];
%     text(0.5, 0.10, text_string, 'Color', 'White', 'HorizontalAlignment', ...
%         'Center', 'FontSize', 5)
end