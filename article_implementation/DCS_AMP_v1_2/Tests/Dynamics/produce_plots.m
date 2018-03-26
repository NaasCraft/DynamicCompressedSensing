% PRODUCE_PLOTS     Produce plots for the dynamics test
%

%
% Coded by: Justin Ziniel, The Ohio State Univ.
% E-mail: ziniel.1@osu.edu
% Last change: 03/24/11
% Change summary: 
%		- Created (03/24/11; JAZ)
% Version 1.2
%

clear; clc

%% Load a results file

uiload

%% Produce the plots

clim = [-25, -10];    % Range for the colormap
algs = {'Support-aware Kalman smoother', 'DCS-AMP', ...
    'Independent MMSE', 'BG-AMP'};    % Algorithms
suffix = {'smooth', 'frame', 'iMMSE', 'naive'};

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Plot of the TNMSE for each approach
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Most likely some of the TNMSE's remain as NaNs, because all trials were
% not completed, thus average only over the n completed trials
for i = 1:N_pz1
    for j = 1:N_alpha
        mean_TNMSE_smooth_dB(i,j) = 10*log10(mean(TNMSE_smooth(i,j,1:n), 3));
        mean_TNMSE_frame_dB(i,j) = 10*log10(mean(TNMSE_frame(i,j,1:n), 3));
        mean_TNMSE_naive_dB(i,j) = 10*log10(mean(TNMSE_naive(i,j,1:n), 3));
        mean_TNMSE_iMMSE_dB(i,j) = 10*log10(mean(TNMSE_iMMSE(i,j,1:n), 3));
        mean_diffTNMSE_frame_dB(i,j) = 10*log10(mean(TNMSE_frame(i,j,1:n) - ...
            TNMSE_smooth(i,j,1:n), 3));
        mean_diffTNMSE_naive_dB(i,j) = 10*log10(mean(TNMSE_naive(i,j,1:n) - ...
            TNMSE_smooth(i,j,1:n), 3));
        mean_TBER_frame(i,j) = mean(TBER_frame(i,j,1:n), 3);
        mean_TBER_naive(i,j) = mean(TBER_naive(i,j,1:n), 3);
    end
end



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   Plot of the TNMSE of the algorithms
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Clear existing figure
figure(1); clf;

% Plot the TNMSE of 3 algorithms for each configuration
for a= 1:2
    ind = [1, 2, 4];    % Index of suffixes (which 3 algs to plot)
    subplot(1, 2, a);
    
%     % *** Use pcolor *** %
%     pcolor(alpha, pz1, eval(['mean_TNMSE_' suffix{ind(a)} '_dB']));
%     set(gca, 'XScale', 'log'); caxis(clim);
%     xlabel('\alpha'); ylabel('p_{01}')
    
%     % *** Use imagesc *** %
%     imagesc(log10(alpha), fliplr(2*pz1*100), eval(['mean_TNMSE_' suffix{ind(a)} '_dB']), clim);
%     xlabel('log_{10}(\alpha)'); ylabel('% Support Change'); colorbar
%     set(gca, 'YTickLabel', flipud(get(gca, 'YTickLabel')));
%     if a == 3, colorbar; end

    % *** Use contour *** %
    [con_out, con_handle] = contour(log10(alpha), 2*100*pz1, ...
        eval(['mean_TNMSE_' suffix{ind(a)} '_dB']));
    clabel(con_out, con_handle, 'Color', 'Black', 'fontsize', 18)
    xlabel('log_{10}(\alpha)'); ylabel('% Support Change  (2p_{01} \times 100)');
    
%     title(sprintf(['N = %d, M = %d, \\lambda = %1.2g, \\sigma^2 = %1.1g,' ...
%         ' SNR_m = %g dB'], N, M(1), lambda_0, kappa_0, SNRmdB))
    title([algs{ind(a)} ' TNMSE [dB]'])
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  Plot of proximity to genie smoother TNMSE
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Clear existing figure
figure(2); clf;

subplot(121); [C2, H2] = contour(log10(alpha), 2*100*pz1, ...
    mean_TNMSE_frame_dB - mean_TNMSE_smooth_dB, 0:0.5:4);
clh1 = clabel(C2, H2); hold on
subplot(122); [C3, H3] = contour(log10(alpha), 2*100*pz1, ...
    mean_TNMSE_naive_dB - mean_TNMSE_smooth_dB, 1:12);
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
    ind = [2, 4];    % Index of suffixes (which 2 algs to plot)
    subplot(1,2,i);
    xlabel('log_{10}(\alpha)'); ylabel('% Support Change  (2p_{01} \times 100)');
    title_string = [algs{ind(i)} ' TNMSE [dB] - SKS TNMSE [dB]'];
    title(title_string)
end


% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% % Plot of Support Error Rate (SER)
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 
% % Clear existing figure
% figure(3); clf;
% 
% % Plot the TSER of multi-timestep BP
% % % *** Use pcolor *** %
% % pcolor(alpha, pz1, mean_TBER_frame);
% % set(gca, 'XScale', 'log'); caxis(clim);
% % xlabel('\alpha'); ylabel('p_{01}')
% 
% % % *** Use imagesc *** %
% % imagesc(log10(alpha), pz1, mean_TBER_frame);
% % xlabel('log_{10}(\alpha)'); ylabel('p_{01}'); colorbar
% 
% % *** Use contour *** %
% [con_out, con_handle] = contour(log10(alpha), pz1, ...
%     mean_TBER_frame, 8);
% clabel(con_out, con_handle, 'Color', 'Black')
% xlabel('log_{10}(\alpha)'); ylabel('p_{01}');
% 
% title(sprintf(['N = %d, M = %d, \\lambda = %1.2g, \\sigma^2 = %1.1g,' ...
%     ' SNR_m = %g dB'], N, M(1), lambda_0, kappa_0, SNRmdB))