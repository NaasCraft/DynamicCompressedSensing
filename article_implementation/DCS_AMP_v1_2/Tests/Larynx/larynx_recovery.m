% LARYNX_RECOVERY       This script will recover a larynx sequence from
% undersampled Fourier measurements using several different recovery
% algorithms.  Most of the code pertaining to experimental setup is
% courtesy of Wei Lu and Namrata Vaswani of Iowa State University.

%% Initialize experiment

% Reset Matlab's state
clear
randn('state',0)
rand('state',0)

% Load the image sequence and obtain sizing information
load larynxsequence  
N0=size(larynximage);
N=N0(1:2);              % Image dimensions
seqlen=N0(3);           % # of timesteps
m=prod(N);              % Dimension of vectorized image

for seq = 1:seqlen
    larynximage(:,:,seq) = larynximage(:,:,seq);
end

%%%%generate n=0.5m sampling mask
% We want to sample more in low frequency regions. genSampling gives 
% more samples in central regions(low frequency regions for fftshifted 2D
% Fourier transform)
[pdf,val] = genPDF(N,2,0.5,2,0,0);  
[mask1,stat,NN] = genSampling(pdf,100,10);
mask1=fftshift(mask1);            

%%%%%generate n=0.16m sampling mask
% genSampling only can generate the r   equired mask for n>=0.25m samples.
% for n<0.25m, we can multiply another mask whose distribution is only 
% generated on the half central regions.
[pdf,val] = genPDF(N,2,0.25,2,0,0);
[mask2,stat,NN] = genSampling(pdf,100,10);
mask2=fftshift(mask2);

maskreduce=ones(N);            %for n<0.25m mask 

[pdf,val] = genPDF(N/2,2,0.30,2,0,0);   %Change the samples for maskreduce to reduce samples for mask2
[maskreduce(N(1)/4+1:N(1)/4*3,N(2)/4+1:N(2)/4*3),stat,NN] = genSampling(pdf,100,10);
maskreduce=fftshift(maskreduce);
mask2=mask2.*maskreduce;

AFr= dftmtx(N(1))/sqrt(N(1));

%%%%%generate wavelet transform matrix
L = 2;                       	%Decomposition levels
wname = 'db4';                  % Wavelet type
dwtmode('per'); 
X=eye(N);
W=zeros(N);                      %wavelet transform matrix for 2D transform
for p=1:N(1)
       [W(:,p),L2] = wavedec(X(:,p),L,'db4'); % wavelet base is Daubecies 4
end

% This is the vectorized version of the underlying compressible signal we
% are trying to recover, i.e., the "ground truth"
for seq=1:seqlen
    larynxseq(:,seq) = wavedec2(larynximage(:,:,seq), L, wname)';
end

% -----------------------------------
% Two-level decomposition
% -----------------------------------
%%%% z is the wavelet coefficient, W is wavelet matrix, mask1 is sampling
%%%% mask. Measurement matrix is Partial Fourier matrix times inverse Wavelet matrix.
%A1 is to do inverse 2D wavelet on z first and generate 2D
%%%% undersasmpled Fourier transform by applying mask1.
A1 = @(z) Fr(z, W, mask1, ones(m,1));        %A1

%%%% A1' which is an inverse operation of A1
At1 = @(z) Frt(z, W, mask1, ones(m,1));      %A1'

A2 = @(z) Fr(z, W, mask2, ones(m,1));        %A2

At2 = @(z) Frt(z, W, mask2, ones(m,1));      %A2'


% This is the vectorized version of the underlying compressible signal we
% are trying to recover, i.e., the "ground truth"
for seq=1:seqlen
    larynxseq(:,seq)=reshape(W*larynximage(:,:,seq)*W',prod(N),1);
end

%%%%%geneate measurements,at t=1 using n0 measurements,at t>1 using n
%%%%%measurements
for seq=1:seqlen
%         % Non-uniform subsampling
%         if seq==1       % Acquire more measurements initially
%              y{seq}=A1(larynxseq(:,seq));
%         else
%              y{seq}=A2(larynxseq(:,seq));
%         end

    % Uniform subsampling
    y{seq}=A2(larynxseq(:,seq));
end

% Create a cell array that contains the needed function handles, for use by
% multi-timestep BP
A_mbp = cell(1,seqlen);
for seq = 1:seqlen
%         % Non-uniform subsampling
%         if seq == 1
%             A_mbp{seq} = @(x,mode) handle_conversion(x, A1, At1, mode);
%         else
%             A_mbp{seq} = @(x,mode) handle_conversion(x, A2, At2, mode);
%         end

    % Uniform subsampling
    A_mbp{seq} = @(x,mode) handle_conversion(x, A2, At2, mode);
end        


%% Run independent ell-1 minimization recovery procedure


etime_cs = tic;     % Start stopwatch
for seq=1:seqlen
       if seq==1
%            A=A1;
%            At=At1;
%            mask=mask1;
            % If same sampling rate for all timesteps
            A=A2;
            At=At2;
            mask=mask2;
       else
           A=A2;
           At=At2;
           mask=mask2;
       end
        
       %%%%Initial estimation
         x0=At(y{seq});
         
       %%%%Do CS
         x = l1eq_pd(x0, A,At, y{seq}, 1e-3,25,1e-10,800); 
         
         xhatcs(:,seq)=x;  %Reconstructed Wavelet Coeff.
         
         xhatcsimage(:,:,seq)=W'*reshape(xhatcs(:,seq),N)*W; %Reconstructed image
         
             
    %%%%Compute rMSE       
    error_cs(seq)=norm(xhatcs(:,seq)-larynxseq(:,seq))/norm(larynxseq(:,seq));
end
etime_cs = toc(etime_cs);   % Record elapsed time for independent ell-1 method


%% Run the modified-CS recovery algorithm

thresholdmodcs=10;      %Threshold for modified-CS to obtain support

etime_modcs = tic;      % Start stopwatch
for seq=1:seqlen
        if seq==1
%             % If higher sampling rate at first timestep
%             A=A1;
%             At=At1;
%             mask=mask1;
            % If same sampling rate for all timesteps
            A=A2;
            At=At2;
            mask=mask2;
            
            x0= xhatcs(:,seq); % Initial estimate for t=1 is CS reconstruction at t=1;
       else
           A=A2;
           At=At2;
           mask=mask2;
           x0=xhatmodcs(:,seq-1); %Intial estimate for t>1 is ModCS reconstruction at t-1
           
        end
       
       T=find(abs(x0)>thresholdmodcs);  %Kown part of support
       
        P=ones(length(x0),1);   %P is a vector used as a parameter to set known support

        P(T)=0;                 %P_{T}=0  P_{T^c}=1;
        
        %%%%Run Modified CS function
        x = l1eqmodcs_large(x0, A,At, y{seq}, P,mask,1e-3,20,1e-9,1200); 
        
        xhatmodcs(:,seq)=x;  %Reconstructed Wavelet Coeff.
         
        xhatmodcsimage(:,:,seq)=W'*reshape(xhatmodcs(:,seq),N)*W; %Reconstructed image
        
        
        %%%%Compute rMSE 
        error_modcs(seq)=norm(xhatmodcs(:,seq)-larynxseq(:,seq))/norm(larynxseq(:,seq))
        
end
etime_modcs = toc(etime_modcs);     % Record elapsed time for modified CS


%% Run DCS-AMP in filtering mode

path(path, '../../Functions')
path(path, '../../ClassDefs')

% See if we can't split coeffs in a way that gives good parameter settings
% Approx coeffs
active_thresh = .25;
larynxseq2d = reshape(larynxseq(:,1), N(1), N(1));
approx_coeffs = larynxseq2d(1:64, 1:64);
approx_coeffs = approx_coeffs(:);
detail_SW = larynxseq2d(65:N(1), 1:64);
detail_NE = larynxseq2d(1:64, 65:N(2));
detail_SE = larynxseq2d(65:N(1), 65:N(2));
detail_coeffs = [detail_SW(:); detail_NE(:); detail_SE(:)];

index_mask = zeros(N(1),N(2));
index_mask(1:64,1:64) = 1;
approx_coeff_group = find(index_mask == 1);     % Indices of approx coeffs
detail_coeff_group = find(not(index_mask) == 1);    % Indices of detail coeffs

% Support parameters
lambda = .01*ones(prod(N),1);
lambda(approx_coeff_group) = 0.99;     % High activity prob. for approx coeffs

% Amplitude parameters
kappa(approx_coeff_group,1) = 1e4;
kappa(detail_coeff_group,1) = 1e3;

% Create object containing all DCS model params
Params = DCSModelParams(lambda, 0.01*ones(prod(N),1), zeros(prod(N),1), ...
    kappa, .05*ones(prod(N),1), 1e-2);

Options.smooth_iter = -1;       % # of fwd/bwd passes (-1 to filter)
Options.eq_iter = 10;           % # of inner AMP iterations
Options.alg = 2;                % AMP
Options.update = 1;             % Update hyperparameters during execution
Options.upd_groups = [{approx_coeff_group}, {detail_coeff_group}];
Options.verbose = 1;            % Print msgs

% Execute multi-timestep BP
etime_mbp = tic;        % Start stopwatch
[x_hat, v_hat, lambda_hat] = sp_multi_frame_fxn(y, A_mbp, Params, Options);
% [x_hat, v_hat, lambda_hat] = sp_parallel_frame_fxn(y, A_mbp, Params, Options);
etime_mbp = toc(etime_mbp);     % Record elapsed time for multi-timestep BP

% Compute error
for seq = 1:seqlen
    error_mbp(seq) = norm(x_hat{seq} - larynxseq(:,seq)) / ...  
        norm(larynxseq(:,seq));
end
mean(error_mbp)
disp(['DCS-AMP TNMSE: ' num2str(10*log10(mean(error_mbp.^2)))])

% Convert recovered wavelet solution back to image solution
for seq = 1:seqlen
    xhatdcsimage(:,:,seq) = W'*reshape(x_hat{seq}, N(1), N(2))*W;
end

figure(1); imagesc(xhatdcsimage(:,:,2)); colormap('gray'); colorbar
figure(2); imagesc(larynximage(:,:,2)); colormap('gray'); colorbar
% figure(1); imagesc(waverec2(x_hat{1}, S, wname)); colorbar
% title(['D-AMP Recovery | Smooth Iters: ' num2str(Options.smooth_iter)])
% figure(2); imagesc(waverec2(larynxseq(:,1), S, wname)); colorbar
% title('True Image')


%% Plot several frames for the ground truth and recoveries

frames = [1, 2, 5, 10];     % List of frames to plot for each method
Nframes = numel(frames);

comparison_image = max(larynximage(:)) * ones(4*N(1) + 6, Nframes*N(2));

for f = 1:Nframes
    % Plot ground truth in top row
    comparison_image(1:N(1),(f-1)*N(2)+1:f*N(2)) = larynximage(:,:,f);
    % Plot basis pursuit recovery in second row
    comparison_image(N(1)+3:2*N(1)+2,(f-1)*N(2)+1:f*N(2)) = ...
        xhatcsimage(:,:,f);
    % Plot Modified-CS solution in third row
    comparison_image(2*N(1)+5:3*N(1)+4,(f-1)*N(2)+1:f*N(2)) = ...
        xhatmodcsimage(:,:,f);
    % Plot DCS-AMP solution in fourth row
    comparison_image(3*N(1)+7:4*N(1)+6,(f-1)*N(2)+1:f*N(2)) = ...
        xhatdcsimage(:,:,f);
end

% Plot the image
colormap('gray')
imagesc(comparison_image, [min(larynximage(:)), max(larynximage(:))])
axis off