%%%%This function is to implement A' 

function x=Frt(y,W,mask,norms)

K = length(y);  %length of measurement vector
N = size(mask); %size of signal

%%%%convert 1D real signal to 2D complex signal
y2D = zeros(N);
if rem(K,2)==1;
   ytmp=[y(1,1);y(2:(K+1)/2)+y((K+3)/2:end)*sqrt(-1)];
   
   y2D(find(mask==1))=ytmp;
    
else
    
    ytmp=y(1:K/2)+y(K/2+1:end)*sqrt(-1);
    
    y2D(find(mask==1))=ytmp;
end


x=real(N(1)*ifft2(y2D)); %Do 2D inverse FFT


x=W*x*W';  %Do 2D wavelet transform

x=x(:);    %pack 2D  signal to 1D signal

x = x./norms;   % Post-multiply by the inverse of the column norms of Fr