%%%This function is to generate measurement matrix using function handle

function y = Fr(x,W,mask,norms)

x = x./norms;    % Pre-multiply x by the inverse of the norm of the columns of Fr

n = round(sqrt(length(x)));  %square root of length of signal

x=W'*reshape(x,n,n)*W;       %Do 2D inverse wavelet transform

yc = 1/n*fft2(reshape(x,n,n));  %Do 2D FFT


%%%%%pack 2D complex signal to a 1D real signal

if mask(1,1)==1
   
   mask(1,1)=0;
   
   y=[yc(1,1);real(yc(find(mask==1)));imag(yc(find(mask==1)))];
else
    
   y=[real(yc(find(mask==1)));imag(yc(find(mask==1)))];
    
   
end
   


