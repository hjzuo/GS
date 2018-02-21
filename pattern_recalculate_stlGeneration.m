%https://www.mathworks.com/matlabcentral/answers/245368-gerchberg-saxton-algorithm
% model=zeros(250);
% model(5:20, 20:121)=1;
% model(24:140, 40:50)=1;
% model(24:140, 85:97)=1;
% figure(1);imagesc(model); colormap(gray)
% DS=model; 
% model=2*pi.*rand(150)-pi; 
% pi_signal=double(imread('pi.bmp'));
% pi_signal=rgb2gray(pi_signal);
% pi_signal=imresize(pi_signal, [150 150]);
% j=fftshift(fft2(pi_signal));
% j=abs(j)/(max(max(abs(j))));
% j=imadjust(abs(j), [0; 0.01], [0; 1]);
% % 
% for k=1:1:200; 
%     step1=fftshift(fft2(model));
%     phase=angle(step1); 
%     Gu=medfilt2(j).*exp(i*phase);
%     gx=fft2(fftshift(Gu));
%     model=pi_signal.*exp(i*angle(gx));
%     figure(7),imagesc(abs(gx)); title(num2str(k));
% end
% %https://www.mathworks.com/matlabcentral/answers/245368-gerchberg-saxton-algorithm
% A = fftshift(ifft2(fftshift(Target)));
% for i=1:25
%   B = abs(Source) .* exp(1i*angle(A));
%   C = fftshift(fft2(fftshift(B)));
%   D = abs(Target) .* exp(1i*angle(C));
%   A = fftshift(ifft2(fftshift(D)));
%     imagesc(abs(C)) %Present current pattern
%     title(sprintf('%d',i));
%     pause(0.5)
% end
% %Before running the code, make sure 'Source' contains your input beam, for example:
% 
% Source = exp(-1/2*(xx0.^2+yy0.^2)/sigma^2);
% %And 'Target' contains your requested pattern.
% 
% %The phase mask can be presented at the end of the for loop:
% % imagesc(angle(A)) 
% %https://www.mathworks.com/matlabcentral/answers/374327-gradient-descent-method-with-gerchberg-saxton-algorithm
% clc
% close all
% clear all
% model=zeros(150);
% model(5:20, 20:121)=1;
% model(24:140, 40:50)=1;
% model(24:140, 85:97)=1;
% figure(1);imagesc(model); colormap(gray)
% DS=model; 
% model=2*pi.*rand(150)-pi; 
% pi_signal1=double(imread('pi.bmp'));
% pi_signal=rgb2gray(pi_signal1);
% pi_signal=imresize(pi_signal, [150 150]);
% j=fftshift(fft2(pi_signal));
% j=abs(j)/(max(max(abs(j))));
% j=imadjust(abs(j), [0; 0.01], [0; 1]);
% for k=1:1:150; 
%   step1=fftshift(fft2(model));%
%   phase=angle(step1); 
%   Gu=medfilt2(j).*exp(i*phase);
%   gx=fft2(fftshift(Gu));
%   model=pi_signal.*exp(i*angle(gx));
%   figure(7),imagesc(abs(gx)); colormap(gray), title(num2str(k));
%   err(k)=-sqrt(sum(abs(gx(1:150,1))-pi_signal1(1:150,1))^2/k);
%   t(k) = 10*k;  %simulation time, where dt is time for one increment of loop 
% end
% figure(3)
% plot(t,err);
%% reference for inversal patern generation:
% https://www.mathworks.com/matlabcentral/fileexchange/65979-gerchberg-saxton-algorithm?s_tid=answers_rc2-2_p5_MLT
clear all
original=double(imread('phase_to_nasa.bmp'));
signal = original;

signal = signal-128;
signal = signal * pi / max(max(signal));
s=150;
input_intensity=ones([s,s]) ;
B = abs(input_intensity) .* exp(1i*signal);
C = fftshift(fft2(fftshift(B)));
% imagesc(abs(C)) 


h_max = 2; %  unit m
stl_array = zeros([s, s]);
for i = 1 : s
    for j = 1 : s
        stl_array(i,j) = (original(i, j)/ 256)* h_max;
        
    end
end
[X,Y] = deal(1:s); 
Z = stl_array;
SOLID_FV = surf2solid(X,Y,Z,'elevation',0);
stlwrite('test.stl',SOLID_FV)        % Save to binary .stl

