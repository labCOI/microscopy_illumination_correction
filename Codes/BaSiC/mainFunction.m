close all; clear; clc;

tic
images_dir = 'C:\Users\Sama\Desktop\Master\Datasets\36-01/';
files = dir([images_dir '*.jpg']);
I = imread(strcat(images_dir,files(1).name));
[M, N, num_ch] = size(I);

 I_R = zeros(M, N,num_ch);
 I_G = zeros(M, N,num_ch);
 I_B = zeros(M, N,num_ch);
 
  
 for i = 1:length(files)
    I = imread([images_dir files(i).name]);
    
    I_R(:,:,i) = I(:,:,1);
    I_G(:,:,i) = I(:,:,2);
    I_B(:,:,i) = I(:,:,3);
end


% Estimation of shading model using BaSiC
shading_model_R = BaSiC(I_R);
clear I_R
shading_model_G = BaSiC(I_G);
clear I_G
shading_model_B = BaSiC(I_B);
clear I_B

totalRGBshading = cat(3, shading_model_R, shading_model_G, shading_model_B);

path_to_save = 'C:\Users\Sama\Desktop\Master\mfiles\Master Thesis\BaSiC-master\36-01\';

for i = 1:length(files)
    I = imread([images_dir files(i).name]);
    I = im2double(I);
    totalRGBshading = im2double(totalRGBshading);
    I = I ./ totalRGBshading;
    imwrite(I, [path_to_save 'BaSiC_'  files(i).name]);
end

toc

%%% visualize_flatfield %%%

totalRGBshading = cat(3, shading_model_R, shading_model_G, shading_model_B);

totalRGB_flatfield_n = (totalRGBshading - min(totalRGBshading(:))) ./ (max(totalRGBshading(:)) - min(totalRGBshading(:)));

%figure, imshow(totalRGB_flatfield_n); %colorbar;

imwrite(totalRGB_flatfield_n, [path_to_save 'BaSiC_flatfield_RGBminmaxscale.jpg'])
