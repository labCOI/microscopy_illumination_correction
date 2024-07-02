close all; clear; clc;

tic
images_dir = 'C:\Users\Sama\Desktop\Master\Datasets\Empty-Zero\33-03/';
files = dir([images_dir '*.jpg']);
files = files(81:90);
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


% Estimation of shading model.
shading_model_R = BaSiC(I_R);
clear I_R
shading_model_G = BaSiC(I_G);
clear I_G
shading_model_B = BaSiC(I_B);
clear I_B

totalRGBshading = cat(3, shading_model_R, shading_model_G, shading_model_B);

path = 'C:\Users\Sama\Desktop\Master\mfiles\Master Thesis\BaSiC-master\33-03\10\80-89\';
%totalRGBshading_1 = imread('C:\Users\Sama\Desktop\Master\mfiles\Master Thesis\BaSiC-master\194-01-70\revise\toatalRGBshading.jpg');
for i = 1:length(files)
    I = imread([images_dir files(i).name]);
    I = im2double(I);
    totalRGBshading = im2double(totalRGBshading);
    I = I ./ totalRGBshading;
    imwrite(I, [path 'BaSiCimg_' num2str(i) '.jpg']);
end

toc