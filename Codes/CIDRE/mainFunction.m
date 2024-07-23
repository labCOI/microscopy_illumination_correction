close all; clear; clc;
dataset = '49-01';
tic

source_path = ['C:\Users\Sama\Desktop\Master\Datasets\' dataset '/'];
source_files = dir([source_path '*.jpg']);

path_r = ['C:\Users\Sama\Desktop\Master\mfiles\Master Thesis\CIDRE results\' dataset '\s_r\'];
path_g = ['C:\Users\Sama\Desktop\Master\mfiles\Master Thesis\CIDRE results\' dataset '\s_g\'];
path_b = ['C:\Users\Sama\Desktop\Master\mfiles\Master Thesis\CIDRE results\' dataset '\s_b\'];

for i = 1:length(source_files)
    img = imread([source_path source_files(i).name]);
    imwrite(img(:, :, 1), [path_r 'CIDREimg' source_files(i).name]);
    imwrite(img(:, :, 2), [path_g 'CIDREimg' source_files(i).name]);
    imwrite(img(:, :, 3), [path_b 'CIDREimg' source_files(i).name]);
end

addpath(genpath('C:\Users\Sama\Desktop\Master\mfiles\Master Thesis\cidre-master\0.1\matlab'));

Source_R = ['C:\Users\Sama\Desktop\Master\mfiles\Master Thesis\CIDRE results\' dataset '\s_r\*.jpg'];
Destination_R = ['C:\Users\Sama\Desktop\Master\mfiles\Master Thesis\CIDRE results\' dataset '\d_r'];
I_R_Corr = cidre(Source_R, Destination_R);


Source_G = ['C:\Users\Sama\Desktop\Master\mfiles\Master Thesis\CIDRE results\' dataset '\s_g\*.jpg'];
Destination_G = ['C:\Users\Sama\Desktop\Master\mfiles\Master Thesis\CIDRE results\' dataset '\d_g'];
I_G_Corr = cidre(Source_G, Destination_G);

Source_B = ['C:\Users\Sama\Desktop\Master\mfiles\Master Thesis\CIDRE results\' dataset '\s_b\*.jpg'];
Destination_B = ['C:\Users\Sama\Desktop\Master\mfiles\Master Thesis\CIDRE results\' dataset '\d_b'];
I_B_Corr = cidre(Source_B, Destination_B);


I_R_Corr_dir = ['C:\Users\Sama\Desktop\Master\mfiles\Master Thesis\CIDRE results\' dataset '\d_r/'];
I_R_Corr_files = dir([I_R_Corr_dir '*.jpg']);

I_G_Corr_dir = ['C:\Users\Sama\Desktop\Master\mfiles\Master Thesis\CIDRE results\' dataset '\d_g/'];
I_G_Corr_files = dir([I_G_Corr_dir '*.jpg']);

I_B_Corr_dir = ['C:\Users\Sama\Desktop\Master\mfiles\Master Thesis\CIDRE results\' dataset '\d_b/'];
I_B_Corr_files = dir([I_B_Corr_dir '*.jpg']);

path_to_save = ['C:\Users\Sama\Desktop\Master\mfiles\Master Thesis\CIDRE results\' dataset '\'];
for i = 1:length(I_R_Corr_files)
    I_R = imread([I_R_Corr_dir I_R_Corr_files(i).name]);
    I_G = imread([I_G_Corr_dir I_G_Corr_files(i).name]);
    I_B = imread([I_B_Corr_dir I_B_Corr_files(i).name]);
    
    I_RGB = cat(3,I_R,I_G,I_B); 
    imwrite(I_RGB, [path_to_save I_R_Corr_files(i).name]);
end

toc

%%% visualize_flatfield %%%
model_r = load(['C:\Users\Sama\Desktop\Master\mfiles\Master Thesis\CIDRE results\' dataset '\d_r\cidre_model.mat']);
f_r = model_r.model.v;
f_r = f_r ./ mean(f_r(:));

model_g = load(['C:\Users\Sama\Desktop\Master\mfiles\Master Thesis\CIDRE results\' dataset '\d_g\cidre_model.mat']);
f_g = model_g.model.v;
f_g = f_g ./ mean(f_g(:));

model_b = load(['C:\Users\Sama\Desktop\Master\mfiles\Master Thesis\CIDRE results\' dataset '\d_b\cidre_model.mat']);
f_b = model_b.model.v;
f_b = f_b ./ mean(f_b(:));

totalRGB_flatfield = cat(3, f_r, f_g, f_b);

totalRGB_flatfield_n = (totalRGB_flatfield - min(totalRGB_flatfield(:))) ./ (max(totalRGB_flatfield(:)) - min(totalRGB_flatfield(:)));

%figure, imshow(totalRGB_flatfield_n); %colorbar;
imwrite(totalRGB_flatfield_n, [path_to_save 'CIDRE_flatfield_RGBminmaxscale.jpg'])
