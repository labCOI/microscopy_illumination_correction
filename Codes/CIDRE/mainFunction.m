close all; clear; clc;
dataset = '49-01';
tic

source_path = ['C:\Users\Sama\Desktop\Master\Datasets\Empty-Zero\' dataset '/'];
source_files = dir([source_path '*.jpg']);
source_files = source_files(81:90);

path_r = ['C:\Users\Sama\Desktop\Master\mfiles\Master Thesis\CIDRE results\' dataset '\s_r\'];
path_g = ['C:\Users\Sama\Desktop\Master\mfiles\Master Thesis\CIDRE results\' dataset '\s_g\'];
path_b = ['C:\Users\Sama\Desktop\Master\mfiles\Master Thesis\CIDRE results\' dataset '\s_b\'];

for i = 1:10%length(source_files)
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

path = ['C:\Users\Sama\Desktop\Master\mfiles\Master Thesis\CIDRE results\' dataset '\'];
for i = 1:length(I_R_Corr_files)
    I_R = imread([I_R_Corr_dir I_R_Corr_files(i).name]);
    I_G = imread([I_G_Corr_dir I_G_Corr_files(i).name]);
    I_B = imread([I_B_Corr_dir I_B_Corr_files(i).name]);
    
    I_RGB = cat(3,I_R,I_G,I_B); 
    imwrite(I_RGB, [path I_R_Corr_files(i).name]);
end

toc