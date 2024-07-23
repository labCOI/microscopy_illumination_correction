%% shading correction using pixel-based algorithm according to:
% Tak, Yoon-Oh, et al. "Simple shading correction method for brightfield
% whole slide imaging." Sensors 20.11 (2020): 3084.

close all; clear; clc;
tic

% File Directory Initialization
directory = 'C:\Hasti Shabani\2. Projects\1. Whole Slide Imaging\Data_WSI\TAK\051-04-80\';
files = dir(directory);
num_files = numel(files);
files = files(3:num_files);
num_Im = numel(files);

% Read first Image to set the size
I = imread(strcat(directory,files(1).name));
[M N num_ch] = size(I);

for i = 1:num_ch
    
    Is = zeros(M, N, num_Im);
    % RGB channels separation
    for j = 1:num_Im
        I = imread(strcat(directory,files(j).name));
        I = im2double(I);
        Is(:,:,j) = I(:,:,i);
    end
    % reduce the size of image to speed up
    %Is = imresize(Is, 0.25);
    Is_sort = sort(Is, 3);
    clear I
    LCoV = zeros(1,num_Im); % Local coefficient of variation
    
    for k = 1:num_Im
        I = Is_sort(:,:,k);
        win_size = 5;
        kernel = ones(win_size)/(win_size^2);
        slidingMean = conv2(I,kernel,'same');
        slidingStd = sqrt( conv2(I.^2,kernel,'same') - slidingMean.^2 );
        LCoV(k) = sum(sum(slidingStd./slidingMean));
    end
    [Min, inds] = min(LCoV);
    m(i) = inds;
    shading = Is_sort(:,:,inds);
    flatfield(:,:,i) = shading ./ mean(shading(:));
end

path_to_save = 'C:\Hasti Shabani\2. Projects\1. Whole Slide Imaging\Data_WSI\TAK\051-04-80\';
for i = 1:num_Im
    I = imread(strcat(directory,files(i).name));
    I = im2double(I);
    I = I ./ flatfield;
    imwrite(I, [path_to_save 'TAK_'  files(i).name]);
end

toc

%%% visualize_flatfield %%%

totalRGB_flatfield_n = (flatfield - min(flatfield(:))) ./ (max(flatfield(:)) - min(flatfield(:));

figure, imshow(totalRGB_flatfield_n); colorbar;

imwrite(totalRGB_flatfield_n, [path_to_save 'TAK_flatfield_RGBminmaxscale.jpg'])
