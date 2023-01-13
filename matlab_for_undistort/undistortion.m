clear;
clc;
addpath(genpath(pwd));

filepath_image = '/home/action/code/PS_49.jpg';
load("calibration_20230112_0111_selected_reorder_mini.mat");

I = imread(filepath_image); 
J1 = undistortImage(I,cameraParams);
figure; 
imshowpair(I,J1,'montage');
title('Original Image (left) vs. Corrected Image (right)');

J2 = undistortImage(I,cameraParams,'OutputView','full');
% J2 = undistortImage(I,cameraParams,'OutputView','same');
figure; 
imshow(J2);
title('Full Output View');
