clc;
clear;

folderpath_input = "/home/action/Pictures/calibration/0111_selected";
folderpath_output = "/home/action/Pictures/calibration/0111_selected_reorder";

Dir = dir(folderpath_input);

for i = 3:size(Dir,1)
    img = imread(fullfile(folderpath_input,Dir(i,1).name));
    str = sprintf("%s/%06d.jpg",folderpath_output,i-2);
    imwrite(img,str);
end