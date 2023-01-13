% Auto-generated by cameraCalibrator app on 12-Jan-2023
%-------------------------------------------------------


% Define images to process
imageFileNames = {'/home/action/Pictures/calibration/0111_selected_reorder_mini/000001.jpg',...
    '/home/action/Pictures/calibration/0111_selected_reorder_mini/000002.jpg',...
    '/home/action/Pictures/calibration/0111_selected_reorder_mini/000003.jpg',...
    '/home/action/Pictures/calibration/0111_selected_reorder_mini/000004.jpg',...
    '/home/action/Pictures/calibration/0111_selected_reorder_mini/000007.jpg',...
    '/home/action/Pictures/calibration/0111_selected_reorder_mini/000010.jpg',...
    '/home/action/Pictures/calibration/0111_selected_reorder_mini/000011.jpg',...
    '/home/action/Pictures/calibration/0111_selected_reorder_mini/000015.jpg',...
    '/home/action/Pictures/calibration/0111_selected_reorder_mini/000016.jpg',...
    '/home/action/Pictures/calibration/0111_selected_reorder_mini/000018.jpg',...
    '/home/action/Pictures/calibration/0111_selected_reorder_mini/000019.jpg',...
    '/home/action/Pictures/calibration/0111_selected_reorder_mini/000020.jpg',...
    '/home/action/Pictures/calibration/0111_selected_reorder_mini/000021.jpg',...
    '/home/action/Pictures/calibration/0111_selected_reorder_mini/000022.jpg',...
    '/home/action/Pictures/calibration/0111_selected_reorder_mini/000023.jpg',...
    '/home/action/Pictures/calibration/0111_selected_reorder_mini/000026.jpg',...
    '/home/action/Pictures/calibration/0111_selected_reorder_mini/000031.jpg',...
    '/home/action/Pictures/calibration/0111_selected_reorder_mini/000062.jpg',...
    '/home/action/Pictures/calibration/0111_selected_reorder_mini/000064.jpg',...
    '/home/action/Pictures/calibration/0111_selected_reorder_mini/000068.jpg',...
    '/home/action/Pictures/calibration/0111_selected_reorder_mini/000069.jpg',...
    '/home/action/Pictures/calibration/0111_selected_reorder_mini/000073.jpg',...
    '/home/action/Pictures/calibration/0111_selected_reorder_mini/000074.jpg',...
    '/home/action/Pictures/calibration/0111_selected_reorder_mini/000075.jpg',...
    '/home/action/Pictures/calibration/0111_selected_reorder_mini/000080.jpg',...
    '/home/action/Pictures/calibration/0111_selected_reorder_mini/000081.jpg',...
    };
% Detect calibration pattern in images
detector = vision.calibration.monocular.CheckerboardDetector();
[imagePoints, imagesUsed] = detectPatternPoints(detector, imageFileNames, 'HighDistortion', true);
imageFileNames = imageFileNames(imagesUsed);

% Read the first image to obtain image size
originalImage = imread(imageFileNames{1});
[mrows, ncols, ~] = size(originalImage);

% Generate world coordinates for the planar pattern keypoints
squareSize = 60;  % in units of 'millimeters'
worldPoints = generateWorldPoints(detector, 'SquareSize', squareSize);

% Calibrate the camera
[cameraParams, imagesUsed, estimationErrors] = estimateCameraParameters(imagePoints, worldPoints, ...
    'EstimateSkew', false, 'EstimateTangentialDistortion', true, ...
    'NumRadialDistortionCoefficients', 3, 'WorldUnits', 'millimeters', ...
    'InitialIntrinsicMatrix', [], 'InitialRadialDistortion', [], ...
    'ImageSize', [mrows, ncols]);

% View reprojection errors
h1=figure; showReprojectionErrors(cameraParams);

% Visualize pattern locations
h2=figure; showExtrinsics(cameraParams, 'CameraCentric');

% Display parameter estimation errors
displayErrors(estimationErrors, cameraParams);

% For example, you can use the calibration data to remove effects of lens distortion.
undistortedImage = undistortImage(originalImage, cameraParams);

% See additional examples of how to use the calibration data.  At the prompt type:
% showdemo('MeasuringPlanarObjectsExample')
% showdemo('StructureFromMotionExample')