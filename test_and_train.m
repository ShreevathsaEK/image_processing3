
% Download CIFAR-10 dataset if not already downloaded
url = 'https://www.cs.toronto.edu/~kriz/cifar-10-matlab.tar.gz';
downloadFolder = tempdir;
filename = fullfile(downloadFolder, 'cifar-10-matlab.tar.gz');
dataFolder = fullfile(downloadFolder, 'cifar-10-batches-mat');
if ~exist(dataFolder, 'dir')
    fprintf('Downloading CIFAR-10 dataset (175 MB)... ');
    websave(filename, url);
    untar(filename, downloadFolder);
    fprintf('Done.\n')
end

% Load CIFAR-10 dataset
[XTrain, YTrain, XTest, YTest] = loadData(downloadFolder);

% Normalize pixel values to [0, 1]
XTrain = double(XTrain) / 255;
XTest = double(XTest) / 255;

% Reshape the training and test data to be 2D
XTrain = reshape(XTrain, [], size(XTrain, 4));
XTest = reshape(XTest, [], size(XTest, 4));

% Define k-NN parameters
k = 3; 

% Train the k-NN model
knnModel = fitcknn(XTrain', YTrain, 'NumNeighbors', k);

% Classify test data using the trained k-NN model
YTestPredicted = predict(knnModel, XTest');

% Calculate and display accuracy
accuracy = sum(YTestPredicted == YTest) / numel(YTest) * 100;
fprintf('Accuracy: %.2f%%\n', accuracy);

% Load and preprocess a new image (adjust as needed)
newImage = imread('DOG6.jpg');
newImage = imresize(newImage, [32, 32]);
newImage = double(newImage(:)') / 255; % Reshape and normalize

% Make a prediction
predictedLabel = predict(knnModel, newImage);
fprintf('Predicted Label: %s\n', char(predictedLabel));
