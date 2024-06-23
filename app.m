%% Set directories for training data
dirClosedEyes = "C:\Users\train\Closed_Eyes";
dirOpenEyes = "C:\Users\train\Open_Eyes";

%% Load images and create labels
[imagesClosed, labelsClosed] = loadImagesAndLabels(dirClosedEyes, 1); % 1 for Closed Eyes
[imagesOpen, labelsOpen] = loadImagesAndLabels(dirOpenEyes, 0); % 0 for Open Eyes

images = cat(4, imagesClosed, imagesOpen);
labels = cat(1, labelsClosed, labelsOpen);

%% Split data into training and testing sets
[idx, numTrain, x_train_eye, y_train_eye, x_test_eye, y_test_eye] = splitData(images, labels, 0.8);

%% Define CNN architecture for eye detection
layers_eye = createCNNLayers();

%% Set training options
options_eye = trainingOptions('sgdm', ...
    'InitialLearnRate', 0.01, ...
    'MaxEpochs', 10, ...
    'Shuffle', 'every-epoch', ...
    'ValidationData', {x_test_eye, y_test_eye}, ...
    'ValidationFrequency', 30, ...
    'Verbose', false, ...
    'Plots', 'training-progress');

%% Train the network
netEye = trainNetwork(x_train_eye, y_train_eye, layers_eye, options_eye);
save('netEyeModel.mat', 'netEye');

%% Evaluate the network
predictedLabels = classify(netEye, x_test_eye);
accuracy = sum(predictedLabels == y_test_eye) / numel(y_test_eye);
fprintf('Accuracy: %.2f%%\n', accuracy * 100);

%% Function to load images and labels
function [images, labels] = loadImagesAndLabels(imageDir, label)
    imageFiles = dir(fullfile(imageDir, '*.png'));
    numImages = length(imageFiles);
    images = zeros(64, 64, 1, numImages, 'uint8');
    labels = repmat(label, numImages, 1);

    for i = 1:numImages
        img = imread(fullfile(imageDir, imageFiles(i).name));
        img = imresize(img, [64, 64]);
        if size(img, 3) == 3
            img = rgb2gray(img);
        end
        images(:, :, 1, i) = img;
    end
end

function [idx, numTrain, x_train, y_train, x_test, y_test] = splitData(images, labels, trainRatio)
    idx = randperm(size(images, 4));
    numTrain = round(trainRatio * numel(labels));
    x_train = images(:, :, :, idx(1:numTrain));
    y_train = labels(idx(1:numTrain));
    x_test = images(:, :, :, idx(numTrain+1:end));
    y_test = labels(idx(numTrain+1:end));

    y_train = categorical(y_train);
    y_test = categorical(y_test);
end

%% Function to create CNN layers
function layers = createCNNLayers()
    layers = [
        imageInputLayer([64 64 1])
        convolution2dLayer(5, 32, 'Padding', 'same')
        batchNormalizationLayer
        reluLayer
        maxPooling2dLayer(2, 'Stride', 2)
        fullyConnectedLayer(2)
        softmaxLayer
        classificationLayer];
end

%% Set directories for yawning data
dirNoYawn = "C:\Users\train2\no_yawn";
dirYawn = "C:\Users\train2\yawn";

%% Load images and create labels
[imagesNoYawn, labelsNoYawn] = loadImagesAndLabels(dirNoYawn, 'no_yawn');
[imagesYawn, labelsYawn] = loadImagesAndLabels(dirYawn, 'yawn');

images = cat(4, imagesNoYawn, imagesYawn);
labels = [labelsNoYawn; labelsYawn];

%% Split data into training and testing sets
[idx, numTrain, x_train_yawn, y_train_yawn, x_test_yawn, y_test_yawn] = splitData(images, labels, 0.8);

%% Define CNN architecture for yawning detection
layers_yawn = createCNNLayers();

%% Set training options
options_yawn = trainingOptions('sgdm', ...
    'InitialLearnRate', 0.01, ...
    'MaxEpochs', 10, ...
    'Shuffle', 'every-epoch', ...
    'ValidationData', {x_test_yawn, y_test_yawn}, ...
    'ValidationFrequency', 30, ...
    'Verbose', true, ...
    'Plots', 'training-progress');

%% Train the network
netYawn = trainNetwork(x_train_yawn, y_train_yawn, layers_yawn, options_yawn);
save('netYawnModel.mat', 'netYawn');

%% Evaluate the network
predictedLabels = classify(netYawn, x_test_yawn);
accuracy = sum(predictedLabels == y_test_yawn) / numel(y_test_yawn);
fprintf('Accuracy: %.2f%%\n', accuracy * 100);

%% Function to detect blinking
function isBlinking = detectBlinking(image, blinkingModel)
    img = preprocessImage(image);
    label = classify(blinkingModel, img);
    isBlinking = (label == "closed");
end

%% Function to detect yawning
function isYawning = detectYawning(image, yawningModel)
    img = preprocessImage(image);
    label = classify(yawningModel, img);
    isYawning = (label == "yawn");
end

%% Function to preprocess images
function img = preprocessImage(image)
    img = imresize(image, [64, 64]);
    if size(img, 3) == 3
        img = rgb2gray(img);
    end
    img = reshape(img, [64, 64, 1, 1]);
end

%% Load trained models
load('netEyeModel.mat', 'netEye');
load('netYawnModel.mat', 'netYawn');

%% Set directory for drowsiness data
imageDir = "C:\Users\Drowsiness_Dataset\Drowsy";
imageFiles = dir(fullfile(imageDir, '*.png'));
images = cell(1, length(imageFiles));

for i = 1:length(imageFiles)
    images{i} = imread(fullfile(imageDir, imageFiles(i).name));
end

%% Eye and mouth detectors
eyeDetector = vision.CascadeObjectDetector('EyePairBig');
mouthDetector = vision.CascadeObjectDetector('Mouth');

closedEyesCount = 0;
yawningCount = 0;

%% Process each image
for i = 0:2000
    image = images{i};
    
    % Eye detection and classification
    bbox = step(eyeDetector, image);
    if ~isempty(bbox)
        croppedEyes = imcrop(image, bbox(1,:));
        if detectBlinking(croppedEyes, netEye)
            closedEyesCount = closedEyesCount + 1;
        end
    end
    
    % Mouth detection and yawning classification
    mouthBbox = step(mouthDetector, image);
    if ~isempty(mouthBbox)
        [~, largestBboxIndex] = max(mouthBbox(:,3).*mouthBbox(:,4));
        croppedMouth = imcrop(image, mouthBbox(largestBboxIndex,:));
        if detectYawning(croppedMouth, netYawn)
            yawningCount = yawningCount + 1;
        end
    end
end

%% Calculate drowsiness score
alpha = 0.6;
beta = 0.12;
drowsinessScore = alpha + beta * (closedEyesCount - 20) + log10(yawningCount + 1);
drowsinessScore = max(0, min(1, drowsinessScore));

%% Determine drowsiness state
threshold = 0.5;
if drowsinessScore > threshold
    disp('Driver is drowsy');
else
    disp('Driver is not drowsy');
end
