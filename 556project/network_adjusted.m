% load the dataset
d = load('data.mat');


trainX  = d.trainX;
testX = d.testX;
trainY = d.trainY;
testY = d.testY;
%plot one training image
imshow(testX(:,:,:,1));

adjust_trainX = double(trainX) / 255. - 0.5;
adjust_testX = double(testX) / 255. - 0.5;
imshow(cast(round((adjust_testX(:,:,:,1)+0.5)*255),'uint8'));
% define the cnn
layers = [
    imageInputLayer([28 28 1])
	
    convolution2dLayer(3,32,'Padding',1)
    %batchNormalizationLayer
    reluLayer
	
    %maxPooling2dLayer(2,'Stride',2)
	
    convolution2dLayer(3,32,'Padding',1)
    %batchNormalizationLayer
    reluLayer
	
    maxPooling2dLayer(2,'Stride',2)
	
    convolution2dLayer(3,64,'Padding',1)
    %batchNormalizationLayer
    reluLayer
	
    convolution2dLayer(3,64,'Padding',1)
    %batchNormalizationLayer
    reluLayer
    
    maxPooling2dLayer(2,'Stride',2)
	
    
    fullyConnectedLayer(200)
    reluLayer
    fullyConnectedLayer(200)
    reluLayer
    fullyConnectedLayer(10)
    softmaxLayer
    classificationLayer];

% define the training miniBatch size (batch gradient descent), and other
% training options
miniBatchSize = 8192;
options = trainingOptions( 'adam',...
    'MiniBatchSize', miniBatchSize,...
    'Plots', 'training-progress');


%train the network
net = trainNetwork(adjust_trainX, categorical(trainY), layers, options);	

% save all the variables
save net_adjusted;



