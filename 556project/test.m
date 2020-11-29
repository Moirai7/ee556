load net;

%this is the network outputs when the inputs are the test images
outputs = predict(net, testX);

%this will give the predicted labels 
predLabelsTest = net.classify(testX);
%this gives the test accuracy
accuracy = sum(predLabelsTest == categorical(transpose(testY))) / numel(testY);

%plot a sample test image 
imshow(testX(:,:,:,1));


