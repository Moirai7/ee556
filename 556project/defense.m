load net_adjusted;

path = 'res_adjust/adam/';

imgs = dir([path,'*ori.png']);
labels = [];
all_ori_imgs_mat = zeros(28,28,1,length(imgs),'double');
for i = 1:length(imgs)
    %all_ori_imgs_mat(:,:,1,i) = load([path imgs(i).name]).input;
    all_ori_imgs_mat(:,:,1,i) = imread([path imgs(i).name]);
    labels(end+1) = 0;
end


imgs = dir([path,'*adv.png']);
all_adv_imgs_mat = zeros(28,28,1,length(imgs));
for i = 1:length(imgs)
    %all_adv_imgs_mat(:,:,1,i) = load([path imgs(i).name]).adv;
    all_adv_imgs_mat(:,:,1,i) = imread([path imgs(i).name]);
    labels(end+1) = 1;
end

alls = cat(4,all_ori_imgs_mat,all_adv_imgs_mat);

cv = cvpartition(size(alls, 4),'HoldOut',0.3);
idx = cv.test;

train_X = alls(:,:,:,~idx);
test_X = alls(:,:,:,idx);
train_Y = labels(~idx);
test_Y = labels(idx);

%defense_net = defense_model_NN(train_X,train_Y,test_X,test_Y);
defense_model_rule(train_X,train_Y,test_X,test_Y);
function [] = defense_model_rule(train_X,train_Y,test_X,test_Y)
    %{
    %all_diff_imgs_mat = all_adv_imgs_mat - all_ori_imgs_mat;
    %all_diff_imgs_mat = reshape(all_diff_imgs_mat, 1,[]);
    %freq1=tabulate(all_diff_imgs_mat(:));
    all_ori_imgs_mat = reshape(all_ori_imgs_mat, 1,[]);
    freq2=tabulate(all_ori_imgs_mat(:));
    all_adv_imgs_mat = reshape(all_adv_imgs_mat, 1,[]);
    freq3=tabulate(all_adv_imgs_mat(:));

    %subplot(2,1,1);
    %h1 = histogram(all_diff_imgs_mat)
    %h1.BinLimits = [-0.45,0.45];

    subplot(2,1,1);
    h2 =histogram(all_ori_imgs_mat)
    %h2.BinLimits = [-0.45,0.45];
    %h2.BinLimits = [2,250];

    subplot(2,1,2);
    h3 =histogram(all_adv_imgs_mat)
    %h3.BinLimits = [-0.45,0.45];
    %h3.BinLimits = [2,250];

    %first, most noises added to the original images are gray color (value from 120 to 140, -0.1 - 0.1)
    %adversarial samples have more gray color (value -0.45 ~ -0.3)
    %calculate the ratio of gray color in an image.

    %}

    
    %find threshold=
    left = 2;
    right = 90;
    %{
    threshold = 20;
    [~,~,score1] = find_threshold(train_X(:,:,:,train_Y==1),threshold,left, right);
    tabulate(score1);
    [~,~,score2] =find_threshold(train_X(:,:,:,train_Y==0),threshold,left, right);
    tabulate(score2);

    h1 = histogram(score1)
    hold on
    h2 = histogram(score2)
    %}

    scores = detect_attack(test_X, left, right);
    [x,y,t,auc]=perfcurve(categorical(transpose(test_Y)),scores, 1);
    disp(auc)
    plot(x,y)
end

function [t, f, score] = find_threshold(mat,threshold, left, right)
    t=0;
    f=0;
    score = [];
    for idx = 1:size(mat, 4)
        img = mat(:,:,1,idx);
        freq = tabulate(reshape(img, 1,[]));
        gray = sum(freq(freq(:,1)<right & freq(:,1)>left, 3));
        score(end+1) = gray;
        if gray >= threshold
            t = t + 1;
        else
            f = f + 1;
        end
    end
end
%%}

function score = detect_attack(mat,left, right)
    score = [];
    for idx = 1:size(mat, 4)
        img = mat(:,:,1,idx);
        freq = tabulate(reshape(img, 1,[]));
        gray = sum(freq(freq(:,1)<right & freq(:,1)>left, 3));
        if gray >=9.
            score(end+1) = 100.;
        elseif gray >= 6.
            score(end+1) = 30.;
        elseif gray >= 2.
            score(end+1) = 10.;
        elseif gray >= 0.
            score(end+1) = 30.;
        end
    end
end

function defense_net = defense_model_NN(train_X,train_Y,test_X,test_Y)
    layers = [
        imageInputLayer([28 28 1])

        convolution2dLayer(3,16,'Padding',1)
        batchNormalizationLayer
        reluLayer

        maxPooling2dLayer(2,'Stride',2)

        convolution2dLayer(3,32,'Padding',1)
        batchNormalizationLayer
        reluLayer

        maxPooling2dLayer(2,'Stride',2)

        convolution2dLayer(3,64,'Padding',1)
        batchNormalizationLayer
        reluLayer

        fullyConnectedLayer(2)
        softmaxLayer
        classificationLayer];

    miniBatchSize = 32;
    options = trainingOptions( 'sgdm',...
        'MiniBatchSize', miniBatchSize,...
        'Plots', 'training-progress');

    defense_net = trainNetwork(train_X, categorical(train_Y), layers, options);	

    scores = predict(defense_net, test_X);
    [x,y,t,auc]=perfcurve(categorical(transpose(test_Y)),scores(:,2), 1);
    disp(auc)
    plot(x,y)
end


