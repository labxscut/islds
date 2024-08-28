tic
load gtruth.mat;
crackTrainingDataset = objectDetectorTrainingData(gTruth);
trainingData  = crackTrainingDataset;
rng(0);
shuffledIdx = randperm(height(trainingData));
trainingData = trainingData(shuffledIdx,:);
imds = imageDatastore(trainingData.imageFilename);
blds = boxLabelDatastore(trainingData(:,2:end));
ds = combine(imds, blds);

 options = trainingOptions('sgdm', ...
     "ExecutionEnvironment","auto",...
      'MiniBatchSize', 1, ...
      'InitialLearnRate', 1e-3, ...
      'MaxEpochs', 1, ...
      'VerboseFrequency',1, ...
      'CheckpointPath', tempdir,...
      "Plots","training-progress");
  
  
 inputImageSize = [1256 500 3];%[500,1256,3]
 numClasses = 1;
%  anchorBoxes = [8,8; 12,12; 24 24];
 anchorBoxes = estimateAnchorBoxes(blds,50);
 network = 'resnet50';
 featureLayer = 'activation_40_relu';
 
 lgraph = fasterRCNNLayers(inputImageSize, numClasses, anchorBoxes,network,featureLayer); 
 detector = trainFasterRCNNObjectDetector(ds,lgraph,options,'NegativeOverlapRange',[0 0.3], ...
        'PositiveOverlapRange',[0.6 1]);
%% 以下是将数据进行保存在RCNNStructure2.mat中的操作
 save('RCNNStructure2','detector');
 toc