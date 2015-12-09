function [outputs, net] = NNfun(Xtrain,Ytrain)
% This script assumes these variables are defined:
%
%   Xtrain - input data.
%   Ytrain - target data.

inputs = Xtrain';
targets = Ytrain';

% Create a Fitting Network
hiddenLayerSize = 3500;
net = patternnet(hiddenLayerSize);

% Setup Division of Data for Training, Validation, Testing
net.divideParam.trainRatio = 70/100;
net.divideParam.valRatio = 15/100;
net.divideParam.testRatio = 15/100;

% Train the Network
[net,~] = train(net,inputs,targets,'useParallel','yes');

% Test the Network
outputs = net(inputs);


end

