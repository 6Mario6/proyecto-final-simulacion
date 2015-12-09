function [outputs, net] = NNfun(Xtrain,Ytrain)
% This script assumes these variables are defined:
%
%   Xtrain - input data.
%   Ytrain - target data.

inputs = Xtrain';
targets = Ytrain';

% Create a Fitting Network
hiddenLayerSize = 5;
net = fitnet(hiddenLayerSize);

% Setup Division of Data for Training, Validation, Testing
net.divideParam.trainRatio = 70/100;
net.divideParam.valRatio = 11/100;
net.divideParam.testRatio = 11/100;

% Train the Network
[net,~] = train(net,inputs,targets,'useParallel','yes');

% Test the Network
outputs = net(inputs);


end

