Xtrain = load('features.train.txt');
Xtest  = load('features.test.txt');

Ytrain = load('target.train.txt');
Ytest  = load('target.test.txt');

Ntrain = size(Xtrain,1);
Dtrain = size(Xtrain,2);


% Batch gradient descent
batchData = batchGD(Xtrain,Ytrain);

% Stochastic gradient descent
stochasticData = stochasticGD(Xtrain,Ytrain);