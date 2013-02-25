Xtrain = load('features.train.txt');
Xtest  = load('features.test.txt');

Ytrain = load('target.train.txt');
Ytest  = load('target.test.txt');

Mtrain = size(Xtrain,1);
Dtrain = size(Xtrain,2);


% Batch gradient descent
C = 100;
eta = 0.0000003;
W = rand(Dtrain,1);
b = rand();

cost = [0];
k = 0;
converged = false;

while ~converged
    [W,b] = batchGD(Xtrain,Ytrain,W,b,C,eta);
end