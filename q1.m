Xtrain = load('features.train.txt');
Xtest  = load('features.test.txt');
X      = load('features.txt');

Ytrain = load('target.train.txt');
Ytest  = load('target.test.txt');
Y      = load('target.txt');

% Batch gradient descent
batchData = batchGD(X,Y,100);

% Stochastic gradient descent
stochasticData = stochasticGD(X,Y,100);


% Mini-batch gradient descent
minibatchData = minibatchGD(X,Y,100);

Cs = [1 10 50 100 200 300 400 500];
errors = [];
for C=Cs
    [res,W,b] = stochasticGD(Xtrain,Ytrain,C);
    preds = Xtest*W+b;
    classes = preds./abs(preds);
    error = 100*sum(classes ~= Ytest)/size(Ytest,1);
    errors = [errors;C error];
end