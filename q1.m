Xtrain = load('features.train.txt');
Xtest  = load('features.test.txt');

Ytrain = load('target.train.txt');
Ytest  = load('target.test.txt');

Ntrain = size(Xtrain,1);
Dtrain = size(Xtrain,2);


% % Batch gradient descent
% C = 100;
% eta = 0.0000003;
% epsilon = 0.25;
% W = zeros(Dtrain,1);
% b = 0;
% 
% lastCost = costFunc(Xtrain,Ytrain,W,b,C);
% k = 0;
% converged = false;
% batchData = [k lastCost 0];
% tic;
% 
% while ~converged
%     k = k+1;
%     [W,b] = batchGD(Xtrain,Ytrain,W,b,C,eta);
%     currentCost = costFunc(Xtrain,Ytrain,W,b,C)
%     DPerc = (100*abs(currentCost - lastCost))/abs(lastCost)
%     batchData = [batchData; k currentCost toc];   
%     lastCost = currentCost;
%     if DPerc < epsilon
%         converged = true;
%     end
% end

% Stochastic gradient descent
C = 100;
eta = 0.0001;
epsilon = 0.001;
W = zeros(Dtrain,1);
b = 0;

lastCost = costFunc(Xtrain,Ytrain,W,b,C);
k = 0;
converged = false;
stochasticData = [k lastCost 0];
DCost = 0;
tic;
idx = randperm(Ntrain);

while ~converged    
    for i = idx
        x = Xtrain(i,:);
        y = Ytrain(i);
        [W,b] = stochasticGD(x,y,W,b,C,eta);
        k = k+1;
        currentCost = costFunc(Xtrain,Ytrain,W,b,C);
        DPerc = (100*abs(currentCost - lastCost))/abs(lastCost);
        DCost = 0.5*(DCost + DPerc);
        stochasticData = [stochasticData; k currentCost toc];   
        lastCost = currentCost;
        if DCost < epsilon
            converged = true;
            break;
        end
    end
    currentCost 
    DCost 
    DPerc
end
