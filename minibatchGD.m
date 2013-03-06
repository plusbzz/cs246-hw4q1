function [minibatchData,W,b] = minibatchGD(X,Y,C)
%MINIBATCHDATA

    N = size(X,1);
    D = size(X,2);    
    
    eta = 0.000001;
    epsilon = 0.01;
    batchsize = 10;
    W = zeros(D,1);
    b = 0;

    lastCost = costFunc(X,Y,W,b,C)
    k = 0;
    converged = false;
    minibatchData = [k lastCost 0];
    DCost = 0;
    tic;

    while ~converged
        idx = randperm(N);

        for s = 1:batchsize:N
            k = k+1;
            batch = idx(s:min(N,s+batchsize-1));
            preds = Y.*(X*W+b);
            %Update W
            for j = 1:D
                gradLw = 0;
                for i = batch % gradient of loss function
                    if preds(i) < 1
                        gradLw = gradLw - Y(i)*X(i,j);
                    end
                end
                W(j) = W(j) - eta*(W(j)+C*gradLw); 
            end
            
            %Update b
            gradB = 0;
            for i = batch
               if preds(i) < 1
                    gradB = gradB - Y(i);
               end
            end        
            b = b - eta*C*gradB;

            currentCost = costFunc(X,Y,W,b,C);
            DPerc = (100*abs(currentCost - lastCost))/abs(lastCost);
            DCost = 0.5*(DCost + DPerc);
            minibatchData = [minibatchData; k currentCost toc];   
            lastCost = currentCost;
            if DCost < epsilon
                converged = true;
                break;
            end
        end
    end
end

