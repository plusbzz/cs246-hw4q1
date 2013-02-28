function minibatchData = minibatchGD(X,Y)
%MINIBATCHDATA

    N = size(X,1);
    D = size(X,2);    
    
    C = 100;
    eta = 0.0001;
    epsilon = 0.001;
    W = zeros(D,1);
    b = 0;

    lastCost = costFunc(X,Y,W,b,C);
    k = 0;
    converged = false;
    minibatchData = [k lastCost 0];
    DCost = 0;
    tic;

    while ~converged
        idx = randperm(N);
        for i = idx
            k = k+1;
            x = X(i,:);
            y = Y(i);
            for j = 1:D
                gradLw = 0;
                pred = y*(x*W + b);
                if pred < 1
                    gradLw = -y*x(j);
                end
                W(j) = W(j) - eta*(W(j)+C*gradLw); 
            end

            gradB = 0;
            pred = y*(x*W + b);
            if pred < 1
                gradB = gradB - y*b;
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

