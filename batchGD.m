function batchData = batchGD(X,Y)
%BATCHGD: One step of batch gradient descent

    % Batch gradient descent
    N = size(X,1);
    D = size(X,2);    
    
    C = 100;
    eta = 0.0000003;
    epsilon = 0.25;
    W = zeros(D,1);
    b = 0;

    lastCost = costFunc(X,Y,W,b,C);
    k = 0;
    converged = false;
    batchData = [k lastCost 0];
    tic;

    while ~converged
        k = k+1;
        preds = Y.*(X*W+b);

        % Update W
        for j = 1:D
            gradLw = 0;
            for i = 1:N  % gradient of loss function
                if preds(i) < 1
                    gradLw = gradLw - Y(i)*X(i,j);
                end
            end
            W(j) = W(j) - eta*(W(j)+C*gradLw); 
        end

        %Update b
        gradB = 0;
        for i = 1:N
           if preds(i) < 1
                gradB = gradB - Y(i)*b;
           end
        end
        b = b - eta*C*gradB;
 
        currentCost = costFunc(X,Y,W,b,C);
        DPerc = (100*abs(currentCost - lastCost))/abs(lastCost);
        batchData = [batchData; k currentCost toc];   
        lastCost = currentCost;
        if DPerc < epsilon
            converged = true;
        end
    end


       
end

