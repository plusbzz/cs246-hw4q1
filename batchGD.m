function [W,b] = batchGD(X,Y,W,b,C,eta)
%BATCHGD: One step of batch gradient descent
    N = size(X,1);
    D = size(X,2);    
    
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
end

