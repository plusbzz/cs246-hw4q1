function [W,b] = batchGD(X,Y,W,b,C,eta)
%BATCHGD: One step of batch gradient descent
    M = size(X,1);
    D = size(X,2);    
    gradB = 0;
    
    for j = 1:D
        w_j = W(j);       
        gradLw = 0;
        % Now calculate batch loss function
        for i = 1:M
            pred = Y(i)*(X(i,:)*W + b);
            if pred >= 1
                gradLw = gradLw - Y(i)*X(i,j);
                gradB = gradB - Y(i)*b;
            end
        end
        W(j) = w_j - eta*(w_j+C*gradLw); 
    end
    b = b - eta*gradB;
end

