function [W,b] = stochasticGD(x,y,W,b)
%STOCHASTICGD
    M = size(X,1);
    D = size(X,2);
    
    W = rand(D,1);
    b = rand();
    gradB = 0;
    
    for j = 1:D
        w_j = W(j);       
        gradLw = 0;
        pred = y*(x*W + b);
        if pred >= 1
            gradLw = -y*x(j);
            gradB = gradB - y*b;
        end
        W(j) = w_j - eta*(w_j+C*gradLw); 
    end
    b = b - eta*gradB;
end

