function [W,b] = stochasticGD(x,y,W,b,C,eta)
%STOCHASTICGD
    D = size(x,2);    
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
end

