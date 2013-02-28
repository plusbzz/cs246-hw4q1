function cost = costFunc(X,Y,W,b,C)
%COSTFUNC  
    cost = 0.5*W'*W + C*sum(max(zeros(size(X,1),1),1-Y.*(X*W+b)));
end

