function cost = costFunc(X,Y,W,b,C)
%COSTFUNC
    N = size(X,1);
    D = size(X,2);    
    
    cost = 0.5*W'*W;
    % Now calculate batch loss function
    preds = Y.*(X*W+b);
    Lw = 0;
    for i = 1:N
        Lw = Lw + max(0,1-preds(i));
    end
    cost = cost + C*Lw; 
end

