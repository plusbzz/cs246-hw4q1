function cost = costFunc(X,Y,W,b,C)
%COSTFUNC
    M = size(X,1);
    D = size(X,2);    
    
    cost = 0.5*W'*W;
    for j = 1:D
        w_j = W(j);  
        Lw = 0;
        % Now calculate batch loss function
        for i = 1:M
            pred = Y(i)*(X(i,:)*W + b);

        end
        cost = cost + C*Lw; 
    end
end

