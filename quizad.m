M = [0.1 0.015 0.01 0.005 1; 0.09 0.016 0.012 0.006 2; .08 .017 .014 .007 3;.07 .018 .015 .008 4;.06 .019 .016 .010 5]
CT = zeros(1,5);

% Calculate CTs
ctcount = 0;
done = false;
while ~done
    for i = 1:3
        round1(:,i) = M(:,1).*M(:,i+1);
    end
    
    [ct,idx] = max(round1);
    ctcount = ctcount + sum(ct)
    for i = 1:3
        M(idx(i),5) = M(idx(i),5) - ct(i)*M(idx(i),1);
        CT(idx(i)) = CT(idx(i)) + ct(i);
    end
    if ctcount >= 101
        CT = round(CT);
        break;
    end

    
    for i = 1:5
        if M(i,5) < 0
            M(i,:) = zeros(5,1);
            CT = round(CT);
        end
    end
    
end



