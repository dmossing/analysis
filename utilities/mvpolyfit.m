function mvpolyfit(X,Y,n)
[npt,ndim] = size(X);
XXcell = cell(n+1,1);
XXcell{1} = ones(npt,1);
XXcell{2} = X;
for ipoly=2:n
    XXcell{ipoly+1} = zeros(npt,ndim^ipoly);
    for iterm=1:ndim^ipoly
        
        XXcell{ipoly+1}(:,iterm) = 1;
    end
end
XX = [XX ones(size(X,1))];