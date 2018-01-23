function aux = sort_by(idx)
if size(idx,1)==1 || size(idx,2)==1
    idx = idx(:);
end
aux = sortrows([idx [1:size(idx,1)]']);
aux = aux(:,end);
end