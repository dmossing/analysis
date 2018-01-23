function beginds = get_beginds(C)
L = cellfun(@(x)size(x,1),C);
beginds = cumsum([1; L]);
end