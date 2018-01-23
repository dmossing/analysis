function M = vec_colon(X,k)
M = repmat(X(:),1,k);
M = M + repmat(0:k-1,numel(X),1);