function N = norm_cols_to_1(M)
N = M./repmat(sum(M,1),size(M,1),1);