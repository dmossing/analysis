function N = norm_cols(M)
N = M./repmat(max(M),size(M,1),1);
N = N - repmat(mean(N),size(N,1),1);
end