function T = transition_mat(x,v)
% given a time series x, returns a transition matrix T s.t. T*p_i = p_i+1
% v are locations of histogram bin centers
T = norm_cols_to_1(hist3([x(1:end-1) x(2:end)],{v,v}))';