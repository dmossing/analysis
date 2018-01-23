function [h,w] = aesthetic_sp_dim(N)
phi = (1+sqrt(5))/2;
L = ceil(sqrt(N/phi));
h = L;
w = floor(L*phi);
w(h.*w < N) = w(h.*w < N) + 1;
h(h.*w >= N+w) = h(h.*w >= N+w) - 1;
end