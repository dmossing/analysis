function [L,idx] = luczak_scrunch(M)
% M is an NxN matrix describing pairwise relationships between N variables.
% L is the same matrix, with the row and column indices shuffled to scrunch
% M along the diagonal.

assert(size(M,1)==size(M,2));
N = size(M,1);

%% average row-wise sorted off-diagonal coefficients
Mp = reshape(M(~eye(N)),N,N-1);
Msorted = fliplr(sort(Mp,2)); % sort off-diagonals in each row of M in descending order
to_fit = mean(Msorted);
% f = fit((1:N-1)',to_fit(:),'exp1');

%% construct a Toeplitz matrix matching the mean row-wise sorted correlation coefficients

T = zeros(N);
for k=1:N-1
%     T((k*N+1):(N+1):end) = f.a*exp(k*f.b);
    T((k*N+1):(N+1):end) = to_fit(k);
end
T = T + T';

%% reshuffle rows and columns to minimize distance of M from T

% need to figure out how to do this in MATLAB!

% distfun = @(X)norm(X-repmat(T,1,1,size(X,3)));
% distfun = @(X)norm(X-T);

Mp = M;
Mp(eye(N)==1) = 0;
[L,idx] = shuf_mat_fmin_anneal(Mp,T);
