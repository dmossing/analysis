function [M,T] = ptcloudSRT(x,y)
% find the scaling, rotation, and translation to take points "x" (d by N)
% to points "y" (d by N). d can be any positive integer (e.g. 2 or 3).
N = size(x,2);
assert(N==size(y,2));
x = [x; ones(1,N)];
A = (x'\y')';
M = A(1:end,1:end-1);
% [U,S,V] = svd(M);
% Sg = diag(sign(diag(S)));
% S = S*Sg; % matrix defining the rescaling
% R = U*Sg*V';
T = A(:,end);