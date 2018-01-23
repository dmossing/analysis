function mkv = markov_model(x)
x = single(x(:));
L = numel(x);
v = unique(x);
T = transition_mat(x,v);
h0 = hist(x,v)';
h0 = h0/L;
Tp = [zeros(size(T,1),1), cumsum(T,2)];
y = zeros(size(x));
randn = rand(size(y));
y(1) = find(histc(randn(1),[0, cumsum(sum(T)/sum(sum(T)))]));
for i=2:numel(y)
    y(i) = find(histc(randn(i),Tp(y(i-1),:)));
end
mkv = y;