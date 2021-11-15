function this_interpolant = softmax_interpolant(x,y,v,decay_constant)
if nargin < 4
    decay_constant = 50;
end
this_interpolant = @(xx,yy) softmax_interpolant_(xx,yy,x,y,v,decay_constant);

function interp_vals = softmax_interpolant_(xx,yy,x,y,v,decay_constant)
Xquery = [xx(:) yy(:)];
Xtrain = [x(:) y(:)];
kneighbors = min(size(Xtrain,1),5);
idx = knnsearch(Xtrain,Xquery,'K',kneighbors);
ff = zeros(size(xx));
distances = zeros(size(Xquery,1),kneighbors);
vals = zeros(size(Xquery,1),kneighbors);
for ineighbor=1:kneighbors
    distances(:,ineighbor) = sqrt(sum((Xquery-Xtrain(idx(:,ineighbor),:)).^2,2));
    vals(:,ineighbor) = v(idx(:,ineighbor));
end
interp_vals = softmax_(vals,distances,decay_constant);
interp_vals = reshape(interp_vals,size(xx));

function interp_vals = softmax_(vals,distance,decay_constant)
weights = exp(-distance/decay_constant);
interp_vals = sum(vals.*weights,2)./sum(weights,2);