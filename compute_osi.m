function osi = compute_osi(oris,r)
% oris in deg, r in (roi index) x ori the (nonnegative) dfof response
th = oris(:)'*pi/180;
n = size(r,1);
nori = size(r,2);
r = r-repmat(min(r,[],2),1,nori); % enforce nonnegativity; bit hack-y!
s = mean(reshape(r,n,nori/2,2),3);
s = mean(s,3);

x = sum(r.*repmat(cos(2*th),n,1),2);
y = sum(r.*repmat(sin(2*th),n,1),2);
norm2d = sqrt(x.^2+y.^2);
norm1d = sum(r,2);
osi = norm2d./norm1d;