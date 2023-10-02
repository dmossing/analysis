function idx1 = mvquadraticfit(X,Y,sz)
% given points X -> Y
XX = quadexpand(X);
p = XX\Y; %% XX*p = Y
[xmesh,ymesh] = meshgrid(1:sz(2),1:sz(1));
Xmesh = [xmesh(:) ymesh(:)];
XXmesh = quadexpand(Xmesh);
XXmesh = XXmesh(:,end-size(p,1)+1:end);
idx1 = round(XXmesh*p);
idx1(:,1) = constrain_mn_mx(idx1(:,1),1,sz(2));
idx1(:,2) = constrain_mn_mx(idx1(:,2),1,sz(1));
idx1 = sub2ind(sz,idx1(:,2),idx1(:,1));
idx1 = reshape(idx1,sz);

function XX = quadexpand(X)
[npt,ndim] = size(X);
XX = zeros(npt,ndim*(ndim+1)/2);
ictr = 1;
for iterm1=1:ndim
    for iterm2=iterm1:ndim
        XX(:,ictr) = X(:,iterm1).*X(:,iterm2);
        ictr = ictr+1;
    end
end
XX = [XX X ones(npt,1)];
if npt < size(XX,2)
    XX = XX(:,end-npt+1:end);
end

function arrc = constrain_mn_mx(arr,mn,mx)
arrc = min(arr,mx);
arrc = max(arrc,mn);