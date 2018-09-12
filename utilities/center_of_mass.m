function com = center_of_mass(img)
[xx,yy] = meshgrid(1:size(img,2),1:size(img,1));
sz = size(img);
xx = repmat(xx,[1,1,sz(3:end)]);
yy = repmat(yy,[1,1,sz(3:end)]);
img = img./repmat(sum(sum(img,1),2),size(img,1),size(img,2));
comx = squeeze(sum(sum(xx.*img,1),2));
comy = squeeze(sum(sum(yy.*img,1),2));

com = zeros([2 size(comx)]);
comx = reshape(comx,[1 size(comx)]);
comy = reshape(comy,[1 size(comy)]);
com = [comy; comx];