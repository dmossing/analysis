function img = gen_color_coded_map(msk,color,options) %colormap,bkgd)
if nargin < 3
    options = [];
end
if ~isfield(options,'colormap')
    options.colormap = @(x)parula(x);
end
if ~isfield(options,'bkgd')
    options.bkgd = 1;
end
img = options.bkgd*ones(512,796,3);
cnorm = norm01(color(:));
cdepth = 256;
c = options.colormap(cdepth);
cindex = 1+round((cdepth-1)*cnorm(:));
for channel=1:3
    temp = img(:,:,channel);
    for i=1:size(msk,3)
        if ~isnan(cindex(i))
            temp(msk(:,:,i)>0) = c(cindex(i),channel);
        else
            temp(msk(:,:,i)>0) = 0.5;
        end
    end
    img(:,:,channel) = temp;
end
end