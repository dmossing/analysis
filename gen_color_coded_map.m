function img = gen_color_coded_map(msk,color,colormap)
if nargin < 3
    colormap = @(x)parula(x);
end
c = colormap(max(color)+1);
img = ones(512,796,3);
for channel=1:3
    temp = img(:,:,channel);
    for i=1:size(msk,3)
        if color(i)
            temp(msk(:,:,i)>0) = c(color(i),channel);
        else
            temp(msk(:,:,i)>0) = 0.5;
        end
    end
    img(:,:,channel) = temp;
end
end