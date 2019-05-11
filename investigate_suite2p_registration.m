%%
fnames = dir('./*.tif');
%%
img = zeros(size(imread(fnames(i).name,1)));
chunksize = 1000;
for i=1:1
    offset = chunksize*(i-1);
    for k=1:chunksize
        to_add = double(imread(fnames(i).name,offset+k))/chunksize;
        img = img + circshift(to_add,round(-[ops.xoff(k),ops.yoff(k)]));
    end
end
%%
figure;
imagesc(img)

%%
plot(F(1,:)/max(F(1,:)));
hold on
plot(ops.xoff/max(ops.xoff));
hold off