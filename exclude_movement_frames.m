%%
im = squeeze(load2P('VIPSSTa_999_001.sbx','Frames',1:100));
%%
for i=1:size(im,3), 
    imagesc(im(:,:,i)), 
    pause, 
end
%%
m = mean(im,3);
%%
sqdiff = zeros(size(im,3),1);
for i=1:size(im,3)
    sqdiff(i) = sum(sum((m-double(im(:,:,i))).^2));
end
%%
m2 = mean(im(:,:,sqdiff<prctile(sqdiff,90)),3);
%%
figure
subplot(1,2,1)
imagesc(m)
subplot(1,2,2)
imagesc(m2)