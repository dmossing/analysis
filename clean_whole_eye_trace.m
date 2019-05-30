%%
data = zeros(size(msk)/2);
%%
data(hulls(hulls>0)) = 1;
imshow(data)
%%
for i=1:10:10000
%     data = zeros(size(msk)/2);
    data(hulls(i,hulls(i,:)>0)) = 1;
    
%     imshow(data)
%     pause(1e-3)
end
imshow(data)