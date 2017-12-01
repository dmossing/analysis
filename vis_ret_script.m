%% load info
filename = 'retinotopy_007.mat';
load(filename)
%%
xray = zeros([size(avg_by_loc{1,1}) size(avg_by_loc)]);
for i=1:ny
    for j=1:nx
        xray(:,:,i,j) = avg_by_loc{i,j};
    end
end
maximg = prctile(reshape(xray,size(xray,1),size(xray,2),[]),95,3);
for i=1:ny
    for j=1:nx
        xray(:,:,i,j) = imgaussfilt(avg_by_loc{i,j},5);
    end
end
visualize_retinotopy(maximg,xray)