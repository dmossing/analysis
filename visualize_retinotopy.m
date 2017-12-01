function visualize_retinotopy(maximg,xray)
% load(filename)
% xray = zeros([size(avg_by_loc{1,1}) size(avg_by_loc)]);
% for i=1:ny
%     for j=1:nx
%         xray(:,:,i,j) = imgaussfilt(avg_by_loc{i,j},5);
%     end
% end
% maximg = prctile(reshape(xray,size(xray,1),size(xray,2),[]),95,3);
subplot(1,2,1)
imagesc(maximg)
anonMove = @(x,y) mouseMove(x,y,maximg,xray);
set (gcf, 'WindowButtonMotionFcn', anonMove);

function mouseMove (object, eventdata,maximg,xray)
C = get (gca, 'CurrentPoint');
x = ceil(C(1,1));
y = ceil(C(1,2));
% newimg = maximg;
% try
%     newimg(y+1:y+ny,x+1:x+nx) = squeeze(xray(y,x,:,:)/sum(sum(xray(y,x,:,:))));
% catch
% end
try
    subplot(1,2,2)
    imagesc(squeeze(xray(y,x,:,:)));
catch
end
% h.CData = newimg;
drawnow;
title(gca, ['(X,Y) = (', num2str(x), ', ',num2str(y), ')']);
subplot(1,2,1)