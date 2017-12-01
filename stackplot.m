function stackplot(arr,offsetby)
offsets = offsetby*repmat(1:size(arr,1),size(arr,2),1);
plot(offsets+arr')