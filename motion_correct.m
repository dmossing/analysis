function im = motion_correct(im,T)
for t=1:size(im,3)
        %     mred1 = mred+double(circshift(imgred(:,:,1,1,i),-T(i,:)));
    im(:,:,t) = circshift(im(:,:,t),T(t,:));
end