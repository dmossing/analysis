% function zp = gen_motion_corrected_movie(fn,tmax,T,movpath)
% z = squeeze(sbxread(fn,0,tmax));
% zp = z;
% if ~exist(movpath,'dir')
%     mkdir(movpath)
% end
% for i=1:tmax
%     zp(:,:,i) = circshift(z(:,:,i),T(i,:));
%     imwrite(zp(:,:,i),[format_fold(movpath) ddigit(i,4) '.tif'])
%     i
% end
function zp = gen_motion_corrected_movie(fn,tmax,T,movpath)
tchunk = 500;
for k=1:ceil(tmax/tchunk)
    thischunk = min(tchunk,tmax-(k-1)*tchunk);
    startat = (k-1)*tchunk;
    z = squeeze(sbxread(fn,startat,thischunk));
    zp = z;
    if ~exist(movpath,'dir')
        mkdir(movpath)
    end
    for i=startat+1:startat+thischunk
        zp(:,:,i-startat) = circshift(z(:,:,i-startat),T(i,:));
        imwrite(zp(:,:,i-startat),[format_fold(movpath) ddigit(i,4) '.tif'])
        i
    end
end