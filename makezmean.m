function zmean = makezmean(fname,zno,framesperz)
try
    % single channel case
    ex = load2P(fname,'frames',1);
    zmean = zeros(zno,size(ex,1),size(ex,2));
    for i=1:zno
        i
        Images = squeeze(load2P(fname,'frames',(i-1)*framesperz+(1:framesperz)));
        zmean(i,:,:) = mean(Images,3);
    end
catch
    % two channel case
    ex = load2P(fname,'frames',1,'channels',[1 2]);
    zmean = zeros(2,zno,size(ex,1),size(ex,2));
    for i=1:zno
        i
        Images = squeeze(load2P(fname,'frames',(i-1)*framesperz+(1:framesperz),'channels',[1 2]));
        zmean(:,i,:,:) = permute(mean(Images,4),[3 1 2]);
    end
end
    % sz = size(Images);
% Images = reshape(Images,sz(1),sz(2),framesperz,zno);
% zmean = permute(Images,[4 1 2 3]);