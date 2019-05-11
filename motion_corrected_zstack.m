%%
framesperz = 50;
%%
% fname = 'M8536_999_000';
fname = 'M9826_999_000';
%%
sbxreadpacked(fname,0,1);
%%
global info
%%
zno = info.max_idx/framesperz;
%%
zm = zeros(512,796,zno);
%%z
for z=1:zno
    
    %%
    im = sbxreadpacked(fname,(z-1)*framesperz,framesperz);
    %%
    zstack = permute(im,[3 1 2]);
    %%
    sz = size(zstack);
    
    z0 = permute(zstack,[2 3 1]); %(:,:,:,z)
    ft = fft2(z0);
    ftm = fft2(mean(z0,3));
    Dm = zeros(framesperz,1);
    [u,v] = deal(zeros(framesperz,1));
    mag = sqrt(sum(sum(abs(ft).^2)));
    magm = sqrt(sum(sum(abs(ftm).^2)));
    mult = zeros(size(z0));
    for i=1:framesperz
%         Dm(i) = abs(sum(sum(conj(ft(:,:,i)).*ftm)))/mag(i)/magm;
        mult(:,:,i) = fftshift(real(ifft2(conj(ft(:,:,i)).*ftm)));
        mt = mult(:,:,i);
        [~,Dm(i)] = max(mt(:));
        [u(i),v(i)] = ind2sub([size(z0,1) size(z0,2)],Dm(i));
        u(i) = u(i) - size(z0,1)/2 - 1;
        v(i) = v(i) - size(z0,2)/2 - 1;
    end
%     z
%     unique(u)
%     unique(v)
    gd = (u==0) & (v==0);
    zm(:,:,z) = prctile(z0(:,:,gd),30,3);
end
%%
for i=1:size(zm,3)
    imagesc(zm(:,:,i));
    pause
end
%%
for i=1:size(zm,2)
    imagesc(squeeze(zm(:,i,:))');
    pause
end
%%
% zstack = makezstack([fname '.sbx'],framesperz,zno);
% %%
% z0 = permute(zstack(:,:,:,end),[2 3 1]);
% ft = fft2(z0);
% %%
% figure;
% imagesc(log(abs(fftshift(ft(:,:,1)))))
% %%
% D = zeros(50,50);
% mag = sqrt(sum(sum(abs(ft).^2)));
% for i=1:50
%     for j=1:50
%         D(i,j) = abs(sum(sum(conj(ft(:,:,i)).*ft(:,:,j))))/mag(i)/mag(j);
%     end
% end
% %%
% ftm = fft2(mean(z0,3));
% %%
% Dm = zeros(50,1);
% mag = sqrt(sum(sum(abs(ft).^2)));
% magm = sqrt(sum(sum(abs(ftm).^2)));
% for i=1:50
%     Dm(i) = abs(sum(sum(conj(ft(:,:,i)).*ftm)))/mag(i)/magm;
% end
% gd = Dm>prctile(Dm,50);
% %%
% figure;
% subplot(1,2,1)
% imagesc(mean(z0,3))
% subplot(1,2,2)
% imagesc(mean(z0(:,:,gd),3))
%%
for i=1:50,
    imagesc(im(:,:,i))
    pause
end
%%
p = zeros(350,2);
figure;
pl1 = 300;
for pl1=1:350
    pl2 = pl1+1;
    A = zm(:,:,pl1);
    B = zm(:,:,pl2);
    scatter(A(:),B(:))
    p(pl1,:) = polyfit(A(:),B(:),1);
end
%%
figure;
hold on
plot(p(:,1)/max(p(:,1)))
plot(p(:,2)/max(p(:,2)))
hold off
%%
zm(:,:,291:295) = zm(:,:,291:295) - 1e4;
zm(:,:,290) = zm(:,:,290) - 7e3;
%%
zm(:,:,291:295) = zm(:,:,291:295) + 1e3;
zm(:,:,290) = zm(:,:,290) - 1e3;
%%
zm(:,:,290) = zm(:,:,290) + 1e3;
%%
for i=1:size(zm,2)
    imagesc(squeeze(zm(:,i,:))');
    pause
end
%%
zmm = squeeze(mean(mean(zm)));
figure
plot(zmm)
zmn = zm - repmat(reshape(zmm,1,1,351),size(zm,1),size(zm,2));
%%
for i=1:size(zmn,2)
    imagesc(flipud(squeeze(zmn(:,i,:))'));
    pause
end
%%
for i=1:size(zmn,2)
    imagesc(zmn(:,:,i));
    pause
end
%%
mkdir('zstack_imgs')
save('M9826_zstack.mat','zmn')
mx = max(zmn(:));
for z=1:351
    imwrite(zmn(:,:,z)/mx,sprintf('zstack_imgs/%03d.tif',z))
end