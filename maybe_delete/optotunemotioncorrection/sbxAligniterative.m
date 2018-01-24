function [m,v,T] = sbxAligniterative(fname,m0,rg1,rg2,thestd,gl,l,Frames,rect)



% Aligns images in fname for all indices in idx

% 

% m - mean image after the alignment

% T - optimal translation for each frame


if ~exist('rect','var') || isempty(rect)
    rect = [];
end

global info

numFrames = size(Frames,2);

T = zeros(numFrames,2);

A = sbxreadpacked(fname,0,1);

m = zeros(length(rg1),length(rg2));

v = zeros(length(rg1),length(rg2));



l = l(rg1,rg2);



parfor ii = 1:numFrames

    A = sbxreadpacked(fname,Frames(1,ii)-1,1);
    
    A = double(A(rg1,rg2));

    A = A - gl(ii)*l;

    A = A./thestd;

    [dx,dy] = fftAlign(A,m0);

    T(ii,:) = [dx,dy];

    

    Ar = circshift(A,[dx, dy]);

    m = m+double(Ar);

    

    Ar = circshift(A.^2,[dx, dy]);

    v = v + double(Ar);



end



m = m/numFrames;

v = m/numFrames;