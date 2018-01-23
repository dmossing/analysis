function [M,T] = guiregister2D(A,B)
% convention: A is LF, B is 2P
% register 2 images in x and y through a scaling, rotation, and translation
% the affine transform described by M,T maps locations in B to
% corresponding locations in A.
subplot(1,2,1)
imagesc(A)
subplot(1,2,2)
imagesc(B)
X = [];
Y = [];

% keepgoing = 1;

ptno = input('How many pairs of points?')

disp(sprintf('click %d pairs of points',ptno))

for i=1:ptno
    subplot(1,2,1)
    [x1,y1] = ginput(1);
    subplot(1,2,2)
    [x2,y2] = ginput(1);
    X = cat(2,X,[x1; y1]);
    Y = cat(2,Y,[x2; y2]);
end

[M,T] = ptcloudSRT(X,Y);
end
