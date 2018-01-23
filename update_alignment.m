function update_alignment(filename)
matfile = load(filename,'-mat');
Tm = mean(matfile.T);
offset = -round(Tm);
T = matfile.T+repmat(offset,size(matfile.T,1),1);
m = circshift(matfile.m,offset);
thestd = circshift(matfile.thestd,offset);
v = circshift(matfile.v,offset);
k = circshift(matfile.k,offset);
sm = circshift(matfile.sm,offset);
c3 = circshift(matfile.c3,offset);
q1 = circshift(reshape(matfile.Q(:,1),512,796),offset);
q2 = circshift(reshape(matfile.Q(:,2),512,796),offset);
Q = [q1(:) q2(:)];
xray = circshift(matfile.xray,[round(offset/2) 0 0]);
save(filename,'T','m','thestd','v','k','sm','c3','Q','xray')