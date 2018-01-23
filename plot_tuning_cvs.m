function plot_tuning_cvs(orilist,ori_cv,mean_dFoF_hi,mean_dFoF_lo,reliable,figno)
N = size(ori_cv,1);
if nargin<5 || isempty(reliable)
    reliable = 1:N;
end
if nargin<6 || isempty(figno)
    figure;
else
    figure(figno)
end
% normfact = repmat(max(ori_cv,[],2),1,size(ori_cv,2));
[x,y] = tuning_cv_xy(orilist,ori_cv);%./normfact);
[xhi,yhi] = tuning_cv_xy(orilist,mean_dFoF_hi);%./normfact);
[xlo,ylo] = tuning_cv_xy(orilist,mean_dFoF_lo);%./normfact);
if min(reliable)==0 && max(reliable)==1
    reliable = find(reliable);
end
scale=ceil(sqrt(numel(reliable)/6));
for j=1:numel(reliable);
    k = reliable(j);
    subplot(2*scale,3*scale,j)
    hold on;
    axis tight equal off;
    fill(xhi(:,k),yhi(:,k),'c');
    fill(xlo(:,k),ylo(:,k),'w');
    plot(x(:,k),y(:,k),'b','LineWidth',2);
    scatter(0,0,'m+');
    title(num2str(k))
    hold off;
end
end
