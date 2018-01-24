function img = gen_ori_tuning_map(msk,ori_tuning,sig,colormap)
if nargin < 4
    colormap = @(x)parula(x);
end
color = zeros(size(msk,3),1);
orionly = ori_tuning.mean_tavg_response(:,1:end/2)+ori_tuning.mean_tavg_response(:,end/2+1:end);
[~,color] = max(orionly,[],2);
color(~sig) = 0;
img = gen_color_coded_map(msk(:,:,ori_tuning.toplot),color(ori_tuning.toplot),colormap);