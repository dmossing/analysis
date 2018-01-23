function plot_tuning_cvs_object(ori_tuning)
try
    polar_plot_tuning_cvs(ori_tuning.orilist,ori_tuning.mean_tavg_response,...
        ori_tuning.mean_tavg_response_hi,ori_tuning.mean_tavg_response_lo,ori_tuning.toplot)
catch
    polar_plot_tuning_cvs(ori_tuning.orilist,ori_tuning.mean_tavg_response,...
        ori_tuning.mean_tavg_response_hi,ori_tuning.mean_tavg_response_lo,ori_tuning.toplot)
end