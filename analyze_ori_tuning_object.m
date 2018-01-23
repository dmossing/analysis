function ori_tuning = analyze_ori_tuning_object(ROIfile,stimfile)
if isempty(strfind(ROIfile,'_ot_'))
    framerate = 15.5;
    addon = 32;
else
    framerate = 3.875;
    addon = 8;
end
% framerate = 15.5;
% addon = 16;
try
    [response,orilist,ori_response,tavg_response,mean_tavg_response,...
        mean_tavg_response_hi,mean_tavg_response_lo,reliable] = ...
        analyze_ori_tuning(ROIfile,stimfile,[],[],[],[],[],[],framerate,addon);
catch
    addon = addon/2;
    [response,orilist,ori_response,tavg_response,mean_tavg_response,...
        mean_tavg_response_hi,mean_tavg_response_lo,reliable] = ...
        analyze_ori_tuning(ROIfile,stimfile,[],[],[],[],[],[],framerate,addon);
end
ori_tuning.response = response;
ori_tuning.orilist = orilist;
ori_tuning.ori_response = ori_response;
ori_tuning.tavg_response = tavg_response;
ori_tuning.mean_tavg_response = mean_tavg_response;
ori_tuning.mean_tavg_response_hi = mean_tavg_response_hi;
ori_tuning.mean_tavg_response_lo = mean_tavg_response_lo;