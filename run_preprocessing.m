function run_preprocessing(foldname,runfoldname,varargin)
arglist = {'floorframes',true};
floorframes = parse_args(varargin,arglist);

if floorframes
    fixframes(foldname)
end
try
    save_npil_corrected(foldname)
catch
    warning(['could not neuropil correct ' foldname])
end
try
    save_trialwise(foldname)
catch
    warning('could not trialize')
end
if nargin > 1 && ~isempty(runfoldname)
    save_running(runfoldname,foldname,floorframes)
end
try
    save_msk(foldname)
    save_roilines(foldname)
catch
    warning(['could not save ROI masks for ' foldname])
end