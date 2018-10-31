function run_preprocessing(foldname,runfoldname,varargin)
arglist = {'floorframes',true};
floorframes = parse_args(varargin,arglist);

if floorframes
    fixframes(foldname)
end
save_npil_corrected(foldname)
try
    save_trialwise(foldname)
catch
    warning('could not trialize')
end
if nargin > 1 && ~isempty(runfoldname)
    save_running(runfoldname,foldname,floorframes)
end
save_msk(foldname)
save_roilines(foldname)