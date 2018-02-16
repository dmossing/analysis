function run_preprocessing(foldname,runfoldname)
fixframes(foldname)
save_npil_corrected(foldname)
try
    save_trialwise(foldname)
catch
    warning('could not trialize')
end
if nargin > 1 && ~isempty(runfoldname)
    save_running(runfoldname,foldname)
end