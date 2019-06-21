%%
fnames = dirnames('.');
old_fold = pwd;
for i=1:numel(fnames)
    if str2num(fnames{i})>181208
        disp(fnames{i})
        cd(fnames{i})
        fnames2 = dirnames('.');
        oldish_fold = pwd;
        for j=1:numel(fnames2)
            try
                cd(fnames2{j})
                gen_eye_tracks('.');
            catch
                disp(['couldnt do ' fnames2{j}])
            end
            cd(oldish_fold)
        end
        cd(old_fold)
    end
end