function createMCdata(fname)
%% Create MCdata variable
vars = whos(matfile([fname,'.align']));
if ~any(strcmp({vars(:).name}, 'MCdata'))
    load([fname, '.align'], 'T', '-mat');
    MCdata.T = T;
    MCdata.type = 'Translation';
    MCdata.date = datestr(now);
    MCdata.FullFilename = [fname, '.sbx'];
    MCdata.Channel2AlignFrom = 1;
    MCdata.Parameters = [];
    save([fname, '.align'], 'MCdata', '-append', '-mat');
end