function outfiles = sbxIdentifyFiles(infiles)
% Outputs information or image file corresponding to input image or
% information file respectively

if ~iscellstr(infiles)
    infiles = cellstr(infiles);   
end

nfiles = length(infiles);
p = cell(nfiles,1);
f = cell(nfiles,1);
e = cell(nfiles,1);
for I = 1:nfiles
    [p{I},f{I},e{I}] = fileparts(infiles{I});
end
outfiles = cell(nfiles,1);
bad = [];
for index = 1:nfiles
    switch e{index}
        case {'.sbx', '.imgs'}
            e{index} = '.mat';
            outfiles{index} = fullfile(cell2mat(p), [cell2mat(f), e{index}]);
            if ~exist(outfiles{index}, 'file')
                bad = [bad, index];
            end
        case '.mat'
            e{index} = '.sbx';
            outfiles{index} = fullfile(cell2mat(p), [cell2mat(f), e{index}]);
            if ~exist(outfiles{index}, 'file')
                bad = [bad, index];
            end
        otherwise
            warning('File extension for file %s not recognized.', infiles{index});
            outfiles{index} = 'not recognized format';
    end
end


if ~isempty(bad)
    for index = bad
        while ~exist(outfiles{index}, 'file');
            [f, p] = uigetfile(['*',e{index}], sprintf('Select corresponding file for: %s',infiles{bad(index)}),outfiles{index});
            if isnumeric(f) %user hit cancel
                outfiles{index} = '';
            else
                outfiles{index} = fullfile(p, f);
            end
        end
    end
end
