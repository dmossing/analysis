%%
foldbase = '/global/scratch/mossing/2Pdata/';
foldnames = {...
%     '/home/mossing/scratch/2Pdata/180412/M7955';... ALREADY DID IN SBX
     %'180530/M8174';... DONE. Threw warning on suite2p analysis; may have fixed.
%    '180720/M8961';... DONE 
%    '180802/M9053';... DONE 
    %     '/home/mossing/scratch/2Pdata/180220/M7254',...
%    '180730/M8961';... DONE
    %'180714/M9053'... DONE; weird bus error on analysis, redoing
%    '180622/M8961' %DONE
%'180704/M8956';... DONE
%'180712/M9053' DONE
'180220/M7254'
    };
%%
endings = {...
%     [0 1 2 3 4];...
     % [0 1 2 3 4];...
%    [3 4 5];...
    %[2 3 4];...
    %     [4 5],...
    %[2 4 5 6];...
    %[2 3 4 5 6]...
%[1 3 4]
%[0 1 2 3];...
%[1 2 3]
[5]
    };
%%
for i=1:numel(foldnames)
    real_endings = cell(size(endings{i}));
    for j=1:numel(endings{i})
        real_endings{j} = ddigit(endings{i}(j),3);
    end
    format_for_suite2p([foldbase foldnames{i}],'endings',real_endings,'target_fold',[foldbase 'suite2P/raw/'])
end
