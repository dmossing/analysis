function transfer_info(filenames1,filenames2)
for i=1:numel(filenames1)
    load(filenames1{i},'info')
    save(filenames2{i},'info','-append')
end