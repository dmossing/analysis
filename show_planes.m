function show_planes(foldname)
d = dir([foldname '/*ot*.align']);
names = {d(:).name};
for i=1:numel(names)
    load(names{i},'-mat','m');
    M{i} = m;
end
for i=1:numel(names)
    subplot(1,numel(names),i)
    imagesc(M{i})
    colormap gray
    axis off
end