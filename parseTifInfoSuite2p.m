function Config = parseTifInfoSuite2p(TifFile)
info = imfinfo(TifFile);
Config.Depth = 1;
Config.Frames = numel(info);
Config.Height = info(1).Height;
Config.Width = info(1).Width;
Config.Channels = 1; % I think suite2p saves only the green channel in registered tifs
Config.FrameRate = 15.49; % will need to write something fancier to make this flexible
Config.DimensionOrder = {'Channels','Width','Height','Frames','Depth'}; % default
Config.Colors = {'green', 'red'};
Config.size = [Config.Height, Config.Width, Config.Depth, Config.Channels, Config.Frames];
Config.type = 'suite2p';
Config.info = info;