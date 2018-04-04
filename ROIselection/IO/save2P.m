function Filename = save2P(Images,Filename,Header)


%% Parse input arguments
if ~exist('Images','var') || isempty(Images) % Prompt for file selection
    directory = cd;
    [Images, p] = uigetfile({'*.imgs;*.sbx;*.tif'}, 'Select images:', directory, 'MultiSelect','on');
    if isnumeric(Images)
        return
    elseif iscell(Images)
        for findex = 1:numel(Images)
            Images{findex} = fullfile(p, Images{findex});
        end
    elseif ischar(Images)
        Images = {fullfile(p, Images)};
    end
    [Images, loadObj] = load2P(Images, 'Type', 'Direct');
end

if ~exist('Filename','var') || isempty(Filename) % No filename input
    directory = cd;
    [Filename, p] = uiputfile({'*.tif'}, 'Save tif file as:', directory);
    if isnumeric(Filename)
        return
    end
    Filename = fullfile(p, Filename);
end

if ~exist('Header','var') || isempty(Header)
    Header='';
end

if ~isa(Images(1), 'uint16')
    Images = uint16(Images);
end
numFrames = size(Images, 5);


%% Save tif file
tiffObject = Tiff(Filename,'w');
fprintf('Saving %d frames as %s...', numFrames, Filename);
for findex=1:numFrames
    if findex~=1
        tiffObject.writeDirectory();
    end
    tiffObject.setTag('ImageLength',size(Images,1));
    tiffObject.setTag('ImageWidth', size(Images,2));
    tiffObject.setTag('Photometric', Tiff.Photometric.MinIsBlack);
    tiffObject.setTag('BitsPerSample', 16);
    tiffObject.setTag('SamplesPerPixel', 1);
    tiffObject.setTag('PlanarConfiguration', Tiff.PlanarConfiguration.Chunky);
    tiffObject.setTag('Software', 'MATLAB');
    if exist('header','var')
        tiffObject.setTag('ImageDescription',Header);
    end
    tiffObject.write(Images(:,:,findex));
end
tiffObject.close();
fprintf('\tComplete\n');

