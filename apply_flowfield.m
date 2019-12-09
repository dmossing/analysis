function flowed = apply_flowfield(image,flow)
% flow: index i,j tells where the content of pixel i,j should come from,
% relative to i,j. Thus, considered as a vector field, the vectors point in
% the apparent direction of flow moving flowed -> image
flowed = image;
[Ny,Nx] = size(image);
for i=1:Ny
    for j=1:Nx
        ii = 1+mod(i+round(flow(i,j,1))-1,Ny);
        jj = 1+mod(j+round(flow(i,j,2))-1,Nx);
        flowed(i,j) = image(ii,jj);
    end
end