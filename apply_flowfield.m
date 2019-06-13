function flowed = apply_flowfield(image,flow)
flowed = image;
[Ny,Nx] = size(image);
for i=1:Ny
    for j=1:Nx
        ii = 1+mod(i+round(flow(i,j,1))-1,Ny);
        jj = 1+mod(j+round(flow(i,j,2))-1,Nx);
        flowed(i,j) = image(ii,jj);
    end
end