function p = gmmpdf(w, mus, sigmas)
% get 2d gmm pdf values (normalized to [0,1]^2)

dgrid = 100;
BOX = 8;
grid1d = linspace(-BOX, BOX, dgrid);
[x1grid, x2grid] = meshgrid(grid1d, grid1d);
xgrid = [x1grid(:) x2grid(:)];

p = zeros(size(xgrid,1),1);
K = length(w);
for k=1:K
    p = p + w(k)*mvnpdf(xgrid,mus(:,k)',sigmas(:,:,k));
end
p = p./mean(p);

end

