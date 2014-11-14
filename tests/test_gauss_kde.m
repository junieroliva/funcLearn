d = 2;
n1 = randi(500);
n2 = randi(500);
L = 850;

% Test inner-product calculation, non-truncated KDE
box = 8;
xgrid = linspace(-box, box, L);
[X1, X2] = meshgrid(xgrid,xgrid);
xe = [X1(:) X2(:)];

x = randn(n1,d);
sigma2x = rand/2;
opts = struct;
opts.sigma2 = sigma2x;
opts.trunc = false;
[kdex, px] = kde_gauss( x, xe, opts );

tol = sqrt(abs( (1-(2*box)^2*mean(px)) ));
L2_grid = (2*box)^2*mean(px.^2);
C = gauss_prod_C(x, sigma2x, x, sigma2x, opts.trunc);
L2_code = sum(sum(exp(-pdist2(x,x).^2/(4*sigma2x)).*C.*(kdex.norma*kdex.norma')));

y = randn(n2,d);
sigma2y = rand/2;
opts = struct;
opts.sigma2 = sigma2y;
opts.trunc = false;
[kdey, py] = kde_gauss( y, xe, opts );

iprod_grid = (2*box)^2*mean(px.*py);
C = gauss_prod_C(x, sigma2x, y, sigma2y, opts.trunc);
iprod_code = sum(sum(exp(-pdist2(x,y).^2/(2*(sigma2x+sigma2y))).*C.*(kdex.norma*kdey.norma')));

% Test inner-product calculation, truncated KDE
xgrid = linspace(1/L, 1, L);
[X1, X2] = meshgrid(xgrid,xgrid);
xe = [X1(:) X2(:)];

x = bsxfun(@minus, x, min(x));
x = bsxfun(@times, x, 1./max(x));
opts = struct;
sigma2x = rand/128
opts.sigma2 = sigma2x;
opts.trunc = true;
[kdex, px] = kde_gauss( x, xe, opts );

tol = sqrt(abs( 1-mean(px) ));
L2_trunc_grid = mean(px.^2);
C = gauss_prod_C(x, sigma2x, x, sigma2x, opts.trunc);
L2_trunc_code = sum(sum(exp(-pdist2(x,x).^2/(4*sigma2x)).*C.*(kdex.norma*kdex.norma')));

y = bsxfun(@minus, y, min(y));
y = bsxfun(@times, y, 1./max(y));
opts = struct;
sigma2y = rand/16;
opts.sigma2 = sigma2y;
opts.trunc = true;
[kdey, py] = kde_gauss( y, xe, opts );

iprod_trunc_grid = mean(px.*py);
C = gauss_prod_C(x, sigma2x, y, sigma2y, opts.trunc);
iprod_trunc_code = sum(sum(exp(-pdist2(x,y).^2/(2*(sigma2x+sigma2y))).*C.*(kdex.norma*kdey.norma')));

% Test Cross-Validation
opts = struct;
opts.trunc = true;
kdex_cv = kde_gauss( x, opts );

si = randi(length(kdex_cv.cv.scores));
sigma2x = kdex_cv.cv.sigma2s(si);
opts.sigma2 = sigma2x;
[kdex, px] = kde_gauss( x, xe, opts );

score_simple = mean(px.^2);
loo_p = nan(n1,1);
for i=1:n1
    [~, loo_p(i)] = kde_gauss( x([1:(i-1) (i+1):n1],:), x(i,:), opts );
end
score_simple = score_simple - 2*mean(loo_p);
score_code = kdex_cv.cv.scores(si);
max(abs(loo_p-kdex_cv.cv.PX(:,si)))