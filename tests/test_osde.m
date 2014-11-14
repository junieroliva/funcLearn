d = 2;
n1 = randi(500);
n2 = randi(500);
L = 850;

% Test inner-product calculation
xgrid = linspace(1/L, 1, L);
[X1, X2] = meshgrid(xgrid,xgrid);
xe = [X1(:) X2(:)];

x = randn(n1,d);
x = bsxfun(@minus, x, min(x));
x = bsxfun(@times, x, 1./max(x));

max_norm = randi(100)+10;
inds = outerprodinds(0:max_norm,d,max_norm);
norms = sqrt(sum(inds.^2,2));
[sv,si] = sort(norms);
ind_used = si(sv<=max_norm);
norms = norms(ind_used);
inds = inds(ind_used,:);

opts = struct;
opts.inds = inds;
opts.eps_trunc = false;
[ospx, px] = osde( x, xe, opts );

tol = sqrt(abs( 1-mean(px) ));
L2_grid = mean(px.^2);
L2_code = sum(ospx.pc.^2);

y = randn(n2,d);
y = bsxfun(@minus, y, min(y));
y = bsxfun(@times, y, 1./max(y));

max_norm = randi(100)+10;
inds = outerprodinds(0:max_norm,d,max_norm);
norms = sqrt(sum(inds.^2,2));
[sv,si] = sort(norms);
ind_used = si(sv<=max_norm);
norms = norms(ind_used);
inds = inds(ind_used,:);

opts = struct;
opts.inds = inds;
opts.eps_trunc = false;
[ospy, py] = osde( x, xe, opts );

iprod_grid = mean(px.*py);
mi = min(length(ospx.pc),length(ospy.pc));
iprod_code = ospx.pc(1:mi)'*ospy.pc(1:mi);

% Test Cross-Validation
opts = struct;
opts.max_norm = 100;
ospx_cv = osde( x, opts );

linds = find(ospx_cv.cv.lastnorms);
si = randi(length(linds));
opts.inds = ospx_cv.cv.inds(1:linds(si),:);
ospx = osde( x, opts );

opts.eps_trunc = false;
score_simple = sum(ospx.pc.^2);
loo_p = nan(n1,1);
for i=1:n1
    [~, loo_p(i)] = osde( x([1:(i-1) (i+1):n1],:), x(i,:), opts );
end
score_simple = score_simple - 2*mean(loo_p);
score_code = ospx_cv.cv.scores(si);
