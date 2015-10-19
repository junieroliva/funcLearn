% dist2real_demo.m
%
% Script to illustrate how to do distribution to real regression with
% funcLearn for low-dimensional distributions (d<=3).
% 
% Here we look to regress the number of GMM components given a sample drawn
% from the GMM. That is, given sample Xj = {Xj_1, ..., Xj_ni} ~iid pj,
% predict Yj = the number of components from pj.
% 
% In general, to do distribution to real regression you will need: a cell
% array Xin (N x 1) where Xin{i} is a ni x d matrix of ni samples drawn
% iid from the ith input distribution; and a vector Y (N x 1) of response
% values where Y(i) is the response for the ith instance.
% 
% Here we use the double-basis estimator where we use an orthonormal basis
% to represent input pdfs pj's (see *), and we use a random basis to map 
% pj's to Yj's (see **). We do this by: 1) getting projection coefficients
% for each sample, 2) running ridge regression with random kitchen sinks.
%
% For more details see: 
% Oliva, Junier B., et al. "Fast distribution to real regression." 
% AISTATS 2014.


% generate synthetic data
fprintf('Generating synthetic data...\n');
N = 20000;
n = 200;
Kmax = 10;
[Xin, ~, Y, d_mus, d_sigmas] = mirror_gmm(Kmax, n, N);

% get projection coefficients for input distributions (1st basis)*
% osde expects a cell array, were Xin{i} is a ni x d matrix
fprintf('Getting projection coefficients for input distributions...\n');
osp = osde(Xin);
PCin = osp.pc;
basis_inds = osp.inds;

% visualize the estimated density for one of the input distributions
[x1grid, x2grid] = meshgrid((1:100)'/100, (1:100)'/100);
xgrid = [x1grid(:) x2grid(:)];
phixgrid = eval_basis(xgrid, basis_inds);
ri = randi(N,1);
phat = phixgrid*PCin(ri,:)';
Kri = size(d_mus{ri},2);
ptrue = gmmpdf(ones(Kri,1)./Kri,d_mus{ri},d_sigmas{ri});
figure;
subplot(1,3,1);
scatter(Xin{ri}(:,1),Xin{ri}(:,2));
axis([0 1 0 1]);
title('Sample X');
subplot(1,3,2);
view_density({x1grid,x2grid}, ptrue, struct('figh',gcf));
title('True p');
subplot(1,3,3);
view_density({x1grid,x2grid}, phat, struct('figh',gcf, 'cscale', max(ptrue)));
title('Estimated p');

% regress the number of components from a gmm sample given projection
% coefficients, (2nd basis)
fprintf('Regressing number of components from GMM samples...\n');
[B, rks, tst_stats, cv_stats] = rks_ridge(PCin, Y, struct('verbose', true));

% get prediction for a new input distribution
fprintf('Making predictions for new sample...\n');
[Xin0, ~, Y0, d_mus0, d_sigmas0] = mirror_gmm(Kmax, n, 1);
% get the projection coefficients for sample (1st basis)
osp0 = osde(Xin0, struct('inds',basis_inds));
% get the random features for projection coefficients (2nd basis)**
z0 = rks.rfeats(osp0.pc);
% get the estimate
Y0_est = z0*B;
% plot
phat = phixgrid*osp0.pc';
Kri = size(d_mus0{1},2);
ptrue = gmmpdf(ones(Kri,1)./Kri,d_mus0{1},d_sigmas0{1});
figure;
subplot(1,3,1);
scatter(Xin0{1}(:,1),Xin0{1}(:,2));
axis([0 1 0 1]);
title('Sample X0');
subplot(1,3,2);
view_density({x1grid,x2grid}, ptrue, struct('figh',gcf));
title(sprintf('True p0 (k=%g)',Y0));
subplot(1,3,3);
view_density({x1grid,x2grid}, phat, struct('figh',gcf, 'cscale', max(ptrue)));
title(sprintf('Estimated p0 (khat=%g)',Y0_est));
