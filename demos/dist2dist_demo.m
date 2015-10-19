% dist2dist_demo.m
%
% Script to illustrate how to do distribution to distribution regression 
% with funcLearn for low-dimensional distributions (d<=3).
% 
% Here we look to regress an output pdf which is the mirror image of an 
% input GMM pdf. That is, given sample Xj = {Xj_1, ..., Xj_ni} ~iid pj,
% predict qj() the output pdf.
% 
% In general, to do distribution to real regression you will need: a cell
% array Xin (N x 1) where Xin{i} is a ni x d matrix of ni samples drawn
% iid from the ith input distribution; and Xout (N x 1) where Xout{i} is a 
% mi x d matrix of ni samples drawn iid from the ith output distribution.
% 
% Here we use the triple-basis estimator where we use an orthonormal basis
% to represent input pdfs pj's (see *), an orthonormal basis
% to represent output pdfs qj's (see **), and we use a random basis to map 
% pj's to qj's (see ***). We do this by: 1) getting projection coefficients
% for each input sample, 2) getting projection coefficients
% for each output sample, 3) running ridge regression with random kitchen 
% sinks mapping the input projection coefficients to the output projection
% coefficients.
%
% For more details see: 
% Oliva, Junier, et al. "Fast Function to Function Regression." 
% AISTATS (2015).
% Oliva, Junier, et al. "Distribution to distribution regression."
% ICML (2013).


% generate synthetic data
fprintf('Generating synthetic data...\n');
N = 20000;
n = 200;
Kmax = 10;
[Xin, Xout, ~, d_mus, d_sigmas] = mirror_gmm(Kmax, n, N);

% get projection coefficients for input distributions (1st basis)*
% osde expects a cell array, were Xin{i} is a ni x d matrix
fprintf('Getting projection coefficients for input distributions...\n');
ospin = osde(Xin);
PCin = ospin.pc;
basis_indsin = ospin.inds;

% get projection coefficients for output distributions (2nd basis)**
fprintf('Getting projection coefficients for output distributions...\n');
ospout = osde(Xout);
PCout = ospout.pc;
basis_indsout = ospout.inds;

% visualize the estimated density for one of the input/output distributions
[x1grid, x2grid] = meshgrid((1:100)'/100, (1:100)'/100);
xgrid = [x1grid(:) x2grid(:)];
phixgridin = eval_basis(xgrid, basis_indsin);
ri = randi(N,1);
phat = phixgridin*PCin(ri,:)';
Kri = size(d_mus{ri},2);
ptrue = gmmpdf(ones(Kri,1)./Kri,d_mus{ri},d_sigmas{ri});
figure;
subplot(2,3,1);
scatter(Xin{ri}(:,1),Xin{ri}(:,2));
axis([0 1 0 1]);
title('Sample Xin');
subplot(2,3,2);
view_density({x1grid,x2grid}, ptrue, struct('figh',gcf));
title('True p');
subplot(2,3,3);
view_density({x1grid,x2grid}, phat, struct('figh',gcf, 'cscale', max(ptrue)));
title('Estimated p');
phixgridout = eval_basis(xgrid, basis_indsout);
qhat = phixgridout*PCout(ri,:)';
Kri = size(d_mus{ri},2);
qtrue = gmmpdf(ones(Kri,1)./Kri,-d_mus{ri},d_sigmas{ri});
subplot(2,3,4);
scatter(Xout{ri}(:,1),Xout{ri}(:,2));
axis([0 1 0 1]);
title('Sample Xout');
subplot(2,3,5);
view_density({x1grid,x2grid}, qtrue, struct('figh',gcf));
title('True q');
subplot(2,3,6);
view_density({x1grid,x2grid}, qhat, struct('figh',gcf, 'cscale', max(qtrue)));
title('Estimated q');

% regress the number of components from a gmm sample given projection
% coefficients, (2nd basis)
fprintf('Regressing number of components from GMM samples...\n');
[B, rks, tst_stats, cv_stats] = rks_ridge(PCin, PCout, struct('verbose', true));

% get prediction for a new input distribution
fprintf('Making predictions for new sample...\n');
[Xin0, Xout0, ~, d_mus0, d_sigmas0] = mirror_gmm(Kmax, n, 1);
% get the projection coefficients for sample (1st basis)
osp0 = osde(Xin0, struct('inds',basis_indsin));
% get the random features for projection coefficients (2nd basis)**
z0 = rks.rfeats(osp0.pc);
% get the estimate, note we only use input sample for estimate of output
% pdf
pc0_est = z0*B; % estimated projection coefficients for output function
qhat = phixgridout*pc0_est';
% plot
phat = phixgridin*osp0.pc';
Kri = size(d_mus0{1},2);
ptrue = gmmpdf(ones(Kri,1)./Kri,d_mus0{1},d_sigmas0{1});
figure;
subplot(2,3,1);
scatter(Xin0{1}(:,1),Xin0{1}(:,2));
axis([0 1 0 1]);
title('Sample X0');
subplot(2,3,2);
view_density({x1grid,x2grid}, ptrue, struct('figh',gcf));
title('True p0');
subplot(2,3,3);
view_density({x1grid,x2grid}, phat, struct('figh',gcf, 'cscale', max(ptrue)));
title('Estimated p0');
qtrue = gmmpdf(ones(Kri,1)./Kri,-d_mus0{1},d_sigmas0{1});
subplot(2,3,5);
view_density({x1grid,x2grid}, qtrue, struct('figh',gcf));
title('True q');
subplot(2,3,6);
view_density({x1grid,x2grid}, qhat, struct('figh',gcf, 'cscale', max(qtrue)));
title('3BE Estimated q0');
