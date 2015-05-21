% Code to illustrate how to use funcLearn for distribution to distribution
% regression. We look to regress a mirrored GMM pdf when given samples
% from the input GMM. 
% 
% We will work over a dataset of P and Q, where P is a length N=30K cell
% array of samples drawn from input pdfs and Q is a length N cell array of 
% samples drawn from correcponding output pdfs. That is, P{i} is a nx2
% matrix of points drawn from the ith input distribution, and Q{i} is a mx2
% matrix of points drawn from the ith output distribution.
%
% The approach, Triple Basis Estimator (3BE), is as follows:
%   - We represent input and output pdfs using a finite set of 
%     coefficients (see lines 61, 62)
%   - We solve a multivariate ridge regression problem using random kitchen
%     sink features (Rahimi & Recht 2007) of input function projection 
%     coefficients as covariates and output function projection 
%     coefficients as responses (see line 83)
%
% The code also illustrates how estimates may be used for unseen data,
% plotting and sampling (see lines 85:end).
%
% Please cite the following works for function to function regression using
% pdfs:
% Fast Function to Function Regression.
% Oliva, J., Neiswanger, W., Póczos, B., & Xing, E., Trac, H., Ho, S.,
% Schneider, J.
% International Conference on AI and Statistics (AISTATS), 2015.
%
% Distribution to Distribution Regression.
% Oliva, J., Póczos, B., & Schneider, J.
% International Conference on Machine Learning (ICML), 2013.


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Generate data

% get the input/output bags
BOX = 8;
fprintf('Generating data...\n');
N = 30000;
[P,Q] = gen_mirror_samples(N);
fprintf('Generated datas.\n');

% plot input/output sample for a random instance
figure;
i = randi(N);
subplot(1,2,1);
scatter(P{i}(:,1),P{i}(:,2));
axis([0 1 0 1]);
title('Example Input Sample Bag');
subplot(1,2,2);
scatter(Q{i}(:,1),Q{i}(:,2));
axis([0 1 0 1]);
title('Example Output Sample Bag');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Build distributions

% get the projection coefficients for input and output functions
fprintf('Getting input/output distribution projection coefficients...\n');
P_osp = osde(P);
Q_osp = osde(Q);
fprintf('Got input/output distribution projection coefficients.\n');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Do regression

% designate test, validation (hold-out), and training sets
tst_set = false(N,1);
tst_set(randperm(N,ceil(.1*N))) = true;
hol_set_s = false(N-sum(tst_set),1);
hol_set_s(randperm(N-sum(tst_set),ceil(.1*N))) = true;
hol_set = false(N,1);
hol_set(~tst_set) = hol_set_s;
trn_set = ~hol_set & ~tst_set;
opts.tst_set = tst_set;
opts.hol_set = hol_set;
opts.trn_set = trn_set;

% perform distribution to distribution regression on the input/output
% projection coefficients
fprintf('Performing distribution to distribution regression...\n');
[Psi, rks, tst_stats, cv_stats] = rks_ridge(P_osp.pc, Q_osp.pc, opts);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Use estimator

% make a grid on [0,1]^2
x1dgrid = (1/100:1/100:1)';
[x1grid, x2grid] = meshgrid(x1dgrid,x1dgrid);
xgrid = [x1grid(:), x2grid(:)];

% make some new unseen input (and output) distribution sample
[P0,Q0,mus,Sigmas] = gen_mirror_samples(1);

% estimate projection coefficients for input function on the same set of
% basis functions as before (indexed by P_osp.inds)
P0_osp = osde(P0, P_osp);

% evaluate input pdf on grid
p0 = eval_basis(xgrid, P_osp.inds)*P0_osp.pc';

% get random kitchen sink features for input distribution
z_P0 = sqrt(2/size(rks.W,2))*cos( bsxfun(@plus, sqrt(1/rks.sigma2)*P0_osp.pc*rks.W, rks.b) );

% get 3BE prediction for output distribution projection coefficients
Q0_pc_hat =  z_P0*Psi;

% evaluate predicted output pdf on grid
q0_hat = eval_basis(xgrid, Q_osp.inds)*Q0_pc_hat';

% draw a sample from the estimated pdf
%   Note that since estimated pdfs are represented by a finite number of
%   basis functions, they will not be normalized, and can have negative
%   values. One can project to the set of valid pdfs by taking the positive
%   part and normalizing.
%   There are several ways one can sample once given predicted output pdf
%   projection coefficients. Perhaps the simplest and fastest is to use a
%   discrete distribution on a grid weighted by the pdf values on the grid.
%   Below we sample using Metropolis-Hastings, which is slower, for
%   illustrative purposes.
f = @(x)(max(x)<=1)*(min(x)>=0)*max(1E-50,eval_basis(x, Q_osp.inds)*Q0_pc_hat');
prop = @(x)(x+.05*randn(1,2));
[smpl,accept] = mhsample(rand(1,2),20000,'pdf',f, 'proprnd',prop,'symmetric',true);
q0_smpl = smpl(100:50:end,:);

% get sample based estimate of output function projection coefficients
Q0_osp = osde(Q0, Q_osp);

% evaluate output pdf on grid
q0 = eval_basis(xgrid, Q_osp.inds)*Q0_osp.pc';

% plot various pdfs and samples on grid
% In order by row:
%   - Top-left: The true input pdf p_0
%   - Top-center: The true output pdf q_0
%   - Middle-left: The estimate, \tilde{p}_0, of p_0 made using a sample  
%     drawn from p_0 (shown below). Note that this is the function 
%     represented by the projection coefficients P0_osp.pc.
%   - Middle-center: The estimate, \tilde{q}_0, of q_0 made using a sample  
%     drawn from q_0 (shown below). Note that this is the function 
%     represented by the projection coefficients Q0_osp.pc.
%   - Middle-right: Our predicted output function, \hat{q}_0, when given
%     \tilde{p}_0 to the 3BE estimator. Note that no samples of q_0 are
%     used for this prediction, we only need a sample from p_0.
%   - Bottom-left: Sample drawn from p_0 used to make \tilde{p}_0.
%   - Bottom-center: Sample drawn from q_0 used to make \tilde{q}_0.
%   - Bottom-right: Sample drawn from our prediction \hat{q}_0.
% get true pdf values on grid
box1dgrid = linspace(-BOX,BOX,100)';
[boxgrid1, boxgrid2] = meshgrid(box1dgrid,box1dgrid);
boxgrid = [boxgrid1(:), boxgrid2(:)];
p0_true = zeros(size(xgrid,1),1);
K = size(mus{1},2);
for k=1:K
    p0_true = p0_true + mvnpdf(boxgrid,mus{1}(:,k)',Sigmas{1}(:,:,k))/K;
end
p0_true = p0_true./mean(p0_true(:));
q0_true = zeros(size(xgrid,1),1);
for k=1:K
    q0_true = q0_true + mvnpdf(boxgrid,-mus{1}(:,k)',Sigmas{1}(:,:,k))/K;
end
q0_true = q0_true./mean(q0_true(:));
% plot
figure;
subplot(3,3,1);
plot_opts.figh = gcf;
view_density({x1grid,x2grid}, p0_true, plot_opts);
title('True $p_0$','Interpreter','latex');
subplot(3,3,2);
plot_opts.figh = gcf;
view_density({x1grid,x2grid}, q0_true, plot_opts);
title('True $q_0$','Interpreter','latex');
subplot(3,3,4);
plot_opts.figh = gcf;
view_density({x1grid,x2grid}, p0, plot_opts);
title('Sample-estimated ${\tilde p}_0$','Interpreter','latex');
subplot(3,3,5);
plot_opts.figh = gcf;
view_density({x1grid,x2grid}, q0, plot_opts);
title('Sample-estimated ${\tilde q}_0$','Interpreter','latex');
subplot(3,3,6);
plot_opts.figh = gcf;
view_density({x1grid,x2grid}, q0_hat, plot_opts);
title('3BE predicted ${\hat q}_0$','Interpreter','latex');
subplot(3,3,7);
scatter(P0{1}(:,1),P0{1}(:,2));
axis([0 1 0 1]);
title('$p_0$ sample','Interpreter','latex');
subplot(3,3,8);
scatter(Q0{1}(:,1),Q0{1}(:,2));
axis([0 1 0 1]);
title('$q_0$ sample','Interpreter','latex');
subplot(3,3,9);
scatter(q0_smpl(:,1),q0_smpl(:,2));
axis([0 1 0 1]);
title('${\hat q}_0$ sample','Interpreter','latex');