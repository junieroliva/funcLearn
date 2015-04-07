function [B, rks, tst_stats, cv_stats, rfeats] = rks_ridge(X, Y, varargin)
%rks_ridge   Ridge regression with random kitchen sinks. Validates the
%       bandwidth and ridge penalty parameters on a hold-out set.
%   Inputs - 
%   X: N x di matrix of input covariates, or N x D matrix of random
%       standard gaussian projections (if opts.input_rand == true)
%   Y: N x do matrix of output responses
%   opts (optional): a struct of options with the following possible fields
%       verbose: print status
%       tst_set, hol_set, trn_set: boolean vectors indicating the instances
%           that belong to the test, hold-out, and training sets 
%           respectively
%       tperc: percentage of instances to use for tst_set and hol_set (if
%           unset)
%       lambdas: values to validate for the ridge penalty
%       input_rand: boolean that is true if X is a matrix of D random
%           standard gaussian projections
%       sigma2s: set of bandwidths to validate
%       sigma2_mult: set of multipliers for a rule of thumb bandwidth to
%           validate if sigma2s is unset or empty
%       D: number of random features (if input_rand is false)

if isempty(varargin)
    opts = struct;
else
    opts = varargin{1};
end

verbose = get_opt(opts,'verbose',false);
reg_func = get_opt(opts,'reg_func',@ridge_reg);

% get training/hold-out/testing sets
[N,di] = size(X);
tst_set = get_opt(opts,'tst_set');
tperc = get_opt(opts,'tperc', .1);
if isempty(tst_set) 
    tst_set = false(N,1);
    tst_set(randperm(N,ceil(N*tperc))) = true;
end
cv = get_opt(opts,'cv','hold');
if strcmp(cv, 'hold')
    hol_set = get_opt(opts,'hol_set');
    trn_set = get_opt(opts,'trn_set');
    if isempty(hol_set) || isempty(trn_set)
        tperc = get_opt(opts,'tperc', .1);
        tst_set = false(N,1);
        tst_set(randperm(N,ceil(N*tperc))) = true;
        trn_set = true(N-ceil(N*tperc),1);
        trn_set(randperm(length(trn_set),ceil(N*tperc))) = false;
        hol_set = ~tst_set;
        hol_set(hol_set) = ~trn_set;
        trn_set = ~hol_set & ~tst_set;
    end
    N_TRAIN = sum(trn_set);
end
% ridge penalties
lambdas = get_opt(opts,'lambdas',[1/64 1/32 1/16 1/8 1/4 1/2 1 2 4 8 16]);
nlambdas = length(lambdas);

% get bandwidths to validate and random features
input_rand = get_opt(opts,'input_rand', false);
if ~input_rand
    sigma2s = get_opt(opts,'sigma2s');
    if isempty(sigma2s)
        sigma2_mult = get_opt(opts,'sigma2_mult',[1/4 1/2 1 2 4]); % multipliers to ROT bandwidth to CV
        N_ROT = 1000; % max number of pairs to estimate rule of thumb bandwidth
        rI = trn_set;
        probI = min(N_ROT/N_TRAIN,1);
        rI(rI) = rand(1,N_TRAIN)<=probI;
        bROT = slmetric_pw(X(rI,:)',X(rI,:)','sqdist');
        bROT = median(bROT(:));
        sigma2s = bsxfun(@times,sigma2_mult,bROT);
    end
    
    D = get_opt(opts,'D', 5000);
    W = randn(di,D);
    XW = X*W;
else
    W = [];
    XW = X;
    D = size(XW,2);
    sigma2s = get_opt(opts,'sigma2s',1);
end
b = (2*pi)*rand(1,D);
rks.W = W;
rks.b = b;

% cross-validate bandwitdth/lambda using kitchen sinks
reg_opts = get_opt(opts, 'reg_opts', struct);
reg_opts.lambdars = lambdas;
reg_opts.cv = 'hold';
reg_opts.trn_set = trn_set(~tst_set);
reg_opts.eigen_decomp = get_opt(reg_opts,'eigen_decomp', false);

stime = tic;
nsigma2s = length(sigma2s);
eyeD = speye(D);
hol_mses = nan(nsigma2s,nlambdas);
min_mse = inf;
mli = nan;
msi = nan;
B = nan;
for si = 1:nsigma2s
    sigma2 = sigma2s(si);
    % features
    Phi = sqrt(2/D)*cos(bsxfun(@plus,sqrt(1/sigma2)*XW(~tst_set,:),b));
    % regress based on random features
    rreg = reg_func(Phi, Y(~tst_set,:), reg_opts);
    hol_mses(si,:) = rreg.cv.lam_mse;
    [mv,mi] = min(rreg.cv.lam_mse);
    if mv<min_mse
        msi = si;
        mli = mi;
        B = rreg.beta;
    end
%     Phi = sqrt(2/D)*cos(bsxfun(@plus,sqrt(1/sigma2)*XW(trn_set,:),b));
%     PhiTPhi = Phi'*Phi;
%     PhiTY = Phi'*Y(trn_set,:);
%     clear Phi
%     % CV lambdas
%     for li = 1:nlambdas
%         lambda = lambdas(li);
%         S = (PhiTPhi+lambda*eyeD)\PhiTY;
%         h_Phi = sqrt(2/D)*cos(bsxfun(@plus,sqrt(1/sigma2)*XW(hol_set,:),b));
%         h_pred = h_Phi*S;
%         hol_mses(si,li) = mean( sum( (Y(hol_set,:)-h_pred).^2, 2 ) );
%         if verbose
%             fprintf('CV: bw = %g, lambda = %g, score:%g \n',sigma2, lambda, hol_mses(si,li));
%         end
%     end
end
% get optimal
cv_stats.hmse = min(hol_mses(:));
cv_stats.hol_mses = hol_mses;
cv_stats.sigma2s = sigma2s;
cv_stats.lambdas = lambdas;
rks.sigma2 = sigma2s(msi);
cv_stats.lambda = lambdas(mli);
cv_stats.sigma2 = sigma2s(msi);

rfeats = sqrt(2/D)*cos(bsxfun(@plus,sqrt(1/sigma2)*XW,b));
% get predicted response for test instances
% Phi = sqrt(2/D)*cos(bsxfun(@plus,sqrt(1/sigma2)*XW(trn_set|hol_set,:),b));
% PhiTPhi = Phi'*Phi;
% PhiTY = Phi'*Y(trn_set|hol_set,:);
% B = (PhiTPhi+lambda*eyeD)\PhiTY;
tst_stats = struct;
if any(tst_set)
    t_Phi = rfeats(tst_set,:);
    pred_projs = t_Phi*B;

    % errors
    mean_pred_mse = mean( sum( bsxfun(@minus,Y(tst_set,:),mean(Y(tst_set,:))).^2, 2 ) );
    tst_mse = mean( sum( (Y(tst_set,:)-pred_projs).^2, 2 ) );
    tst_stats.mse = tst_mse;
    tst_stats.mean_pred_mse = mean_pred_mse;
    tst_stats.tst_set = tst_set;
    if verbose
        fprintf('TEST: bw = %g, lambda = %g, score: %g, mean_pred score: %g (CVed in %g secs)\n',sigma2, lambda, tst_mse, mean_pred_mse, toc(stime));
    end
end
end
