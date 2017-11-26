function [ B, tst_stats, cv_stats ] = cv_inputs_ridge( Xs, Y, varargin )
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
if isempty(varargin)
    opts = struct;
else
    opts = varargin{1};
end

verbose = get_opt(opts,'verbose',false);
reg_func = get_opt(opts,'reg_func',@ridge_reg);
hol_set = [];
trn_set = [];

% get training/hold-out/testing sets
N = size(Xs{1}, 1);
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
end

% ridge penalties
lambdas = get_opt(opts,'lambdas',[1/64 1/32 1/16 1/8 1/4 1/2 1 2 4 8 16]);
nlambdas = length(lambdas);

% cross-validate inputs/lambda
reg_opts = get_opt(opts, 'reg_opts', struct);
reg_opts.lambdars = lambdas;
reg_opts.cv = 'hold';
reg_opts.trn_set = trn_set(~tst_set);
reg_opts.eigen_decomp = get_opt(reg_opts,'eigen_decomp', false);

stime = tic;
ninps = length(Xs);
hol_mses = nan(ninps,nlambdas);
min_mse = inf;
mli = nan;
msi = nan;
for si = 1:ninps
    % regress based on random features
    rreg = reg_func(Xs{si}(~tst_set,:), Y(~tst_set,:), reg_opts);
    hol_mses(si,:) = rreg.cv.lam_mse;
    [mv,mi] = min(rreg.cv.lam_mse);
    if mv<min_mse
        min_mse = mv;
        msi = si;
        mli = mi;
    end
    if verbose
        fprintf('CV: inp = %i, best score: %g (%g secs)\n', si, mv, toc(stime));
    end
end

% get optimal
cv_stats.trn_set = trn_set;
cv_stats.hol_set = hol_set;
cv_stats.tst_set = tst_set;
cv_stats.hmse = min(hol_mses(:));
cv_stats.hol_mses = hol_mses;
cv_stats.lambdas = lambdas;
cv_stats.lambda = lambdas(mli);
cv_stats.input_ind = msi;
lambda = cv_stats.lambda;

% get predicted response for test instances
Phi = Xs{msi}(~tst_set,:);
B = (Phi'*Phi+lambda*speye(size(Phi,2)))\(Phi'*Y(~tst_set,:));
tst_stats = struct;
if any(tst_set)
    t_Phi = Xs{msi}(tst_set,:);
    pred_projs = t_Phi*B;

    % errors
    mean_pred_mse = mean( sum( bsxfun(@minus,Y(tst_set,:),mean(Y(tst_set,:))).^2, 2 ) );
    tst_mse = mean( sum( (Y(tst_set,:)-pred_projs).^2, 2 ) );
    tst_stats.pred = pred_projs;
    tst_stats.mse = tst_mse;
    tst_stats.mean_pred_mse = mean_pred_mse;
    tst_stats.tst_set = tst_set;
    if verbose
        fprintf('TEST: inp = %i, lambda = %g, score: %g, mean_pred score: %g (CVed in %g secs)\n',cv_stats.input_ind, lambda, tst_mse, mean_pred_mse, toc(stime));
    end
end

end

