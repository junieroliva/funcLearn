function [ lambdar, lambdars, MSE, hol_MSEs, beta ] = cv_rdg_lam_FuSSO( Y, PC, varargin )
%cv_supp_FuSSO Summary of this function goes here
%   Detailed explanation goes here
if isempty(varargin)
    opts = struct;
else
    opts = varargin{1};
end
[N,d] = size(PC);
verbose = get_opt(opts,'verbose',false);
% get lambdas
intercept = get_opt(opts,'intercept',true);
lambdars = get_opt(opts,'lambdars',2.^(30:-1:-30));
nlambdars = length(lambdars);
% get training/hold-out sets
trn_set = get_opt(opts,'trn_set',[]);
if isempty(trn_set)
    trn_perc = get_opt(opts,'trn_perc',.8);
    trn_set = false(N,1);
    trn_set(randperm(N,ceil(N*trn_perc))) = true;
end
N_trn = sum(trn_set);
N_hol = sum(~trn_set);
if intercept
    PC_hol = [PC(~trn_set,:) ones(N_hol,1)];
    PC = [PC(trn_set,:) ones(N_trn,1)];
else
    PC_hol = PC(~trn_set,:);
    PC = PC(trn_set,:);
end
Y_hol = Y(~trn_set);
Y = Y(trn_set);

best_hol_MSE = inf;
best_lambdar = nan;
best_beta = nan(d,1);
stime = tic;

[U,S] = eig(PC*PC');
S = diag(S);
PCtU = PC'*U;
PCtY = PC'*Y;
UtPCPCtY = PCtU'*PCtY;
hol_MSEs = nan(nlambdars,1);
for lr=1:nlambdars
    lambdar = lambdars(lr);
    %beta_act = (PC'*PC+lambdar*eye(d))\(PC'*Y);
    %beta_act = (1/lambdar)*(Ig-PC'*U*diag(1./(S+lambdar))*U'*PC)*(PC'*Y);
    beta_act = (1/lambdar)*(PCtY-PCtU*(UtPCPCtY./(S+lambdar)));
    hol_MSE = mean( (Y_hol-PC_hol*beta_act).^2 );
    hol_MSEs(lr) = hol_MSE;
    if hol_MSE<best_hol_MSE
        best_hol_MSE = hol_MSE;
        best_lambdar = lambdars(lr);
        best_beta = beta_act;
    end
    if verbose
        fprintf('[lr:%g] hol_mse: %g elapsed:%f \n', lambdars(lr), hol_MSEs(lr), toc(stime));
    end
end

lambdar = best_lambdar;
MSE = best_hol_MSE;
beta = best_beta;
end

