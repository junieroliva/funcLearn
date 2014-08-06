function [ active, sqerr, lambda, lambdae, lambdar, lambdas, lambdaes, lambdars ] = ...
    cv_mse_FuSSO( Y, PC, p, varargin )
%cv_mse_FuSSO Summary of this function goes here
%   Detailed explanation goes here
if isempty(varargin)
    opts = struct;
else
    opts = varargin{1};
end
N = size(PC,1);
M_n = size(PC,2)/p;
verbose = get_opt(opts,'verbose',false);
% get lambdas
intercept = get_opt(opts,'intercept',true);
opts.intercept = intercept;
lambdas = get_opt(opts,'lambdas',[]);
if isempty(lambdas)
    nlambdas = get_opt(opts,'nlambdas',100);
    min_lambda_ratio = get_opt(opts,'min_lambda_ratio',1E-2);
    if intercept
        Y_0 = Y-mean(Y);
    else
        Y_0 = Y;
    end
    max_lambda = max(sqrt(sum(reshape(PC'*Y_0,M_n,[]).^2)));
    b = max_lambda*min_lambda_ratio;
    B = max_lambda;
    lambdas = b*((B/b).^([(nlambdas-1):-1:0]/(nlambdas-1)));
    opts.lambdas = lambdas;
end
lambdars = get_opt(opts,'lambdars',10.^(15:-1:-15));
opts.lambdars = lambdars;
lambdaes = get_opt(opts,'lambdaes',[0 4.^(1:3)]);
opts.lambdaes = lambdaes;

active = nan(N,p);
sqerr = nan(N,1);
lambda = nan(N,1);
lambdae = nan(N,1);
lambdar = nan(N,1);
stime = tic;
for i = 1:N
    trn_set = true(N,1);
    trn_set(i) = false;
    [ active(i,:), supp, lambda(i), lambdae(i), lambdar(i) ] = cv_supp_FuSSO( Y(trn_set), PC(trn_set,:), p, opts );
    nactive = sum(active(i,:));
    if nactive>0
        if intercept
            PC_act = [PC(trn_set,supp) ones(N-1,1)];
        else
            PC_act = PC(trn_set,supp);
        end
        [U,S] = eig(PC_act*PC_act');
        S = diag(S);
        PCtU = PC_act'*U;
        PCtY = PC_act'*Y(trn_set);
        UtPCPCtY = PCtU'*PCtY;
        beta_act = (1/lambdar(i))*(PCtY-PCtU*(UtPCPCtY./(S+lambdar(i))));
        if intercept
            sqerr(i) = (Y(i)-PC(i,supp)*beta_act(1:end-1)-beta_act(end)).^2;
        else
            sqerr(i) = (Y(i)-PC(i,supp)*beta_act).^2;
        end
    else
        if intercept
            sqerr(i) = (Y(i)-mean(Y(trn_set))).^2 ;
        else
            sqerr(i) = Y(i)^2;
        end
    end
    
    if verbose
        fprintf('###### [i: %i] active: %i, sqerr: %g elapsed:%f \n', i, nactive, sqerr(i), toc(stime));
    end
end


end

