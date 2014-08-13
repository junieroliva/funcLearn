function [ Y_pred, active, sqerr, lambda, lambdae, lambdar, lambdas, lambdaes, lambdars ] = ...
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
    max_lambda = max(sqrt(sum(reshape(PC'*Y_0,M_n,[]).^2,1)));
    b = max_lambda*min_lambda_ratio;
    B = max_lambda;
    lambdas = b*((B/b).^([(nlambdas-1):-1:0]/(nlambdas-1)));
    opts.lambdas = lambdas;
end
lambdars = get_opt(opts,'lambdars',2.^(20:-1:-20));
opts.lambdars = lambdars;
lambdaes = get_opt(opts,'lambdaes',[0 4.^(1:2)]);
opts.lambdaes = lambdaes;
ntrls = get_opt(opts,'ntrls',3);

active = nan(N,p);
Y_pred = nan(N,1);
sqerr = nan(N,1);
lambda = nan(N,1);
lambdae = nan(N,1);
lambdar = nan(N,1);
stime = tic;
parfor i = 1:N
    trn_set = true(N,1);
    trn_set(i) = false;
    cv_lambda = nan(ntrls,1);
    cv_lambdae = nan(ntrls,1);
    cv_lambdar = nan(ntrls,1);
    for trl=1:ntrls
        if verbose
            fprintf('*** [i: %i] trial: %i elapsed:%f \n', i, trl, toc(stime));
        end
        [ ~, ~, cv_lambda(trl), cv_lambdae(trl), cv_lambdar(trl) ] = cv_supp_FuSSO( Y(trn_set), PC(trn_set,:), p, opts );
    end
    lambdae(i) = mean(cv_lambdae);
    lambdar(i) = mean(cv_lambdar);
    [~,li] = min(abs(lambdas-mean(cv_lambda)));
    lambda(i) = lambdas(li);
    topts = opts;
    topts.lambdas = lambdas(1:li);
    topts.lambdae = lambdae(i);
    [~,norms] = eval_FuSSO( Y(trn_set), PC(trn_set,:), p, topts );
    active(i,:) = norms(end,:)>0;
    supp = repmat(active(i,:),M_n,1);
    supp = supp(:)>0;
    
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
            Y_pred(i) = PC(i,supp)*beta_act(1:end-1)+beta_act(end);
        else
            Y_pred(i) = PC(i,supp)*beta_act;
        end
    else
        if intercept
            Y_pred(i) = mean(Y(trn_set)) ;
        else
            Y_pred(i) = 0;
        end
    end
    sqerr(i) = (Y(i)-Y_pred(i)).^2;
    
    if verbose
        fprintf('###### [i: %i] active: %i, sqerr: %g elapsed:%f \n', i, nactive, sqerr(i), toc(stime));
        fprintf('###### [i: %i] lambda: %g, lambdae: %g lambdar:%g \n', i, lambda(i), lambdae(i), lambdar(i));
    end
end


end

