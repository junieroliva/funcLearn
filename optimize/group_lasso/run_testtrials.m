function [ resids, errs, activevoxs, test_sets ] = ....
    run_testtrials( Y, K, g, ntrls, varargin )
%UNTITLED3 Summary of this function goes here
%   Detailed explanation goes here

% options
if isempty(varargin)
    opts = struct;
else
    opts = varargin{1};
end
verbose = get_opt(opts, 'verbose', false);
cv_opts = get_opt(opts, 'cv_opts', struct);
intercept = get_opt(cv_opts,'intercept',true);
class_prob = do_classify(Y);

% set up groups
N = size(K,1);
if length(g)==1
    if mod(size(K,2),g)~=0
        error('#covariates not divisible by #groups indicated.');
    end
    p = g;
else
    p = length(g);
end

% run trials
Ntst = get_opt(opts, 'Ntst', max( 1, floor(.05*N) ));
test_sets = get_opt(opts, 'test_sets');
if isempty(test_sets)
    test_sets = nan(ntrls,Ntst);
    for i=1:ntrls
        test_sets(i,:) = randperm(N,Ntst);
    end
end
resids = cell(ntrls,1);
errs = nan(ntrls,1);
activevoxs = nan(ntrls,p);
stime = tic;
parfor trl=1:ntrls
    if verbose
        fprintf('[[ Trial (%i/%i) ]] (%gs)\n', trl, ntrls, toc(stime));
    end
    
    tst_set = false(N,1);
    tst_set(test_sets(trl,:)) = true;
    
    [activevoxs(trl,:),~,~,~,~,~,~,~,beta] = ...
      cv_supp_grplasso( Y(~tst_set,:), K(~tst_set,:), g, cv_opts );
    
    if intercept
        preds = K(tst_set,:)*beta(1:end-1)+beta(end);
    else
        preds = K(tst_set,:)*beta;
    end
    if class_prob
        preds = preds>=0;
    end
    
    resids{trl} = preds-Y(tst_set);
    errs(trl) = mean( resids{trl}.^2 );
    
    if verbose
        fprintf('\t{ Trial (%i/%i) } err: %g (%gs)\n', trl, ntrls, errs(trl), toc(stime));
    end
end

end

