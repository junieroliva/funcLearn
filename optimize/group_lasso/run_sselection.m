function [ lambdas, gnorms, trn_sets ] = ....
    run_selection( Y, K, g, B, varargin )
%UNTITLED3 Summary of this function goes here
%   Detailed explanation goes here

% options
if isempty(varargin)
    opts = struct;
else
    opts = varargin{1};
end
verbose = get_opt(opts, 'verbose', false);
eval_opts = get_opt(opts, 'eval_opts', struct);
intercept = get_opt(eval_opts,'intercept',true);
class_prob = do_classify(Y);

% set up groups
N = size(K,1);
if length(g)==1
    if mod(size(K,2),g)~=0
        error('#covariates not divisible by #groups indicated.');
    end
    gsize = size(K,2)/g;
    ginds = [];
    gmult = [];
    p = g;
else
    gsize = nan;
    ginds = g;
    gmult = get_opt( opts, 'gmult', sqrt(ginds-[0; ginds(1:end-1)]) );
    p = length(g);
end

% setup sets
trn_sets = cell(2,B);
for i=1:B
    rprm = randperm(1:N);
    trn_sets{1,i} = rprm(1:floor(N/2));
    trn_sets{2,i} = rprm(floor(N/2)+1:end);
end
trn_sets = trn_sets(:);

% run trials
ntrls = 2*B;
lambdas = get_opt(eval_opts,'lambdas');
if isempty(lambdas)
    groups.gsize = gsize;
    groups.ginds = ginds;
    groups.gmult = gmult;
    lambdas = get_lambda_range(Y, K, groups, eval_opts);
    eval_opts.lambdas = lambdas;
end
nlambdas = length(lambdas);
gnorms = nan(ntrls,nlambdas,p);
stime = tic;
parfor trl=1:ntrls
    if verbose
        fprintf('[[ Trial (%i/%i) ]] (%gs)\n', trl, ntrls, toc(stime));
    end
    
    trn_set = trn_sets(trl);
    
    [~,gn] = ...
      eval_grplasso( Y(trn_set,:), K(trn_set,:), g, eval_opts );
    gnorms(trl,:,:) = gn;
    
    if verbose
        fprintf('\t{ Trial (%i/%i) } (%gs)\n', trl, ntrls, toc(stime));
    end
end

end

