function [B, rks, tst_stats, cv_stats] = rks_conddens(X, Y, varargin)
%rks_ridge   Ridge regression with random kitchen sinks. Validates the
%       bandwidth and ridge penalty parameters on a hold-out set.
%   Inputs - 
%   X: N x di matrix of input covariates, or N x D matrix of random
%       standard gaussian projections (if opts.input_rand == true)
%   Y: N x 1 matrix of output responses
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

% get training/testing sets
[N,di] = size(X);
tst_set = get_opt(opts,'tst_set');
hol_set = get_opt(opts,'hol_set');
trn_set = get_opt(opts,'trn_set');
if isempty(hol_set) || isempty(trn_set)
    tperc = get_opt(opts,'tperc', .1);
    hperc = get_opt(opts,'hperc', .1);
    [trn_set, hol_set, tst_set] = split_data(N, tperc, hperc);
end
N_trn = sum(trn_set);
N_hol = sum(hol_set);
N_tst = sum(tst_set);

% ridge penalties
lambdas = get_opt(opts,'lambdas',2.^(5:-1:-6));
nlambdas = length(lambdas);

% get bandwidths to validate and random features
input_rand = get_opt(opts,'input_rand', false);
if ~input_rand
    sigma2s = get_opt(opts,'sigma2s');
    if isempty(sigma2s)
        sigma2_mult = get_opt(opts,'sigma2_mult',[4 2 1 1/2 1/4]); % multipliers to ROT bandwidth to CV
        N_ROT = 1000; % max number of pairs to estimate rule of thumb bandwidth
        rI = trn_set;
        probI = min(N_ROT/N_trn,1);
        rI(rI) = rand(1,N_trn)<=probI;
        bROT = slmetric_pw(X(rI,:)',X(rI,:)','sqdist');
        bROT = median(bROT(:));
        sigma2s = bsxfun(@times,sigma2_mult,bROT);
    end
    
    D = get_opt(opts,'D', 2500);
    W = randn(di,D);
    XW = X*W;
else
    W = [];
    XW = X;
    D = size(XW,2);
    sigma2s = get_opt(opts,'sigma2s',1);
end
rks.W = W;

% 
basis = get_opt(opts,'basis','trig');
do_trig = strcmp(basis,'trig');
phi_inds = get_opt(opts,'phi_inds',2.^(2:8));
ninds = length(phi_inds);
inds = 1:phi_inds(end);
if do_trig
inds = [inds;-inds];
end
inds = [0;inds(:)];
PhiY = eval_basis( Y, inds, basis);

% cross-validate bandwitdth/lambda using kitchen sinks
eyeD = speye(2*D);
stime = tic;
nsigma2s = length(sigma2s);
hol_scores = nan(nsigma2s,ninds,nlambdas);
for si = 1:nsigma2s
    sigma2 = sigma2s(si);
    
    % features
    Z = sqrt(1/D)*[cos(sqrt(1/sigma2)*XW(trn_set,:)) sin(sqrt(1/sigma2)*XW(trn_set,:))];
    Z_hol = sqrt(1/D)*[cos(sqrt(1/sigma2)*XW(hol_set,:)) sin(sqrt(1/sigma2)*XW(hol_set,:))];
    
    % regress based on random features
    ZTZ = Z'*Z;
    ZTZ_hol = Z_hol'*Z_hol;
    
    for phii=1:ninds
        if do_trig
            pind = 1+2*phi_inds(phii);
        else
            pind = 1+phi_inds(phii);
        end
        ZTPhiY = Z'*PhiY(trn_set,1:pind);
        ZTPhiY_hol = Z_hol'*PhiY(hol_set,1:pind);

        % CV lambdas
        for li = 1:nlambdas
            lambda = lambdas(li);
            B = (ZTZ+lambda*eyeD)\ZTPhiY;
            
            hol_scores(si,phii,li) = (1/N_hol)*(.5*sum(sum(ZTZ_hol.*(B*B'))) -sum(sum(ZTPhiY_hol.*B)));
            if verbose
                fprintf('CV: bw = %g, lambda = %g, ind = %g, score:%g \n',...
                    sigma2, lambda, phi_inds(phii), hol_scores(si,phii,li));
            end
        end
    end
end

% get optimal
ii = find(hol_scores(:)==min(hol_scores(:)));
[si,phii,li] = ind2sub([nsigma2s,ninds,nlambdas],ii(1));
sigma2 = sigma2s(si);
rks.sigma2 = sigma2;
Z = sqrt(1/D)*[cos(sqrt(1/sigma2)*XW(trn_set|hol_set,:)) sin(sqrt(1/sigma2)*XW(trn_set|hol_set,:))];
ZTZ = Z'*Z;
if do_trig
    pind = 1+2*phi_inds(phii);
else
    pind = 1+phi_inds(phii);
end
ZTPhiY = Z'*PhiY(trn_set|hol_set,1:pind);
lambda = lambdas(li);
B = (ZTZ+lambda*eyeD)\ZTPhiY;

if any(tst_set)
    Z_tst = sqrt(1/D)*[cos(sqrt(1/sigma2)*XW(tst_set,:)) sin(sqrt(1/sigma2)*XW(tst_set,:))];
    ZTPhiY_tst = Z_tst'*PhiY(tst_set,1:pind);
    tst_score = (1/N_tst)*(.5*sum(sum((Z_tst'*Z_tst).*(B*B'))) -sum(sum(ZTPhiY_tst.*B)));
    
    ywin = get_opt(opts,'ywin',(1:100)'/100);
    phiywin = eval_basis(ywin, inds(1:pind), basis);
    
    tst_stats = struct;
    tst_stats.pcs = Z_tst*B;
    tst_stats.pdfs = tst_stats.pcs*phiywin';
    tst_stats.tst_score = tst_score;
    tst_stats.tst_set = tst_set;
    if verbose
        fprintf('TEST: bw = %g, lambda = %g, ind = %g, score: %g (CVed in %g secs)\n',sigma2, lambda, phi_inds(phii), tst_score, toc(stime));
    end
end

cv_stats.trn_set = trn_set;
cv_stats.tst_set = tst_set;
cv_stats.min_score = min(hol_scores(:));
cv_stats.trn_scores = hol_scores;
cv_stats.sigma2s = sigma2s;
cv_stats.lambdas = lambdas;
cv_stats.pind = pind;
cv_stats.basis = basis;

end
