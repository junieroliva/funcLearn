function [tst_stats, cv_stats] = nwe(X, Y, varargin)

if isempty(varargin)
    opts = struct;
else
    opts = varargin{1};
end

N_ROT = 1000; % max number of pairs to estimate rule of thumb bandwidth
N_BAND_MULT = get_opt(opts,'N_BAND_MULT',[1E-20 1E-10 1/4096 1/512 1/256 1/128 1/64 1/32 1/16 1/8 1/4 1/2 1 2 4 8 32 64 128]);
% verbose = get_opt(opts,'verbose',false);

% get training/hold-out/testing sets
X = X';  % TODO: avoid transpose
Y = Y';
[di, N] = size(X);
tst_set = get_opt(opts,'tst_set');
tperc = get_opt(opts,'tperc', .1);
if isempty(tst_set) 
    tst_set = false(N,1);
    tst_set(randperm(N,ceil(N*tperc))) = true;
end

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
N_H = sum(hol_set);
N_T = sum(tst_set);

% get a ROT estimate of bandwidth/ bandwidths to CV
bands = get_opt(opts,'bands');
if isempty(bands)
    rI = trn_set;
    probI = min(N_ROT/N_TRAIN,1);
    rI(rI) = rand(1,N_TRAIN)<=probI;
    bROT = slmetric_pw(X(:,rI), X(:,rI),'sqdist');
    bROT = median(bROT(:));
    bands = bsxfun(@times,N_BAND_MULT,bROT);
end
nbands = length(bands);

% cross-validate bandwitdth
cv_stats.trn_set = trn_set;
cv_stats.hol_set = hol_set;
cv_stats.tst_set = tst_set;

ht_score = nan(N_H,nbands);
ho_scores = nan(nbands,1);
% get pairwise distance to the test/holdout
pDists = slmetric_pw(X(:,trn_set),X(:,hol_set),'sqdist');
pDists = bsxfun(@minus,pDists,min(pDists));
for bi = 1:nbands
    sigma_sq = bands(bi);
    w = exp(-pDists/(2*sigma_sq));
    w = bsxfun(@times,w,1./sum(w));
    pred_projs = Y(:,trn_set)*w;
    % for now just use the distance from predicted to projection series
    % estimator
    ht_score(:,bi) = sum((pred_projs-Y(:,hol_set)).^2,1);
    ho_scores(bi) = mean(ht_score(:,bi));
    fprintf('CV: bw = %g, score:%g \n',sigma_sq, ho_scores(bi));
end
cv_stats.ht_score = ht_score;
cv_stats.ho_scores = ho_scores;

% get optimal
bi = find(ho_scores==min(ho_scores(:)));
pDists = slmetric_pw(X(:,trn_set|hol_set), X(:,tst_set), 'sqdist');
pDists = bsxfun(@minus,pDists,min(pDists));

sigma_sq = bands(bi);
w = exp(-pDists/(2*sigma_sq));
w = bsxfun(@times,w,1./sum(w));
pred_projs = Y(:,trn_set|hol_set)*w;
tst_score = mean(sum((pred_projs-Y(:,tst_set)).^2,1));

ts = tic;
N_time = get_opt(opts,'N_time', 100);
test_inds = find(tst_set);
test_nw_pred_time_per = 0;
for i = 1:N_time
    tic;
    pDists = slmetric_pw(X(:,trn_set|hol_set),X(:,test_inds(i)),'sqdist');
    pDists = bsxfun(@minus,pDists,min(pDists));
    w = exp(-pDists/(2*sigma_sq));
    w = bsxfun(@times,w,1./sum(w));
    Y(:,trn_set|hol_set)*w;
    test_nw_pred_time_per = test_nw_pred_time_per + toc;
end
test_nw_pred_time_per = test_nw_pred_time_per/N_time;
tst_stats.pred_time = test_nw_pred_time_per;
 
mean_pred = mean( sum( bsxfun(@minus,Y(:,tst_set),mean(Y(:,trn_set|hol_set),2)).^2, 1) );
fprintf('TEST: bw = %g, score: %g, mean_pred score: %g\n',sigma_sq, tst_score, mean_pred);

tst_stats.tst_score = tst_score;
tst_stats.mean_pred = mean_pred;
end

