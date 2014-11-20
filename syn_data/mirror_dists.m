% Junier Oliva
% ------------
% Generate input/output samples from random 2d Gaussian mixtures, and their
% projection coefficients. Input distribution is mixture, output is the
% input distribution mirrored.

DIMS = 2; % dimension of data
BOX = 8; % truncation box for data

N_MIX = 5; % number of mixture components
N_SAMP_MIX = 200; % samples per mixture component
N_SAMP = N_SAMP_MIX*N_MIX; % total number of samples
if ~exist('N_DISTS','var')
    N_DISTS = 100000; % number of total input/output distribution pairs
    N_TRAIN = 96000; % number of pairs for training set (rest are hold-out/test)
end
N_HT = N_DISTS-N_TRAIN;

% max 2-norm of projection coeficient indices
if ~exist('M','var')
    M = 18;
end
[i1, i2] = ind2sub([M M], (1:M^2)');
inds = [i1 i2];
inds = inds(sum(inds.^2,2)<=M^2,:);
M_n = size(inds,1);
% projection estimator basis functions
phi_k = @(x,k) (k==1)+(k>1)*sqrt(2)*cos((k-1)*pi*x);
phi_m = ( ((1:M)-1)*pi )';
phi_c = [ 1; sqrt(2)*ones(M-1,1)];

% mixture statistics
d_sigmas = cell(1,N_DISTS);
d_mus = cell(1,N_DISTS);

% projection coeficients
d_projcoef_inp = nan(M_n,N_DISTS);
d_projcoef_out = nan(M_n,N_DISTS);

% generate samples
parfor distro = 1:N_DISTS
    sigmas = zeros(DIMS,DIMS,N_MIX);
    mus = zeros(DIMS,N_MIX);
    samples1 = nan(N_SAMP,DIMS);
    samples2 = nan(N_SAMP,DIMS);
    % add samples from each mixture component
    for i=1:N_MIX
        A = (rand+1)*(2*rand(DIMS)-1); 
        A = A*A'+diag(rand(DIMS,1));
        sigmas(:,:,i) = A;
        B = 10*rand(1,DIMS)-5;
        mus(:,i) = B;
        samples1((i-1)*(N_SAMP_MIX)+1:i*(N_SAMP_MIX),:) = mvnrnd(B, A,N_SAMP_MIX);
        samples2((i-1)*(N_SAMP_MIX)+1:i*(N_SAMP_MIX),:) = mvnrnd(-B, A,N_SAMP_MIX);
    end
    % save parameters
    d_sigmas{distro} = sigmas;
    d_mus{distro} = mus;
    % dispose of samples out of box, scale to unit box
    in_samples1 = (-BOX<=samples1(:,1))&(samples1(:,1)<= BOX) & (-BOX<=samples1(:,2))&(samples1(:,2)<= BOX);
    in_samples2 = (-BOX<=samples2(:,1))&(samples2(:,1)<= BOX) & (-BOX<=samples2(:,2))&(samples2(:,2)<= BOX);
    samples1 = samples1(in_samples1,:);
    samples2 = samples2(in_samples2,:);
    samples1 = ((samples1 - (-BOX))/(BOX - (-BOX)))';
    samples2 = ((samples2 - (-BOX))/(BOX - (-BOX)))';
    % input distribution projection estimate 
    % compute 1d projection coefficients
    Phi1 =  bsxfun(@times,cos(bsxfun(@times,samples1(1,:),phi_m)),phi_c);
    Phi2 =  bsxfun(@times,cos(bsxfun(@times,samples1(2,:),phi_m)),phi_c);
    % do tensor-like product and take mean
    B = (Phi1(inds(:,1),:).*Phi2(inds(:,2),:));
    d_projcoef_inp(:,distro) = mean(B,2);
    % output distribution projection estimate 
    % compute 1d projection coefficients
    Phi1 =  bsxfun(@times,cos(bsxfun(@times,samples2(1,:),phi_m)),phi_c);
    Phi2 =  bsxfun(@times,cos(bsxfun(@times,samples2(2,:),phi_m)),phi_c);
    % do tensor-like product and take mean
    B = (Phi1(inds(:,1),:).*Phi2(inds(:,2),:));
    d_projcoef_out(:,distro) = mean(B,2);
end

