function [Xin, Xout, Y, d_mus, d_sigmas] = mirror_gmm(maxmix, n, N)
%generate_gmm_dists     Generate input samples from random 2d 
%  Gaussian mixtures. Input distribution is mixture, output is the number
%  of mixture components.
%  Inputs -
%    maxmix: max number of possible mixtures
%    n: max number of points per sample
%    N: number of distributions to generate
%  Outputs - 
%    X: N x 1 cell of sample sets
%    Y: N x 1 vector of indicating number of mixture for each distribution
%    d_mus: N x 1 cell of mixture means
%    d_sigmas: N x 1 cell of mixture precision matrices

d = 2; % dimension of data
BOX = 8; % truncation box for data

% mixture statistics
d_sigmas = cell(N,1);
d_mus = cell(N,1);

% projection coeficients
Xin = cell(N,1);
Xout = cell(N,1);
Y = nan(N,1);

% generate samples
parfor distro = 1:N
    nmix = randi(maxmix);
    sigmas = zeros(d,d,nmix);
    mus = zeros(d,nmix);
    samples = nan(nmix*ceil(n/nmix),d);
    samples2 = nan(nmix*ceil(n/nmix),d);
    
    % add samples from each mixture component
    N_SAMP_MIX = ceil(n/nmix);
    for i=1:nmix
        A = (rand+1)*(2*rand(d)-1); 
        A = A*A'+diag(rand(d,1));
        sigmas(:,:,i) = A;
        B = 10*rand(1,d)-5;
        mus(:,i) = B;
        samples((i-1)*(N_SAMP_MIX)+1:i*(N_SAMP_MIX),:) = mvnrnd(B, A,N_SAMP_MIX);
        samples2((i-1)*(N_SAMP_MIX)+1:i*(N_SAMP_MIX),:) = mvnrnd(-B, A,N_SAMP_MIX);
    end
    samples = samples(randperm(nmix*ceil(n/nmix)),:);
    samples2 = samples2(randperm(nmix*ceil(n/nmix)),:);
    
    % save parameters
    d_sigmas{distro} = sigmas;
    d_mus{distro} = mus;
    
    % dispose of samples out of box, scale to unit box
    samples = truncate(samples, BOX);
    samples2 = truncate(samples2, BOX);
    
    Xin{distro} = samples;
    Xout{distro} = samples2;
    Y(distro) = nmix;
    
    if mod(distro,1000)==0
        fprintf('{made distro %g}\n',distro);
    end
end

end

function trun_samp = truncate(samples, BOX)
    in_samples1 = (-BOX<=samples(:,1))&(samples(:,1)<= BOX) & (-BOX<=samples(:,2))&(samples(:,2)<= BOX);
    samples = samples(in_samples1,:);
    trun_samp = ((samples - (-BOX))/(BOX - (-BOX)));
end
