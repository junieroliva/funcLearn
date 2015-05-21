function [P, Q, d_mus, d_sigmas] = gen_mirror_samples(N, varargin)
% gen_mirror_samples Generate input/output samples where input samples come
%   from a random GMM, and output samples come from the mirrored input GMM.
% Inputs -
%   N: number of input/output sample bags to generate
% Outputs -
%   P: length N cell array of input sample bags; P{i} contains a nx2 matrix
%      of samples
%   Q: length N cell array of output sample bags
%   d_mus: length N cell array of input GMM means
%   d_sigmas: length N cell array of input GMM covariances

DIMS = 2; % dimension of data
BOX = 8; % truncation box for data

if isempty(varargin)
    opts = struct;
else
    opts = varargin{1};
end

% TODO: document these
N_MIX = get_opt(opts,'N_MIX',5); % number of mixture components
n_SAMP_MIX = get_opt(opts,'n_SAMP',100); % samples per mixture component
n = n_SAMP_MIX*N_MIX; % total number of samples

% mixture statistics
d_sigmas = cell(N,1);
d_mus = cell(N,1);

% samples
P = cell(N,1);
Q = cell(N,1);

% generate samples
for distro = 1:N
    sigmas = zeros(DIMS,DIMS,N_MIX);
    mus = zeros(DIMS,N_MIX);
    samples1 = nan(n,DIMS);
    samples2 = nan(n,DIMS);
    % add samples from each mixture component
    for i=1:N_MIX
        A = (rand+1)*(2*rand(DIMS)-1); 
        A = A*A'+diag(rand(DIMS,1));
        sigmas(:,:,i) = A;
        B = 10*rand(1,DIMS)-5;
        mus(:,i) = B;
        samples1((i-1)*(n_SAMP_MIX)+1:i*(n_SAMP_MIX),:) = mvnrnd(B, A,n_SAMP_MIX);
        samples2((i-1)*(n_SAMP_MIX)+1:i*(n_SAMP_MIX),:) = mvnrnd(-B, A,n_SAMP_MIX);
    end
    % save parameters
    d_sigmas{distro} = sigmas;
    d_mus{distro} = mus;
    % dispose of samples out of box, scale to unit box
    in_samples1 = (-BOX<=samples1(:,1))&(samples1(:,1)<= BOX) & (-BOX<=samples1(:,2))&(samples1(:,2)<= BOX);
    in_samples2 = (-BOX<=samples2(:,1))&(samples2(:,1)<= BOX) & (-BOX<=samples2(:,2))&(samples2(:,2)<= BOX);
    samples1 = samples1(in_samples1,:);
    samples2 = samples2(in_samples2,:);
    P{distro} = ((samples1 - (-BOX))/(BOX - (-BOX)));
    Q{distro} = ((samples2 - (-BOX))/(BOX - (-BOX)));
end

end

