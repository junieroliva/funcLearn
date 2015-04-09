% load data
fprintf('Loading data...\n')
A = importdata('/shared/downloads/facial_keypnts_training.csv');
fprintf('\tOpened file\n')
A = cellfun(@(C)strsplit(C,',','CollapseDelimiters',false),A,'UniformOutput',false);
fprintf('\tGot fields\n')
keys = cell2mat(cellfun(@(C)str2double(C(1:30)),A(2:end),'UniformOutput',false));
fprintf('\tGot keypoints\n')
imgs = cell2mat(cellfun(@(C)str2double(strsplit(C{end},' ')),A(2:end),'UniformOutput',false));
fprintf('\tGot images\n')
fprintf('Done loading data...\n')

% get first basis projection coefficients
fprintf('Getting projection coefficients...\n')
nd = 96;
xgrid = 1/nd:1/nd:1;
[x1,x2] = meshgrid(xgrid,xgrid);
x = [x1(:),x2(:)];
os_opts.inds = outerprodinds(0:50,2,50); % the set of indices for basis funcs
                                         % could CV, but keep fixed for now
PCs = osfe(x,imgs,os_opts);
fprintf('Got projection coefficients...\n')

% get the first key points
fprintf('Training...\n')
ki = 1;
nnan = ~isnan(keys(:,ki));
[B, rks, tst_stats, cv_stats, rfeats] = rks_ridge(PCs(nnan,:),keys(nnan,ki));