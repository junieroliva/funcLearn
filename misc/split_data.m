function [ trn_set, hol_set, tst_set ] = split_data( N, tprec, hprec )
%split_data Splits data into sets for training, validation, and testing
% Inputs - 
%   N: number of instances
%   tprec: percent (to be rounded) of instances to use for testing
%   hprec: percent (to be rounded) of instances to use for validation

tst_set = false(N,1);
tst_set(randperm(N,ceil(tprec*N))) = true;
Ntst = sum(tst_set);

hol_set = false(N,1);
hol_set_nt = false(N-Ntst,1);
hol_set_nt(randperm(N-Ntst,ceil(hprec*N))) = true;
hol_set(~tst_set) = hol_set_nt;

trn_set = ~hol_set & ~tst_set;
end

