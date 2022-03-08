%% Estimate the model assuming all owners are myopic
%{

    Author: Dai Yao (dai@yaod.ai; http://www.yaod.ai)

%}

clear

addpath('OwnerAcceptance');
addpath('OwnerAcceptance/utils/');

data_type = 1;
filename = "data/data-"+num2str(data_type)+".mat";
load(filename);

%- load some core variables
SetupCoreEnv;

% the utility coefficients
thetas0 = zeros(1, size(RentalInfo, 2));

%-- estimation
% estimation precision
rho = 1e-4;
options = optimset('Display','iter', 'TolX', rho, 'TolFun', rho, 'MaxIter', 50);
%-- first using all the data
% iDrops - set all to -1
iDrops = -1 * ones(1, length(OwnerNumRequests));
ObjM = @(thetas) ObjMyopicModel(...
    thetas, beta, ...
    OwnerStartIndices, OwnerNumRequests, ...
    RentalStates, RentalInfo, OwnerDecisions,...
    iDrops, 0, []);
[thetas, LL, ExitFlag] = fminunc(ObjM, thetas0, options);

filename = "results/est-myopic-data-"+num2str(data_type)+".mat";
save(filename, ...
    'thetas', 'LL');
