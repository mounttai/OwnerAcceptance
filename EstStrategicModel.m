%% Estimate the model assuming all owners are strategic
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
%- the W array
% obtain the sizes of all possible owner characteristics
len_g_owner = length(OwnerGenders);
len_a_owner = length(OwnerAgeGroups);
len_t_owner = length(OwnerTenureGroups);
% obtain the sizes of various state variables
len_Lambda = size(Lambdas, 2);
len_B = size(B, 1);

len_c = length(LeadTimes);
len_d = length(Durations);

len_g = length(RenterGenders);
len_a = length(RenterAgeGroups);
len_t = length(RenterTenureGroups);
% dimension of Ws
numW = len_g_owner * len_a_owner * len_t_owner;
WIndices = zeros(numW, 3);
tmpIndex = 1;
for g=1:len_g_owner
    for a=1:len_a_owner
        for t=1:len_t_owner
            WIndices(tmpIndex,:) = [g, a, t];
            tmpIndex = tmpIndex + 1;
        end
    end
end
Ws = zeros([numW, ...
    len_Lambda, ...
    len_B, len_c, len_d, ...
    len_g, len_a, len_t, 2]);
save("results-test/tmp/Ws.mat", 'Ws');

% estimation precision
rho = 1e-4;
options = optimset('Display','iter', 'TolX', rho, 'TolFun', rho, 'MaxIter', 50);
%-- first using all the data
% iDrops - set all to -1
iDrops = -1 * ones(1, length(OwnerNumRequests));
ObjS = @(thetas) ObjStrategicModel(...
    thetas, beta, Ws, WIndices,...
    OwnerIndices, OwnerStartIndices, OwnerNumRequests, ...
    RentalStates, RenterIndices, OwnerDecisions,...
    iDrops, ...
    Lambdas, ...
    LeadTimes, Durations, B, ...
    OwnerGenders, OwnerAgeGroups, OwnerTenureGroups, ...
    RenterGenders, RenterAgeGroups, RenterTenureGroups, ...
    ProbRentalChar, ...
    ReqFsbState, ...
    0, []);

[thetas, LL, ExitFlag] = fminunc(ObjS, thetas0, options);

filename = "results-test/est-strategic-data-"+num2str(data_type)+".mat";
save(filename, ...
    'thetas', 'LL');

