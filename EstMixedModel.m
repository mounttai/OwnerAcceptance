%% Estimate the model assuming part of the owners are myopic, and the rest are strategic
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

%- set up the initial parameters (current parameters)
num_thetas = size(RentalInfo, 2);
filename = "results/est-myopic-data-"+num2str(data_type)+".mat";
load(filename, 'thetas');
cur_thetas_mp = thetas;
filename = "results/est-strategic-data-"+num2str(data_type)+".mat";
load(filename, 'thetas');
cur_thetas_fl = thetas;
num_phis = size(OwnerInfo, 2);
cur_phis = zeros(1, num_phis);
cur_paras = [cur_thetas_mp, cur_thetas_fl, cur_phis];

%- initial segment membership probabilities
NumOwners = size(OwnerInfo, 1);
expGroupWgts = exp(OwnerInfo * cur_phis');

cur_pMOwners = 1.0*ones(NumOwners, 1) ./ (1.0+expGroupWgts);
cur_pSOwners = 1 - cur_pMOwners;

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
%save("results-test/tmp/Ws.mat", 'Ws');

% estimation precision
rho = 1e-5;
options = optimset('Display','iter', 'TolX', rho1, 'TolFun', rho1, 'MaxIter', 50);
options_nodisp = optimset('TolX', rho1, 'TolFun', rho1, 'MaxIter', 50);
iDrops = -1 * ones(1, length(OwnerTypes));

iter = 1;

while 1
    
    iter

    prev_paras = cur_paras;
    
    %{
        Step 1: obtain the optimal utility coefficients for myopic owners
        and strategic owners separately
    %}
    % model assuming myopic owners
    ObjM = @(thetas) ObjMyopicModel(...
        thetas, beta, ...
        OwnerStartIndices, OwnerNumRequests, ...
        RentalStates, RentalInfo, OwnerDecisions,...
        iDrops, ...
        1, cur_pMOwners);
    cur_thetas_mp = fminunc(ObjM, cur_thetas_mp, options);

    % model assuming strategic owners
    %load("results-test/tmp/Ws.mat", 'Ws');
    ObjS = @(thetas) ObjStrategicModel(...
        thetas, beta, Ws, WIndices, ...
        OwnerIndices, OwnerStartIndices, OwnerNumRequests, ...
        RentalStates, RenterIndices, OwnerDecisions,...
        iDrops, ...
        Lambdas, ...
        LeadTimes, Durations, B, ...
        OwnerGenders, OwnerAgeGroups, OwnerTenureGroups, ...
        RenterGenders, RenterAgeGroups, RenterTenureGroups, ...
        ProbRentalChar, ...
        ReqFsbState, ...
        1, cur_pSOwners);
    cur_thetas_fl = fminunc(ObjS, cur_thetas_fl, options);
    
    %{
        Step 2: obtain the optimal coefficients for segment membership
    %}
    ObjSeg = @(phis) ObjSegMembership(...
        phis, OwnerInfo, ...
        cur_pMOwners, cur_pSOwners);
    [cur_phis, LL] = fminunc(ObjSeg, cur_phis, options_nodisp);
    
    % put the new parameters together
    cur_paras = [cur_thetas_mp, cur_thetas_fl, cur_phis];
    cur_paras
    
    %{
        Step 3: update the posterior probability
    %}
    % new probabilities of decisions, assuming myopic owners
    [~, LLMOwnerDecisions] = ObjMyopicModel(...
        cur_thetas_mp, beta, ...
        OwnerStartIndices, OwnerNumRequests, ...
        RentalStates, RentalInfo, OwnerDecisions,...
        iDrops, ...
        0, []);
    % new probabilities of decisions, assuming strategic owners
    %load("results-test/tmp/Ws.mat", 'Ws');
    [~, LLSOwnerDecisions] = ObjStrategicModel(...
        cur_thetas_fl, beta, Ws, WIndices, ...
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
    % new probabilities of segment membership
    expGroupWgts = exp(OwnerInfo * cur_phis');
    LLMOwners = log(ones(NumOwners, 1) ./ (1.0+expGroupWgts));
    LLSOwners = log(expGroupWgts ./ (1+expGroupWgts));
    % update the segment membership
    cur_pMOwners = exp(LLMOwners + LLMOwnerDecisions) ./ (...
        exp(LLMOwners + LLMOwnerDecisions) + exp(LLSOwners + LLSOwnerDecisions));
    cur_pSOwners = 1 - cur_pMOwners;
    
    mean(cur_pMOwners)
    
    LL = sum(cur_pMOwners.*LLMOwnerDecisions + cur_pSOwners.*LLSOwnerDecisions);
    
    %{
        Step 4: determine if the algorithm should stop
    %}
    
    if max(abs(cur_paras-prev_paras), [], 'all') < rho
        break
    end
    
    iter = iter + 1;
    
end

cur_paras
LL

filename = "results/est-mixed-data-"+num2str(data_type)+".mat";
save(filename, ...
    'cur_paras', 'LL', 'cur_pMOwners', 'cur_pSOwners', ...
    'LLMOwners', 'LLSOwners', 'LLMOwnerDecisions', 'LLSOwnerDecisions');

