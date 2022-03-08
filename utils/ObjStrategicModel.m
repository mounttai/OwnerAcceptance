%% Objective function of the model with all strategic owners
%{
    iDrops - the observations to drop from each owner
        - all are set as -1 for the full data
        - for boostrapping

    Author: Dai Yao (dai@yaod.ai; http://www.yaod.ai)
%}
function [LL, LLOwners] = ObjStrategicModel(...
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
    bMixed, pSOwners)

NumOwners = length(OwnerStartIndices);

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

CPUs = zeros(len_g_owner, len_a_owner, len_t_owner, len_c, len_d, len_g, len_a, len_t);
% compute the current period utilities (CPUs)
for go=1:len_g_owner
    for ao=1:len_a_owner
        for to=1:len_t_owner
            for c=1:len_c
                for d=1:len_d
                    for g=1:len_g
                        for a=1:len_a
                            for t=1:len_t
                                CPUs(go, ao, to, c, d, g, a, t) = ComputeU(...
                                    go, ao, to, c, d, g, a, t, ...
                                    thetas, beta, len_g, len_a, len_t);
                            end
                        end
                    end
                end
            end
        end
    end
end

% MOST TIME
numW = len_g_owner * len_a_owner * len_t_owner;
parfor w=1:numW
    Ws(w,:,:,:,:,:,:,:,:) = ComputeW(...
        WIndices(w, 1), WIndices(w, 2), WIndices(w, 3), Lambdas, ...
        LeadTimes, Durations, B, ...
        RenterGenders, RenterAgeGroups, RenterTenureGroups, ...
        ProbRentalChar, ...
        ReqFsbState, ...
        CPUs, beta);
end
%save("results-test/tmp/Ws.mat", 'Ws');

%- computing log-likelihood
LLOwners = zeros(NumOwners, 1);

parfor o=1:NumOwners
    tmpStart = OwnerStartIndices(o);
    tmpNumRequests = OwnerNumRequests(o);
    tmpOwnerIndex = OwnerIndices(o, :);
    tmpOwnerGender = tmpOwnerIndex(1);
    tmpOwnerAgeGroup = tmpOwnerIndex(2);
    tmpOwnerTenureGroup = tmpOwnerIndex(3);
    % W is the same as long as [OwnerGender, OwnerAgeGroup, OwnerTenureGroup] are the same
    w_index = (tmpOwnerGender-1)*(len_a_owner*len_t_owner) + ...
        (tmpOwnerAgeGroup-1)*len_t_owner + tmpOwnerTenureGroup;
    thisOwnerW = reshape(...
        Ws(w_index,:,:,:,:,:,:,:,:), ...
        [len_Lambda, len_B, len_c, len_d, len_g, len_a, len_t, 2]);
    thisOwnerRentalStates = RentalStates(tmpStart+(1:tmpNumRequests), :);
    thisOwnerRenterIndices = RenterIndices(tmpStart+(1:tmpNumRequests), :);
    thisOwnerDecisions = OwnerDecisions(tmpStart+(1:tmpNumRequests));
    % calculate the log-likelihood
    LLOwners(o) = ComputeLLStrategic(...
        thisOwnerW, ...
        thisOwnerRentalStates, thisOwnerRenterIndices, ...
        thisOwnerDecisions, ...
        iDrops(o));
end

% if we estimate a mixture model
if bMixed==1
    LLOwners = LLOwners .* pSOwners;
end

LL = - sum(LLOwners);
