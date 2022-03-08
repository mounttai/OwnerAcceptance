%% Compute the loglikelihood for strategic owners
%{
    iDrop: when it is -1, don't drop any, otherwise, drop the [iDrop]'s element

    Author: Dai Yao (dai@yaod.ai; http://www.yaod.ai)
%}
function LL = ComputeLLStrategic(...
    thisOwnerW, ...
    thisOwnerRentalStates, thisOwnerRenterIndices, ...
    thisOwnerDecisions, ...
    iDrop)

thisOwnerNumRequests = length(thisOwnerDecisions);
LLRequests = zeros(thisOwnerNumRequests);

% iterate all the requests for current owner
for r=1:thisOwnerNumRequests
    
    tmpRentalState = thisOwnerRentalStates(r,:);
    
    tmp_T = tmpRentalState(1);
    tmp_b = tmpRentalState(2);
    tmp_c = tmpRentalState(3);
    tmp_d = tmpRentalState(4);
    
    tmpRenterIndex = thisOwnerRenterIndices(r, :);
    tmpRenterGender = tmpRenterIndex(1);
    tmpRenterAgeGroup = tmpRenterIndex(2);
    tmpRenterTenureGroup = tmpRenterIndex(3);
    
    exp_utils = ones(1, 2);
    
    exp_utils(1) = exp(...
        thisOwnerW(tmp_T, tmp_b, tmp_c, tmp_d, ...
        tmpRenterGender, tmpRenterAgeGroup, tmpRenterTenureGroup, ...
        2));
    exp_utils(2) = exp(...
        thisOwnerW(tmp_T, tmp_b, tmp_c, tmp_d, ...
        tmpRenterGender, tmpRenterAgeGroup, tmpRenterTenureGroup, ...
        1));
    
    % accept: log(exp(1)/[exp(1) + exp(2)])
    % reject: log(exp(2)/[exp(1) + exp(2)])
    LLRequests(r) = thisOwnerDecisions(r) * ...
        (log(exp_utils(1)) - log(sum(exp_utils))) + ...
        (1-thisOwnerDecisions(r)) * ...
        (log(exp_utils(2)) - log(sum(exp_utils)));
end

calibrateIndices = (1:thisOwnerNumRequests)~=iDrop;
LL = sum(LLRequests(calibrateIndices));
