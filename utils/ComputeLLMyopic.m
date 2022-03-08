%% Compute the loglikelihood for myopic owners
%{
    iDrop: when it is -1, don't drop any, otherwise, drop the [iDrop]'s element

    Author: Dai Yao (dai@yaod.ai; http://www.yaod.ai)
%}
function LL = ComputeLLMyopic(...
    thetas, beta, ...
    thisOwnerRentalStates, thisOwnerRentalInfo, thisOwnerDecisions, ...
    iDrop)

thisOwnerNumRequests = length(thisOwnerDecisions);
LLRequests = zeros(thisOwnerNumRequests);

% iterate all the requests for current owner
for r=1:thisOwnerNumRequests
   
    % duration of the rental
    tmp_d = thisOwnerRentalStates(r, 4);
    % rental information
    tmpRentalInfo = thisOwnerRentalInfo(r, :);
    
    exp_utils = ones(1, 2);
    exp_utils(1) = exp(...
        tmp_d * sum(thetas .* tmpRentalInfo));
    
    % accept: log(exp(1)/[exp(1) + exp(0)])
    % reject: log(exp(0)/[exp(1) + exp(0)])
    LLRequests(r) = thisOwnerDecisions(r) * ...
        (log(exp_utils(1)) - log(sum(exp_utils))) + ...
        (1-thisOwnerDecisions(r)) * ...
        (log(exp_utils(2)) - log(sum(exp_utils)));
end

calibrateIndices = (1:thisOwnerNumRequests)~=iDrop;
LL = sum(LLRequests(calibrateIndices));
