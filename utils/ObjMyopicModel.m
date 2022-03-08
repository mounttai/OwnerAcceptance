%% Objective function of the model with all myopic owners
%{
    iDrops - the observations to drop from each owner
        - all are set as -1 for the full data
        - for boostrapping

    Author: Dai Yao (dai@yaod.ai; http://www.yaod.ai)
%}
function [LL, LLOwners] = ObjMyopicModel(...
    thetas, beta, ...
    OwnerStartIndices, OwnerNumRequests, ...
    RentalStates, RentalInfo, OwnerDecisions,...
    iDrops, ...
    bMixed, pMOwners)

NumOwners = length(OwnerStartIndices);
LLOwners = zeros(NumOwners, 1);

parfor o=1:NumOwners
    tmpStart = OwnerStartIndices(o);
    tmpNumRequests = OwnerNumRequests(o);
    thisOwnerRentalStates = RentalStates(tmpStart+(1:tmpNumRequests), :);
    thisOwnerRentalInfo = RentalInfo(tmpStart+(1:tmpNumRequests), :);
    thisOwnerDecisions = OwnerDecisions(tmpStart+(1:tmpNumRequests));
    LLOwners(o) = ComputeLLMyopic( ...
        thetas, beta, ...
        thisOwnerRentalStates, thisOwnerRentalInfo, thisOwnerDecisions,...
        iDrops(o));
end

% if we estimate a mixture model
if bMixed==1
    LLOwners = LLOwners .* pMOwners;
end

LL = - sum(LLOwners);
