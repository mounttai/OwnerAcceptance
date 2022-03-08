%% Objective function of the weighted segment membership
%{

    Author: Dai Yao (dai@yaod.ai; http://www.yaod.ai)

%}
function LL = ObjSegMembership(...
    phis, OwnerInfo,...
    pMOwners, pSOwners)

expGroupWgts = exp(OwnerInfo * phis');

LLOwners = pMOwners .* log(1.0 ./ (1+expGroupWgts)) + ...
    pSOwners .* log(expGroupWgts ./ (1+expGroupWgts));

LL = - sum(LLOwners);
