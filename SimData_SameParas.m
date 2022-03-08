%% simulate the data for estimation
%{
    data type:
        - 1, myopic
        - 2, strategic
        - 3, mixed

    Author: Dai Yao (dai@yaod.ai; http://www.yaod.ai)
%}

clear;

addpath('OwnerAcceptance');
addpath('OwnerAcceptance/utils/');

data_type = 1;  % all myopic owners
data_type = 2;  % all strategic owners
data_type = 3;  % some myopic, and the rest strategic

NumOwners = 1000;
AvgNumRequests = 10;

%-- setup the core environment
SetupCoreEnv;

% [Intercept, 
%   RenterGender=F, RenterGender=OwnerGender, 
%   RenterAge=1, RenterAge=OwnerAge, 
%   RenterTenure=1, RenterTenure=OwnerTenure]
thetas_mp = [0.1, 0.3, -0.7, 0, 0.2, 0.3, 0.1];
thetas_fl = [0.1, 0.3, -0.7, 0, 0.2, 0.3, 0.1];
% [Intercept, OwnerGender=F, OwnerAgeGroup=2, OwnerTenureGroup=2]
phis = [-1, -0.5, 0.2, 0.8];

% OwnerInfo record the owner information that can be readily multiplied by phis
OwnerInfo = zeros(NumOwners, length(phis));
% OwnerIndices record the values of OwnerGender, OwnerAgeGroup, and OwnerTenureGroup
OwnerIndices = zeros(NumOwners, 3);
OwnerTypes = zeros(1, NumOwners);
% the starting index for each owner
OwnerStartIndices = zeros(1, NumOwners);
% number of requests for each owner
OwnerNumRequests = zeros(1, NumOwners);

parfor o=1:NumOwners
    % generate random owner gender, age group, and tenure group
    tmpOwnerGender = binornd(1, pg_owner(2))+1;
    tmpOwnerAgeGroup = binornd(1, pa_owner(2))+1;
    tmpOwnerTenureGroup = binornd(1, pt_owner(2))+1;
    OwnerIndices(o, :) = [tmpOwnerGender, tmpOwnerAgeGroup, tmpOwnerTenureGroup];
    
    % owner gender in OwnerInfo
    tmpOwnerInfo = zeros(1, len_g_owner + len_a_owner + len_t_owner);
    tmpOwnerInfo(1, tmpOwnerGender) = 1;
    tmpOwnerInfo(1, len_g_owner + tmpOwnerAgeGroup) = 1;
    tmpOwnerInfo(1, len_g_owner+len_a_owner + tmpOwnerTenureGroup) = 1;
    
    tmpOwnerInfo = tmpOwnerInfo(1, [(2:len_g_owner), ...
        len_g_owner+(2:len_a_owner), ...
        len_g_owner+len_a_owner+(2:len_t_owner)]);
    OwnerInfo(o, :) = [1, tmpOwnerInfo];  % add the intercept
    
    % number of requests for this owner
    OwnerNumRequests(o) = max(...
        4, AvgNumRequests + round(2 * randn())); 
end

% label the starting observation for each owner in the whole data
NumTotalRentals = 0;
for o=1:NumOwners
    % record the starting index of this owner
    OwnerStartIndices(o) = NumTotalRentals;
    NumTotalRentals = NumTotalRentals + OwnerNumRequests(o);
end

% RentalStates record the latent state characterized by [T, b, c, d]
RentalStates = zeros(NumTotalRentals, 4);
% RentalInfo record the rental information that can be multiplied by thetas
% [1; RenterGender, SameGender; RenterAgeGroup, SameAgeGroup; RenterTenureGroup, SameTenureGroup]
RentalInfo = zeros(NumTotalRentals, length(thetas_mp));
% RenterIndices = [RenterGender, RenterAgeGroup, and RenterTenureGroup]
RenterIndices = zeros(NumTotalRentals, 3);

% acceptance decisions of the owners
OwnerDecisions = zeros(NumTotalRentals, 1);

Ws = [];
if data_type>1
    CPUs = zeros(len_g_owner, len_a_owner, len_t_owner, len_c, len_d, len_g, len_a, len_t);
    % compute the current period utilities (CPUs)
    parfor go=1:len_g_owner
        for ao=1:len_a_owner
            for to=1:len_t_owner
                for c=1:len_c
                    for d=1:len_d
                        for g=1:len_g
                            for a=1:len_a
                                for t=1:len_t
                                    CPUs(go, ao, to, c, d, g, a, t) = ComputeU(...
                                        go, ao, to, c, d, g, a, t, ...
                                        thetas_fl, beta, len_g, len_a, len_t);
                                end
                            end
                        end
                    end
                end
            end
        end
    end
    
    % dimension of Ws
    Ws = zeros([len_g_owner, len_a_owner, len_t_owner, ...
        len_Lambda, ...
        len_B, len_c, len_d, ...
        len_g, len_a, len_t, 2]);

    parfor g=1:len_g_owner
        for a=1:len_a_owner
            for t=1:len_t_owner
                Ws(g, a, t,:,:,:,:,:,:,:,:) = ComputeW(...
                    g, a, t, ...
                    Lambdas, ...
                    LeadTimes, Durations, B, ...
                    RenterGenders, RenterAgeGroups, RenterTenureGroups, ...
                    ProbRentalChar, ...
                    ReqFsbState, ...
                    CPUs, beta);
            end
        end
    end
end

% generate the details of the rentals, and owners' decisions
for o=1:NumOwners
    %- retrieve the owner information
    tmpOwnerIndex = OwnerIndices(o, :);
    tmpOwnerGender = tmpOwnerIndex(1);
    tmpOwnerAgeGroup = tmpOwnerIndex(2);
    tmpOwnerTenureGroup = tmpOwnerIndex(3);
    
    %- determine which type the owner is
    o_thetas = [];
    o_type = data_type;
    if data_type==1                 % all are myopic owners
        o_thetas = thetas_mp;
    elseif data_type==2             % all are strategic owners
        o_thetas = thetas_fl;
    else                            % some are myopic, the rest are strategic
        tmpGroupWgt = exp(sum(OwnerInfo(o,:) .* phis));
        o_type = binornd(1, tmpGroupWgt/(1.0+tmpGroupWgt)) + 1;
        if o_type==1                % myopic owner
            o_thetas = thetas_mp;
        else                        % strategic owner
            o_thetas = thetas_fl;
        end
    end
    OwnerTypes(o) = o_type;
    
    %- if owner is stragic, retrieve W from Ws
    W = [];
    if o_type==2
        W = Ws(tmpOwnerGender, tmpOwnerAgeGroup, ...
            tmpOwnerTenureGroup,:,:,:,:,:,:,:,:);
    end
    
    for r=1:OwnerNumRequests(o)
        
        % generate (b, c, d), make sure the request (c,d) is feasible given B(b,:)
        while 1
            tmp_b = randi(len_B, 1, 1);
            tmp_c = randi(len_c, 1, 1);
            tmp_d = randi(len_d, 1, 1);
            if ReqFsbState(tmp_b, tmp_c, tmp_d)==1
                break
            end
        end
        
        % randomly generate 1<=T<=len_Lambda
        tmp_T = randi(len_Lambda, 1, 1);
        
        % record RentalStates
        RentalStates(OwnerStartIndices(o)+r, :) = [tmp_T, tmp_b, tmp_c, tmp_d];
        
        % randomly generate renter characteristics
        tmpRenterGender = binornd(1, pg(2)) + 1;
        tmpRenterAgeGroup = binornd(1, pa(2)) + 1;
        tmpRenterTenureGroup = binornd(1, pt(2)) + 1;
        
        % record RenterIndices
        RenterIndices(OwnerStartIndices(o)+r, :) = ...
            [tmpRenterGender, tmpRenterAgeGroup, tmpRenterTenureGroup];
        
        % [OwnerInfo == RenterInfo]
        tmpSameGender = 1 * (tmpRenterGender == tmpOwnerGender);
        tmpSameAgeGroup = 1 * (tmpRenterAgeGroup == tmpOwnerAgeGroup);
        tmpSameTenureGroup = 1 * (tmpRenterTenureGroup == tmpOwnerTenureGroup);
        
        % variables in RentalInfo
        tmpRentalInfo = zeros(1, (len_g+1)+(len_a+1)+(len_t+1));
        tmpRentalInfo(tmpRenterGender) = 1; % only one element from 1:len_g will be assigned 1
        tmpRentalInfo(len_g+1) = tmpSameGender; % same gender
        tmpRentalInfo((len_g+1)+tmpRenterAgeGroup) = 1; % only one element from (len_g+1)+(1:len_a) will be assigned 1
        tmpRentalInfo((len_g+1)+(len_a+1)) = ...
            tmpSameAgeGroup; % same age group
        tmpRentalInfo((len_g+1)+(len_a+1)+tmpRenterTenureGroup) = 1; % only one element from (len_g+1)+(len_a+1)+(1:len_t) will be assigned 1
        tmpRentalInfo((len_g+1)+(len_a+1)+(len_t+1)) = ...
            tmpSameTenureGroup;  % same tenure group

        % ignore RenterGender=1, RenterAgeGroup=1, and RenterTenureGroup=1
        tmpRentalInfo = tmpRentalInfo([2:(len_g+1), ...
            (len_g+1)+(2:(len_a+1)), ...
            (len_g+1)+(len_a+1)+(2:(len_t+1))]);
        RentalInfo(OwnerStartIndices(o)+r, :) = [1, tmpRentalInfo];  % add the intercept
        
        exp_utils = ones(1, 2);        % [accept, reject]
        if o_type==1                    % myopic owner
            % calculate the exp(utility)
            exp_utils(1) = exp(...
                tmp_d * sum( ...
                    o_thetas .* RentalInfo(OwnerStartIndices(o)+r, :) ...
                ));
            % exp_utils(2) = 1;         % no need to update
        else                            % strategic owner
            exp_utils(1) = exp(...
                W(1, 1, 1, tmp_T, tmp_b, tmp_c, tmp_d, ...
                tmpRenterGender, tmpRenterAgeGroup, tmpRenterTenureGroup, 2));
            exp_utils(2) = exp(...
                W(1, 1, 1, tmp_T, tmp_b, tmp_c, tmp_d, ...
                tmpRenterGender, tmpRenterAgeGroup, tmpRenterTenureGroup, 1));
        end
        
        OwnerDecisions(OwnerStartIndices(o)+r) = ...
            binornd(1, exp_utils(1)/sum(exp_utils));
        
    end
end

%---- save the data: OwnerInfo, OwnerTypes, OwnerNumRequests, RentalInfo, OwnerDecisions
filename = "data/data-"+num2str(data_type)+".mat";
save(filename, ...
    'OwnerIndices', 'OwnerInfo', 'OwnerTypes', 'OwnerStartIndices', 'OwnerNumRequests', ...
    'RenterIndices', 'RentalStates', 'RentalInfo', 'OwnerDecisions');

tabulate(OwnerDecisions)

if data_type==3
    tabulate(OwnerTypes)
end
