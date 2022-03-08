%% set up the core environment for estimation of the dynamic model
%{

    Author: Dai Yao (dai@yaod.ai; http://www.yaod.ai)

%}

%{
    Owner characteristics
%}
OwnerGenders = 0:1;
len_g_owner = length(OwnerGenders);
pg_owner = [0.8, 0.2];
OwnerAgeGroups = 0:1;
len_a_owner = length(OwnerAgeGroups);
pa_owner = [0.45, 0.55];
OwnerTenureGroups = 0:1;
len_t_owner = length(OwnerTenureGroups);
pt_owner = [0.65, 0.35];

%{
    Renter characteristics
%}
RenterGenders = 0:1;
RenterAgeGroups = 0:1;
RenterTenureGroups = 0:1;
len_g = length(RenterGenders);
%pg = ones(1, len_g) / len_g;
if len_g==1
    pg = 1;
else
    pg = [0.7, 0.3];
end
len_a = length(RenterAgeGroups);
%pa = ones(1, len_a) / len_a;
pa = [];
if len_a==1
    pa = 1;
elseif len_a==2
    pa = [0.65, 0.35];
else
    pa = ones(1, len_a) / len_a;
end
len_t = length(RenterTenureGroups);
pt = [];
if len_t==1
    pt = 1;
elseif len_t==2
    pt = [0.9, 0.1];
else
    pt = ones(1, len_a) / len_a;
end

% discount factor
beta = 0.9997;

len_Lambda = 1;
Lambdas = zeros(1, len_Lambda);
Lambdas(1) = 0.3;
if len_Lambda>1
    for l=2:len_lambda
        Lambdas(l) = Lambdas(l-1)+0.1;
        if Lambdas(l)>1
            Lambdas(l) = 1;
        end
    end
end

% probabilities of observing lead time
LeadTimes = 1:2;
len_c = length(LeadTimes);
pc = ones(1, len_c) / len_c;
% probabilities of observing duration
Durations = 1:2;
len_d = length(Durations);
pd = ones(1, len_d) / len_d;

% the availability vector B
len_B = len_c + len_d - 1;
B = de2bi(0:(2^len_B-1));
sz_B = size(B);
len_B = sz_B(1);

%-- given B, whether (c, d) is feasible, i.e., not in conflict with B
ReqFsbState = ones(len_B, len_c, len_d);
for b=1:len_B
    b_bi = B(b, :);
    for c=1:len_c
        for d=1:len_d
            if (sum(b_bi(c:(c+d-1)))<d)
                ReqFsbState(b, c, d) = 0;
            end
        end
    end
end

%-- the probabilities of observing (c, d, g);
ProbRentalChar = zeros(1, len_c, len_d, len_g, len_a, len_t);
for c=1:len_c
    for d=1:len_d
        for g=1:len_g
            for a=1:len_a
                for t=1:len_t
                    ProbRentalChar(1, c, d, g, a, t) = pc(c) * pd(d) * pg(g) * pa(a) * pt(t);
                end
            end
        end
    end
end