%% Compute the current period utility (CPU)
%{
    This is only determined by:
    - lead time and duration: c, d
    - owner information: gender, age, tenure
    - renter information: gender, age, tenure

    Author: Dai Yao (dai@yaod.ai; http://www.yaod.ai)
%}
function CPU = ComputeU(...
    g_owner, a_owner, t_owner, ...
    c, d, ...
    g, a, t, ...
    thetas, beta, ...
    len_g, len_a, len_t)

%- utility in the current period
CPU = 0;
for tau=(c+1):(c+d)
    U_flow = 0;
    % flow utility by renter characteristics
    theta_index = 2;
    if len_g>1
        if g>1	% only when RenterGender > 1 (i.e., only when RenterGender=Female)
            U_flow = U_flow + thetas(theta_index);
        end
        theta_index = theta_index + 1;
        %- SimilarGender=1
        if g_owner==g
            U_flow = U_flow + thetas(theta_index);
        end
        theta_index = theta_index + 1;
    end
    
    if len_a>1
        if a>1	% only when RenterAgeGroup > 1
            U_flow = U_flow + thetas(theta_index + (a-2));
        end
        theta_index = theta_index + (len_a-1);
        if a_owner==a      % same age group
            U_flow = U_flow + thetas(theta_index);
        end
        theta_index = theta_index + 1;
    end
    
    if len_t>1
        if t>1	% only when RenterTenureGroup > 1
            U_flow = U_flow + thetas(theta_index + (t-2));
        end
        theta_index = theta_index + (len_t-1);
        if t_owner==t   % same tenure group
            U_flow = U_flow + thetas(theta_index);
        end
        % theta_index = theta_index + 1;
    end
    
    CPU = CPU + (...
        thetas(1) + beta^(tau-1) * U_flow);

end