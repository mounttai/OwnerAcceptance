%% This function calcuates the W(*) values using successive approximation
%{
 Inputs:
    - LambdaVector: arrival probabilities
    - B: availability states - 2-dim
    - LeadTimes: (0, 1, 2, 3)
    - Durations: (1, 2, 3, 4)

    - RenterGenders: (0 - male, 1, female)
    - RenterAgeGroups: (0, 1)
    - RenterTenureGroups: (0, 1)

    - ProbRentalChar: probability of renter characteristics

    - ReqFsbState: whether a request is feasible

    - thetas: the parameters
    - beta: the discount factor (set as 0.9997)

    Author: Dai Yao (dai@yaod.ai; http://www.yaod.ai)
%}

function W = ComputeW(...
    thisOwnerGender, thisOwnerAgeGroup, thisOwnerTenureGroup, ...
    LambdaVector, LeadTimes, Durations, B, ...
    RenterGenders, RenterAgeGroups, RenterTenureGroups, ...
    ProbRentalChar, ...
    ReqFsbState, ...
    Us, beta)

% obtain the sizes of various state variables
len_Lambdas = length(LambdaVector);
len_B = size(B, 1);
len_c = length(LeadTimes);
len_d = length(Durations);
len_g = length(RenterGenders);
len_a = length(RenterAgeGroups);
len_t = length(RenterTenureGroups);

%-- whether the rental (request + renter) is feasible
RentalFsbStates = zeros([size(ReqFsbState), len_g, len_a, len_t]);
for g=1:len_g
    for a=1:len_a
        for t=1:len_t
            RentalFsbStates(:,:,:,g,a,t) = ReqFsbState;
        end
    end
end

% dimension of W
W = zeros(len_Lambdas, ...
    len_B, len_c, len_d, ...
    len_g, len_a, len_t, 2);

%iter = 1;
while 1  % break when successive approximation is achieved.
    
    W_prev = W;  % the previous values
    eW = exp(W);
    
    for b=1:len_B  % each availability state
        for c=1:len_c  % each rental lead time
            for d=1:len_d  % each rental duration
                
                % only when (c,d) is feasible given B(b,:)
                if (ReqFsbState(b, c, d)==1)
                    %--- compute W(*, a=1) and W(*, a=0)
                    % F(B+(c,d)) is the new state for a=1
                    b_bi = B(b,:);
                    b_bi(c:(c+d-1)) = 0;   %---- OVERLAY Operator
                    b_bi_accept = [b_bi(2:end),1];  %---- FORWARD Operator
                    b_accept = bi2de(b_bi_accept)+1;
                    % F(B) is the new state for a=0
                    b_bi = B(b,:);
                    b_bi_reject = [b_bi(2:end),1];  %---- FORWARD Operator
                    b_reject = bi2de(b_bi_reject)+1;
                    
                    for g=1:len_g % renter gender
                        for a=1:len_a % renter age group
                            for t=1:len_t % renter tenure group

                                %- utility in the current period
                                W_cur = Us(...
                                    thisOwnerGender, thisOwnerAgeGroup, thisOwnerTenureGroup, ...
                                    c, d, g, a, t);

                                %- value function for a=1, lambdas(1)
                                %{
                                eW1 = reshape(eW(1, b_accept, :, :, :, :, :, 1), ...
                                    1, len_c, len_d, len_g, len_a, len_t);
                                eW2 = reshape(eW(1, b_accept, :, :, :, :, :, 2),...
                                    1, len_c, len_d, len_g, len_a, len_t);
                                %}
                                sum_p = sum(RentalFsbStates(b_accept,:,:,:,:,:) .* ...
                                    ProbRentalChar(1,:,:,:,:,:), 'all');
                                W_t1_accept = 0;
                                if sum_p>0
                                    log_sum_eW = reshape(...
                                        log(eW(1, b_accept, :, :, :, :, :, 1) + ...
                                        eW(1, b_accept, :, :, :, :, :, 2)), ...
                                        1, len_c, len_d, len_g, len_a, len_t);
                                    W_t1_accept = sum((1/sum_p)* ...
                                        RentalFsbStates(b_accept,:,:,:,:,:) .* ...
                                        ProbRentalChar(1,:,:,:,:,:) .* log_sum_eW, 'all');
                                end

                                %- value function for a=0, lambdas(1)
                                %{
                                eW1 = reshape(eW(1, b_reject, :, :, :, :, :, 1), ...
                                    1, len_c, len_d, len_g, len_a, len_t);
                                eW2 = reshape(eW(1, b_reject, :, :, :, :, :, 2),...
                                    1, len_c, len_d, len_g, len_a, len_t);
                                %}
                                sum_p = sum(RentalFsbStates(b_reject,:,:,:,:,:) .* ...
                                    ProbRentalChar(1,:,:,:,:,:), 'all');
                                W_t1_reject = 0;
                                if sum_p>0                                    
                                    log_sum_eW = reshape(...
                                        log(eW(1, b_reject, :, :, :, :, :, 1) + ...
                                        eW(1, b_reject, :, :, :, :, :, 2)), ...
                                        1, len_c, len_d, len_g, len_a, len_t);
                                    W_t1_reject = sum((1/sum_p)* ...
                                        RentalFsbStates(b_reject,:,:,:,:,:) .* ...
                                        ProbRentalChar(1,:,:,:,:,:) .* log_sum_eW, 'all');
                                end
                                
                                %-------------------------------------------------
                                % for each t (# of days since last request),
                                % compute its value functions
                                %-------------------------------------------------
                                for T=1:len_Lambdas

                                    %-------------------------------------------------
                                    % look ahead [len_Lambdas] days
                                    %-------------------------------------------------
                                    %- probability of observing a request on each day
                                    probs_tk = zeros(1, len_Lambdas);
                                    %- utility
                                    Ws_accept_tk = zeros(1, len_Lambdas);
                                    Ws_reject_tk = zeros(1, len_Lambdas);
                                    % discount factor
                                    df_tk = 1;

                                    %- one period ahead following acceptance/rejection
                                    probs_tk(1) = LambdaVector(T);
                                    Ws_accept_tk(1) = df_tk * probs_tk(1) * W_t1_accept;
                                    Ws_reject_tk(1) = df_tk * probs_tk(1) * W_t1_reject;

                                    k = 1;

                                    %-------------------------------------------------
                                    % stop whenever an request arrives with 100% certainty.
                                    %-------------------------------------------------
                                    p_no_request_till_tk = (1-LambdaVector(T));
                                    if (len_Lambdas>1) && (p_no_request_till_tk>0)

                                        %- current discount factor
                                        df_tk = df_tk * beta;

                                        %- availability states in next period, in the case of no request arrival
                                        b_bi_accept_tk = [b_bi_accept(2:end),1];
                                        b_bi_reject_tk = [b_bi_reject(2:end),1];
                                        b_accept_tk = bi2de(b_bi_accept_tk)+1;
                                        b_reject_tk = bi2de(b_bi_reject_tk)+1;

                                        for k=2:len_Lambdas
                                            % current period (t), period for the delayed request (t+k)
                                            T2 = min(len_Lambdas, T+k);
                                            % stop whenever an request arrives with 100% certainty.
                                            if LambdaVector(T2)==1
                                                break
                                            end
                                            %
                                            %{
                                            eW1 = reshape(eW(T2,b_accept_tk,:,:,:,:,:,1), ...
                                                1, len_c, len_d, len_g, len_a, len_t);
                                            eW2 = reshape(eW(T2,b_accept_tk,:,:,:,:,:,2),...
                                                1, len_c, len_d, len_g, len_a, len_t);
                                            %}
                                            sum_p = sum(RentalFsbStates(b_accept_tk,:,:,:,:,:) .* ...
                                                ProbRentalChar(1,:,:,:,:,:), 'all');
                                            W_accept_tk = 0;
                                            if sum_p>0
                                                log_sum_eW = reshape(...
                                                    log(eW(T2, b_accept_tk, :, :, :, :, :, 1) + ...
                                                    eW(T2, b_accept_tk, :, :, :, :, :, 2)), ...
                                                    1, len_c, len_d, len_g, len_a, len_t);
                                                W_accept_tk = sum((1/sum_p)* ...
                                                    RentalFsbStates(b_accept_tk,:,:,:,:,:) .* ...
                                                    ProbRentalChar(1,:,:,:,:,:) .* log_sum_eW, 'all');
                                            end

                                            %
                                            %{
                                            eW1 = reshape(eW(T2,b_reject_tk,:,:,:,:,:,1), ...
                                                1, len_c, len_d, len_g, len_a, len_t);
                                            eW2 = reshape(eW(T2,b_reject_tk,:,:,:,:,:,2),...
                                                1, len_c, len_d, len_g, len_a, len_t);
                                            %}
                                            sum_p = sum(RentalFsbStates(b_reject_tk,:,:,:,:,:) .* ...
                                                ProbRentalChar(1,:,:,:,:,:), 'all');
                                            W_reject_tk = 0;
                                            if sum_p>0
                                                log_sum_eW = reshape(...
                                                    log(eW(T2, b_reject_tk, :, :, :, :, :, 1) + ...
                                                    eW(T2, b_reject_tk, :, :, :, :, :, 2)), ...
                                                    1, len_c, len_d, len_g, len_a, len_t);
                                                W_reject_tk = sum((1/sum_p)* ...
                                                    RentalFsbStates(b_reject_tk,:,:,:,:,:) .* ...
                                                    ProbRentalChar(1,:,:,:,:,:) .* log_sum_eW, 'all');
                                            end

                                            %
                                            % the probability of observing a request
                                            probs_tk(k) = p_no_request_till_tk * LambdaVector(T2);
                                            % the utilities when initial decision is acceptance/rejection
                                            Ws_accept_tk(k) = df_tk * probs_tk(k) * W_accept_tk;
                                            Ws_reject_tk(k) = df_tk * probs_tk(k) * W_reject_tk;

                                            %-------------------------------------------------
                                            % prepare the parameters for one period further
                                            %-------------------------------------------------

                                            %- update discount factor
                                            df_tk = df_tk * beta;
                                            %- update prob(no request until t+k+1)
                                            p_no_request_till_tk = p_no_request_till_tk * (1-LambdaVector(T2));
                                            %- update availability state
                                            b_bi_accept_tk = [b_bi_accept_tk(2:end),1];
                                            b_bi_reject_tk = [b_bi_reject_tk(2:end),1];
                                            b_accept_tk = bi2de(b_bi_accept_tk)+1;
                                            b_reject_tk = bi2de(b_bi_reject_tk)+1;
                                        end

                                    end

                                    %-------------------------------------------------
                                    % compute the average utility weighted by arrival probabilities
                                    %-------------------------------------------------
                                    probs_tk(1:k) = probs_tk(1:k) / sum(probs_tk(1:k));
                                    W_weighted_accept = sum(Ws_accept_tk(1:k) .* probs_tk(1:k));
                                    W_weighted_reject = sum(Ws_reject_tk(1:k) .* probs_tk(1:k));

                                    %- value of acceptance
                                    W(T, b, c, d, g, a, t, 1) = W_cur + ... 
                                        beta * W_weighted_accept;
                                    %- value of rejection
                                    W(T, b, c, d, g, a, t, 2) = 0 + ...
                                        beta * W_weighted_reject;

                                end
                                
                                % end state space defined by [LambdaVector]
                                
                            end
                        end
                        
                        % end state space defined by [renter gender, age group, tenure group]
                        
                    end
                    
                    % end if
                    
                end
                
                % end state space defined by (B, c, d)
                
            end
        end
    end
    
    % end computation of W
    
    diff = max(abs(W-W_prev), [], 'all');
    if diff < 1e-5
        break
    end
    
    % [iter, diff]
    
    % iter = iter + 1;
end




