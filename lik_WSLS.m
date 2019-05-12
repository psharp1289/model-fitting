function lik = lik_WSLS(P,data)


    % Win-stay-lose-shift (WSLS) likelihood function
    %
    % USAGE: lik = lik_WSLS (P,data)
    %
    % INPUTS:
    %   P - structure of S parameter samples, with the following fields:
    %           .lose_shift - [S x 1] bernoulli parameter for lose-shift rate
    %           .win_stay - [S x 1] bernoulli parameter for win-stay rate
  
    %   data - structure with the following fields:
    %          .C - [N x 1] choices
    %          .O - [N x 1] rewards
    %
    % OUTPUTS:
    %   lik - [S x 1] log-likelihoods
    %
    % Paul Sharp, May 2019
    
    S = size(P.win_stay,1); % number of samples of parameters
    win_stay=P.win_stay;
    lose_shift=P.lose_shift;
    lik=zeros(S,1);

  for t = 2:data.T 
        prev_c=data.C(t-1);
        c = data.C(t); 
        prev_o = data.O(t-1);
        %choice here is swictch (1) or stay (0)
        if c ~= prev_c
            choice=1;
        else
            choice=0;
        end
        if prev_o==0
            if choice==1
                lik=lik+log(1-win_stay);
            else
                lik=lik+log(win_stay);
            end
        else
            if choice==1
                lik=lik+log(lose_shift);
            else
                lik=lik+log(1-lose_shift);
            end    
        end
  end
end
