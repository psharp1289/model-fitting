function lik = lik_moodyRL(P,data)
    
    % Likelihood function for Moody RL learning agent on k-armed bandit
    %
    % USAGE: lik = lik_moodyRL(P,data)
    %
    % INPUTS:
    %   P - structure of S parameter samples, with the following fields:
    %           .lrv - 
    %           .lrm - 
    %   data - structure with the following fields:
    %          .C - [N x 1] choices
    %          .O - [N x 1] rewards
    %
    % OUTPUTS:
    %   lik - [S x 1] log-likelihoods
    %
    % Eran Eldar, June 2018
    
    S = size(P.invtemp,1); % number of parameters
    Nc = max(unique(data.C)); % number of options
    
    % initial values, likelihood, mood
    q = zeros(S,Nc);
    lik = zeros(S,1);
    mood=zeros(S,1);
    
    %parameters
    invtemp = P.invtemp;
    lrv = P.lrv;
    lrm = P.lrm;
    mood_bias=P.moodbias;
    
    
    for t = 1:data.T 
        c = data.C(t); 
        o = data.O(t);
        lik = lik + invtemp.*q(:,c) - mfUtil.logsumexp(bsxfun(@times,invtemp,q),2);
        uf=tanh(mood); %nonlinear transformation of mood by sigmoid
        perceived_r=o+(mood_bias.*uf);
        pe = perceived_r - q(:,c);
        mood=mood.*(1-lrm) + lrm.*pe;
        q(:,c) = q(:,c) + lrv.*pe;      % update values for chosen option
    end
end