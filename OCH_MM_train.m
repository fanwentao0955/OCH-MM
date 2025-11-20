function [H1,H2,MM,VV] = OCH_MM_train(X, Y, param, LChunk, MM, VV,chunki)

X1 = X'; X2 =Y'; L = LChunk{chunki,:};
oL = cell2mat(LChunk(1:(chunki-1),:));
oV = cell2mat(VV(1:end,1));
%% paramter setting
bit = param.bit;
maxIter = param.maxIter;
lambda = param.lambda;
alpha1 = param.alpha1;
alpha2 = param.alpha2;
mu = param.mu;
numTrain = size(L, 1);
sampleColumn = (chunki-1)*param.sample;
sampleColumn_ = param.sample;
theta = param.theta;
%% initialization

B_opt = ones(numTrain, bit);
B_opt(randn(numTrain, bit) < 0) = -1;

V = ones(numTrain, bit);
V(randn(numTrain, bit) < 0) = -1;

H1 = randn(bit,size(X1,1));
H2 = randn(bit,size(X2,1));

E1 = randn(bit,numTrain);
E2 = randn(bit,numTrain);

c = size(L,2);
R = ones(c, bit);
R(randn(c, bit) < 0) = -1;
%% hash learning
for epoch = 1:maxIter
    
    Sc = randperm(size(oV,1), sampleColumn);
    Sno = L * oL(Sc, :)' > 0;
    
    Sc_ = randperm(numTrain, sampleColumn_);
    Sno_ = L * L(Sc_,:)'>0;
    
    B_opt = updateB(B_opt, oV, V, Sno, Sc, Sno_, Sc_, bit, lambda, sampleColumn, sampleColumn_, alpha1, alpha2, H1', H2', X1', X2', E1', E2');
    B = B_opt';
    
    SY = L(Sc_, :) * L' > 0;
    V = updateV(V, B_opt, SY, Sc_, bit, lambda, sampleColumn_); 
    

    H1 = ((B+E1)*X1'+ MM{1,2})/(X1*X1'+ MM{1,1} + theta * eye(size(X1,1))); %1e-3
    H2 =((B+E2)*X2'+ MM{1,4})/(X2*X2'+ MM{1,3}+ theta * eye(size(X2,1)));
    
    
    E1 = sign(H1*X1 - B).*max((abs(H1*X1 - B)- mu),0); 
    E2 = sign(H2*X2 - B).*max((abs(H2*X2 - B)- mu),0); 

end



M1 = X1*X1';
M2 = (B+E1)*X1';
M3 = X2*X2';
M4 = (B+E2)*X2';


    
MM{1,1} = MM{1,1} + M1;
MM{1,2} = MM{1,2} + M2;
MM{1,3} = MM{1,3} + M3;
MM{1,4} = MM{1,4} + M4;

VV{chunki,1} = V;


function newB = updateB(newB, oldV, V, S, Sc, S_, Sc_, bit, lambda, sampleColumn, sampleColumn_, alpha1, alpha2,  H1, H2, X1, X2, E1, E2)
m = sampleColumn;
m_ = sampleColumn_;
n = size(newB, 1);
for k = 1: bit
    TX = lambda * newB * oldV(Sc, :)' / bit;
    AX = 1 ./ (1 + exp(-TX));
    ovjk = oldV(Sc, k)';
    TX_ = lambda * newB * V(Sc_, :)'/ bit;
    AX_ = 1 ./ (1 + exp(-TX_));
    vjk = V(Sc_,k)';
    pa = lambda * ((S - AX) .* repmat(ovjk, n, 1)) * ones(m, 1) / bit;
    pa_ = 1e-1 * lambda * ((S_ - AX_) .* repmat(vjk, n, 1)) * ones(m_, 1) / bit;
    pb = ((m + 1e-1 * m_) * lambda^2 + 8*bit^2 * (alpha1 + alpha2))* newB(:, k) / (4 * bit^2);
    pc = 2*alpha1*(E1(:,k)+newB(:,k)- X1*(H1(:,k)) ) + 2*alpha2*(E2(:,k)+newB(:,k)- X2*(H2(:,k)));
    p = pa+pa_+pb-pc;
    newB_opt = ones(n, 1);
    newB_opt(p < 0) = -1;
    newB(:, k) = newB_opt;
end
end


function newV = updateV(newV, newB, S, Sc, bit, lambda, sampleColumn)
m = sampleColumn;
n = size(newV, 1);
for k = 1: bit
    TX1 = lambda * newB(Sc, :) * newV' / bit;
    AX1 = 1 ./ (1 + exp(-TX1));
    Bjk = newB(Sc, k)';  
    p = lambda * ((S' - AX1') .* repmat(Bjk, n, 1)) * ones(m, 1)  / bit + m * lambda^2 * newV(:, k) / (4 * bit^2);
    newV_opt = ones(n, 1);
    newV_opt(p < 0) = -1;
    newV(:, k) = newV_opt;
end
end


end