function [H1,H2,MM,VV] = OCH_MM_train0(X, Y, param, L)

X1 = X'; X2 =Y';

%% paramter setting
bit = param.bit;
maxIter = param.maxIter;
lambda = param.lambda;
alpha1 = param.alpha1;
alpha2 = param.alpha2;
theta = param.theta;
mu = param.mu;

numTrain = size(L, 1);
sampleColumn = param.sample;

%% initialization
V = ones(numTrain, bit);
V(randn(numTrain, bit) < 0) = -1;

B_opt = ones(numTrain, bit);
B_opt(randn(numTrain, bit) < 0) = -1;

H1 = randn(bit,size(X1,1));
H2 = randn(bit,size(X2,1));

E1 = randn(bit,numTrain);
E2 = randn(bit,numTrain);

c = size(L,2);
R = ones(c, bit);
R(randn(c, bit) < 0) = -1;
%% hash learning
for epoch = 1:maxIter
    
    
    Sc = randperm(numTrain, sampleColumn);
    SX = L * L(Sc, :)' > 0;
    
    B_opt = updateB(B_opt, V, SX, Sc, bit, lambda, sampleColumn, alpha1, alpha2, H1', H2', X1', X2', E1', E2');
    B = B_opt';
    
    SY = L(Sc, :) * L' > 0;
    V = updateV(V, B_opt, SY, Sc, bit, lambda, sampleColumn); 
    
        
    H1 = (B+E1)*X1'/(X1*X1'+ theta * eye(size(X1,1))); 
    H2 = (B+E2)*X2'/(X2*X2'+ theta * eye(size(X2,1)));
     
    E1 = sign(H1*X1 - B).*max((abs(H1*X1 - B)- mu),0); 
    E2 = sign(H2*X2 - B).*max((abs(H2*X2 - B)- mu),0);

    
end


M1 = X1*X1';
M2 = (B+E1)*X1';
M3 = X2*X2';
M4 = (B+E2)*X2';


MM{1,1} = M1;
MM{1,2} = M2;
MM{1,3} = M3;
MM{1,4} = M4;

VV{1,1} = V;


function newB = updateB(newB, newV, S, Sc, bit, lambda, sampleColumn, alpha1, alpha2,  H1, H2, X1, X2, E1, E2)
m = sampleColumn;
n = size(newB, 1);
for k = 1: bit
    TX = lambda * newB * newV(Sc, :)' / bit;
    AX = 1 ./ (1 + exp(-TX));
    newV_jk = newV(Sc, k)';
    pa = lambda * ((S - AX) .* repmat(newV_jk, n, 1)) * ones(m, 1) / bit;
    pb = (m * lambda^2 + 8*bit^2 * (alpha1 + alpha2))* newB(:, k) / (4 * bit^2);
    pc = 2*alpha1*(E1(:,k)+newB(:,k)- X1*(H1(:,k)) ) + 2*alpha2*(E2(:,k)+newB(:,k)- X2*(H2(:,k)));
    p = pa+pb-pc;
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