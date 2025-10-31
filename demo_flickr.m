close all; clear;
addpath(genpath(fullfile('utils/')));
seed = 0;rng('default');rng(seed);
param.seed = seed;
%% load dataset
dataname = 'flickr';
param.dataname = dataname;
dataset = load_data(dataname);

%% initialization
param.maxIter = 15;
param.mu = 1.5;                                                                                                                                                                                                 
param.sample = 50;
param.theta = 5e1; 

X = dataset.XDatabase; Y = dataset.YDatabase; L = dataset.databaseL;
sampleInds = randperm(size(L,1));
param.chunk_size = 2000;
param.nchunks = floor(length(sampleInds)/param.chunk_size);
XChunk = cell(param.nchunks,1);
YChunk = cell(param.nchunks,1);
LChunk = cell(param.nchunks,1);

for subi = 1:param.nchunks-1
    XChunk{subi,1} = X(sampleInds(param.chunk_size*(subi-1)+1:param.chunk_size*subi),:);
    YChunk{subi,1} = Y(sampleInds(param.chunk_size*(subi-1)+1:param.chunk_size*subi),:);
    LChunk{subi,1} = L(sampleInds(param.chunk_size*(subi-1)+1:param.chunk_size*subi),:);
end

XChunk{param.nchunks,1} = X(sampleInds(param.chunk_size*subi+1:end),:);
YChunk{param.nchunks,1} = Y(sampleInds(param.chunk_size*subi+1:end),:);
LChunk{param.nchunks,1} = L(sampleInds(param.chunk_size*subi+1:end),:);

XTest = dataset.XTest; YTest = dataset.YTest; LTest = dataset.testL;
clear X Y L
%% OCH-MM
% param.bit = 16; param.lambda = 5; param.alpha1 = 1e-2; param.alpha2 = param.alpha1; 
 param.bit = 32; param.lambda = 7; param.alpha1 = 1e-1; param.alpha2 = param.alpha1;
% param.bit = 64; param.lambda = 9; param.alpha1 = 1e-1; param.alpha2 = param.alpha1;
[B,Hx,Hy,LTrain]=OCH_MM(XChunk,YChunk,LChunk,XTest,YTest,LTest,param);