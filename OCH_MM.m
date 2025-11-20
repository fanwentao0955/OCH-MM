function [B,Hx,Hy,LTrain]=OCH_MM(XChunk,YChunk,LChunk,XTest,YTest,LTest,param)

I2Tsum = 0; 
T2Isum = 0; 

for chunki = 1:param.nchunks
    
     LTrain = cell2mat(LChunk(1:chunki,:));
     XTrain_new = XChunk{chunki,:};
     YTrain_new = YChunk{chunki,:};
     LTrain_new = LChunk{chunki,:};
     
     if chunki == 1
         [Hx,Hy,MM,VV] = OCH_MM_train0(XTrain_new,YTrain_new,param,LTrain_new);
     else
         [Hx,Hy,MM,VV] = OCH_MM_train(XTrain_new,YTrain_new,param,LChunk, MM, VV,chunki);
     end
     
   
                  
    B = cell2mat(VV(1:end,1));
     
    tBX = sign(XTest * (Hx)');
    tBY = sign(YTest * (Hy)');
    sim_ti = B * tBX';
    sim_it = B * tBY';
    R_ = size(B,1);
    
    ImgToTxt = mAP(sim_ti,LTrain,LTest,R_);
    TxtToImg = mAP(sim_it,LTrain,LTest,R_);
    fprintf('OCH-MM %d bits -- round: %d,   ImgToTxt: %f,    TxtToImg: %f \n',param.bit ,chunki, ImgToTxt, TxtToImg);
    
    I2Tsum = I2Tsum + ImgToTxt;
    T2Isum = T2Isum + TxtToImg;
    
   
   clear ImgToTxt TxtToImg


end

fprintf('OCH-MM %d bits -- average,   ImgToTxt: %f,    TxtToImg: %f \n',param.bit , I2Tsum/param.nchunks, T2Isum/param.nchunks);
end