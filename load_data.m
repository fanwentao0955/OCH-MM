function dataset = load_data(dataname)
switch dataname
    case 'flickr'
        load('flickr.mat');
        dataset.XTest = XTest;
        dataset.YTest = YTest;
        dataset.XDatabase = XTrain;
        dataset.YDatabase = YTrain;
        dataset.testL = LTest;
        dataset.databaseL = LTrain;
    case 'NUSWIDE'
        load('NUSWIDE.mat');
        dataset.XTest = XTest;
        dataset.YTest = YTest;
        dataset.XDatabase = XTrain;
        dataset.YDatabase = YTrain;
        dataset.testL = LTest;
        dataset.databaseL = LTrain;
    case 'NUSWIDE_CNN'
        load('NUSWIDE_CNN.mat');
        dataset.XTest = XTest;
        dataset.YTest = YTest;
        dataset.XDatabase = XTrain;
        dataset.YDatabase = YTrain;
        dataset.testL = LTest;
        dataset.databaseL = LTrain;
    case 'IAPRTC_12'
        load('IAPRTC_12.mat');
        dataset.XTest = XTest;
        dataset.YTest = YTest;
        dataset.XDatabase = XTrain;
        dataset.YDatabase = YTrain;
        dataset.testL = LTest;
        dataset.databaseL = LTrain;
    case 'WIKI10'
        load('WIKI10.mat');
        dataset.XTest = XTest;
        dataset.YTest = YTest;
        dataset.XDatabase = XTrain;
        dataset.YDatabase = YTrain;
        dataset.testL = LTest;
        dataset.databaseL = LTrain;
    case 'LabelMe'
        load('LabelMe.mat');
        dataset.XTest = XTest;
        dataset.YTest = YTest;
        dataset.XDatabase = XTrain;
        dataset.YDatabase = YTrain;
        dataset.testL = LTest;
        dataset.databaseL = LTrain;

end
end

