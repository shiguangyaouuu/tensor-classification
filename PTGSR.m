function [computedClass recogRate] = PTGSR(trainSet,testSet,options)

k = options.k;
TrainT = trainSet.input;
gnd_tr = trainSet.output';
TestT = testSet.input;
gnd_te = testSet.output';
clear trainSet testSet;

tol = options.tol;
Position_TR = options.Position_TR;
Position_TE = options.Position_TE;

TrainT2 = options.WTrain2;
TestT2 = options.WTest2;

[p,m,n] = size(TestT);

TrainT_mean1 = mean(TrainT2,3);
TrainT_mean2 = mean(TrainT,3);
TestT_mean1 = mean(TestT2,3);
TestT_mean2 = mean(TestT,3);

W = zeros(n);
computedClass = [];
for i = 1:p
    i
    pos_test = Position_TE(i,:);
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    err_total1 = [];
    for j = 1:length(unique(gnd_tr))
        k0 = k;
        indw = find(gnd_tr == j);
        if length(indw)<k0
            k0 = length(indw);
        end
        data_indw = TrainT_mean1(indw,:);
        distt = EuDist2(TestT_mean1(i,:),data_indw);
        pos_train = Position_TR(indw,:);
        spatial_dis = pdist2(pos_test,pos_train)+0.0001;
        distt = distt.*spatial_dis;
        D = distt;
        
        [temp id] = sort(D,2,'ascend');
        z = repmat(TestT_mean1(i,:),k0,1)-data_indw(id(1:k0),:);
        C = z*z';
        C = C + eye(k0,k0)*tol*trace(C);
        temp = (C\ones(k0,1))';
        W_temp = temp./sum(temp);
        err_temp = TestT_mean1(i,:) - W_temp*data_indw(id(1:k0),:);
        err_total1 = [err_total1 err_temp*err_temp'];
    end
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    err_total2 = [];
    for j = 1:length(unique(gnd_tr))
        k0 = k;
        indw = find(gnd_tr == j);
        if length(indw)<k0
            k0 = length(indw);
        end
        data_indw = TrainT_mean2(indw,:);
        distt = EuDist2(TestT_mean2(i,:),data_indw);
        pos_train = Position_TR(indw,:);
        spatial_dis = pdist2(pos_test,pos_train)+0.0001;
        distt = distt.*spatial_dis;
        D = distt;
        
        [temp id] = sort(D,2,'ascend');
        z = repmat(TestT_mean2(i,:),k0,1)-data_indw(id(1:k0),:);
        C = z*z';
        C = C + eye(k0,k0)*tol*trace(C);
        temp = (C\ones(k0,1))';
        W_temp = temp./sum(temp);
        err_temp = TestT_mean2(i,:) - W_temp*data_indw(id(1:k0),:);
        err = err_temp*err_temp';
        err_total2 = [err_total2 err];
    end
    
    err_total = err_total1.*err_total2;
    [a1,a2] = min(err_total);
    computedClass = [computedClass;a2];
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
end
recogRate = sum(computedClass == gnd_te)/size(gnd_te,1);
end



