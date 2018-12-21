clc
clear all
% close all

%Exit lane
% logName = './Logs/181130_160730_driving_exit_lane_Cpuct_0p1_Dpw_0p3_Big_replay_Truck_dim/evalResults2.txt';
logName = './Logs/181203_174746_driving_exit_lane_Cpuct_0p1_Dpw_0p3_Big_replay_Truck_dim_Terminal_state_Est_v0/evalResults2.txt';
logName = './Logs/181215_121952_driving_exit_lane_Cpuct_0p5_Dpw_0p3_Big_replay_Truck_dim_Terminal_state_Est_v0_R_plus_19/evalResults2.txt';


data = dlmread(logName,' ');

evalFreq = 1000;
firstEval = 1001; %401;
exitPos = 1000;

m = 9; %Number of lines for each worker and eval run
for i=1:size(data,1)/m
    worker(i) = data((i-1)*m+1,1);
    step(i) = data((i-1)*m+1,2);
%     if step(i) == 201 %Fix because first eval happens after 201 steps and not 1 steps
%         step(i) = 1;
%     end
    sum_reward(i) = data((i-1)*m+1,3);
    reward(i,:) = data((i-1)*m+2,:);
    x(i,:) = data((i-1)*m+3,:);
    y(i,:) = data((i-1)*m+4,:);
    v(i,:) = data((i-1)*m+5,:);
    lc(i,:) = data((i-1)*m+6,:);
    t_set(i,:) = data((i-1)*m+7,:);
    v_set(i,:) = data((i-1)*m+8,:);
    a(i,:) = data((i-1)*m+9,:);
    
    [x_end, i_end] = max(x(i,:));
    T_ = (i_end-1)*0.75;
    v_end = v(i,i_end);
    T = T_ - (x_end-exitPos)/v_end;

    idx = (step(i)-firstEval+evalFreq)/evalFreq+1;
    if idx < 1.5
        idx = 1;
    end
    idx = round(idx);

    sortedData(worker(i)-2,1,idx) = idx;            %sortedData dims: worker #, property, generation
    sortedData(worker(i)-2,2,idx) = sum_reward(i);
    sortedData(worker(i)-2,3,idx) = max(x(i,:));
    sortedData(worker(i)-2,4:9,idx) = hist(a(i,:),[0 1 2 3 4 5]);
    sortedData(worker(i)-2,10,idx) = T; %Time to exit
    sortedData(worker(i)-2,11,idx) = y(i,i_end)==1; %Exit reached
    sortedData(worker(i)-2,12,idx) = sum(reward(i,1:i_end-2),2,'omitnan'); %Sum reward except for last one
end
%Replace zeros in reward and x with nans
tmp = sortedData(:,1:3,:);
tmp(tmp==0) = nan;
sortedData(:,1:3,:) = tmp;
%Replace worker with 0 actions with nans
tmp2 = sortedData(:,4:9,:);
for i=1:size(tmp2,3)
    for j=1:size(tmp2,1)
        if min(tmp2(j,:,i)==0)==1
            tmp2(j,:,i) = tmp2(j,:,i)*nan;
        end
    end
end
sortedData(:,4:9,:) = tmp2;


%Move first step index to 0
step = step - (step(1)==step).*(step(1)-0); %Move first one to 0, since then no training has been done

uniqueSteps = [];
stepCount = [];
% sumValue = [];
values = [];
totalSteps = [];
k=0;
for i=1:size(step,2)
%     if ~ismember(step(i),uniqueSteps)
    if length(uniqueSteps)==0 || ~( min(abs(uniqueSteps-step(i))) < 100 ) %then unique step
        k = k+1;
        uniqueSteps(k) = step(i);
        stepCount(k) = 1;
%         sumValue(k) = sum_reward(i);
        values(k,1) = sum_reward(i);
    else
%         idx = find(uniqueSteps==step(i));
        [dummy, idx] = min(abs(uniqueSteps-step(i)));
        stepCount(idx) = stepCount(idx)+1;
        values(idx,stepCount(idx)) = sum_reward(i);
%         sumValue(idx) = sumValue(idx)+sum_reward(i);
    end
end


sumReward = squeeze(sortedData(:,2,:));
sumReward = sumReward(~isnan(sumReward(:)));

meanValue = sum(squeeze(sortedData(:,2,:)),1,'omitnan')./stepCount;

values(values==0)=nan; %If worker dies, replace zeros with nans

totalSteps = cumsum([0,diff(uniqueSteps)].*stepCount);
totalStepsVec = [];
k=0;
for i=1:size(stepCount,2)
    totalStepsVec(k+1:k+stepCount(i)) = ones(1,stepCount(i))*totalSteps(i);
    k = k + stepCount(i);
end

nWorkers = stepCount(1);
nSimSteps = size(reward(1,:),2)-1;

dt = 0.75;

disp('Average number of actions per episode for the generations: ')
idx = 0;
for i=1:length(stepCount)
    disp( squeeze(mean(sortedData(:,5:9,i),'omitnan')) )
end

% aStats = a(end-stepCount(end)-stepCount(end-1)+1:end-stepCount(end),:);
% [n,aa] = hist(aStats',[0 1 2 3 4 5]);



%% Ref model
rm = load('./Logs/dpwAndRefAndIdleModelsDistance_exit_lane_181129_103940_exit_lane.txt');
rm = load('./Logs/dpwAndRefAndIdleModelsDistance_exit_lane_181207_160457_.txt');


dpwReward = rm(:,2);
dpwDistance = rm(:,5);
refReward = rm(:,3);
refDistance = rm(:,6);
idleReward = rm(:,4);
idleDistance = rm(:,7);
dpwTime = rm(:,8);
refTime = rm(:,9);
idleTime = rm(:,10);
dpwCorrectedTime = rm(:,11);
refCorrectedTime = rm(:,12);
idleCorrectedTime = rm(:,13);
dpwEndLane = rm(:,14);
refEndLane = rm(:,15);
idleEndLane = rm(:,16);
dpwSuccess = rm(:,17);
refSuccess = rm(:,18);
idleSuccess = rm(:,19);


normIdleTime = idleCorrectedTime./refCorrectedTime;
normDpwTime = dpwCorrectedTime./refCorrectedTime;
normTime = squeeze(sortedData(:,10,:))'./refCorrectedTime';

normWrtDpwIdle = idleCorrectedTime./dpwCorrectedTime;
normWrtDpwRef = refCorrectedTime./dpwCorrectedTime;
normWrtDpwAz = squeeze(sortedData(:,10,:))'./dpwCorrectedTime';

normWrtIdleRef = refCorrectedTime./idleCorrectedTime;
normWrtIdleAz = squeeze(sortedData(:,10,:))'./idleCorrectedTime';
normWrtIdleDpw = dpwCorrectedTime./idleCorrectedTime;


%% NN policy
nnActions = [];
if exist([logName(1:end-16),'prior_policy_result.txt'],'file')
    nn = load([logName(1:end-16),'prior_policy_result.txt']);

    for i=1:size(nn,1)/nWorkers
        nnReward(i,:) = nn(nWorkers*(i-1)+1:nWorkers*i,3)';
        nnDistance(i,:) = nn(nWorkers*(i-1)+1:nWorkers*i,4)';
        nnActions(i,:,:) = nn(nWorkers*(i-1)+1:nWorkers*i,5:9);
        nnSample(i) = nn(nWorkers*(i-1)+1,2)*nWorkers;
        nnTime(i,:) = nn(nWorkers*(i-1)+1:nWorkers*i,10)';
        nnCorrectedTime(i,:) = nn(nWorkers*(i-1)+1:nWorkers*i,11)';
        nnEndLane(i,:) = nn(nWorkers*(i-1)+1:nWorkers*i,12)';
        nnSuccess(i,:) = nn(nWorkers*(i-1)+1:nWorkers*i,13)';
    end
    
    %sorting
    [nnSample, idx] = sort(nnSample);
    nnReward = nnReward(idx,:);
    nnDistance = nnDistance(idx,:);
    nnActions = nnActions(idx,:,:);
    nnTime = nnTime(idx,:);
    nnCorrectedTime = nnCorrectedTime(idx,:);
    nnEndLane = nnEndLane(idx,:);
    nnSuccess = nnSuccess(idx,:);
    
    normNnTime = nnCorrectedTime./refCorrectedTime';
    normWrtDpwNn = nnCorrectedTime./dpwCorrectedTime';

%     normNnDistance = nnDistance./refDistance';
%     normWrtDpwNn = nnDistance./dpwDistance';
%     normWrtIdleNn = nnDistance./idleDistance';
end

disp('Average number of actions per episode for the generations with NN policy: ')
idx = 0;
for i=1:size(nnActions,1)
    disp( squeeze(mean(nnActions(i,:,:),'omitnan'))' )
end



%% Plots
% Rings - solved cases
% Crosses - failed cases
% Sold line - mean of solved cases
% Dashed line - mean of all cases



figWidth = 600;
figHeight = 600;


figure(1)
clf(1)
hold on
solvedScenarios = squeeze(sortedData(:,11,:));
plot(totalSteps,mean(solvedScenarios,1),'x-')
plot(nnSample,mean(nnSuccess,2),'cx-')
plot(totalSteps,mean(refSuccess)*ones(1,length(totalSteps)),'g')
plot(totalSteps,mean(dpwSuccess)*ones(1,length(totalSteps)),'m')


figure(3)
clf(3)
hold on
sumReward = squeeze(sortedData(:,12,:));
sumReward = sumReward(~isnan(sumReward(:)));
meanReward = sum(squeeze(sortedData(:,12,:)),'omitnan')./stepCount;
plot(totalStepsVec,sumReward/nSimSteps,'x')
errorbar(totalSteps,meanReward/nSimSteps,std(squeeze(sortedData(:,12,:)),'omitnan')/nSimSteps,'b')
plot(totalSteps,meanReward/nSimSteps,'b')

plot(totalSteps,squeeze(sortedData(:,2,:))/nSimSteps,'ro')
plot(totalSteps,sum(squeeze(sortedData(:,2,:)),1)./stepCount/nSimSteps,'r')

% if exist([logName(1:end-16),'prior_policy_result.txt'],'file')% && size(totalSteps(2:end),2) == size(nnReward,1)
%     plot(nnSample-2000, nnReward/nSimSteps,'co')
%     plot(nnSample-2000,mean(nnReward,2)/nSimSteps,'c')
% end

xlabel('Training steps')
ylabel('Reward per step')
title('Reward per step (excluding last step)')



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
figure(4)
clf(4)
hold on

for j=1:size(sortedData,3)
    sumTime = 0;
    for i=1:nWorkers    
        if sortedData(i,11,j) == 1
            plot(totalSteps(j),sortedData(i,10,j),'bo')
            sumTime = sumTime + sortedData(i,10,j);
        else
            plot(totalSteps(j),sortedData(i,10,j),'bx')
        end
    end
    meanTimeVec(j) = sumTime/sum(sortedData(:,11,j));
end

plot(totalSteps,meanTimeVec,'b')
plot(totalSteps,squeeze(mean(sortedData(:,10,:),'omitnan')),'b--')

for i=1:length(idleSuccess)
    if idleSuccess(i)
        plot(2000,idleCorrectedTime(i),'ro')
    else
        plot(2000,idleCorrectedTime(i),'rx')
    end
end
for i=1:nWorkers
    if refSuccess(i)
        plot(4000,refCorrectedTime(i),'go')
    else
        plot(4000,refCorrectedTime(i),'gx')
    end
end
for i=1:nWorkers
    if refSuccess(i)
        plot(6000,dpwCorrectedTime(i),'mo')
    else
        plot(6000,dpwCorrectedTime(i),'mx')
    end
end
tmp = idleCorrectedTime.*idleSuccess;
idleOnlySuccess = tmp(tmp~=0);
plot(totalStepsVec,ones(1,length(step))*mean(idleOnlySuccess),'r')
plot(totalStepsVec,ones(1,length(step))*mean(idleCorrectedTime),'r--')
tmp = refCorrectedTime.*refSuccess;
refOnlySuccess = tmp(tmp~=0);
plot(totalStepsVec,ones(1,length(step))*mean(refOnlySuccess),'g')
plot(totalStepsVec,ones(1,length(step))*mean(refCorrectedTime),'g--')
tmp = dpwCorrectedTime.*dpwSuccess;
dpwOnlySuccess = tmp(tmp~=0);
plot(totalStepsVec,ones(1,length(step))*mean(dpwOnlySuccess),'m')
plot(totalStepsVec,ones(1,length(step))*mean(dpwCorrectedTime),'m--')

if exist([logName(1:end-16),'prior_policy_result.txt'],'file')% && size(totalSteps(2:end),2) == size(nnDistance,1)
%     plot(nnSample-2000,nnCorrectedTime,'ko')
%     plot(nnSample-2000,mean(nnCorrectedTime,2),'k--')
    
    for j=1:length(nnSample)
        sumNnTime = 0;
        for i=1:nWorkers    
            if nnSuccess(j,i) == 1
                plot(nnSample(j)-2000,nnCorrectedTime(j,i),'co')
                sumNnTime = sumNnTime + nnCorrectedTime(j,i);
            else
                plot(nnSample(j)-2000,nnCorrectedTime(j,i),'cx')
            end
        end
        meanNnTimeVec(j) = sumNnTime/sum(nnSuccess(j,:));
    end
    plot(nnSample-2000,meanNnTimeVec,'c')
    plot(nnSample-2000,mean(nnCorrectedTime,2),'c--')
end

xlabel('Training steps')
ylabel('Time')
% axis([0 max(totalStepsVec) 18 25.2])
title('Time to exit')

% legend('IDM+MOBIL', 'IDM+MOBIL mean', 'MCTS+NN', 'MCTS+NN mean')



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
figure(5)
clf(5)
hold on
for i=1:length(dpwSuccess)
    if idleSuccess(i)
        plot(2000,normIdleTime(i),'ro')
    else
        plot(2000,normIdleTime(i),'rx')
    end
end
% plot(2000,normIdleTime,'ro')
for i=1:nWorkers
    if dpwSuccess(i)
        plot(4000+1000,normDpwTime(i),'mo')
    else
        plot(4000+1000,normDpwTime(i),'mx')
    end
    if dpwSuccess(i) && refSuccess(i)
        plot(4000,normDpwTime(i),'mo')
    else
        plot(4000,normDpwTime(i),'mx')
    end
end
% plot(4000,normDpwTime,'mo')
for j=1:length(totalSteps)
    for i=1:nWorkers
        if sortedData(i,11,j)
            plot(totalSteps(j)+1000,normTime(j,i),'bo')
        else
            plot(totalSteps(j)+1000,normTime(j,i),'bx')
        end
    end
end
for j=1:length(totalSteps)
    for i=1:nWorkers
        if refSuccess(i) && sortedData(i,11,j)
            plot(totalSteps(j),normTime(j,i),'bo')
        else
            plot(totalSteps(j),normTime(j,i),'bx')
        end
    end
end
% plot(totalSteps,normTime,'bx')
plot(totalStepsVec,ones(1,length(step)),'g')
tmp = normIdleTime.*idleSuccess.*refSuccess;
normIdleOnlySuccess = tmp(tmp~=0);
plot(totalStepsVec,ones(1,length(step))*mean(normIdleOnlySuccess),'r')
plot(totalStepsVec,ones(1,length(step))*mean(normIdleTime),'r--')
tmp = normDpwTime.*dpwSuccess.*refSuccess;
normDpwOnlySuccess = tmp(tmp~=0);
plot(totalStepsVec,ones(1,length(step))*mean(normDpwOnlySuccess),'m')
plot(totalStepsVec,ones(1,length(step))*mean(normDpwTime),'m--')

for i=1:length(totalSteps)
    tmp = normTime(i,:).*squeeze(sortedData(:,11,i))'.*refSuccess';
    tmp(tmp==0) = NaN;
    normOnlySuccess(i,:) = tmp;
end

plot(totalSteps,mean(normOnlySuccess,2,'omitnan'),'b*')
plot(totalSteps+1000,mean(normTime,2,'omitnan'),'b--')
errorbar(totalSteps,mean(normOnlySuccess,2,'omitnan'),std(normOnlySuccess','omitnan')/sqrt(length(uniqueSteps)),'b')

% if exist([logName(1:end-16),'prior_policy_result.txt'],'file') %&& size(totalSteps(2:end),2) == size(normNnDistance,1)
%     plot(nnSample-2000,normNnDistance,'co')
%     plot(nnSample-2000,mean(normNnDistance,2),'c')
% end

xlabel('Training steps')
ylabel('Normalized mean speed')
title('Time to exit normalized with reference model performance')
axis([0 max(totalStepsVec)+5000 0.8 1.25])



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Norm with dpw performance
figure(15)
clf(15)
hold on

for i=1:length(dpwSuccess)
    if idleSuccess(i)
        plot(2000,normWrtDpwIdle(i),'ro')
    else
        plot(2000,normWrtDpwIdle(i),'rx')
    end
end
% plot(2000,normWrtDpwIdle,'ro')
for i=1:nWorkers
    if refSuccess(i)
        plot(4000+1000,normWrtDpwRef(i),'go')
    else
        plot(4000+1000,normWrtDpwRef(i),'gx')
    end
    if refSuccess(i) && dpwSuccess(i)
        plot(4000,normWrtDpwRef(i),'go')
    else
        plot(4000,normWrtDpwRef(i),'gx')
    end
end
% plot(4000,normWrtDpwRef,'go')
for j=1:length(totalSteps)
    for i=1:nWorkers
        if sortedData(i,11,j)
            plot(totalSteps(j)+1000,normWrtDpwAz(j,i),'bo')
        else
            plot(totalSteps(j)+1000,normWrtDpwAz(j,i),'bx')
        end
    end
end
for j=1:length(totalSteps)
    for i=1:nWorkers
        if dpwSuccess(i) && sortedData(i,11,j)
            plot(totalSteps(j),normWrtDpwAz(j,i),'bo')
        else
            plot(totalSteps(j),normWrtDpwAz(j,i),'bx')
        end
    end
end


% plot(totalSteps,normWrtDpwAz,'bx')
plot(totalStepsVec,ones(1,length(step)),'m')

tmp = normWrtDpwIdle.*dpwSuccess.*idleSuccess;
normWrtDpwIdleOnlySuccess = tmp(tmp~=0);
plot(totalStepsVec,ones(1,length(step))*mean(normWrtDpwIdleOnlySuccess),'r')
plot(totalStepsVec,ones(1,length(step))*mean(normWrtDpwIdle),'r--')
tmp = normWrtDpwRef.*refSuccess.*dpwSuccess;
normWrtDpwRefOnlySuccess = tmp(tmp~=0);
plot(totalStepsVec,ones(1,length(step))*mean(normWrtDpwRefOnlySuccess),'g')
plot(totalStepsVec,ones(1,length(step))*mean(normWrtDpwRef),'g--')

for i=1:length(totalSteps)
    tmp = normWrtDpwAz(i,:).*squeeze(sortedData(:,11,i))'.*dpwSuccess';
    tmp(tmp==0) = NaN;
    normWrtDpwOnlySuccess(i,:) = tmp;
end

plot(totalSteps,mean(normWrtDpwOnlySuccess,2,'omitnan'),'b')
plot(totalSteps,mean(normWrtDpwAz,2,'omitnan'),'b--')
errorbar(totalSteps,mean(normWrtDpwOnlySuccess,2,'omitnan'),std(normWrtDpwOnlySuccess','omitnan')/sqrt(length(uniqueSteps)),'b')

xlabel('Training steps')
ylabel('Normalized mean speed')
title('Time to exit normalized with DPW performance')
axis([0 max(totalStepsVec+5000) 0.8 1.25])

if exist([logName(1:end-16),'prior_policy_result.txt'],'file')% && size(totalSteps(2:end),2) == size(nnDistance,1)
%     plot(nnSample-2000,normWrtDpwNn,'ko')
%     plot(nnSample-2000,mean(normWrtDpwNn,2),'k--')
    
    for j=1:length(nnSample)
        sumNormNnTime = 0;
        for i=1:nWorkers    
            if nnSuccess(j,i) == 1
                plot(nnSample(j)-2000,normWrtDpwNn(j,i),'co')
                sumNormNnTime = sumNormNnTime + normWrtDpwNn(j,i);
            else
                plot(nnSample(j)-2000,normWrtDpwNn(j,i),'cx')
            end
        end
        meanNormNnTimeVec(j) = sumNormNnTime/sum(nnSuccess(j,:));
    end
    plot(nnSample-2000,meanNormNnTimeVec,'c')
    plot(nnSample-2000,mean(normWrtDpwNn,2),'c--')
end


% %Norm with idle performance
% figure(16)
% clf(16)
% hold on
% plot(4000,normWrtIdleDpw,'mo')
% plot(2000,normWrtIdleRef,'go')
% plot(totalSteps,normWrtIdleAz,'bx')
% plot(totalStepsVec,ones(1,length(step)),'r')
% plot(totalStepsVec,ones(1,length(step))*mean(normWrtIdleDpw),'m')
% plot(totalStepsVec,ones(1,length(step))*mean(normWrtIdleRef),'g')
% plot(totalSteps,mean(normWrtIdleAz,2,'omitnan'),'b')
% errorbar(totalSteps,mean(normWrtIdleAz,2,'omitnan'),std(normWrtIdleAz','omitnan')/sqrt(length(uniqueSteps)),'b')
% xlabel('Training steps')
% ylabel('Normalized mean speed')
% title('Distance driven normalized with idle performance')
% axis([0 max(totalStepsVec) 0.8 1.25])
% 
% if exist([logName(1:end-16),'prior_policy_result.txt'],'file') %&& size(totalSteps(2:end),2) == size(normWrtDpwNn,1)
%     plot(nnSample-2000,normWrtIdleNn,'co')
%     plot(nnSample-2000,mean(normWrtIdleNn,2),'c')
% end

set(figure(1), 'Position', [100, 500, figWidth, figHeight])
set(figure(3), 'Position', [100, 100, figWidth, figHeight])
set(figure(4), 'Position', [500, 100, figWidth, figHeight])
set(figure(5), 'Position', [1000, 100, figWidth, figHeight])
set(figure(15), 'Position', [1500, 100, figWidth, figHeight])
% set(figure(16), 'Position', [2000, 100, figWidth, figHeight])


%%
% Reward for all the workers over generations
figure(6)
clf(6)
hold on
for i=1:size(sortedData,1)
%     plot(squeeze(sortedData(i,1,:)),squeeze(sortedData(i,2,:)))
    plot(totalSteps,squeeze(sortedData(i,2,:)))
end
% plot(squeeze(sortedData(i,1,:)),squeeze(mean(sortedData(:,2,:),1,'omitnan')),'r--x')
plot(totalSteps,squeeze(mean(sortedData(:,2,:),1,'omitnan')),'r--x')
plot(totalSteps,squeeze(mean(sortedData(1:20,2,:),1,'omitnan')),'b--x')
title('Reward for all the workers over generations')

% Distance for all the workers over generations
figure(7)
clf(7)
hold on
for i=1:size(sortedData,1)
    plot(totalSteps,squeeze(sortedData(i,3,:)))
end
plot(totalSteps,squeeze(mean(sortedData(:,3,:),1,'omitnan')),'r--x')
plot(totalSteps,squeeze(mean(sortedData(1:20,3,:),1,'omitnan')),'b--x')
title('Distance for all the workers over generations')


%Do nothing actions over generations
figure(8)
clf(8)
hold on
for i=1:size(sortedData,1)
    plot(totalSteps,squeeze(sortedData(i,5,:)))
end
title('Do nothing actions over generations')
%4 - Void, action 0, happens at last step
%5 - Do nothing
%6 - Brake
%7 - Acc
%8 - Change left (or right?)
%9 - Change right (or left?)

%Change lane left actions over generations
figure(9)
clf(9)
hold on
for i=1:size(sortedData,1)
    plot(totalSteps,squeeze(sortedData(i,9,:)))
end
plot(totalSteps,squeeze(mean(sortedData(:,9,:))),'rx')
% plot(totalSteps,totalSteps*0+20,'r')
title('Change lane left actions over generations')

%Change lane left actions over generations
figure(10)
clf(10)
hold on
for i=1:size(sortedData,1)
    plot(totalSteps,squeeze(sortedData(i,8,:)))
end
plot(totalSteps,squeeze(mean(sortedData(:,8,:))),'rx')
% plot(totalSteps,totalSteps*0+20,'r')
title('Change lane right actions over generations')

%Number of active workers
figure(20)
clf(20)
hold on
plot(totalSteps,stepCount,'x-')
title('Number of active workers')