clc
clear all
% close all

% Fixed maxpool in NN

% logName = './Logs/181024_154644_driving_Change_pen_0p03_Loss_weights_1_100_Cpuct_0p1_Fixed_maxpool_Change_back_pen_0p03/evalResults2.txt';
% logName = './Logs/181024_155209_driving_Change_pen_0p03_Loss_weights_1_100_Cpuct_0p1_Fixed_maxpool_Change_back_pen_0p03_Update_3_times_per_sample/evalResults2.txt';

% logName = './Logs/181101_171935_driving_Change_pen_0p03_Loss_weights_1_100_Cpuct_0p1_Fixed_maxpool_Change_back_pen_0p03_Update_3_times_per_sample_More_dpw/evalResults2.txt';

%Added lane change in ego state
% logName = './Logs/181115_205258_driving_Change_pen_0p03_Cpuct_0p1_Dpw_0p3_N_final_32_Lane_change_in_ego_state_V_min_10/evalResults2.txt';
% logName = './Logs/181115_205959_driving_Change_pen_0p03_Cpuct_0p1_Dpw_0p3_N_final_32_Lane_change_in_ego_state_V_min_10_Mean_z_q_target/evalResults2.txt';
% logName = './Logs/181116_155907_driving_Change_pen_0p03_Cpuct_0p1_Dpw_0p3_N_final_32_Lane_change_in_ego_state_V_min_10_No_batchnorm/evalResults2.txt';

%
logName = './Logs/181119_180615_driving_Change_pen_0p03_Cpuct_0p1_Dpw_0p3_N_final_32_Lane_change_in_ego_state_V_min_10_Added_set_V_set_T_ego_state/evalResults2.txt';
% logName = './Logs/181120_104942_driving_Change_pen_0p03_Cpuct_0p1_Dpw_0p3_N_final_32_Lane_change_in_ego_state_V_min_10_Added_set_V_set_T_ego_state_Lane_change_4_samples/evalResults2.txt';

%truck size
logName = './Logs/181126_154437_driving_Cpuct_0p1_Dpw_0p3_V_min_10_Big_replay_Truck_dim/evalResults2.txt';
% logName = './Logs/181126_155336_driving_Cpuct_0p1_Dpw_0p3_V_min_10_Big_replay_Truck_dim_Weights_1_10/evalResults2.txt';




% nWorkers = 4;

data = dlmread(logName,' ');

episodeLength = 200;
evalEpisodePeriod = 5; %1;
firstEval = 1001; %401;

m = 9;
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

    idx = (step(i)-firstEval+evalEpisodePeriod*episodeLength)/(evalEpisodePeriod*episodeLength)+1;
    if idx < 1.5
        idx = 1;
    end
    idx = round(idx);
%     idx = (step(i)-1)/(evalEpisodePeriod*episodeLength);
    sortedData(worker(i)-2,1,idx) = idx;            %sortedData dims: worker #, property, generation
    sortedData(worker(i)-2,2,idx) = sum_reward(i);
%     sortedData(worker(i)-2,3,idx) = x(i,end);
    sortedData(worker(i)-2,3,idx) = max(x(i,:));
    sortedData(worker(i)-2,4:9,idx) = hist(a(i,:),[0 1 2 3 4 5]);
    
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
% rm = load('./Logs/refModelResults2.m');
% refReward = rm(1:2:end);
% refDistance = rm(2:2:end);

% rm = load('./Logs/refAndIdleModelsDistance.txt');
% refReward = rm(:,2);
% refDistance = rm(:,4);
% idleReward = rm(:,3);
% idleDistance = rm(:,5);

% rm = load('./Logs/dpwAndRefAndIdleModelsDistance.txt');
% rm = load('./Logs/dpwAndRefAndIdleModelsDistance181020_185045.txt');
% rm = load('./Logs/dpwAndRefAndIdleModelsDistance181024_155626_.txt');
rm = load('./Logs/dpwAndRefAndIdleModelsDistance181120_115958_.txt'); %"Original, for car size simulations"
% rm = load('./Logs/dpwAndRefAndIdleModelsDistance181122_092739_lane_change_4_samples.txt');
rm = load('./Logs/dpwAndRefAndIdleModelsDistance181126_160317_truck_size.txt');

dpwReward = rm(:,2);
dpwDistance = rm(:,5);
refReward = rm(:,3);
refDistance = rm(:,6);
idleReward = rm(:,4);
idleDistance = rm(:,7);


normIdleDistance = idleDistance./refDistance;
normDpwDistance = dpwDistance./refDistance;

normDistance = squeeze(sortedData(1:20,3,:))'./refDistance';


normWrtDpwIdle = idleDistance./dpwDistance;
normWrtDpwRef = refDistance./dpwDistance;
normWrtDpwAz = squeeze(sortedData(1:20,3,:))'./dpwDistance';

normWrtIdleRef = refDistance./idleDistance;
normWrtIdleAz = squeeze(sortedData(1:20,3,:))'./idleDistance';
normWrtIdleDpw = dpwDistance./idleDistance;


%% NN policy
nnActions = [];
if exist([logName(1:end-16),'prior_policy_result.txt'],'file')
    nn = load([logName(1:end-16),'prior_policy_result.txt']);

    for i=1:size(nn,1)/20
        nnReward(i,:) = nn(20*(i-1)+1:20*i,3)';
        nnDistance(i,:) = nn(20*(i-1)+1:20*i,4)';
        nnActions(i,:,:) = nn(20*(i-1)+1:20*i,5:9);
        nnSample(i) = nn(20*(i-1)+1,2)*20;
    end
    
    %sorting
    [nnSample, idx] = sort(nnSample);
    nnReward = nnReward(idx,:);
    nnDistance = nnDistance(idx,:);
    nnActions = nnActions(idx,:,:);
    

    normNnDistance = nnDistance./refDistance';
    normWrtDpwNn = nnDistance./dpwDistance';
    normWrtIdleNn = nnDistance./idleDistance';
end

disp('Average number of actions per episode for the generations with NN policy: ')
idx = 0;
for i=1:size(nnActions,1)
    disp( squeeze(mean(nnActions(i,:,:),'omitnan'))' )
end



%% Plots
figWidth = 600;
figHeight = 600;

% figure(1)
% hold on
% plot(step*nWorkers,sum_reward,'x')
% plot(uniqueSteps*nWorkers,meanValue,'rx')
% plot(uniqueSteps*nWorkers,mean(values,2),'go')
% errorbar(uniqueSteps*nWorkers,meanValue,std(values'))
% 
% figure(2)
% errorbar(uniqueSteps*nWorkers,meanValue,std(values'))


figure(3)
clf(3)
hold on
sumReward = squeeze(sortedData(:,2,:));
sumReward = sumReward(~isnan(sumReward(:)));
plot(totalStepsVec,sumReward/nSimSteps,'x')
errorbar(totalSteps,meanValue/nSimSteps,std(squeeze(sortedData(:,2,:)),'omitnan')/nSimSteps,'b')
plot(totalSteps,meanValue/nSimSteps,'b')

if exist([logName(1:end-16),'prior_policy_result.txt'],'file')% && size(totalSteps(2:end),2) == size(nnReward,1)
    plot(nnSample-2000, nnReward/nSimSteps,'co')
    plot(nnSample-2000,mean(nnReward,2)/nSimSteps,'c')
end

xlabel('Training steps')
ylabel('Reward per step')
title('Reward per step')
axis([0 max(totalStepsVec) 0.8 1.0])


figure(4)
clf(4)
hold on
plot(6000,dpwDistance/nSimSteps/dt,'mo')
plot(4000,refDistance/nSimSteps/dt,'go')
plot(2000,idleDistance/nSimSteps/dt,'ro')
plot(totalStepsVec,mean(dpwDistance)/nSimSteps/dt*ones(1,length(step)),'m')
plot(totalStepsVec,mean(refDistance)/nSimSteps/dt*ones(1,length(step)),'g')
plot(totalStepsVec,mean(idleDistance)/nSimSteps/dt*ones(1,length(step)),'r')
plot(totalStepsVec,x(:,end)/nSimSteps/dt,'bx')
plot(totalSteps,squeeze(mean(sortedData(1:20,3,:),'omitnan'))/nSimSteps/dt,'b')         %ZZZ Manually set to use 20 workers

if exist([logName(1:end-16),'prior_policy_result.txt'],'file')% && size(totalSteps(2:end),2) == size(nnDistance,1)
    plot(nnSample-2000,nnDistance/nSimSteps/dt,'co')
    plot(nnSample-2000,mean(nnDistance,2)/nSimSteps/dt,'c')
end

xlabel('Training steps')
ylabel('Mean speed')
axis([0 max(totalStepsVec) 18 25.2])
title('Distance driven')

% legend('IDM+MOBIL', 'IDM+MOBIL mean', 'MCTS+NN', 'MCTS+NN mean')


figure(5)
clf(5)
hold on
plot(2000,normIdleDistance,'ro')
plot(4000,normDpwDistance,'mo')
plot(totalSteps,normDistance,'bx')
plot(totalStepsVec,ones(1,length(step)),'g')
plot(totalStepsVec,ones(1,length(step))*mean(normIdleDistance),'r')
plot(totalStepsVec,ones(1,length(step))*mean(normDpwDistance),'m')
plot(totalSteps,mean(normDistance,2,'omitnan'),'b')
errorbar(totalSteps,mean(normDistance,2,'omitnan'),std(normDistance','omitnan')/sqrt(length(uniqueSteps)),'b')

if exist([logName(1:end-16),'prior_policy_result.txt'],'file') %&& size(totalSteps(2:end),2) == size(normNnDistance,1)
    plot(nnSample-2000,normNnDistance,'co')
    plot(nnSample-2000,mean(normNnDistance,2),'c')
end

xlabel('Training steps')
ylabel('Normalized mean speed')
title('Distance driven normalized with IDM/MOBIL performance')
axis([0 max(totalStepsVec) 0.8 1.25])

%Norm with dpw performance
figure(15)
clf(15)
hold on
plot(2000,normWrtDpwIdle,'ro')
plot(4000,normWrtDpwRef,'go')
plot(totalSteps,normWrtDpwAz,'bx')
plot(totalStepsVec,ones(1,length(step)),'m')
plot(totalStepsVec,ones(1,length(step))*mean(normWrtDpwIdle),'r')
plot(totalStepsVec,ones(1,length(step))*mean(normWrtDpwRef),'g')
plot(totalSteps,mean(normWrtDpwAz,2,'omitnan'),'b')
errorbar(totalSteps,mean(normWrtDpwAz,2,'omitnan'),std(normWrtDpwAz','omitnan')/sqrt(length(uniqueSteps)),'b')
xlabel('Training steps')
ylabel('Normalized mean speed')
title('Distance driven normalized with DPW performance')
axis([0 max(totalStepsVec) 0.8 1.25])

if exist([logName(1:end-16),'prior_policy_result.txt'],'file') %&& size(totalSteps(2:end),2) == size(normWrtDpwNn,1)
    plot(nnSample-2000,normWrtDpwNn,'co')
    plot(nnSample-2000,mean(normWrtDpwNn,2),'c')
end


%Norm with idle performance
figure(16)
clf(16)
hold on
plot(4000,normWrtIdleDpw,'mo')
plot(2000,normWrtIdleRef,'go')
plot(totalSteps,normWrtIdleAz,'bx')
plot(totalStepsVec,ones(1,length(step)),'r')
plot(totalStepsVec,ones(1,length(step))*mean(normWrtIdleDpw),'m')
plot(totalStepsVec,ones(1,length(step))*mean(normWrtIdleRef),'g')
plot(totalSteps,mean(normWrtIdleAz,2,'omitnan'),'b')
errorbar(totalSteps,mean(normWrtIdleAz,2,'omitnan'),std(normWrtIdleAz','omitnan')/sqrt(length(uniqueSteps)),'b')
xlabel('Training steps')
ylabel('Normalized mean speed')
title('Distance driven normalized with idle performance')
axis([0 max(totalStepsVec) 0.8 1.25])

if exist([logName(1:end-16),'prior_policy_result.txt'],'file') %&& size(totalSteps(2:end),2) == size(normWrtDpwNn,1)
    plot(nnSample-2000,normWrtIdleNn,'co')
    plot(nnSample-2000,mean(normWrtIdleNn,2),'c')
end

set(figure(3), 'Position', [100, 100, figWidth, figHeight])
set(figure(4), 'Position', [500, 100, figWidth, figHeight])
set(figure(5), 'Position', [1000, 100, figWidth, figHeight])
set(figure(15), 'Position', [1500, 100, figWidth, figHeight])
set(figure(16), 'Position', [2000, 100, figWidth, figHeight])


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