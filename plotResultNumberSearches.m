clc
clear all
% close all

% % Fixed maxpool in NN
% 
% % logName = './Logs/181024_154644_driving_Change_pen_0p03_Loss_weights_1_100_Cpuct_0p1_Fixed_maxpool_Change_back_pen_0p03/evalResults2.txt';
% % logName = './Logs/181024_155209_driving_Change_pen_0p03_Loss_weights_1_100_Cpuct_0p1_Fixed_maxpool_Change_back_pen_0p03_Update_3_times_per_sample/evalResults2.txt';
% 
% % logName = './Logs/181101_171935_driving_Change_pen_0p03_Loss_weights_1_100_Cpuct_0p1_Fixed_maxpool_Change_back_pen_0p03_Update_3_times_per_sample_More_dpw/evalResults2.txt';
% 
% %Added lane change in ego state
% % logName = './Logs/181115_205258_driving_Change_pen_0p03_Cpuct_0p1_Dpw_0p3_N_final_32_Lane_change_in_ego_state_V_min_10/evalResults2.txt';
% % logName = './Logs/181115_205959_driving_Change_pen_0p03_Cpuct_0p1_Dpw_0p3_N_final_32_Lane_change_in_ego_state_V_min_10_Mean_z_q_target/evalResults2.txt';
% % logName = './Logs/181116_155907_driving_Change_pen_0p03_Cpuct_0p1_Dpw_0p3_N_final_32_Lane_change_in_ego_state_V_min_10_No_batchnorm/evalResults2.txt';
% 
% % %
% % logName = './Logs/181119_180615_driving_Change_pen_0p03_Cpuct_0p1_Dpw_0p3_N_final_32_Lane_change_in_ego_state_V_min_10_Added_set_V_set_T_ego_state/evalResults2.txt';
% % logName = './Logs/181120_104942_driving_Change_pen_0p03_Cpuct_0p1_Dpw_0p3_N_final_32_Lane_change_in_ego_state_V_min_10_Added_set_V_set_T_ego_state_Lane_change_4_samples/evalResults2.txt';
% 
% %truck size
% % logName = './Logs/181126_154437_driving_Cpuct_0p1_Dpw_0p3_V_min_10_Big_replay_Truck_dim/evalResults2.txt';
% logName = './Logs/181126_155336_driving_Cpuct_0p1_Dpw_0p3_V_min_10_Big_replay_Truck_dim_Weights_1_10/evalResults2.txt';
% % logName = './Logs/181215_123610_driving_Cpuct_0p1_Dpw_0p3_V_min_10_Big_replay_Truck_dim_Bigger_net/evalResults2.txt';
% 
% %New action space
% logName = './Logs/181221_153735_driving_Cpuct_0p1_Bigger_net_New_action_space/evalResults2.txt'; %Use this for the paper!
% % % %logName = './Logs/190211_153417_driving_Cpuct_0p1_Bigger_net_New_action_space_No_batchnorm/evalResults2.txt';
% % logName = './Logs/190218_082611_driving_Cpuct_0p1_Bigger_net_New_action_space_No_batchnorm/evalResults2.txt';

logName = './Logs/181221_153735_driving_Cpuct_0p1_Bigger_net_New_action_space/Reruns/evalResults2_searches_13111.txt';

%% Ref model
%100 eval runs
rm = load('./Logs/dpwAndRefAndIdleModelsDistance_continuous_driving_190319_082827_ 100evalRuns_forPaper.txt');


%%
data = dlmread(logName,' ');

nSearches = [1 10 50 100 400 1000 2000 10000]; %%%%%%%%%%%%%%%% TMP, CHANGE ORDER %%%%%%%%%%%%%%%%%%%%%%%%%

m = 9;
for i=1:size(data,1)/m
    worker(i) = data((i-1)*m+1,1);
    step(i) = data((i-1)*m+1,2);
    sum_reward(i) = data((i-1)*m+1,3);
    reward(i,:) = data((i-1)*m+2,:);
    x(i,:) = data((i-1)*m+3,:);
    y(i,:) = data((i-1)*m+4,:);
    v(i,:) = data((i-1)*m+5,:);
    lc(i,:) = data((i-1)*m+6,:);
    t_set(i,:) = data((i-1)*m+7,:);
    v_set(i,:) = data((i-1)*m+8,:);
    a(i,:) = data((i-1)*m+9,:);

    idx = find(nSearches==step(i));
    
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


uniqueSteps = [];
stepCount = [];
% sumValue = [];
values = [];
totalSteps = [];
k=0;
for i=1:size(step,2)
%     if ~ismember(step(i),uniqueSteps)
    if length(uniqueSteps)==0 || ~( min(abs(uniqueSteps-step(i))) < 1 ) %then unique step
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

totalSteps = uniqueSteps;
totalStepsVec = [];
k=0;
for i=1:size(stepCount,2)
    totalStepsVec(k+1:k+stepCount(i)) = ones(1,stepCount(i))*totalSteps(i);
    k = k + stepCount(i);
end

% if evalRuns
%     totalStepsVec = totalStepsVec+10000;
% end

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


dpwReward = rm(:,2);
dpwDistance = rm(:,5);
refReward = rm(:,3);
refDistance = rm(:,6);
idleReward = rm(:,4);
idleDistance = rm(:,7);

if size(rm,2)>7
    actionsDpw = rm(:,8:12);
    actionsRef = rm(:,13:15);
    disp('MCTS: ')
    disp( mean(actionsDpw,1)/200 )
    disp('IDM/MOBIL: ')
    disp( mean(actionsRef,1)/200 )   %Remember that some episodes are weird, changing lanes too many times. Manually fix there.
end


normIdleDistance = idleDistance./refDistance;
normDpwDistance = dpwDistance./refDistance;

normDistance = squeeze(sortedData(:,3,:))'./refDistance';


normWrtDpwIdle = idleDistance./dpwDistance;
normWrtDpwRef = refDistance./dpwDistance;
normWrtDpwAz = squeeze(sortedData(:,3,:))'./dpwDistance';

normWrtIdleRef = refDistance./idleDistance;
normWrtIdleAz = squeeze(sortedData(:,3,:))'./idleDistance';
normWrtIdleDpw = dpwDistance./idleDistance;

normWrtIdle25 = 25*200*0.75./idleDistance;
normWrtRef25 = 25*200*0.75./refDistance;
normWrtDpw25 = 25*200*0.75./dpwDistance;






%% Plots
figWidth = 600;
figHeight = 600;

plotNN = 0;
plotNN = 1;

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

if plotNN && exist([logName(1:end-16),'prior_policy_result.txt'],'file')% && size(totalSteps(2:end),2) == size(nnReward,1)
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
plot(60,dpwDistance/nSimSteps/dt,'mo')
plot(40,refDistance/nSimSteps/dt,'go')
plot(20,idleDistance/nSimSteps/dt,'ro')
plot(totalStepsVec,mean(dpwDistance)/nSimSteps/dt*ones(1,length(step)),'m')
plot(totalStepsVec,mean(refDistance)/nSimSteps/dt*ones(1,length(step)),'g')
plot(totalStepsVec,mean(idleDistance)/nSimSteps/dt*ones(1,length(step)),'r')
plot(totalStepsVec,x(:,end)/nSimSteps/dt,'bx')
plot(totalSteps,squeeze(mean(sortedData(:,3,:),'omitnan'))/nSimSteps/dt,'b')         %ZZZ Manually set to use 20 workers

if plotNN && exist([logName(1:end-16),'prior_policy_result.txt'],'file')% && size(totalSteps(2:end),2) == size(nnDistance,1)
    plot(nnSample-2000,nnDistance/nSimSteps/dt,'co')
    plot(nnSample-2000,mean(nnDistance,2)/nSimSteps/dt,'c')
end

xlabel('Number of MCTS iterations, $$n$$','Interpreter','Latex')
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
plot(totalStepsVec,ones(1,length(step))*25*0.75*200/mean(refDistance),'k') %Empty road
plot(totalStepsVec,ones(1,length(step))*mean(normIdleDistance),'r')
plot(totalStepsVec,ones(1,length(step))*mean(normDpwDistance),'m')
plot(totalSteps,mean(normDistance,2,'omitnan'),'b')
errorbar(totalSteps,mean(normDistance,2,'omitnan'),std(normDistance','omitnan')/sqrt(size(normDistance,2)),'b')

if plotNN && exist([logName(1:end-16),'prior_policy_result.txt'],'file') %&& size(totalSteps(2:end),2) == size(normNnDistance,1)
    plot(nnSample-2000,normNnDistance,'co')
    plot(nnSample-2000,mean(normNnDistance,2),'c')
end

xlabel('Number of MCTS iterations, $$n$$','Interpreter','Latex')
ylabel('Normalized mean speed')
title('Distance driven normalized with IDM/MOBIL performance')
axis([0 max(totalStepsVec) 0.8 1.25])



set(figure(3), 'Position', [100, 100, figWidth, figHeight])
set(figure(4), 'Position', [500, 100, figWidth, figHeight])
set(figure(5), 'Position', [1000, 100, figWidth, figHeight])



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
plot(totalSteps,squeeze(mean(sortedData(:,2,:),1,'omitnan')),'b--x')
title('Reward for all the workers over generations')

% Distance for all the workers over generations
figure(7)
clf(7)
hold on
for i=1:size(sortedData,1)
    plot(totalSteps,squeeze(sortedData(i,3,:)))
end
plot(totalSteps,squeeze(mean(sortedData(:,3,:),1,'omitnan')),'r--x')
plot(totalSteps,squeeze(mean(sortedData(:,3,:),1,'omitnan')),'b--x')
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