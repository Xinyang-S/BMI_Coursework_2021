function [modelParameters] = positionEstimatorTraining(training_data)
%% Find firing rates

xvel = [];
yvel = [];
spike_rate = [];
firingRate = [];
trainingData = struct([]);
Vel = struct([]);

dt = 10; % bin size

for k = 1:8
    for i = 1:98
        for n = 1:length(training_data)
            for t = 300:dt:570-dt
                % For every trial, compute spike rate
                number = length(find(training_data(n,k).spikes(i,t:t+dt)==1));
                spike_rate = cat(2, spike_rate, number/(dt*0.001));
                % compute the velocity of hand in x,y direction
                if i==1
                    x_vel = (training_data(n,k).handPos(1,t+dt) - training_data(n,k).handPos(1,t)) / (dt*0.001);
                    y_vel = (training_data(n,k).handPos(2,t+dt) - training_data(n,k).handPos(2,t)) / (dt*0.001);
                    xvel = cat(2, xvel, x_vel);
                    yvel = cat(2, yvel, y_vel);
                end
            end
            
            % Append spike rate in firingRate and reset spike_rate
            firingRate = cat(2, firingRate, spike_rate);
            spike_rate = [];
        end
        Vel(k).x = xvel;
        Vel(k).y = yvel;        
        trainingData(i,k).firingRates = firingRate;
        firingRate = [];
    end
    xvel = [];
    yvel = [];
end

%% Linear Regression
% we used Linear Regression for velocity and KNN for position
beta = struct([]);

for k=1:8
    firingRate = [];
    vel = [Vel(k).x; Vel(k).y];
    for i=1:98
    firingRate = cat(1, firingRate, trainingData(i,k).firingRates);
    end
    beta(k).reachingAngle = lsqminnorm(firingRate',vel');
end

%% KNN Classifier

spike_number = zeros(length(training_data),98);
Angle = [];
spike = [];

for k = 1:8
    for i = 1:98
        for n = 1:length(training_data)
                spike_number(n,i) = length(find(training_data(n,k).spikes(i,1:320)==1));
        end
    end
    spike = cat(1, spike, spike_number);
    angle_temp(1:length(training_data)) = k;
    Angle = cat(2, Angle, angle_temp);
    
end

knn = fitcecoc(spike,Angle);

modelParameters = struct('beta',beta,'knnModel',knn); 
  
end
