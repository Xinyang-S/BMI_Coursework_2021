function [x, y, newModelParameters] = positionEstimator(test_data, modelParameters)
%% Estimate direction

spike_number = zeros(98,1);

if length(test_data.spikes) <= 320
    for i = 1:98
        spike_number(i) = length(find(test_data.spikes(i,1:320)==1));
    end
    %estimate direciton of hand
    direction = mode(predict(modelParameters.knnModel,spike_number'));
else
    %otherwise use previous model
    direction = modelParameters.direction;
end

%% Estimate velocity

% time window
t = 20;
t_min = length(test_data.spikes) - 1*t;
t_max = length(test_data.spikes);

firingRate = zeros(98,1);
for i = 1:98
    firingRate(i) = 1*length(find(test_data.spikes(i,t_min:t_max)==1))/(t*0.001);
end

%estimate velocity
velocity_x = 1*firingRate'*modelParameters.beta(direction).reachingAngle(:,1);
velocity_y = 1*firingRate'*modelParameters.beta(direction).reachingAngle(:,2);

%% Update Model

if length(test_data.spikes) <= 320
    x = test_data.startHandPos(1);
    y = test_data.startHandPos(2);
else
    % previous position + velocity * 20ms (to get position)
    x = test_data.decodedHandPos(1,length(test_data.decodedHandPos(1,:))) + 1*velocity_x*(t*0.001);
    y = test_data.decodedHandPos(2,length(test_data.decodedHandPos(2,:))) + 1*velocity_y*(t*0.001);
end

newModelParameters.beta = modelParameters.beta;
newModelParameters.direction = direction;
newModelParameters.knnModel = modelParameters.knnModel;
end