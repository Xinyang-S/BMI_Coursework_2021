function [decodedPosX, decodedPosY, error] = positionEstimator(input, parameters,y_pred)

if isempty(input.decodedHandPos)
    prev_pos =  input.startHandPos;
else
    prev_pos = input.decodedHandPos(:,end);
end

test_size = size(input.spikes,2);
X = test_data(input.spikes(:,test_size-319:end));
X = X - parameters.mean_X; %Normalize
X = X./parameters.std_X;
W = parameters.W;
B = parameters.B;

for N = 1:(parameters.layer_numbers-3)
    weight = strcat('w', num2str(N));
    b = strcat('b', num2str(N));
    
    Z = X * W(weight) + B(b); % X = a where a0 is the training set
    Z(Z<0) = 0;
    X = Z; %for next iteration
end
for N = (parameters.layer_numbers-2):(parameters.layer_numbers-1)
    weight = strcat('w', num2str(N));
    b = strcat('b', num2str(N));
    
    Z = X * W(weight) + B(b); % X = a where a0 is the training set
    Z(Z<0) = 0;
    X = Z; %for next iteration
end

%Output layer
weight = strcat('w', num2str(N+1));
b = strcat('b', num2str(N+1));

Z = X * W(weight) + B(b); %N+1 is the last layer
decodedPosX = prev_pos(1,:) +  Z(:,1);
decodedPosY = prev_pos(2,:) + Z(:,2);
weight = strcat('w', num2str(N+2));
b = strcat('b', num2str(N+2));
[p,label] = max(cosh(Z)./sum(cosh(Z),2),[],2);
error = label - y_pred;
error = (1-length(error(error ~= 0))/length(label))*100;
%% Data Split
    function X_test  = test_data(testData)
        Set_spikes = mat2cell(testData, 98, [20*ones(1,16)]);
        % Spike Rate
        for Set = 1:16
            % Spike Rate:
            Set_spikes{1, Set} = sum(Set_spikes{1, Set},2)/20;
        end
        
        % Now the X matrix can be constructed.
        X_test = reshape(transpose(cell2mat(Set_spikes)),[1,98*Set]);
        
    end

end

