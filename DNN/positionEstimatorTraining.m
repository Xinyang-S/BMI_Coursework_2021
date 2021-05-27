function  [parameters]= positionEstimatorTraining(train_data)
[X_dataset, Y_dataset] = Preprocessing(train_data);
[X,Y,mean_X,std_X] = Normalization(X_dataset, Y_dataset);
X_size = size(X,2);
[param,W,B,Ad] = initialization(4, [100 100 100 2 8],X_size,1e-2,...
    1e-3,0.12,6,512);

[~,W,B] = training(X,Y,param,W,B,Ad);

parameters = struct();
parameters.std_X = 1*std_X;
parameters.mean_X = 1*mean_X;
parameters.layer_numbers = 1*param.layer_numbers;
parameters.W = W;
parameters.B = B;


%% Preprocessing-extract data every 20ms
    function [X, Y]  = Preprocessing(training_Data)

        %initialize x and y, preassign x a large size for speed
        X = zeros(50000,1568);
        Y = [];
        interval= 20;% extract data every 20ms

        % Create new dataset with extracted data from original one
        i = 1;
        for trials = 1:size(training_Data,1)

            for movement = 1:size(training_Data,2)

                % Number of Extraction times for the trial:
                number = 1*floor(size(training_Data(trials,movement).spikes,2)/20);

                % Set the last extraction:
                if sum(20*ones(1,number)) < size(training_Data(trials,movement).spikes,2)
                    last_extraction = 1*size(training_Data(trials,movement).spikes,2) - 1*sum(20*ones(1,number));
                elseif sum(20*ones(1,number)) > size(training_Data(trials,movement).spikes,2)
                    last_extraction = 1*sum(20*ones(1,number)) - 1*size(training_Data(trials,movement).spikes,2);
                else
                    last_extraction = [];
                end

                % save temporary data points in cell matrix:
                Cell_Y{trials,movement} = mat2cell(training_Data(trials,movement).handPos, 3, [20*ones(1,number), last_extraction]);
                Cell_X{trials,movement} = mat2cell(training_Data(trials,movement).spikes, 98, [20*ones(1,number),last_extraction]);
                
                % Calculation in every cell
                for set = 1:size(Cell_X{trials, movement}, 2)
                    % Spike rate and position
                    Cell_Y{trials, movement}{1, set} =  [Cell_Y{trials, movement}{1, set}(1,end);
                    Cell_Y{trials, movement}{1, set}(2,end);
                    Cell_Y{trials, movement}{1, set}(2,end)];
                    Cell_X{trials, movement}{1, set} = sum(Cell_X{trials, movement}{1, set},2)/interval;
                    % Velocity:
                    if set ~= 1
                        Cell_dY{trials, movement}{1, set} =  Cell_Y{trials, movement}{1, set} - Cell_Y{trials, movement}{1, set-1};
                    else
                        Cell_dY{trials, movement}{1, set} = Cell_Y{trials, movement}{1, set};
                    end
                end

                % Bind the data together for every 300ms(15*20ms)
                for period = 1:size(Cell_X{trials, movement}, 2) - 16
                    Y(i,1:3) = transpose(cell2mat(Cell_dY{trials, movement}(:,15+period)));
                    Y(i,4) = movement;
                    X(i,:) = reshape(transpose(cell2mat(Cell_X{trials, movement}(period:15+period))),[1,98*16]);
                    i = i+1;
                end
            end
        end
    end
%% Normalization
    function [X,Y,mean_X,std_X] = Normalization(X_dataset, Y_dataset)
        
        %extract Y from all Y data
        Y = Y_dataset(:,1:2);
        Y(:,3) = Y_dataset(:,4); % direction no
        
        s = RandStream('mt19937ar','Seed',1); %Fix a seed
        RandStream.setGlobalStream(s)
        Seeds = randperm(size(Y,1));
        X = X_dataset(Seeds,:);
        Y = Y(Seeds,:);
        %Normalization using mean and standardised data
        mean_X = mean(X,1);
        std_X = std(X);
        X = (X - mean_X)./std_X;
    end

%% Neural Network
    function [Parameters,Weight,Bias,Adam_Parameters] = initialization(LayerNumbers,NeuronLayer,X_size,LearningRate,...
            Regularization,StdWeight,epochs,batchsize)   
        Parameters = struct(); %set the parameters
        Std_Weight = StdWeight;
        Weight_Row = X_size;

        Parameters.beta_2 = 0.999; %Fixed hyperparameters beta2
        Parameters.beta_1 = 0.9; %Fixed hyperparameters beta1
        Parameters.layer_of_neuron = NeuronLayer;% the neuron layer
        Parameters.layer_numbers = LayerNumbers;% numbers of layers
        Parameters.batchsize = batchsize;%the batch size
        Parameters.epoch = epochs;%epochs
        Parameters.regularization = Regularization; %regularization factor
        Parameters.learningrate = LearningRate; %learning rate
        
        Adam_Parameters = containers.Map('UniformValues',false);%store the adam parameters
        Weight = containers.Map('UniformValues',false);%store the weight
        Bias = containers.Map('UniformValues',false);%store the bias
        
        for Layer_Num = 1:5
            
            weight = strcat('w', num2str(Layer_Num));
            Weight(weight) = Std_Weight * randn(Weight_Row,NeuronLayer(Layer_Num));
            Weight_Row = NeuronLayer(:,Layer_Num) ;
            bias = strcat('b', num2str(Layer_Num));
            Bias(bias) = zeros(1,NeuronLayer(Layer_Num));
            first_moment_estimate = strcat('m',num2str(Layer_Num));
            Adam_Parameters(first_moment_estimate)=0;
            first_bias_corrected_moment_estimate = strcat('mt',num2str(Layer_Num));
            Adam_Parameters(first_bias_corrected_moment_estimate)=0;
            second_moment_estimate = strcat('v',num2str(Layer_Num));
            Adam_Parameters(second_moment_estimate)=0;
            second_bias_corrected_moment_estimate = strcat('vt',num2str(Layer_Num));
            Adam_Parameters(second_bias_corrected_moment_estimate)=0;
            first_moment_estimate_b = strcat('m_b',num2str(Layer_Num));
            Adam_Parameters(first_moment_estimate_b)=0;
            first_bias_corrected_moment_estimate_b = strcat('mt_b',num2str(Layer_Num));
            Adam_Parameters(first_bias_corrected_moment_estimate_b)=0;
            second_moment_estimate_b = strcat('v_b',num2str(Layer_Num));
            Adam_Parameters(second_moment_estimate_b)=0;
            second_bias_corrected_moment_estimate_b = strcat('vt_b',num2str(Layer_Num));
            Adam_Parameters(second_bias_corrected_moment_estimate_b)=0;
            
        end
        
        Weight(weight) = Std_Weight * randn(100,8);
        Bias(bias) = zeros(1,8);
        
    end
    function [Loss_Function,Activation_Function] = forwardpass(Parameters,Input_X,Output_Y,Weight,Bias)
        
        Activation_Function = containers.Map('UniformValues',false); % A = f(Z) where f is the activation function
        Input_Weighted_Sum = containers.Map('UniformValues',false); %Linear function A * X + B
        NeuronLayer = Parameters.layer_of_neuron;
        LayerNumber = Parameters.layer_numbers;
        
        
        
        
        %layer 1
        layer1_bias = strcat('b', num2str(1));
        layer1_weight = strcat('w', num2str(1));
        layer1_z = strcat('z', num2str(1)); %z = w*a
        
        Input_Weighted_Sum(layer1_z) = Input_X * Weight(layer1_weight) + Bias(layer1_bias); % X = a where a0 is the training set
        activation = Input_Weighted_Sum(layer1_z);
        activation(activation<=0 ) = 0; %ReLU function
        layer1_a = strcat('a', num2str(1)); %a = f(z)
        Activation_Function(layer1_a) = activation;
        Input_X = Activation_Function(layer1_a); %for next iteration

        %layer 2
        layer2_bias = strcat('b', num2str(2));
        layer2_weight = strcat('w', num2str(2));
        layer2_z = strcat('z', num2str(2)); %z = w*a

        Input_Weighted_Sum(layer2_z) = Input_X * Weight(layer2_weight) + Bias(layer2_bias); % X = a where a0 is the training set
        activation = Input_Weighted_Sum(layer2_z);
        activation(activation<=0 ) = 0; %ReLU function
        layer2_a = strcat('a', num2str(2)); %a = f(z)
        Activation_Function(layer2_a) = activation;
        Input_X = Activation_Function(layer2_a); %for next iteration

        %layer 3
        layer3_bias = strcat('b', num2str(3));
        layer3_weight = strcat('w', num2str(3));
        layer3_z = strcat('z', num2str(3)); %z = w*a
        
        Input_Weighted_Sum(layer3_z) = Input_X * Weight(layer3_weight) + Bias(layer3_bias); % X = a where a0 is the training set
        activation = Input_Weighted_Sum(layer3_z);
        activation(activation<=0 ) = 0; %ReLU function
        layer3_a = strcat('a', num2str(3)); %a = f(z)
        Activation_Function(layer3_a) = activation;
        Input_X = Activation_Function(layer3_a); %for next iteration

        
        
        %layer 4
        layer4_z = strcat('z', num2str(4)); %z = w*a
        layer4_weight = strcat('w', num2str(4));
        layer4_bias = strcat('b', num2str(4));
        Input_Weighted_Sum(layer4_z) = Input_X * Weight(layer4_weight) + Bias(layer4_bias); %N+1 is the last layer
        layer4_a = strcat('a', num2str(4)); %a = f(z)
        Activation_Function(layer4_a) = Input_Weighted_Sum(layer4_z);
        
        %Output Layer
        variable_z = strcat('z', num2str(5)); %z = w*a
        weight = strcat('w', num2str(5));
        bias = strcat('b', num2str(5));
        Input_Weighted_Sum(variable_z) = Input_X * Weight(weight) + Bias(bias);
        variable_a = strcat('a', num2str(5)); %a = f(z)
        Activation_Function(variable_a)=1./(1.0+exp(-1.0*Input_Weighted_Sum(variable_z)));
        final_output = Activation_Function(variable_a);
        Total_Weights = sum(cellfun(@(x) sum(x(:)), cellfun(@(x)x.^2,values(Weight),'UniformOutput',false)));%square all elements of each weight matrix

        Loss_Function = sum(-log(final_output(sub2ind([length(Output_Y) NeuronLayer(:,end)],(1:numel(Output_Y(:,end)))',Output_Y(:,end)))))/1*length(Output_Y) + Parameters.regularization*0.5*Total_Weights*1;
    end


    function [dW,dB] = backwardpass(Input_X,Output_Y,Activation_Function,Bias,Weight,Parameters)
        %Backpropagation function to calculate the gradient.
        
        %LayerNumber = Parameters.layer_numbers;
        
        %layer 4
        layer4_activation = strcat('a',num2str(4));
        DeltaK = Activation_Function(layer4_activation); %f(z) for last layer
        DeltaK = 2.5*(DeltaK-Output_Y(:,1:2)); %(y_hat-y)
       
        dW = containers.Map('UniformValues',false);
        dB = containers.Map('UniformValues',false);
        layer4_dw = strcat('dw', num2str(4));
        layer3_activation = strcat('a',num2str(3));
        layer4_weight = strcat('w',num2str(4));
        dW(layer4_dw) = 1*transpose(Activation_Function(layer3_activation)) * DeltaK + Parameters.regularization*Weight(layer4_weight);
        layer4_bias = strcat('b',num2str(4));
        layer4_db = strcat('db', num2str(4));
        dB(layer4_db) = 1*sum(DeltaK,1) + Parameters.regularization*Bias(layer4_bias);
        
        % Classification (Output)
        output_dw = strcat('dw', num2str(5));
        output_db = strcat('db', num2str(5));
        output_activation = strcat('a',num2str(5));
        output_deltak = Activation_Function(output_activation);
        output_deltak(sub2ind(size(output_deltak),(1:numel(Output_Y(:,end)))',Output_Y(:,end))) = output_deltak(sub2ind(size(output_deltak),(1:numel(Output_Y(:,end)))',Output_Y(:,end))) -1;
        %layer3_activation = strcat('a',num2str(LayerNumber-1));
        output_weight = strcat('w',num2str(5));
        output_bias = strcat('b',num2str(5));
        
        
        
        output_deltak = 1*output_deltak/length(Output_Y);
        dW(output_dw) = 1*transpose(Activation_Function(layer3_activation)) * output_deltak + Parameters.regularization*Weight(output_weight)*0.5*2;
        dB(output_db) = sum(output_deltak,1) + Parameters.regularization*Bias(output_bias)*1;
        
        hidlayer = 3;

        for Layer_Num = 1:2
            hidlayer_bias = strcat('b',num2str(hidlayer));
            hidlayer_weight = strcat('w',num2str(hidlayer));
            hidlayer_next_weight = strcat('w',num2str(hidlayer+1));
            hidlayer_last_activation = strcat('a',num2str(hidlayer-1));
            hidlayer_activation = strcat('a',num2str(hidlayer));
            hidlayer_db = strcat('db', num2str(hidlayer));
            hidlayer_dw = strcat('dw', num2str(hidlayer));
            
            hidlayer = hidlayer - 1;
            
            
            if Layer_Num == 1
                %delta = 0.1*1.2 * DeltaK * transpose(Weight(w)) + 0.1*0.8 * delta_k_class * transpose(Weight(w_class));
                hidlayer_delta = 0.5*1.2 * DeltaK * transpose(Weight(hidlayer_next_weight)) + 0.5*0.8 * output_deltak * transpose(Weight(output_weight));
                hidlayer_delta(Activation_Function(hidlayer_activation)<=0) = 0;
                dW(hidlayer_dw) = 1*transpose(Activation_Function(hidlayer_last_activation)) * hidlayer_delta + Parameters.regularization*Weight(hidlayer_weight)*1;
                dB(hidlayer_db) = sum(hidlayer_delta,1) + 1*Parameters.regularization*Bias(hidlayer_bias);
                delta_k = hidlayer_delta;
            else
                %delta = 10000*delta_k * transpose(Weight(w));
                hidlayer_delta = 50000*delta_k * transpose(Weight(hidlayer_next_weight));
                hidlayer_delta(Activation_Function(hidlayer_activation)<=0) = 0;
                dW(hidlayer_dw) = 1*transpose(Activation_Function(hidlayer_last_activation)) * hidlayer_delta + Parameters.regularization*Weight(hidlayer_weight)*1;
                dB(hidlayer_db) = sum(hidlayer_delta,1) + 1*Parameters.regularization*Bias(hidlayer_bias);
                delta_k = hidlayer_delta;
            end
            
        end

        
        %First layer backprop (layer 1)
        layer1_next_weight = strcat('w', num2str(2));
        delta = delta_k * transpose(Weight(layer1_next_weight));
        layer1_activation = strcat('a', num2str(1));
        delta(Activation_Function(layer1_activation) <=0 ) = 0;
        
        layer1_weight = strcat('w',num2str(1));
        layer1_dw = strcat('dw', num2str(1));
        dW(layer1_dw) = transpose(Input_X) * delta + Parameters.regularization*Weight(layer1_weight)*1;
        layer1_db = strcat('db', num2str(1));
        layer1_bias = strcat('b',num2str(1));
        dB(layer1_db) = sum(delta,1) + Parameters.regularization*Bias(layer1_bias)*1;
    end

    function [Loss,W,B] = training(X,Y,param,W,B,Ad)
        %Training function which iterate over a specified number of epoch. Takes as
        %input the training input and label, param struct, the initial weight and
        %bias, the gradient descent method to be used (update either 'sgd' or
        %'adam'), the adam parameters and the validation set. Returns the loss,
        %accuracy for  training, weights and biases.
        %Mini-batch are used instead of feeding all the training data at each
        %forward/backward pass.
        
        %epoch = batchsize * iteration/12000; %one epoch is one full sweep through all the data
        %X=zeros(10683,1568);
        iteration = (param.epoch*length(Y))/param.batchsize;
        X_ini = X; Y_ini = Y;
        
        %A
        Number_of_layer = param.layer_numbers; %Unroll variables from structure
        
        %Decay rate
        %wd = 0.00059;% wd = 0.0000025;,RMSE = 13.2827

        wd = 0.0000045; % RMSE = 13.2827
        
        
        
        for i = 1:iteration
            %     Mini-batch
            %X=zeros(6511-(iteration-1)*128,1568)
            Random_indexes = randperm(size(X,1));%size(X,1) = 13323
            %shuffle_indexes = randperm(r);
            Random_indexes = Random_indexes(1:param.batchsize);
            X_batch = X(Random_indexes, :);
            Y_batch = Y(Random_indexes,:);
            
            
            
            [loss,A] = forwardpass(param,X_batch,Y_batch,W,B);
            [dW,dB] = backwardpass(X_batch,Y_batch,A,B,W,param);
            
            Loss(:,i) = loss;
            
            
            for i1 = 1:Number_of_layer+1 %Update the weights and biases for each layer
                
                vt = strcat('vt',num2str(i1));
                v = strcat('v',num2str(i1));
                w = strcat('w', num2str(i1));
                dw = strcat('d',w);
                b = strcat('b', num2str(i1));
                db = strcat('d',b);
                
                m = strcat('m',num2str(i1));
                mt = strcat('mt',num2str(i1));
                

                
                %Ad(m) = param.beta1.*Ad(m) + (1-param.beta1).* dW(dw);
                Ad(m) = param.beta_1*Ad(m) + (1-param.beta_1).*(dW(dw) + wd .* W(w));
                Ad(mt) = Ad(m) ./ (1 - param.beta_1.^i1); 
                
                
                Ad(v) = param.beta_2 * Ad(v) + (1 - param.beta_2) * (dW(dw) .^ 2);
               
                Ad(vt) = Ad(v) / (1 - param.beta_2 .^ i1);
                if Ad(vt) > Ad(v)
                    Max = Ad(vt);
                else
                    Max = Ad(v);
                end
                M1 = W(w) - param.learningrate * Ad(mt) ./ (sqrt(Ad(vt)) + 1e-9);
                %AdamW
                M2 = W(w) - param.learningrate * (Ad(mt) ./ (sqrt(Ad(vt)) + 1e-9) - wd .* W(w));
                %W(w) = W(w) - param.n * Ad(mt) ./ (sqrt(Max) + 1e-9);
                %W(w) = W(w) -(param.n ./ (1 - param.beta1 .^ i1)) .* Ad(mt) ./ Ad(vt);
                %Nadam
                %W(w) = W(w) - (param.n .*(1 ./ (sqrt(Max + 1e-9))) .* (param.beta1 * Ad(mt) + ((1 - param.beta1).* dW(dw)) ./ (1 - param.beta1 .^ i1)));
                if M1 < M2
%                     if M1 < M3
                     W(w) = M1;
%                     else
%                         W(w) = M3;
%                     end
                else
                    W(w) = M2;
                end
                m_b = strcat('m_b',num2str(i1));
                Ad(m_b) = param.beta_1.*Ad(m_b) + (1-param.beta_1)*(dB(db) + wd .* B(b));
                mt_b = strcat('mt_b',num2str(i1));
                Ad(mt_b) = Ad(m_b) ./ (1 - param.beta_1 .^ i1);
                v_b = strcat('v_b',num2str(i1));
                Ad(v_b) = param.beta_2 * Ad(v_b) + (1 - param.beta_2) * (dB(db) .^ 2);
                vt_b = strcat('vt_b',num2str(i1));
                Ad(vt_b) = Ad(v_b) / (1 - param.beta_2 .^ i1);
                
                
                
                
                
                if Ad(vt_b) > Ad(v_b)
                    Max_b = Ad(vt_b);
                else
                    Max_b = Ad(v_b);
                end
                N1 = B(b) - param.learningrate * Ad(mt_b) ./ (sqrt(Ad(vt_b)) + 1e-9);
                %AdamW
                N2 = B(b) - param.learningrate * (Ad(mt_b) ./ (sqrt(Ad(vt_b)) + 1e-9) - wd .* B(b));
                %B(b) = B(b) - param.n * Ad(mt_b) ./ (sqrt(Max_b) + 1e-9);
                N3 = B(b) - (param.learningrate ./ (1 - param.beta_1 .^i1)) .* Ad(mt_b) ./ Ad(vt_b);
                %Nadam
                %W(w) = B(b) - (param.n .*(1 ./ (sqrt(Max_b + 1e-9)) .* (param.beta1 .* Ad(mt_b) + ((1 - param.beta1) .* dB(db)) ./ (1 - param.beta1 .^i1))));
                if N1 < N2
%                     if N1 < N3
                        B(b) = N1;
%                     else
%                         B(b) = N3;
%                     end
                else
                    B(b) = N2;
                end
                
                
            end
            
        end
    end
end