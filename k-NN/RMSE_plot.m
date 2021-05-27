t=[6.5740,7.5837,9.8419,7.6132,13.3399,7.6936]
rmse=[12.912,23.4918,19.1693,27.9621,21.3837,19.6062]

figure;
for i=1:6
    scatter(t(i),rmse(i),'filled')
    hold on
end
xlabel('Time')
ylabel('RMSE')
legend('DNN','KNN','ECOC','Tree','Ensemble','Discriminant');
title('Time vs. RMSE of different methods')
