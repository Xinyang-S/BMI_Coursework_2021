function estimated_plot(trial,times)

figure
for t = times
    [idx,~] = find(trial.spikes(:,t) == 1);
    if idx ~= 0
        t_plot(1:length(idx),:) = t;
        plot(t_plot,idx,'.','Color','b')
        hold on
        clearvars t_plot
    end
end

hold on
xlabel('Time in ms')
ylabel('Neurons units')
plot([300 300],[0 100],'Color','r','LineWidth',1.5)
hold on
plot([size(trial.spikes,2)-100 size(trial.spikes,2)-100],[1 100],'Color','r','LineWidth',1.5)
legend
end
