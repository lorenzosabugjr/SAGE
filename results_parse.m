list_problem = {'least-squares'};
list_noise   = [1e-6 1e-3 1.0];
close all;

% Plotting the per-evaluation
for problem = list_problem
    if strcmp(problem,'diabetes')
        list_condnum = [1];
        log_en = 0;
    else
        list_condnum = [1 1e4 1e8];
        log_en = 1;
    end
    figure;
    for condnum_i = 1:length(list_condnum)
        condnum = list_condnum(condnum_i);
        for bfgs = [0 1]            
            subplot(2,length(list_condnum),bfgs*length(list_condnum)+condnum_i);
            title(sprintf('BFGS: %d, CN: %d', bfgs, condnum));
            for noise_i = 1:length(list_noise)
                fname = sprintf('%s-%d-%d-sb1g-%.6f.mat', problem{1}, condnum, bfgs, list_noise(noise_i));
                load(fname);
                res_hist = reshape(res_hist, 489, []);
                res_hist = sort(res_hist, 2);
                mean_100 = mean(res_hist,2);
                med_100  = median(res_hist,2);
                [r,q] = iqr(res_hist');
                hold on;
                plot(med_100,'Color',[0.0 0.0 0.0],'LineWidth',noise_i,'DisplayName',sprintf('Noise: %.6f',list_noise(noise_i)));

                fname = sprintf('%s-%d-%d-ffd-%.6f.mat', problem{1}, condnum, bfgs, list_noise(noise_i));
                load(fname);
                res_hist = reshape(res_hist, 499, []);
                res_hist = sort(res_hist, 2);
                mean_100 = mean(res_hist,2);
                med_100  = median(res_hist,2);
                hold on;
                plot(med_100,'Color',[0.2 0.2 1.0],'LineWidth',noise_i);

                fname = sprintf('%s-%d-%d-cfd-%.6f.mat', problem{1}, condnum, bfgs, list_noise(noise_i));
                load(fname);
                res_hist = reshape(res_hist, 499, []);
                res_hist = sort(res_hist, 2);
                mean_100 = mean(res_hist,2);
                med_100  = median(res_hist,2);
                hold on;
                plot(med_100,'Color',[1.0 0.2 0.2],'LineWidth',noise_i);
            end
            grid on; 
            if log_en
                yscale('log');
                ylim([1e-5 1]); 
            end
            hold off;
        end
    end
end

% Plotting the per-iteration
for problem = list_problem
    if strcmp(problem,'diabetes')
        list_condnum = [1];
        log_en = 0;
    else
        list_condnum = [1 1e4 1e8];
        log_en = 1;
    end
    figure;
    for condnum_i = 1:length(list_condnum)
        condnum = list_condnum(condnum_i);
        for bfgs = [0 1]            
            subplot(2,length(list_condnum),bfgs*length(list_condnum)+condnum_i);
            title(sprintf('BFGS: %d, CN: %d', bfgs, condnum));
            for noise_i = 1:length(list_noise)
                fname = sprintf('%s-%d-%d-sb1g-%.6f.mat', problem{1}, condnum, bfgs, list_noise(noise_i));
                load(fname);
                res_hist = reshape(res_hist, 489, []);
                iter_hist = cellfun(@unique, num2cell(res_hist, 1), 'UniformOutput', false);
                min_length = max(50, min(cellfun(@length, iter_hist)));
                max_length = max(cellfun(@length, iter_hist));
                padded_values = cellfun(@(x) [x; zeros(max_length - length(x), 1)], iter_hist, 'UniformOutput', false);
                iter_hist = cell2mat(padded_values);
                iter_hist = sort(iter_hist, 1,'descend');
                iter_hist(iter_hist==0) = NaN;
                iter_hist_mean = mean(iter_hist,2,'omitnan');
                hold on;
                stairs(iter_hist_mean,'Color',[0.0 0.0 0.0],'LineWidth',noise_i,'DisplayName',sprintf('Noise: %.6f',list_noise(noise_i)));
                xlim([0 min_length]);

                fname = sprintf('%s-%d-%d-ffd-%.6f.mat', problem{1}, condnum, bfgs, list_noise(noise_i));
                load(fname);
                res_hist = reshape(res_hist, 499, []);
                iter_hist = cellfun(@unique, num2cell(res_hist, 1), 'UniformOutput', false);
                max_length = max(cellfun(@length, iter_hist));
                padded_values = cellfun(@(x) [x; zeros(max_length - length(x), 1)], iter_hist, 'UniformOutput', false);
                iter_hist = cell2mat(padded_values);
                iter_hist = sort(iter_hist, 1,'descend');
                iter_hist(iter_hist==0) = NaN;
                iter_hist_mean = mean(iter_hist,2,'omitnan');
                hold on;
                stairs(iter_hist_mean,'Color',[0.2 0.2 1.0],'LineWidth',noise_i);

                fname = sprintf('%s-%d-%d-cfd-%.6f.mat', problem{1}, condnum, bfgs, list_noise(noise_i));
                load(fname);
                res_hist = reshape(res_hist, 499, []);
                iter_hist = cellfun(@unique, num2cell(res_hist, 1), 'UniformOutput', false);
                max_length = max(cellfun(@length, iter_hist));
                padded_values = cellfun(@(x) [x; zeros(max_length - length(x), 1)], iter_hist, 'UniformOutput', false);
                iter_hist = cell2mat(padded_values);
                iter_hist = sort(iter_hist, 1,'descend');
                iter_hist(iter_hist==0) = NaN;
                iter_hist_mean = mean(iter_hist,2,'omitnan');
                hold on;
                stairs(iter_hist_mean,'Color',[1.0 0.2 0.2],'LineWidth',noise_i);
            end
            grid on; 
            if log_en
                yscale('log');
                ylim([1e-5 1]); 
            end
            hold off;
        end
    end
end