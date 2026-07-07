% results_plot_grad.m
%
% MATLAB companion to utils/plot_grad_results.py. Loads grad-bmk-*.mat
% gradient-benchmark result files and plots two-panel (rel_err, cos_sim)
% histograms per (dim, problem, condnum, noise_type, noise_param)
% combination.
%
% Edit the settings below, then run this script from the repo root.

%% ------------------------- User settings -----------------------------
source_dirs = {'results/2026-07-05 14-26-07'};  % folders to search for .mat files
recursive   = false;                            % also search subfolders

dims         = [20];
problems     = {'least-squares'};               % hyphenated names are OK
condnums     = [1.0, 1.0e4];
noise_types  = {'uniform'};
noise_params = [0.0];
estimators   = {'ffd', 'cfd'};                  % compared in this order

output_dir    = 'plots';
rel_err_bins  = 80;
cos_sim_bins  = 80;
cos_sim_range = [-1.0, 1.0];
overwrite     = true;
%% -----------------------------------------------------------------------

files = grad_discover_files(source_dirs, recursive);
parsed = grad_parse_files(files, estimators);

for di = 1:numel(dims)
    for pi = 1:numel(problems)
        for ci = 1:numel(condnums)
            for ni = 1:numel(noise_types)
                for qi = 1:numel(noise_params)
                    combo = struct( ...
                        'dim', dims(di), ...
                        'problem', problems{pi}, ...
                        'condnum', condnums(ci), ...
                        'noise_type', noise_types{ni}, ...
                        'noise_param', noise_params(qi));

                    combo_files = grad_resolve_combo(parsed, combo, estimators);
                    if combo_files.Count == 0
                        fprintf('WARNING: no data found for %dD %s cond=%g %s noise=%g\n', ...
                            combo.dim, combo.problem, combo.condnum, ...
                            combo.noise_type, combo.noise_param);
                        continue;
                    end
                    grad_plot_combo(combo, combo_files, estimators, output_dir, ...
                        rel_err_bins, cos_sim_bins, cos_sim_range, overwrite);
                end
            end
        end
    end
end

%% ------------------------- Local functions -----------------------------
function files = grad_discover_files(source_dirs, recursive)
% Search each source dir (exact top-level, or recursive) for .mat files.
    files = {};
    for k = 1:numel(source_dirs)
        d = source_dirs{k};
        if ~isfolder(d)
            error('grad_plot:baddir', 'source_dir does not exist or is not a directory: %s', d);
        end
        if recursive
            listing = dir(fullfile(d, '**', '*.mat'));
        else
            listing = dir(fullfile(d, '*.mat'));
        end
        for j = 1:numel(listing)
            if listing(j).isdir
                continue;
            end
            files{end+1} = fullfile(listing(j).folder, listing(j).name); %#ok<AGROW>
        end
    end
end

function parsed = grad_parse_files(files, estimators)
% Parse every discovered file, discarding any that don't match the pattern.
    parsed = struct('dim', {}, 'problem', {}, 'condnum', {}, ...
        'estimator', {}, 'noise_type', {}, 'noise_param', {}, 'path', {});
    for k = 1:numel(files)
        [~, name, ext] = fileparts(files{k});
        m = grad_parse_filename([name ext], estimators);
        if isempty(m)
            continue;
        end
        m.path = files{k};
        parsed(end+1) = m; %#ok<AGROW>
    end
end

function match = grad_parse_filename(filename, estimators)
% Parse "grad-bmk-{D}D-{problem}-{condnum}-{estimator}-{noise_type}-
% {noise:.6f}.mat". Since problem may itself contain hyphens (e.g.
% "least-squares"), the configured estimator names anchor the split of
% the ambiguous middle section. Returns [] if the filename doesn't match.
    match = [];
    prefix = 'grad-bmk-';
    suffix = '.mat';
    if ~startsWith(filename, prefix) || ~endsWith(filename, suffix)
        return;
    end

    core = filename(length(prefix)+1 : end-length(suffix));
    tokens = strsplit(core, '-');
    if numel(tokens) < 5
        return;
    end

    dim_token = tokens{1};
    if ~endsWith(dim_token, 'D')
        return;
    end
    dim = str2double(dim_token(1:end-1));
    if isnan(dim)
        return;
    end

    rest = tokens(2:end);
    if numel(rest) < 4
        return;
    end

    noise_param = str2double(rest{end});
    noise_type = rest{end-1};
    if isnan(noise_param)
        return;
    end

    body = rest(1:end-2);
    for i = numel(body):-1:2
        if ~ismember(body{i}, estimators)
            continue;
        end
        condnum = str2double(body{i-1});
        if isnan(condnum)
            continue;
        end
        problem_tokens = body(1:i-2);
        if isempty(problem_tokens)
            continue;
        end
        match = struct( ...
            'dim', dim, ...
            'problem', strjoin(problem_tokens, '-'), ...
            'condnum', condnum, ...
            'estimator', body{i}, ...
            'noise_type', noise_type, ...
            'noise_param', noise_param);
        return;
    end
end

function combo_files = grad_resolve_combo(parsed, combo, estimators)
% Match each configured estimator to exactly one file for this combo.
% Errors on duplicate matches, listing all duplicate paths. Missing
% matches are simply omitted from the returned map (equivalent to the
% Python tool's missing_policy: warn_skip).
    combo_files = containers.Map('KeyType', 'char', 'ValueType', 'any');
    for k = 1:numel(estimators)
        est = estimators{k};
        matches = {};
        for j = 1:numel(parsed)
            p = parsed(j);
            if strcmp(p.estimator, est) && strcmp(p.problem, combo.problem) && ...
                    strcmp(p.noise_type, combo.noise_type) && p.dim == combo.dim && ...
                    abs(p.condnum - combo.condnum) < 1e-9 * max([1, abs(p.condnum), abs(combo.condnum)]) && ...
                    abs(p.noise_param - combo.noise_param) < 5e-7
                matches{end+1} = p.path; %#ok<AGROW>
            end
        end
        if numel(matches) > 1
            error('grad_plot:duplicate', ...
                'Duplicate files for %dD %s cond=%g %s noise=%g estimator=%s:\n%s', ...
                combo.dim, combo.problem, combo.condnum, combo.noise_type, ...
                combo.noise_param, est, strjoin(sort(matches), sprintf('\n')));
        elseif numel(matches) == 1
            combo_files(est) = matches{1};
        end
    end
end

function grad_plot_combo(combo, combo_files, estimators, output_dir, ...
        rel_err_bins, cos_sim_bins, cos_sim_range, overwrite)
% Render and save the two-panel (rel_err, cos_sim) PDF for one combo.
    present = {};
    for k = 1:numel(estimators)
        if isKey(combo_files, estimators{k})
            present{end+1} = estimators{k}; %#ok<AGROW>
        end
    end

    rel_data = containers.Map('KeyType', 'char', 'ValueType', 'any');
    cos_data = containers.Map('KeyType', 'char', 'ValueType', 'any');
    combined_rel = [];

    for k = 1:numel(present)
        est = present{k};
        s = load(combo_files(est), 'rel_err', 'cos_sim');
        rel = double(s.rel_err(:));
        cos = double(s.cos_sim(:));

        rel_valid = rel(isfinite(rel) & rel > 0);
        cos_valid = cos(isfinite(cos) & cos >= cos_sim_range(1) & cos <= cos_sim_range(2));

        rel_data(est) = rel_valid;
        cos_data(est) = cos_valid;
        combined_rel = [combined_rel; rel_valid]; %#ok<AGROW>
    end

    fig = figure('Visible', 'off');

    ax1 = subplot(2, 1, 1);
    hold(ax1, 'on');
    if ~isempty(combined_rel)
        lo = min(combined_rel);
        hi = max(combined_rel);
        if lo == hi
            lo = lo / 2;
            hi = hi * 2;
        end
        rel_edges = logspace(log10(lo), log10(hi), rel_err_bins + 1);
    else
        rel_edges = [];
    end
    for k = 1:numel(present)
        est = present{k};
        values = rel_data(est);
        n_valid = numel(values);
        label = sprintf('%s (n=%d)', est, n_valid);
        if ~isempty(rel_edges) && n_valid > 0
            weights = ones(n_valid, 1) / n_valid;
            histogram(ax1, values, rel_edges, 'DisplayStyle', 'stairs', ...
                'Normalization', 'count', 'Weights', weights, 'DisplayName', label);
        else
            plot(ax1, nan, nan, 'DisplayName', label);
        end
    end
    set(ax1, 'XScale', 'log');
    xlabel(ax1, 'relative error');
    ylabel(ax1, 'probability');
    title(ax1, sprintf('%dD %s cond=%g %s noise=%g', ...
        combo.dim, combo.problem, combo.condnum, combo.noise_type, combo.noise_param));
    legend(ax1, 'show', 'Location', 'best');
    hold(ax1, 'off');

    ax2 = subplot(2, 1, 2);
    hold(ax2, 'on');
    cos_edges = linspace(cos_sim_range(1), cos_sim_range(2), cos_sim_bins + 1);
    for k = 1:numel(present)
        est = present{k};
        values = cos_data(est);
        n_valid = numel(values);
        label = sprintf('%s (n=%d)', est, n_valid);
        if n_valid > 0
            weights = ones(n_valid, 1) / n_valid;
            histogram(ax2, values, cos_edges, 'DisplayStyle', 'stairs', ...
                'Normalization', 'count', 'Weights', weights, 'DisplayName', label);
        else
            plot(ax2, nan, nan, 'DisplayName', label);
        end
    end
    xlabel(ax2, 'cosine similarity');
    ylabel(ax2, 'probability');
    legend(ax2, 'show', 'Location', 'best');
    hold(ax2, 'off');

    if ~isfolder(output_dir)
        mkdir(output_dir);
    end
    out_name = sprintf('grad-hist-%dD-%s-cond%s-%s-noise%s.pdf', ...
        combo.dim, combo.problem, grad_fmt_num(combo.condnum), ...
        combo.noise_type, grad_fmt_num(combo.noise_param));
    out_path = fullfile(output_dir, out_name);

    if isfile(out_path) && ~overwrite
        close(fig);
        error('grad_plot:exists', 'Output already exists and overwrite is false: %s', out_path);
    end

    exportgraphics(fig, out_path, 'ContentType', 'vector');
    close(fig);
    fprintf('-> saved %s\n', out_path);
end

function s = grad_fmt_num(x)
% Trimmed %g formatting, matching the Python tool's output filenames.
    s = strtrim(sprintf('%g', x));
end
