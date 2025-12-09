% Oscillator system (mass-spring-damper model)

% 1 DOF: position x
% Input: force F

% Alessandro Antonucci @AlexRookie
% University of Trento

% Model attributes
% 'linear': DEFAULT
% 'nonlinstiff': non linear stiffness coefficient (ax^3)
% 'coupled': coupled spring-damper characteristic (aẋx^2)
% 'deadzone': deadzone position-dependent (k(x)x)
% 'saturation': saturation velocity-dependent (c(ẋ)ẋ)

clear all;
clc;
close all;

%==========================================================================

% Parameters

model_attributes = {'linear'}; %, 'nonlinstiff', 'coupled', 'deadzone', 'saturation'};

num_sim = 100; % number of simulation
T = 20;        % simulation time (s)
dt = 0.001;     % simulation sampling time (s)

dt_data = 0.01; % dataset sampling time (s)

% m = [1, 3];     % mass (kg)
% k = [0.5, 2.5]; % stiffness  (N/m)
% c = [0.1, 0.9]; % viscous damping coefficient (N-s/m)
% a1 = [-3, 3];   % non linear stiffness coefficient (-)
% a2 = [0, 10];   % coupling coefficient (-)
% d;              % deadzone thresholds
% s;              % saturation thresholds
m = 1;
k = 3;
c = 0.175;
a1 = 3;
a2 = 2;
d  = 0.5;
s  = 1;

x0 = [0, 1];   % initial position (m)
v0 = [0, 0.5]; % initial velocity (m/s)

force = [-3, 3]; % input force (N)
u_types = 2; % input types: step, triangle

data_folder = 'msd-data';

% Options
options.plot = true;
options.save = true;

%==========================================================================

% Define and create data folder
outFolder = ['', data_folder, '/'];
if options.save == true
    if not(isfolder(outFolder))
        mkdir(outFolder);
    end
    if not(isfolder([outFolder, 'data/']))
        mkdir([outFolder, 'data/']);
    end
end

% Open parameter file
if options.save == true
    paramsfile = [outFolder, 'params.txt'];
    fid = fopen(paramsfile, 'w');
end

% Plot
if options.plot == true
    figure(1);
    hold on, box on, grid on;
    xlabel('Time (s)', 'interpreter', 'latex', 'fontsize', 18);
    ylabel('System response', 'interpreter', 'latex', 'fontsize', 18);
end

% Anonymous functions
deadzone_function = @(k,x,d) (k.*(x-d)).*(x>=d) + (0).*((-d<x)&(x<d)) + (k*(x+d)).*(x<=-d);
saturation_function = @(c,v,s) (c.*s).*(v>=s) + (c.*v).*((-s<v)&(v<s)) + (-c.*v).*(v<=-s);

% Time vector
t = (0:dt:T)';

downsample = round(dt_data/dt);
t_down = t(1:downsample:end);

samples = 0;
tic;

for i = 1:num_sim
    if mod(i,10) == 0
        disp(i);
    end
    
    %---------------------------------------------------------------------%
    
    % Get variable parameters
    
    % m_i = m(1) + (m(2)-m(1)).*rand(1); % 1;
    % k_i = k(1) + (k(2)-k(1)).*rand(1); % 3;
    % c_i = c(1) + (c(2)-c(1)).*rand(1); % 0.175;
    m_i = m;
    k_i = k;
    c_i = c;
    a1_i = a1;
    a2_i = a2;
    d_i = d;
    s_i = s;
    
    x0_i = x0(1) + (x0(2)-x0(1)).*rand(1);
    v0_i = v0(1) + (v0(2)-v0(1)).*rand(1);
    
    u_type_i = randi(u_types);
    u_mod_i = force(1) + (force(2)-force(1)).*rand(1);
    
    tu_start_i = round(1 + (length(t)/2-1)*rand(1));
    tu_end_i = round(1 + (length(t)/2-1)*rand(1));
    if tu_start_i > tu_end_i
        [tu_start_i,tu_end_i] = deal(tu_end_i,tu_start_i);
    end
    
    %---------------------------------------------------------------------%
    
    % Input vector
    U = zeros(length(t),1);
    if u_type_i == 1
        % Step
        U(tu_start_i:tu_end_i) = u_mod_i;
    elseif u_type_i == 2
        % Triangle
        tu_middle = floor((tu_start_i+tu_end_i)/2);
        U(tu_start_i:tu_middle) = linspace(0, u_mod_i, numel(tu_start_i:tu_middle));
        U(tu_middle+1:tu_end_i) = linspace(u_mod_i, 0, numel(tu_middle+1:tu_end_i));
    end
    
    % State vector
    Y = zeros(length(t),2);
    Y(1,:) = [x0_i, v0_i]; % initial state
    
    % Simulate
    for it = 1:length(t)-1
        x_acc = (1 / m_i) * ( U(it) - c_i * Y(it,2) - k_i * Y(it,1) );
        if any(strcmp(model_attributes, 'deadzone')) && any(strcmp(model_attributes, 'saturation'))
            x_acc = (1 / m_i) * ( U(it) - saturation_function(c_i,Y(it,2),s_i) - deadzone_function(k_i,Y(it,1),d_i) );
        elseif any(strcmp(model_attributes, 'deadzone'))
            x_acc = (1 / m_i) * ( U(it) - c_i * Y(it,2) - deadzone_function(k_i,Y(it,1),d_i) );
        elseif any(strcmp(model_attributes, 'saturation'))
            x_acc = (1 / m_i) * ( U(it) - saturation_function(c_i,Y(it,2),s_i) - k_i * Y(it,1) );
        end
        if any(strcmp(model_attributes, 'nonlinstiff'))
            x_acc = x_acc - a1_i * Y(it,1).^3;
        end
        if any(strcmp(model_attributes, 'coupled'))
            x_acc = x_acc - a2_i * Y(it,2) * Y(it,1).^2;
        end
        %error('Invalid model attribute: %s.', model_attributes{j});
        
        Y(it+1,1) = Y(it,1) + dt * Y(it,2);
        Y(it+1,2) = Y(it,2) + dt * x_acc;
    end
    %         if strcmp(model, 'linear')
    %             Y(it+1,1) = Y(it,1) + dt*Y(it,2);
    %             Y(it+1,2) = Y(it,2) + dt*( (1/m_i) * (U(it) - c_i*Y(it,2) - k_i*Y(it,1)) );
    %         elseif strcmp(model, 'duffing')
    %             Y(it+1,1) = Y(it,1) + dt*Y(it,2);
    %             Y(it+1,2) = Y(it,2) + dt*( (1/m_i)*U(it) - (c_i/m_i)*Y(it,2) - (k_i/m_i)*Y(it,1) - a1_i*Y(it,1).^3 - a2_i*Y(it,2)*Y(it,1).^2 );
    %         elseif strcmp(model, 'vanderpol')
    %             Y(it+1,1) = Y(it,1) + dt*Y(it,2);
    %             Y(it+1,2) = Y(it,2) + dt*( (1/m_i) * (U(it) + a2_i*Y(it,2) - a2_i*Y(it,2)*Y(it,1).^2 - Y(it,1)) );
    %         end
    %         if any(isnan(Y(it+1,:)))
    %             error('Invalid simulation step at it: %d', it);
    %         end
    
    %---------------------------------------------------------------------%
    
    % Downsample data
    U = U(1:downsample:end);
    Y = Y(1:downsample:end,:);
    
    % Plot
    if options.plot == true
        %cla;
        plot(t_down, U, 'g', t_down, Y(:,1), 'b-', t_down, Y(:,2), 'r--');
        %legend('Input (N)', 'x (m)', 'v (m/s)', 'fontsize', 12);
        %pause(1);
        drawnow;
    end
    
    samples = samples + size(Y,1);
    
    % Save params
    if options.save == true
        fprintf(fid, '%d;%d;%f;%f;%f;%f;%f;%f;%f;%f;%f;%f;%d;%f;%f\n', ...
            i, size(Y,1), m_i, k_i, c_i, a1_i, a2_i, d_i, s_i, x0_i, v0_i, u_mod_i, u_type_i, tu_start_i*dt, tu_end_i*dt);
    end
    
    % Save results
    if options.save == true
        simfile = [outFolder, 'data/', num2str(i), '.txt'];
        writematrix([t_down, Y, U], simfile, 'Delimiter', ';');
    end
    
end

Toc = toc;
fprintf('Elapsed time: %10.3f sec\n', Toc);

% Close text file
if options.save == true
    fclose(fid);
end

% Save statistics
if options.save == true
    clearvars str;
    str.model_attributes = model_attributes;
    str.num_simulations = num_sim;
    str.elapsed_time = Toc;
    str.total_samples = samples;
    str.params.time = T;
    str.params.sampling_time = dt;
    str.params.data_sampling_time = dt_data;
    str.params.m = m;
    str.params.k = k;
    str.params.c = c;
    str.params.a1 = a1;
    str.params.a2 = a2;
    str.params.d = d;
    str.params.s = s;
    str.params.x0 = x0;
    str.params.v0 = v0;
    str.params.force = force;
    statsfile = [outFolder, 'stats.txt'];
    fid = fopen(statsfile, 'w');
    fprintf(fid, jsonencode(str));    
    %     fprintf(fid, 'model;%s\nsimulations;%d\ntotal_samples;%d\nparam_time;%4.3f\nparam_sampling_time;%4.5f\nparam_downsampling;%4.5f', ...
    %         model, num_sim, samples, T, dt, dt_data);
    %     fprintf(fid, '\n');
    %     if strcmp(model, 'linear')
    %         fprintf(fid, 'm;%4.3f\nk;%4.3f\nc;%4.3f\nx_0;%4.3f;%4.3f\nv_0;%4.3f;%4.3f\nforce;%4.3f;%4.3f', ...
    %             m, k, c, x0, v0, force);
    %     elseif strcmp(model, 'duffing')
    %         fprintf(fid, 'm;%4.3f\nk;%4.3f\nc;%4.3f\na1;%4.3f\na2;%4.3f\nx_0;%4.3f;%4.3f\nv_0;%4.3f;%4.3f\nforce;%4.3f;%4.3f', ...
    %             m, k, c, a1, a2, x0, v0, force);
    %     elseif strcmp(model, 'vanderpol')
    %         fprintf(fid, 'm;%4.3f\na2;%4.3f\nx_0;%4.3f;%4.3f\nv_0;%4.3f;%4.3f\nforce;%4.3f;%4.3f', ...
    %             m, a2, x0, v0, force);
    %     end
    %fprintf(fid, 'm;%4.3f;%4.3f\nk;%4.3f;%4.3f\nc;%4.3f;%4.3f\nx_0;%4.3f;%4.3f\nv_0;%4.3f;%4.3f\nu;%4.3f;%4.3f', ...
    %    m, k, c, x0, v0, force);
    fclose(fid);
end

% Save plot
if options.plot == true && options.save == true
    savefig([outFolder, 'plot.fig']);
end

% Clear
clearvars fid paramsfile simfile statsfile;
