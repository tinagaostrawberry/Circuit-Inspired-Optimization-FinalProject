%% Prelim
seed = 0;
rng(seed)

% Discretization params (calcualted from ciropt)
h = 6.6666666636820855;
% Ciruit params
R = 1;
C = 10;


%% Define objective function

% Dim of problem
m = 30;
n = 100;

% Params
data = 'fromGithub'; % fromGithub newlyGen
switch data
    case 'fromGithub'
        load('fromPython.mat')
        c = c.';
        b = b.';
        z0 = z.';
    case 'newlyGen'
        A0 = randn(m,n);
        A = A0/min(eig(A0*(A0.'))); % For (1/2)-strongly convex
        b = randn(m,1);
        c = randn(n,1);
end

% Obj fun in dual
gobj = @(x) -huberPenaltyVec_conj(-(A.')*x,c) - (b.')*x;


%% Perform descent

% Iterations
maxIters = 30;
% Set up save vars
gobjVals = zeros(maxIters,1);
% Err tolerance
relErrThresh = 1e-3;

% Initialization
switch data
    case 'fromGithub'
        z = z0;
    case 'newlyGen'
        z = randn(m,1);
end
e2 = zeros(m,1);

% Iterate
x = zeros(m,1);
for iters = 1:maxIters

    x_prev = x;
    e2_prev = e2;
    z_prev = z;

    % Perform descent
    x = prox_huberDual(z_prev,(R/2),A,b,c,relErrThresh);
    y = (2/R)*(z_prev-x);
    e2 = e2_prev - (h/(2*C*R))*(R*y+3*e2_prev);
    z = z_prev - (h/(4*C*R))*(5*R*y+3*e2_prev);

    % Save
    gobjVals(iters) = gobj(x);

end

%% Plot

switch data
    case 'fromGithub'
        % Do nothing, f_star already loaded
    case 'newlyGen'
        f_star = gobjVals(iters);
end
fobjRelErr = abs(gobjVals(1:iters) - f_star)/abs(f_star);

% NOTE: Results don't converge as nicely as in paper because MATLAB's
% numerical conditioning (or maybe Windows numerical conditioning?) not as
% good.... Solving proximal should ensure that abs(-A.'*y) is never greater
% than 2 in order for dual obj fun to not shoot to negative inf. However,
% due to numerical rounding, sometimes it does. This does not happen when
% running in Python on Linux VM (and in fact is able to reproduce results
% of paper). 
figure
semilogy(1:iters,fobjRelErr,'.-')
xlabel('k')
ylabel('|g(z)-g*|/|g*|')

