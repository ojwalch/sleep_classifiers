function [tc,y] = circadianModel(lightStruct,tau)
% CIRCADIANMODEL  Simulates the model of the circadian clock given light
% input in a struct and a value for tau; assumes light sufficiently padded
% to remove initial condition effects
%
% [T,Y] = circadianModel(LIGHT,TAU) returns the timesteps in T and output
% variables (x, xc, n) in Y
%
% Example:
% [t,y] = circadianModel(struct('dur',dur,'time',time,'light',light_vec),24.2);


% Random initial conditions
ics = rand(1,3);
ics(1:2) = ics(1:2)*2 - 1;
dt = 0.1; % hours

tc = 0:dt:lightStruct.dur;
y = ode4(@simple,tc,ics,lightStruct);

%[tc,y] = ode23s(@simple, 0:dt:lightStruct.dur,ics,[],lightStruct);

    function dydt = simple(t,y,u)
        % Forger, 1999 - This is the one we use in the paper
        x = y(1);
        xc = y(2);
        n = y(3);
        
        alph = u.light(find(u.time < t, 1, 'last' ));
        
        if(isempty(alph))
            alph = 0;
        end        
        
        tx = tau; % 24.2;
        G = 19.875;
        k = .55;
        mu = .23;
        b = 0.013;
        
        Bh = G*(1-n)*alph;
        B = Bh*(1 - .4*x)*(1 - .4*xc);
        
        dydt(1) = pi/12*(xc + B);
        dydt(2) = pi/12*(mu*(xc - 4*xc^3/3) - x*((24/(.99669*tx))^2 + k*B));
        dydt(3) = 60*(alph*(1-n) - b*n);
        
        dydt = dydt';
    end


    function dydt = nonphotic(t,y,u)
        % St. Hilaire Model 2007
        x = y(1);
        xc = y(2);
        n = y(3);
        alph = interp1(u.time,u.light,t);
        
        tx = tau; % 24.2;
        
        G = 19.875;
        k = .55;
        b = 0.013;
        
        mu = .1300;
        q = 1/3;
        rho = 0.032;
        Bh = G*alph*(1-n);
        B = Bh*(1-0.4*x)*(1-0.4*xc);
        
        sw = sign(alph); % Sleep/wake, currently guessed from light exposure
        Nsh = rho*(1/3 - sw);
        Ns = Nsh*(1 - tanh(10*x));
        
        if (x < -0.3 && x > -0.9)
            Nsh = rho*(1/3);
            Ns = Nsh*(1 - tanh(10*x));
        end
        
        dydt(1) = pi/12* (xc + mu*(1/3*x+4/3*x^3-256/105*x^7) + B + Ns);
        dydt(2) = pi/12* (q*B*xc - x*((24/(0.99729*tx))^2 + k*B));
        dydt(3) = 60*(alph*(1-n) - b*n);
        
        dydt = dydt';
    end

    function dydt = kronauerJewett(t,y,u)
        % Kronauer-Jewett Model
        x = y(1);
        xc = y(2);
        n = y(3);
        
        alph = interp1(u.time,u.light,t);
        tx = tau; % 24.2;
        G = 19.875;
        k = .55;
        b = 0.013;
        
        mu = .1300;
        q = 1/3;
        Bh = G*alph*(1-n);
        B = Bh*(1-0.4*x)*(1-0.4*xc);
        dydt(1) = pi/12* (xc + mu*(1/3*x+4/3*x^3-256/105*x^7) + B);
        dydt(2) = pi/12* (q*B*xc - x*((24/(0.99729*tx))^2 + k*B));
        dydt(3) = 60*(alph*(1-n) - b*n);
        
        dydt = dydt';
    end

end



