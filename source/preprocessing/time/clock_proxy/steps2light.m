function light = steps2light(steps, startTimeForAverageDay, dtSeconds)
light = [];
step_threshold = 20;

for i = 1:length(steps)
    value = 500;
    
    if mod(startTimeForAverageDay + i*dtSeconds/3600,24) > 10 && mod(startTimeForAverageDay + i*dtSeconds/3600,24) < 16
        value = 1000;
    end
    if mod(startTimeForAverageDay + i*dtSeconds/3600,24) < 7 || mod(startTimeForAverageDay + i*dtSeconds/3600,24) > 22
        value = 50;
        
    end
    
    if steps(i) < step_threshold
        steps(i) = 0;
    end
    
    light(i) = value*sign(steps(i));
    
end

end