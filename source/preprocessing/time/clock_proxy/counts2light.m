function light = counts2light(counts, startTimeForCountVector, dtPSG)
count_threshold = 100;

for i = 1:length(counts)
    value = 500;
    if mod(startTimeForCountVector + i*dtPSG/3600,24) > 10 && mod(startTimeForCountVector + i*dtPSG/3600,24) < 16
        value = 1000;
    end
    if mod(startTimeForCountVector + i*dtPSG/3600,24) < 7 || mod(startTimeForCountVector + i*dtPSG/3600,24) > 22
        value = 50;
        
    end
    
    if counts(i) < count_threshold
        counts(i) = 0;
    end
    
    light(i) = value*sign(counts(i));
    
end



end
