secondsPerMinute = 60;
secondsPerHour = 3600;
secondsPerDay = secondsPerHour*24;
numHoursToSaveAsOutput = 9;
outputStepSizeInSeconds = 30;
hoursOverhangToCrop = 10;

maxDaysToAverage = 5; % Window of days to use in computing "average day"
binSizeForStepsInMinutes = 10; % Bin size for integration

close all;
valid_subjects = [1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16, 19, 20, 22, 23, 25, 27, 28, 29, 30, 32, 33, 34, 35, 38, 39, 41, 42];
    
for subject_num = valid_subjects
    fprintf('Running subject %d...\n',subject_num)
    
    % Load data
    stepsFilename = ['../../../../data/steps/' int2str(subject_num) '_steps.txt'];
    scoresFilename = ['../../../../outputs/cropped/' int2str(subject_num) '_cleaned_psg.out'];
    
    if exist(stepsFilename, 'file') == 2
        if exist(scoresFilename, 'file') == 2
            
            stepsData = csvread(stepsFilename);
            scoreData = dlmread(scoresFilename,' ');
            
            stepsData = cleanStepsData(stepsData);
            dataDurationInDays = (max(stepsData(:,1)) - min(stepsData(:,1)))/secondsPerDay;
            
            if(dataDurationInDays > maxDaysToAverage)
                dataDurationInDays = maxDaysToAverage;
            else
                % Repeat data to reach maxDaysToAverage
                basePaddingStartPoint = max(stepsData(:,1)) - maxDaysToAverage*secondsPerDay;                
                padding = [basePaddingStartPoint, interp1(stepsData(:,1), stepsData(:,2), basePaddingStartPoint + maxDaysToAverage*secondsPerDay - secondsPerDay)];
                paddingStartPoint = basePaddingStartPoint;
                
                cumulativeSum = 0;
                while paddingStartPoint < min(stepsData(:,1))
                    cumulativeSum = cumulativeSum + binSizeForStepsInMinutes*60;
                    paddingStartPoint = basePaddingStartPoint + cumulativeSum;
                    newPadding = [paddingStartPoint, interp1(stepsData(:,1), stepsData(:,2), basePaddingStartPoint + mod(cumulativeSum, secondsPerDay) + maxDaysToAverage*secondsPerDay - secondsPerDay)];
                    padding = [padding; newPadding];
                end
                
                stepsData = [padding; stepsData];
                dataDurationInDays = (max(stepsData(:,1)) - min(stepsData(:,1)))/secondsPerDay;

            end
            
            % Bin and average data across days
            dt = binSizeForStepsInMinutes*secondsPerMinute;
            numberOfBinsPerDay = secondsPerDay/dt;
            averageSteps = zeros(1,numberOfBinsPerDay);
            cumulativeBinCount = zeros(1,numberOfBinsPerDay);
            
            % Add buffer of days at the end
            numBufferDays = 3;
            
            croppedStartPoint = max(stepsData(:,1)) - secondsPerDay*floor(dataDurationInDays);
            endPointForLightWindow = (max(stepsData(:,1)) - dt/2);
            endTimestamp = max(stepsData(:,1)) + numBufferDays*secondsPerDay;
            
            if max(scoreData(:,1)) < max(stepsData(:,1))
                endTimestamp = max(scoreData(:,1)) + numBufferDays*secondsPerDay;
                endPointForLightWindow = (max(scoreData(:,1)) - dt/2);
                croppedStartPoint = max(scoreData(:,1)) - secondsPerDay*floor(dataDurationInDays);
            end
            
            binIndex = 1;
            
            for t = (croppedStartPoint + dt/2):dt:endPointForLightWindow
                currentBin = mod(binIndex - 1, numberOfBinsPerDay) + 1; % Index for current hour
                indicesInBin = intersect(find(stepsData(:,1) >= t - dt/2),find(stepsData(:,1) < t + dt/2));
                averageSteps(currentBin) = averageSteps(currentBin) + sum(stepsData(indicesInBin,2));
                cumulativeBinCount(currentBin) = cumulativeBinCount(currentBin) + 1;
                binIndex = binIndex + 1;
            end
            
            averageSteps = averageSteps./cumulativeBinCount; % Normalize
            
            dt_ob = datetime(croppedStartPoint, 'convertfrom','posixtime');
            dt_ob.TimeZone = 'America/New_York';
            tz_diff = hours(tzoffset(dt_ob));
            startTimeForAverageDay = mod(hour(dt_ob) + tz_diff + 24,24) + minute(dt_ob)/60;
            
            
            dtStartPSG = datetime(min(scoreData(:,1)), 'convertfrom','posixtime');
            dtStartPSG.TimeZone = 'America/New_York';
           
            
            psgOffset = hour(dtStartPSG) + tz_diff + minute(dtStartPSG)/60;
            
            stepsToLightOutput = steps2light(averageSteps, startTimeForAverageDay,dt);
            numDaysToSimulate = 60; % Remove initial condition effects
            lightForSimulation = repmat(stepsToLightOutput,[1 numDaysToSimulate]);
            
            lightForSimulation = lux2alpha(lightForSimulation);
            
            timestampsForSimulation = fliplr(endTimestamp:-dt:(endTimestamp - (numDaysToSimulate)*secondsPerDay + dt));
            
            % Include pre-PSG dark collection from DLMO
            numHoursInDark = 0;
            darkPeriodPrePSGIndices = timestampsForSimulation > min(scoreData(:,1)) - numHoursInDark*secondsPerHour;
            lightForSimulation(darkPeriodPrePSGIndices) = 0;
            
            % Crop anything after 10 hours from PSG start
            overhangIndices = find(timestampsForSimulation > hoursOverhangToCrop*secondsPerHour + min(scoreData(:,1)));
            timestampsForSimulation(overhangIndices) = [];
            lightForSimulation(overhangIndices) = [];
            
            % Plot local window before night of sleep
            plotIndices = find(timestampsForSimulation > min(scoreData(:,1) - secondsPerDay*2));
            figure(2); hold on;
            stairs(timestampsForSimulation(plotIndices) - timestampsForSimulation(plotIndices(1)),lightForSimulation(plotIndices));
            
            figure(3);
            stairs(timestampsForSimulation(plotIndices) - timestampsForSimulation(plotIndices(1)),lightForSimulation(plotIndices));

            % Scale for simulation
            shiftedTimestampsForSimulation = timestampsForSimulation - min(timestampsForSimulation);
            timestampsForSimulationScaledHours = shiftedTimestampsForSimulation/secondsPerHour;
            durationHours = max(timestampsForSimulationScaledHours);
            
            lightStruct = struct('dur',durationHours,'time',timestampsForSimulationScaledHours,'light',lightForSimulation);
            [tc,y] = circadianModel(lightStruct,24.2);
            
            % Convert simulation time to seconds
            timestampsCircadianOutput = tc*secondsPerHour + min(timestampsForSimulation);
            outputTimestampsSeconds = min(scoreData(:,1)):outputStepSizeInSeconds:(min(scoreData(:,1)) + numHoursToSaveAsOutput*secondsPerHour);
            
            outputCircadianX = interp1(timestampsCircadianOutput,y(:,1),outputTimestampsSeconds);
            outputCircadianXC = interp1(timestampsCircadianOutput,y(:,2),outputTimestampsSeconds);
            
            output = [outputTimestampsSeconds; outputCircadianX; outputCircadianXC];
            output = output';
            
            figure(1);
            plot(outputTimestampsSeconds - min(outputTimestampsSeconds),outputCircadianX); hold on; drawnow;

            outputFilename = ['../../../../data/circadian_predictions/' int2str(subject_num) '_clock_proxy.txt'];
            dlmwrite(outputFilename,output,'delimiter',',','precision',12); % w/o precision, won't save correctly
            
        end
    end
end