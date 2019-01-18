secondsPerMinute = 60;
secondsPerHour = 3600;
secondsPerDay = secondsPerHour*24; 
maxDaysToAverage = 5;
binSizeForStepsInMinutes = 60;
numHoursToSaveAsOutput = 9;
outputStepSizeInSeconds = 30;

for subject_num = 1:42
    
    % Load data
    stepsFilename = ['../../data/cleaned_data/' int2str(subject_num) '_steps.out'];
    scoresFilename = ['../../data/cleaned_data/' int2str(subject_num) '_scores.out'];
    
    if exist(stepsFilename, 'file') == 2
        if exist(scoresFilename, 'file') == 2
            
            stepsData = csvread(stepsFilename);
            scoreData = csvread(scoresFilename);
            
            stepsData = cleanStepsData(stepsData);
            dataDurationInDays = (max(stepsData(:,1)) - min(stepsData(:,1)))/secondsPerDay;
            
            if(dataDurationInDays > maxDaysToAverage)
                dataDurationInDays = maxDaysToAverage;
            end
                        
            % Bin and average data across days
            dt = binSizeForStepsInMinutes*secondsPerMinute;
            numberOfBinsPerDay = secondsPerDay/dt;
            averageSteps = zeros(1,numberOfBinsPerDay);
            cumulativeBinCount = zeros(1,numberOfBinsPerDay);
            
            croppedStartPoint = max(stepsData(:,1)) - secondsPerDay*floor(dataDurationInDays); 
            binIndex = 1;

            for t = (croppedStartPoint + dt/2):dt:(max(stepsData(:,1)) - dt/2)
                currentBin = mod(binIndex - 1, numberOfBinsPerDay) + 1; % Index for current hour
                overhangIndices = intersect(find(stepsData(:,1) >= t - dt/2),find(stepsData(:,1) < t + dt/2)); 
                averageSteps(currentBin) = averageSteps(currentBin) + sum(stepsData(overhangIndices,2)); 
                cumulativeBinCount(currentBin) = cumulativeBinCount(currentBin) + 1;
                binIndex = binIndex + 1;
            end
            
            averageSteps = averageSteps./cumulativeBinCount; % Normalize
            
            numDaysToSimulate = 50; % Remove initial condition effects
            lightForSimulation = repmat(steps2light(averageSteps),[1 numDaysToSimulate]);
            
            lightForSimulation = lux2alpha(lightForSimulation);
            
            % Add buffer at end to make sure we have enough coverage
            numBufferDays = 3;
            endTimestamp = max(stepsData(:,1)) + numBufferDays*24*60*60;
            timestampsForSimulation = fliplr(endTimestamp:-dt:(endTimestamp - (numDaysToSimulate)*24*60*60 + dt));
            
            % Include pre-PSG dark collection from DLMO
            numHoursInDark = 6;
            darkPeriodPrePSGIndices = timestampsForSimulation > min(scoreData(:,1)) - numHoursInDark*60*60;
            lightForSimulation(darkPeriodPrePSGIndices) = 0;
            
            % Crop anything after 10 hours from PSG start
            overhangIndices = find(timestampsForSimulation > 10*60*60 + min(scoreData(:,1)));
            timestampsForSimulation(overhangIndices) = [];
            lightForSimulation(overhangIndices) = [];

            % Scale for simulation
            shiftedTimestampsForSimulation = timestampsForSimulation - min(timestampsForSimulation);
            timestampsForSimulationScaledHours = shiftedTimestampsForSimulation/(3600);
            durationHours = max(timestampsForSimulationScaledHours);
            
            lightStruct = struct('dur',durationHours,'time',timestampsForSimulationScaledHours,'light',lightForSimulation);
            [tc,y] = circadianModel(lightStruct,24.2);
            
            % Convert simulation time to seconds
            timestampsCircadianOutput = tc*3600 + min(timestampsForSimulation);
            outputTimestampsSeconds = min(scoreData(:,1)):outputStepSizeInSeconds:(min(scoreData(:,1)) + numHoursToSaveAsOutput*secondsPerHour);
            
            outputCircadianX = interp1(timestampsCircadianOutput,y(:,1),outputTimestampsSeconds);
            
            output = [outputTimestampsSeconds; outputCircadianX];
            output = output';
            
            figure(1);
            plot(outputTimestampsSeconds - min(outputTimestampsSeconds),outputCircadianX); hold on; drawnow;

            outputFilename = ['../../data/cleaned_data/' int2str(subject_num) '_clock_proxy.out'];
            dlmwrite(outputFilename,output,'delimiter',',','precision',12); % w/o precision, won't save correctly
            
        end
    end
end