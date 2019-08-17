close all;
secondsPerHour = 3600;
hoursPerDay = 24;
secondsPerDay = secondsPerHour*hoursPerDay;
secondsPerMinute = 60;
psgCropWindowHours = 14;
hoursToPlot = 10;

defaultTau = 24.2;
dtPSG = 30; % Seconds
dtOutput = 30/secondsPerHour; % Hours

numDays = 5;
binSizeForCountsInMinutes = 10;
numRepDays = 20;
isNanThresh = 0.1; % Fraction of data allowed to be NaN

pathToActigraphy = '../../../../data/mesa/actigraphy/';
actigraphyFileList = [dir([pathToActigraphy '*csv'])];
actigraphyFileNames = {actigraphyFileList.name};

fid = fopen('../../../../data/mesa/overlap/mesa-actigraphy-psg-overlap.csv');
actigraphyPSGOverlap = textscan(fid, '%d%d%s%s','Delimiter',',','Headerlines',1);
mesaSubjectIDs = actigraphyPSGOverlap{1};
overlapLines = actigraphyPSGOverlap{2};
fclose(fid);

for i = 1:length(actigraphyFileNames)
    
    filename = [pathToActigraphy actigraphyFileNames{i}];
    
    if exist(filename, 'file') == 2
        
        % Format of textscan line: mesaid	line	linetime	offwrist	activity	
        % marker	whitelight	redlight	greenlight	bluelight	wake	
        % interval	dayofweek	daybymidnight	daybynoon

        fid = fopen(filename);
        input = textscan(fid, '%d%d%s%d%f%f%f%f%f%f%f%s%f%f%f','Delimiter',',','Headerlines',1);
        fclose(fid);
        
        mesaStringID = filename(end-7:end-4);
        mesaNumericalID = str2double(mesaStringID);
        
        person = struct(); 
        person.light = input{7};
        person.steps = input{5};
        person.times = input{3};
        person.sleep = input{8};
        person.dates = input{3};
        
        psgStartLine = overlapLines(mesaSubjectIDs == mesaNumericalID);
        
        if(~isempty(psgStartLine))

            splitStartTime = strsplit(person.times{psgStartLine},':');
            startTimeForCountVector = str2double(splitStartTime{1}) +  str2double(splitStartTime{2})/60;
            lightData = counts2light(person.steps(psgStartLine:(psgStartLine + secondsPerDay*numDays/dtPSG)), startTimeForCountVector, dtPSG);
            timestampData = 0:dtPSG:(length(lightData)*dtPSG - dtPSG);
            
            dt = secondsPerMinute*binSizeForCountsInMinutes; 
            
            timeForSimulation = [];
            lightForSimulation = [];
            
            % Bin light data
            for t = 0:dt:(max(timestampData)-1)
                indicesInBin = intersect(find(timestampData >= t),find(timestampData < t + dt));
                timeForSimulation = [timeForSimulation; t];
                lightForSimulation = [lightForSimulation; max(lightData(indicesInBin))];
            end
            
            figure(5);
            plot(lightForSimulation); hold off;
            
            lightForSimulation = lux2alpha(lightForSimulation);
            lightForSimulation = repmat(lightForSimulation,[numRepDays,1]);
            timeForSimulation = 0:dt:(length(lightForSimulation)*dt - 1);
            shiftedTimestampsForSimulation = timeForSimulation - min(timeForSimulation);
            timestampsForSimulationScaledHours = shiftedTimestampsForSimulation/secondsPerHour;
            
            duration = max(timestampsForSimulationScaledHours);
            
            if(sum(isnan(lightForSimulation)) < isNanThresh*length(lightForSimulation))
                
                % Clear NaNs
                temporaryLight = lightForSimulation;
                temporaryTime = timeForSimulation;
                temporaryLight(isnan(lightForSimulation)) = [];
                temporaryTime(isnan(lightForSimulation)) = [];
                lightForSimulation = interp1(temporaryTime,temporaryLight,timeForSimulation);
              
                % Create light struct
                lightStruct = struct('dur',duration,'time',timestampsForSimulationScaledHours,'light',lightForSimulation);
                [tc,y] = circadianModel(lightStruct,defaultTau);
                
                timestampsCircadianOutput = tc*secondsPerHour + min(timeForSimulation);

                figure(1); hold on;
                outputTimestampStart = duration - numDays*hoursPerDay;
                psg_range = outputTimestampStart:dtOutput:(outputTimestampStart + psgCropWindowHours); 
                
                outputCircadianX = interp1(tc,y(:,1),psg_range);
                                outputCircadianXC = interp1(tc,y(:,2),psg_range);

                outputTimestampsSeconds = (psg_range - min(psg_range))*secondsPerHour;
                indicesForPlotting = find(outputTimestampsSeconds < hoursToPlot*secondsPerHour);
                
                plot(outputTimestampsSeconds(indicesForPlotting),outputCircadianX(indicesForPlotting));
                drawnow; hold on;
                output = [outputTimestampsSeconds', outputCircadianX', outputCircadianXC'];
                
                outputFilename = ['../../../../data/mesa/clock_proxy/' mesaStringID '_clock_proxy.out'];
                dlmwrite(outputFilename,output,'delimiter',',','precision',12); 
                
            else
                
                fprintf('Invalid data for %s: Too many NaNs in actigraphy...\n', mesaStringID);
                outputFilename = ['../../../../data/mesa/clock_proxy/' mesaStringID '_clock_proxy.out'];
                dlmwrite(outputFilename,[-1,-1],'delimiter',',','precision',12); 

            end
        end
    end
end
