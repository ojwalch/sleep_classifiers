function data = cleanStepsData(data)

% Remove data where timestamps = 0 
data(data(:,1) == 0,:) = [];

% If timestamps in milliseconds instead of seconds, convert to seconds
if(max(data(:,1)) > 1.5e10)
    data(:,1) = data(:,1)/1000;
end

end