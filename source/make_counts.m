function make_counts
%%% Converts acceleration to activity counts
% % Reads files from folder with extension _motion.out
% % Saves file to folder with activity counts and extension counts.out
% % Uses code from (and thanks to):
% % https://github.com/maximosipov/actant/blob/master/sleep/oakley.m and
% % https://github.com/maximosipov/actant/blob/master/other/max2epochs.m
%%%

input_dir = '../outputs/cropped/*motion.out';
output_dir = '../outputs/cropped/';

% Get all files that have _motion.out extensions
file_list = [dir(input_dir)];

% Track how many aren't sampled at 50 Hz
cumulative_gaps = 0;
cumulative_time = 0;

threshold = 1; % Seconds, threshold for finding gaps in data dropped on server side

for f_index = 1:length(file_list)
    file = file_list(f_index);

    if file.bytes > 0
        fname = file.name;
        folder = file.folder;
        if exist([folder '/' fname], 'file') == 2

            % Read data
            data = dlmread([folder '/' fname]);

            % Timestamp differentials
            timestamps = sort(data(:,1));
            diff_time = timestamps(2:end) - timestamps(1:end-1);

            fprintf('Fraction of samples with significant gaps: %f\n',length(find(abs(diff_time) > threshold))/length(diff_time));
            fprintf('Fraction of time with significant gaps: %f\n',sum(abs(diff_time(find(abs(diff_time) > threshold))))/(sum(abs(diff_time))));

            cumulative_gaps = cumulative_gaps + sum(abs(diff_time(find(abs(diff_time) > threshold))));
            cumulative_time = cumulative_time + sum(abs(diff_time));

            % Resample to be 50 Hz
            fs = 50; % Hz

            time = min(data(:,1)):1/fs:max(data(:,1));
            z_data = interp1(data(:,1),data(:,4),time);

            % From here to end of file, code taken from https://github.com/maximosipov/actant/

            % Set filter specifications
            cf_low = 3;               % lower cut off frequency (Hz)
            cf_hi  = 11;              % high cut off frequency (Hz)
            order  = 5;               % filter order
            pass   = 'bandpass';      % filter type
            w1     = cf_low/(fs/2);   % normalized frequency low
            w2     = cf_hi/(fs/2);    % normalized frequency high
            [b, a] = butter(order, [w1 w2], pass);

            % Filter z data only
            z_filt = filtfilt(b, a, z_data);

            % Convert data to 128 bins between 0 and 5
            z_filt = abs(z_filt);
            topEdge = 5;
            botEdge = 0;
            numBins = 128;

            binEdges = linspace(botEdge, topEdge, numBins+1);
            [~, binned] = histc(z_filt, binEdges);

            % Convert to counts/epoch
            epoch = 15;
            counts = max2epochs(binned, fs, epoch);

            % NOTE: Please be aware that the algorithm used here has only been
            % validated for 15 sec epochs and 50 Hz raw accelerometery (palmar-dorsal
            % z-axis data. The formula (1) used below
            % is based on these settings. The longer the epoch, the higher the
            % constant offset/residual noise will be(18 in this case). Sampling frequencies
            % will probably affect the constant offset less. However, due
            % to the band-pass of 3-11 Hz used above and human movement frequencies
            % of up to 10 Hz, a sampling of less than 30 Hz is not reliable.

            % Subtract constant offset and multiply with factor for distal location
            counts = (counts-18).*3.07;                   % ---> formula (1)

            % Set any negative values to 0
            indices = counts < 0;
            counts(indices) = 0;
            time_counts = linspace(min(data(:,1)),max(data(:,1)),length(counts));

            counts = counts(:);
            time_counts = time_counts(:);

            output = [time_counts counts];

            % Plot (for debugging)
            stairs(time_counts,counts); drawnow

            save_name = [output_dir fname(1:end-10) 'counts.out'];
            fprintf('Saving to %s\n',save_name);
            dlmwrite(save_name,output,'delimiter',',','precision',15);

        end
    end
end

fprintf('\n\nFraction data with gaps: %f\n',cumulative_gaps/cumulative_time);

end


%% Code source: https://github.com/maximosipov/actant/blob/master/other/max2epochs.m
function epochdata = max2epochs(data, fs, epoch)
% MAX2EPOCHS Aggregates maximal values per second across epochs
%
% Description:
%   The function converts a time series to epochs of length epoch. It will
%   pick the peak values per second and sums these values over the epoch
%   This is an pre-processing step that is performed online on the
%   CamNtech/Respironics AWD/AWL devices
%
% Arguments:
%   data - input data timeseries
%   fs - sampling frequency of the data
%   epoch - required epoch length in seconds
%
% Results:
%   epochdata - series of epochs
%
% See also MEAN2EPOCHS
%
% Copyright (C) 2011-2013, Bart te Lindert
%
% All rights reserved.
%
% Redistribution and use in source and binary forms, with or without modification,
% are permitted provided that the following conditions are met:
%
%  - Redistributions of source code must retain the above copyright notice, this
%    list of conditions and the following disclaimer.
%  - Redistributions in binary form must reproduce the above copyright notice,
%    this list of conditions and the following disclaimer in the documentation
%    and/or other materials provided with the distribution.
%  - Neither the name of the University of Oxford nor the names of its
%    contributors may be used to endorse or promote products derived from this
%    software without specific prior written permission.
%
% THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
% ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
% WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
% IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
% INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
% BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
% DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
% LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE
% OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED
% OF THE POSSIBILITY OF SUCH DAMAGE.

% force column vector
data = data(:);

% length in full seconds
seconds = floor(length(data)/fs);

% rectify
data = abs(data);

% reshape data to samples-by-seconds matrix
data = data(1:seconds*fs);
data = reshape(data, fs, seconds);

% find max per second (i.e. across column)
data = max(data, [], 1);

% reshape data to epoch-by-epochs matrix
data = data(:);
N = length(data);
nepochs = floor(N/epoch);
data = data(1:nepochs*epoch);

% sum per epoch (i.e. across column)
data = reshape(data, epoch, nepochs);
epochdata = sum(data, 1);

epochdata = epochdata(:);   % force column vector

end
