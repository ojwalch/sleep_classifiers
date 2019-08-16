function light = counts2light(counts)
    light = 850*tanh(0.015*counts);
end