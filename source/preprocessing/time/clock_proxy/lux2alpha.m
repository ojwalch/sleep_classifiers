function alpha = lux2alpha(I)

I0 = 9500;
p = .6;
a0 = .16;
alpha = a0*(I.^p/I0.^p);

end

