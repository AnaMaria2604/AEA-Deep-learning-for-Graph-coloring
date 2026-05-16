from scipy.stats import sem, t
from numpy import mean

#confidence = 0.95
data = [10, 19, 11, 12, 15, 19, 9, 17, 1, 22, 9, 8]

n = len(data)
print("n: ", n)
m = mean(data)
print("mean: ", m)

#calculates the standard error of the mean
std_err = sem(data)

for confidence in [0.90, 0.95, 0.99]:
    #t.ppf = a Student’s T continuous random variable
    h = std_err * t.ppf((1 + confidence) / 2, n - 1)
    start = m - h
    end = m + h
    print("confidence:", confidence, "start: ", start, " end: ", end)

# n:  12
# mean:  12.666666666666666
# confidence: 0.9 start:  9.59297796408904  end:  15.740355369244293
# confidence: 0.95 start:  8.8996415486546  end:  16.433691784678732
# confidence: 0.99 start:  7.351023759181143  end:  17.98230957415219