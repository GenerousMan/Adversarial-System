import numpy as np
import math

images1 = np.load("./results/CW-untarget-slim-k30-1000.npy")[:50]
images2 = np.load("./results/slim-10000-images.npy")[:50]
images2 = images2/255-0.5

# l2_distance = np.sqrt(np.sum((images2 - images1)*(images2 - images1)))
# print(l2_distance)
sum = 0
for i in range(10):
    images1_new = images1[i]
    images2_new = images2[i]
    l2_distance_s = np.sqrt(np.sum((images1_new - images2_new)*(images1_new - images2_new)))/3
    sum = sum + l2_distance_s
    print(l2_distance_s)
print(sum/10)