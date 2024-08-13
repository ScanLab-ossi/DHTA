import numpy as np
from scipy.spatial.distance import jensenshannon

# Example lists
list1 = [0.1, 0.2, 0.3, 0.4]
list2 = [0.2, 0.2, 0.3, 0.3]

# Initialize an empty list to store the results
result_list = []

# Calculate the Jensen-Shannon distance for each pair
for p, q in zip(list1, list2):
    # Convert to 2-element arrays (probability distributions)
    p_dist = np.array([p, 1 - p])
    q_dist = np.array([q, 1 - q])
    
    # Calculate the Jensen-Shannon distance
    js_distance = jensenshannon(p_dist, q_dist)
    
    # Append the result to the list
    result_list.append(js_distance)

# Display the resulting list
print(result_list)