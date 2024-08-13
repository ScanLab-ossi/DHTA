import numpy as np
import numpy.ma as ma 
from   scipy.spatial.distance import jensenshannon
from scipy.special import kl_div 
from scipy.stats import entropy
import math


def fix_vectors(p,q):
    p = [0.0 if math.isnan(x) else x for x in p]
    q = [0.0 if math.isnan(x) else x for x in q]

    
    # Ensure the input arrays are numpy arrays and normalized
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    
    
    
    # Pad the shorter array with zeros to match the length of the longer array
    if len(p) < len(q):
        p = np.pad(p, (0, len(q) - len(p)), 'constant')
    elif len(q) < len(p):
        q = np.pad(q, (0, len(p) - len(q)), 'constant')
    
    return (p,q)

def calculate_kl_divergence (p,q):
    # Input : 2  probability distributions
    #p = np.array([0.2, 0.4, 0.4])
    #q = np.array([0.1, 0.5, 0.4])
    p,q=fix_vectors(p,q)
    # Calculate element-wise KL divergence
    kl_divergence = kl_div(p, q)

    # Sum the element-wise divergences to get the total KL divergence
    total_kl_divergence = np.sum(kl_divergence)

    print(f"Element-wise KL divergence: {kl_divergence}")
    print(f"Total KL divergence: {total_kl_divergence}")
    return kl_divergence # total_kl_divergence

def calculate_entropy(p,q): 
    p,q=fix_vectors(p,q)
    # Calculate KL divergence using scipy.stats.entropy
    kl_divergence = entropy(p, q)
    return kl_divergence 

def calculate_js_distance(p, q):
    """
    Calculate the Jensen-Shannon distance between two probability distributions.

    Parameters:
    p (numpy array): First probability distribution.
    q (numpy array): Second probability distribution.

    Returns:
    float: Jensen-Shannon distance.
    """
    p,q=fix_vectors(p,q)
    # Normalize the distributions (in case they are not already normalized)
    p /= np.sum(p)
    q /= np.sum(q)
    
    # Calculate the Jensen-Shannon distance using scipy's jensenshannon function
    js_distance = jensenshannon(p, q)
    
    return js_distance 

def calculate_jsd_sig(p, q):
    """
    Calculate the Jensen-Shannon distance between two probability distributions.

    Parameters:
    p (numpy array): First probability distribution.
    q (numpy array): Second probability distribution.

    Returns:
    float: Jensen-Shannon distance.
    """
    p,q=fix_vectors(p,q)
    #conversion to list

    plist = p.tolist()
    qlist = q.tolist()

    # Normalize the distributions (in case they are not already normalized)
    result_list = []
    # Calculate the Jensen-Shannon distance for each pair
    for p, q in zip(plist, qlist):
        # Convert to 2-element arrays (probability distributions)
        p_dist = np.array([p, 1 - p])
        q_dist = np.array([q, 1 - q])
        
        # Calculate the Jensen-Shannon distance
        js_distance = jensenshannon(p_dist, q_dist)
        
        # Append the result to the list
        result_list.append(js_distance)

        # Display the resulting list
    print(result_list)
    return(result_list) 

"""
p = np.array([0.1,0.4,0.0001, 0.3, 0.20009])
q = np.array([0.1, 0.5, 0.0001,0.2,0.2])
print ('JSD: ', calculate_js_distance(p,q) ) 
print ('calculate_kl_divergence: ', calculate_kl_divergence(p,q) ) 
print ('calculate_entropy: ',calculate_entropy(p,q) ) 
"""