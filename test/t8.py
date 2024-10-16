import numpy as np
import   tensorflow as tf
ar =  np.array([1, 2, 3,4,5,6,7,8,9])
ar2  = np.array([1, 2, 3])
ar3 =  ar*ar2
print(ar3)
tf.reduce_max(ar3)