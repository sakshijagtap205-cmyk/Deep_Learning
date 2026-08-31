import numpy as np

#Step 1 : Define imput features ie X
#                 [X1,X2,X3]
input = np.array([2.0,3.0,4.0])
print("X:", input)

#step2 : Deefine weights ie. W
                    #[W1,W2,W3]
weights = np.array([0.5,0.3,0.2])
print("W:", weights)

#step3 = Define bias ie b
#       b
bias = 0.1
print(":b", bias)

#step4 : Calculated weighted sum ie Z
#  Z = x1w1 + x2w2 + x3w3 +b
# z = (2.0*0.5) + (3.0*0.3) + (4.0*0.2) + 1.0

z = np.dot(input ,weights)+bias   #dot is method

#step 5 : Activation function (ReLU)

def ReLU(x):
    return max(0,x)

#step 6 : Final Output 
Y  = ReLU(z)
print("Y:", Y)

