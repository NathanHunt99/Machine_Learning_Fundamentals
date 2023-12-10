
import numpy as np
import matplotlib.pyplot as plt

def REG_SGD():
    np.random.seed(10)
    x = np.random.rand(100) #100 numbers from 0 to 100
    w = 1 + 0.05*np.random.randn(100) # w with noise
    b = 3 + 0.05*np.random.randn(100) # b with noise
    y = w*x + b
    
    epochs = 20
    l_rate = 0.1
    
    
    w_guess = 4 #Initialize
    b_guess = 7
    
    
    for _ in range(epochs):
        for i in range(len(x)):
            xi = x[i]
            yi = y[i]
            y_pred = w_guess * xi + b_guess
            w_guess -= xi*l_rate*(y_pred - yi)
            b_guess -= l_rate*(y_pred - yi)
        
    plt.plot(x, y, '*')
    plt.plot(x, w_guess*x + b_guess, 'r')
    plt.grid()
    plt.show()
    return w_guess, b_guess


def REG_BGD():
    np.random.seed(10)
    x = np.random.rand(100) #100 numbers from 0 to 100
    w = 1 + 0.05*np.random.randn(100) # w with noise
    b = 3 + 0.05*np.random.randn(100) # b with noise
    y = w*x + b
    
    epochs = 20
    l_rate = 0.1
    batch_size = 4
    
    
    w_guess = 4 #Initialize
    b_guess = 7
    
    
    for _ in range(epochs):
        for i in range(0, len(x)):
            batch_end = min(i+batch_size, len(x))
            x_batch = x[i:batch_end]
            y_batch = y[i:batch_end]
            y_pred_batch = w_guess * x_batch + b_guess
            w_guess -= np.mean(x_batch)*l_rate*np.mean(y_pred_batch - y_batch)
            b_guess -= l_rate*np.mean(y_pred_batch - y_batch)
        
    plt.plot(x, y, '*')
    plt.plot(x, w_guess*x + b_guess, 'r')
    plt.grid()
    plt.show()
    return w_guess, b_guess

w, b = REG_SGD()
print("SGD: w = ", w, ", b = ", b)
w, b = REG_BGD()
print("BGD: w = ", w, ", b = ", b)

