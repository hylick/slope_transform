"""
Documenation: This file gneerates batch data input to the TF linear
              linear regression model (run.py and model.py)

Description: This code creates the input and output files to train the
             linear regression model with. It creates weights and biases
             if they haven't been created yet.
             
             The weights and biases for this geenrator are stored by run.py
             for continued training or test/inference using the trained model.
"""

import numpy as np

def gen_data(batch_size,num_inputs,w=[],b=[]):
  y_output = np.zeros((batch_size,num_inputs))
  x_input = np.zeros((batch_size,num_inputs))
  if len(w) == 0:
    w = np.random.randint(10, size=(num_inputs,num_inputs))
    b = np.random.randint(5, size=(1,num_inputs))
#  print(w)
#  print(b)
  for i in range(batch_size):
    x = np.random.randint(10, size=(1,num_inputs))
#    print('----',x)
    y_output[i] = np.add(np.matmul(x,w),b)
    x_input[i] = x
#  print(y_output)
#  print(x_input)
  return w,b,x_input,y_output

#def main():
#  gen_data(3,2)

#if __name__ == '__main__':
#  main()
