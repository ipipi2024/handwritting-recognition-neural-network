import numpy
import matplotlib.pyplot as mathplot

# Open and read the dataset
data_file = open("mnist_dataset/mnist_train_100.csv", 'r')
data_list = data_file.readlines()
data_file.close()

# Process the data
all_values = data_list[1].split(',')
image_array = numpy.asarray(all_values[1:], dtype=float).reshape((28, 28))

# # Display the image
# mathplot.imshow(image_array, cmap='Greys', interpolation='None')
# mathplot.show()  # This ensures the plot is displayed


#input preparation
#divide by 255 to scale inputs to be in range [0,1]
#multiply by 0.99 so that range [0,0.99]
#shift to right 0.01 so that range [0.01, 1.00]
#we avoid 0 input that can kill weight updates 0 x wieight = 0
scaled_input = (numpy.asarray(all_values[1:], dtype=float) /255.0 * 0.99) + 0.01

#output nodes is 10 since we are trying to predict 1 out ten 
#possible numbers
onodes = 10
#set the all the nodes to 0 and shift right by 0.01 to avoid 0 outputs
#which can saturate network-> output close to min and max bound which cause gradient to diminish

targets = numpy.zeros(onodes) + 0.01 

#set the label data to have strong signal
targets[int(all_values[0])] = 0.99 

print(targets)


