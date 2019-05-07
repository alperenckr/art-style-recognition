import numpy as np
import sys #int min

class ConvLayer:
    def conv(self, image ,filter ,stride : int ,padding : int):
        if(padding > 0):
            image = self.make_padding(image, padding)
        
        x, y = image.shape
        result = np.zeros(((image.shape[0] - filter.shape[0]) // stride + 1, (image.shape[1]-filter.shape[1]) // stride + 1),dtype=np.int32)
        for i in range(0,x-filter.shape[0]+1, stride):
            for j in range(0,y-filter.shape[1]+1, stride):
                result[i//stride][j//stride] = self.conv_prod(i,j,image,filter)
        return result

    def make_padding(self, image, padding : int):
        image = np.insert(image, [0]*padding, 0, axis = 1)
        image = np.insert(image, [image.shape[1]]*padding, 0, axis = 1)
        image = np.insert(image, [0]*padding, [0], axis = 0)
        image = np.insert(image, [image.shape[0]]*padding, [0], axis = 0)
        return image

    def conv_prod(self, x : int, y : int, image, filter):
        filter_x, filter_y = filter.shape
        sum = 0
        for i in range(filter_x):
            for j in range(filter_y):
                sum += image[x+i][y+j] * filter[i][j]
        return sum

    def compute_layer(self,images,filters,bias,stride,padding):

        result = np.zeros((filters.shape[0], (image.shape[0] - filter.shape[0]) // stride + 1, (image.shape[1]-filter.shape[1]) // stride + 1),dtype=np.int32)
        biases = np.zeros((filters.shape[0], (image.shape[0] - filter.shape[0]) // stride + 1, (image.shape[1]-filter.shape[1]) // stride + 1),dtype=np.int32)
        biases += bias

        for i in range(filters.shape[0]):
            for j in range(images.shape[0]):
                result[i] = result[i] + self.conv(images[j],filters[i], stride, padding)
        result += biases
        return result

    def max_pooling(self, image, filter_size : int, stride : int):
        x, y = image.shape
        result = np.zeros(((image.shape[0] - filter_size) // stride + 1, (image.shape[1]-filter_size) // stride + 1),dtype=np.int32)
        for i in range(0,x-filter_size+1, stride):
            for j in range(0,y-filter_size+1, stride):
                result[i//stride][j//stride] = self.get_max(i,j,image,filter_size)
        return result

    def get_max(self, x : int, y : int, image, filter_size : int):
        max = -sys.maxsize-1 #int min
        for i in range(filter_size):
            for j in range(filter_size):
                if(image[x+i][y+j] > max):
                    max = image[x+i][y+j]
        return max



layer = ConvLayer()

#image = np.zeros((28,28),dtype=np.int32)
image = np.array([[1,1,1],[2,2,2],[3,3,3]],dtype=np.int32)

filter = np.array([[2,4],[1,5]],dtype=np.int32)
#filter = np.array((2,2),dtype=np.int32)

#layer.compute_layer()

result = layer.conv(image,filter,1,1)
print(result)
print(result.shape)
result = layer.max_pooling(result, 2, 1)
print(result)
