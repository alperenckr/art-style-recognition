import numpy as np


class ConvLayer:
    def conv(self, image ,filter ,stride : int ,padding : int):
        x, y = image.shape
        result = np.zeros(((image.shape[0] - filter.shape[0]) // stride + 1, (image.shape[1]-filter.shape[1]) // stride + 1),dtype=np.int32)
        for i in range(0,x-filter.shape[0]+1, stride):
            for j in range(0,y-filter.shape[1]+1, stride):
                result[i//stride][j//stride] = self.conv_prod(i,j,image,filter)
        return result
       
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
                result[i] = result[i] + self.conv(images[j],filters[i])
        result += biases
        return result


layer = ConvLayer()

image = np.zeros((5,28,28),dtype=np.int32)
#image = np.array([[1,1,1],[2,2,2],[3,3,3]],dtype=np.int32)

#filter = np.array([[2,4,3],[1,5,1],[1,2,3]],dtype=np.int32)
filter = np.array((10,3,3),dtype=np.int32)

layer.compute_layer()

result = layer.conv(image,filter,1,0)
print(result)
print(result.shape)