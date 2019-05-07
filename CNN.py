import numpy as np
import im2col

class ConvLayer:

    def __init__(self,input, filter_count : int, filter_size : tuple(), filter_channel, stride: tuple(), padding: int,bias, activation):
        self.input = input
        self.input_shape = input.shape
        self.stride = stride
        self.filter_size = filter_size
        self.filter_count = filter_count
        self.filter_channel = filter_channel
        self.padding = padding
        self.activation = activation
        self.bias = bias
        self.filters = self.initialize_parameters()

    def initialize_parameters(self,filter_size,filter_channel,filter_count):
        return 0.01 * np.random.rand(self.filter_count,self.filter_channel,self.filter_size[0],self.filter_size[1])

    def output_shape(self):
        return ((self.filter_count, 
                 self.output_channel,
                 (self.input.shape[0] + 2 * self.padding - self.filter_size[0]) // self.stride + 1, 
                 (self.input.shape[1] + 2 * self.padding - self.filter_size[1]) // self.stride + 1))

    def forward_compute(self):
        n_filters, d_filter, h_filter, w_filter = self.filters.shape
        n_x, d_x, h_x, w_x = self.input.shape
        h_out = (h_x - h_filter + 2 * padding) // stride + 1
        w_out = (w_x - w_filter + 2 * padding) // stride + 1


        input_col = im2col(self.input,h_filter,w_filter,padding=self.padding,stride=self.stride)
        filter_col = W.reshape(n_filters, -1)

        out = W_col @ X_col + b
        out = out.reshape(n_filters, h_out, w_out, n_x)
        out = out.transpose(3, 0, 1, 2)
        self.output = out


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

        result = np.zeros((filters.shape[0], (image.shape[1] - filter.shape[1]) // stride + 1, (image.shape[2]-filter.shape[2]) // stride + 1),dtype=np.int32)
        biases = np.zeros((filters.shape[0], (image.shape[1] - filter.shape[1]) // stride + 1, (image.shape[2]-filter.shape[2]) // stride + 1),dtype=np.int32)
        biases += bias

        for i in range(filters.shape[0]):
            for j in range(images.shape[0]):
                result[i] = result[i] + self.conv(images[j],filters[i],stride,padding)
        result += biases
        return result


layer = ConvLayer()

image = np.zeros((5,28,28),dtype=np.int32)
#image = np.array([[1,1,1],[2,2,2],[3,3,3]],dtype=np.int32)

#filter = np.array([[2,4,3],[1,5,1],[1,2,3]],dtype=np.int32)
filter = np.zeros((10,3,3),dtype=np.int32)

result = layer.compute_layer(image,filter,0,1,0)

#result = layer.conv(image,filter)


print(result)
print(result.shape)