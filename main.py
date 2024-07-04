from PIL import Image
import numpy
import struct


# * from https://stackoverflow.com/a/63041700
# Assuming the image has channels as the last dimension.
# filter.shape -> (kernel_size, kernel_size, channels)
# image.shape -> (width, height, channels)
def convolve(image, filter, padding = (1, 1)):
    # For this to work neatly, filter and image should have the same number of channels
    # Alternatively, filter could have just 1 channel or 2 dimensions
    
    if(image.ndim == 2):
        image = numpy.expand_dims(image, axis=-1) # Convert 2D grayscale images to 3D
    if(filter.ndim == 2):
        filter = numpy.repeat(numpy.expand_dims(filter, axis=-1), image.shape[-1], axis=-1) # Same with filters
    if(filter.shape[-1] == 1):
        filter = numpy.repeat(filter, image.shape[-1], axis=-1) # Give filter the same channel count as the image
    
    #print(filter.shape, image.shape)
    assert image.shape[-1] == filter.shape[-1]
    size_x, size_y = filter.shape[:2]
    width, height = image.shape[:2]
    
    output_array = numpy.zeros(((width - size_x + 2*padding[0]) + 1, 
                             (height - size_y + 2*padding[1]) + 1,
                             image.shape[-1])) # Convolution Output: [(W−K+2P)/S]+1
    
    padded_image = numpy.pad(image, [
        (padding[0], padding[0]),
        (padding[1], padding[1]),
        (0, 0)
    ])
    
    for x in range(padded_image.shape[0] - size_x + 1): # -size_x + 1 is to keep the window within the bounds of the image
        for y in range(padded_image.shape[1] - size_y + 1):

            # Creates the window with the same size as the filter
            window = padded_image[x:x + size_x, y:y + size_y]

            # Sums over the product of the filter and the window
            output_values = numpy.sum(filter * window, axis=(0, 1)) 

            # Places the calculated value into the output_array
            output_array[x, y] = output_values
            
    return output_array



img = Image.open('d20.png')

org = numpy.asarray(img)

filter_arr = numpy.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])

# sharp = convolve(org, filter_arr)

flat = numpy.floor_divide(org.sum(2), [[383 for _ in range(org.shape[1])] for _ in range(org.shape[0])])

flat_shape = flat.shape

x_dim = (flat_shape[1] // 2) * 2
y_dim = (flat_shape[0] // 4) * 4


output = ""

b_pattern = [(0,0),(1,0),(2,0),(0,1),(1,1),(2,1),(3,0),(3,1)]

for line in range(0, y_dim, 4):
    for block in range(0, x_dim, 2):
        bs = ""
        for cur in b_pattern:
            c = flat[line + cur[0]][block + cur[1]].astype(int)
            if c >= 1:
                bs = "1" + bs
            else:
                bs = "0" + bs
        output += chr(int(bs, 2) + int("2800", 16))
    output += "\n"

with open('output.txt', 'w', encoding="utf-8") as file:
    file.write(output)
