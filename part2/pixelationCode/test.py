#original pixelated code from: https://gist.github.com/danyshaanan/6754465
# Author: Dany Shaanan ->  danyshaanan
from PIL import Image
import os
backgroundColor = (0,)*3
pixelSize = 12


for i in range(10000):
    filename = "%s.jpg"%i
    # Image.open(filename).convert('RGB').save(filename'new.jpeg')
    if os.path.isfile(filename):
        image = Image.open(filename)
        image.convert('RGB').save("Original/%s.jpg"%i)
        image = image.resize((image.size[0]//pixelSize, image.size[1]//pixelSize), Image.NEAREST)
        image = image.resize((image.size[0]*pixelSize, image.size[1]*pixelSize), Image.NEAREST)
        pixel = image.load()

        image.convert('RGB').save("Pixelated/%s.jpg"%i)
    if i%100 == 0:
        print(i)
