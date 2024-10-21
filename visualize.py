import numpy as np

from PIL import Image
import matplotlib.pyplot as plt
import os
import random

def make_plot(number_of_images):
    rows, col = number_of_images//2, number_of_images//2
    fig, ax = plt.subplots(nrows=rows, ncols=col)
    for i in rows:
        for j in col:
            pass




if __name__=="__main__":

    folders = "gestures"

    sub_folders = [os.path.join(os.getcwd(),f.path) for f in os.scandir(folders) if f.is_dir()]

    total_images = []

    print(len(sub_folders))

    for dirname in list(sub_folders):
        images = [os.path.join(dirname,f) for f in os.listdir(dirname) if f.endswith(".png") or f.endswith(".jpeg")]
        total_images.extend(images)

    print(total_images)
    """
    for i in range(10):
        print(len(total_images))
        print(random.randint(0, int(len(total_images))))
        print(total_images[random.randint(0,len(total_images))])
        img = np.array(Image.open(total_images[random.randint(0,len(total_images))]))
        print(img.shape)
        plt.imshow(img)
        plt.show()"""




    

            