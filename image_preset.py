from PIL import Image
import os
from tqdm import tqdm

def is_png(str):
    if (".png" in str):
        return (True)
    else:
        return (False)
dir_list = list(filter(os.path.isdir, os.listdir(".")))
folder_2_files = {} # folder name : list of image files

for folder in (dir_list):
    folder_2_files.update({folder:list(filter(is_png, os.listdir(folder)))})

for dir_name, images in (folder_2_files.items()):
    for img in (images):
        curr = Image.open(dir_name + "/" + img)
        curr.resize((224, 224), Image.ANTIALIAS).save(dir_name + "/" + img)

for folder in dir_list:
    train_num = int(len(folder_2_files[folder]) * 0.65)
    val_num = int(len(folder_2_files[folder]) * 0.25)
    test_num = len(folder_2_files[folder]) - val_num - train_num
    print(train_num)
    print(val_num)
    print(test_num)

# let the machine multiplicate the training set

from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode="nearest")

ttv = ["train", "validation", "test"]

for dir in (dir_list):
    train_num = int(len(folder_2_files[dir]) * 0.65)
    val_num = int(len(folder_2_files[dir]) * 0.25)
    test_num = len(folder_2_files[dir]) - val_num - train_num
    curr = "train"
    for folder in (ttv):
        if not os.path.exists(dir+"/"+folder):
            os.makedirs(dir + "/" + folder)
    for num, image in tqdm(enumerate(folder_2_files[dir])):
        img = load_img(dir+"/"+image)
        x = img_to_array(img)
        x = x.reshape((1,) + x.shape)
        if (num > train_num):
            curr = "validation"
        if (num > train_num + val_num):
            curr = "test"
        try:
            for idx, batch in enumerate(datagen.flow(x, batch_size=1, save_to_dir=dir+"/"+curr, save_prefix=dir, save_format="png")):
                if (idx == 20):
                    raise NotImplementedError
        except:
            print("Treatement for", image, "done as ", curr, "set.")