import os
import random
import shutil

SOURCE_DIR = "data/raw"
DEST_DIR = "data"
SPLIT_RATIO = 0.8  

source_dir = os.path.join(SOURCE_DIR)

cats = [f for f in os.listdir(source_dir) if f.startswith("cat") and f.endswith(".jpg")]
dogs = [f for f in os.listdir(source_dir) if f.startswith("dog") and f.endswith(".jpg")]

random.shuffle(cats)
random.shuffle(dogs)

index_cat = int(len(cats) * SPLIT_RATIO)
index_dog = int(len(dogs) * SPLIT_RATIO)

train_cats = cats[:index_cat]
test_cats = cats[index_cat:]
train_dogs = dogs[:index_dog]
test_dogs = dogs[index_dog:]

train_dir = os.path.join(DEST_DIR, "train")
test_dir = os.path.join(DEST_DIR, "test")

os.makedirs(os.path.join(train_dir, "cats"), exist_ok=True)
os.makedirs(os.path.join(train_dir, "dogs"), exist_ok=True)
os.makedirs(os.path.join(test_dir, "cats"), exist_ok=True)
os.makedirs(os.path.join(test_dir, "dogs"), exist_ok=True)

for cat in train_cats:
    shutil.move(os.path.join(source_dir, cat), os.path.join(train_dir, "cats", cat))

for dog in train_dogs:
    shutil.move(os.path.join(source_dir, dog), os.path.join(train_dir, "dogs", dog))

for cat in test_cats:
    shutil.move(os.path.join(source_dir, cat), os.path.join(test_dir, "cats", cat))

for dog in test_dogs:
    shutil.move(os.path.join(source_dir, dog), os.path.join(test_dir, "dogs", dog))
