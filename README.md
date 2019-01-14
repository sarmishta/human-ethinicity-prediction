# Human Ethnicity Prediction

This project is an implementation of the VGGFace model described in ["Deep Face Recognition"](https://www.robots.ox.ac.uk/~vgg/publications/2015/Parkhi15/parkhi15.pdf) for classifying images in the [UTKFace](http://aicip.eecs.utk.edu/wiki/UTKFace) dataset based on ethnicity/race using Keras with TensorFlow backend.

## Organizing the data

UTKFace contains the following images which are labelled incorrectly:

```
39_1_20170116174525125.jpg.chip.jpg  
61_1_20170109150557335.jpg.chip.jpg  
61_1_20170109142408075.jpg.chip.jpg
```

These images are removed and the remaining data is organized into 5 folders: `Asian`, `Black`, `Indian`, `Others`, and `White` using the following commands:

```
mv *_*_0_*.jpg White
mv *_*_1_*.jpg Black
mv *_*_2_*.jpg Asian
mv *_*_3_*.jpg Indian
mv *_*_4_*.jpg Others
```

After this step, the distribution of the images is as follows:

```
White: 10,078
Black: 4,526
Asian: 3,434
Indian: 3,975
Others: 1,692
```

60% of the images in each of these folders are taken for training, 20% for validation, and the final 20% for testing. For example, 6047 (~60%) `White` images are randomly selected and moved to `White` in `Train` as in the first command below.

```
cd White
ls | shuf -n 6047 | xargs -i mv {} ../Train/White

cd ../Black
ls | shuf -n 2716 | xargs -i mv {} ../Train/Black

cd ../Asian
ls | shuf -n 2060 | xargs -i mv {} ../Train/Asian

cd ../Indian
ls | shuf -n 2385 | xargs -i mv {} ../Train/Indian

cd ../Others
ls | shuf -n 1015 | xargs -i mv {} ../Train/Others
```

A similar process is followed for `Val`:

```
cd ../White
ls | shuf -n 2016 | xargs -i mv {} ../Val/White

cd ../Black
ls | shuf -n 905 | xargs -i mv {} ../Val/Black

cd ../Asian
ls | shuf -n 687 | xargs -i mv {} ../Val/Asian

cd ../Indian
ls | shuf -n 795 | xargs -i mv {} ../Val/Indian

cd ../Others
ls | shuf -n 338 | xargs -i mv {} ../Val/Others
```

All other images are moved to `Test`:

```
cd ../White
mv * ../Test/White

cd ../Black
mv * ../Test/Black

cd ../Asian
mv * ../Test/Asian

cd ../Indian
mv * ../Test/Indian

cd ../Others
mv * ../Test/Others
```

## Preprocessing and model building

Each image is normalized by a factor of 1/255 before use. All training images are shuffled after each epoch so that the model generalizes well and random horizontal flips of images are taken to prevent the model from becoming sensitive to small variations.


## Performance

This version of VGGFace which is customized for ethnicity prediction achieves an accuracy of `84.5%` on the test set and other specific performance measures can be observed in the confusion matrix and the classification report.
