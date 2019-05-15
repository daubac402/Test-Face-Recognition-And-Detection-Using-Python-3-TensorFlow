# Test Face Recognition And Detection Using Python 3, TensorFlow

## Refer
1. https://medium.com/@somaniswastik/face-recognition-using-tensorflow-pre-trained-model-opencv-91184efa4aaf
2. https://www.tensorflow.org/tutorials
3. https://www.tensorflow.org/hub/tutorials/image_retraining

## Detach the face and convert to BW images
1. Prepare people images in labeled folder in /people
for eg: /people/person_a/images...
2. Execute
```sh
python3 image_normalizer.py
```

## Prepare normalized images into labled folder in /retrain_images
for eg: /people/person_a/images...
```sh
python3 retrain.py --image_dir=retrain_images --model_dir=retrained_models --summaries_dir=retrained_sumary --output_graph=retrained_graph.pb --bottleneck_dir=retrained_bottlenecks --how_many_training_steps=10000
```

## If you want to Check the modal, then have a look at http://localhost:6006/
```sh
tensorboard --logdir retrained_sumary
```

## Guest an image for ? people (with a percentage result) using modal
1. Prepare labels text file => check people_labels.txt
```sh
python3 label_image.py --graph=retrained_graph.pb --input_layer=Placeholder --output_layer=final_result --labels=people_labels.txt --image=test_image_path
```