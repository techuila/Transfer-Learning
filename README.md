# Transfer Learning
Script template for retraining a model. 

## run.py
Edit the following file path before runing the script: <br />
``
train_path = ''<br />
test_path = ''<br />
export_path = ''<br />
model_analytics_path = ''
``

## listfile.py
Edit the folder path to your training folder to list all classes on a txt file

#### Default model: MobileNetV2
#### Output file: saved_model.pb

## Tflite Conversion
Use tensorflow's tflite_convert cli:<br />
`tflite_convert  --saved_model_dir=output_folder --output_file=output_folder/model.tflite`
