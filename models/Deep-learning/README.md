# cas13d-guide-efficiency

###Getting Started
Installing Requirements
```
pip3 install -r requirements.txt
```
Running a Model on the survival screen dataset with all features
```
python3 train.py --dataset guide_all_features_9f --model guide_nolin_ninef --kfold 9 --split 0
```
Note: The whole workflow for model hyperparameter tuning, model feature engineering, model testing and interpretation is described in the "model.sh" script in the main repository. 

See options.py for extra arguments you can pass into the train script, like `-r` for reggression and `-f` for focal loss


### Creating a new model
Make a new model in the models directory. Make sure the name of the file and the name of the model match (and if it is a function, ends in _model).
From then on, when running train.py you can add --model \<your model name without _model\> to run your model

### Creating a new dataset
Very similar to model
Make a new dataset in the dataset directory. Make sure the name of the file and the name of the dataset match, ends in _dataset.
From then on, when running train.py you can add --dataset \<your dataset name without _dataset\> to run your dataset

### Creating Options
If you want to add command-line arguments for hyperparameters to the model, you can add them in the options file and pass them into your model generator.