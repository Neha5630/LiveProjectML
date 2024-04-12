A project to demonstrate the creation of ML Pipeline

step1: create a virtual environment
```
conda create -p venv python==3.8
```
step2:create a.gitignore file
```
create the file by right click and include the venc in it
```

step 3: Create a requiremenet.txt file
```
pip install ir requirements.txt
```
step4: create setup.py file
```
This is to install the entire project as a package.Additionally,write a function to read the package from requirements.txt
```
step5: create a folder src
```
Include exception, logger, and utils python files. Make this folder as a package by including __init__.py file. The scr folder will include another folder with name components will be created. Include __init__.py also
```
step 5.1:Create a folder components
```
Include data_ingestion, data_transformation, model trainer, and __init_.py. These components are to be interconnected in future.
```
step5.2:Create a folder called pipeline
```
Create two python files training_pipeline and prediction_pipeline with __init__.py folder
```
step6:create a folder called notebooks
```
Create a folder called data and include the dataset. Additionally, create a EDA.ipynb file to do the EDA analysis.
```
