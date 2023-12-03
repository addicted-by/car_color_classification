# MLOps practicing. (Classification of Cars Colors)

This repository made for MLOps techniques practicing.

## Task Description

The task is to develop a machine learning model that can accurately classify the colors of cars based on input images. The model should take an image of a car as input and predict the color of the car from a predefined set of colors.

## Setup

1. Clone the repository

```bash
git clone https://github.com/addicted-by/car_color_classification.git
```

2. Setup dependencies via poetry:

```bash
poetry install
```

3. Run the train process

```bash
poetry run python commands.py train
```

3.1. In case if something went wrong with data fetching perform:

```bash
poetry run dvc pull
```

4. Run the inference via the command

```bash
poetry run python commands.py infer
```

You can run the commands above with specified config files on your own.
Do not forget to set `tracking_uri` before the launch if you dont want to start it locally. The logs are saved in the default directory: mlruns. If you are using the standard MLFlow server, then run it before the training.

### Deployment [TO BE CONTINUED]
