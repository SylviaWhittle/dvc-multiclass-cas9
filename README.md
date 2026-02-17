# dvc-multiclass-cas9

This repo is an attempt at making a plug-and-play DVC repository for people wanting to train U-Net models on their data.

## Setup

- Clone this repo
- Create a new conda environment (works on 3.10 for me)
- Install the required packages pip install -r requirements.txt
- Add your data to the data/all_data/ directory (this should be your sequentially named images and mask files: image_X.npy, mask_X.npy)
- You can then have a look at the experiments via dvc exp show and dvc exp list
- You can have a look at the processing pipeline via dvc dag and cat .dvc.yaml
- Run an experiment simply by running dvc exp run. It's likely that it will say that there are no updates and so it won't run anything - this is intended! It will only run an experiment if the result will change, so try editing a parameter like batch size or epochs: dvc exp run --set-param "train.batch_size=10"
- Multiple experiments can be queued via dvc exp run --queue --set-param "train.batch_size=8,16,24" for example. (Not done this yet but it's in the docs).
- Also use the Visual Studio Code DVCLive extension to see the cool graphs and tables comparing experiments!

## Branching

- If you want to upload your own experiment, try creating a branch with a descriptive name and running one, change the code if you like too, DVC seems to be able to handle that. Then message me to add you as a contributor and you can push your experiment.
