# MAVRIC Trajectron Evaluation on NuScenes, Lyft, Argoverse, and Waymo Dataset Winter 2020
~ Nam Gyu Kil

## Project Goal
This project is to apply the [Trajectron++ code](https://github.com/StanfordASL/Trajectron-plus-plus) for vehicle trajectory predictions on various datasets. The four datasets that we tested Trajectron++ were [NuScenes](https://www.nuscenes.org/), [Lyft](https://self-driving.lyft.com/level5/data/), [Argoverse](https://www.argoverse.org/), and [Waymo](https://waymo.com/open/data/). The databases contain information on the ego vehicle, as well other vehicles and pedestrians for every timestep. The scenes for the databases range from 10-60 seconds. The databases have been recorded in different frequencies:
- NuScenes: 2 Hz
- Lyft: 5 Hz
- ArgoVerse: 5 Hz
- Waymo: 10 Hz  

The databases all have their own unique format which have been converted to a common format using [MAVRIC 2020 Summer Project](https://github.com/jskumaar/MAVRIC_Interaction_Modeling). This standard format is stored in a .csv file which is then processed into .pkl and fed into the Trajectron++ pipeline as input. 

## How to run the code
The code is stored in the folder ./experiments/MAVRIC as Jupyter Notebook files ./ipnb. There are multiple notebook which does the following:
0. Process:
