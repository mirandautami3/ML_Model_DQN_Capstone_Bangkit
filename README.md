# Machine Learning
This is the documentation for ML models. This model attempts to solve VRP problem, We use Reinforcement Learning - Deep Q Network for this problem.

## Inputs and Outputs
1. The inputs are longitude and latitude coordinates, which will be taken from Open Street Maps(OSM) API
2. The outputs are the routes which can be visualize into OSM API, and the distance which then we can get the time, etc.

## Files and Dependencies
There are few files in this project, the main one is vrp_dqn_model.tflite which contains the RL-DQN algorithm and few python files containing the algorithm so that the model can produce routes for VRP.

1. libaries = this file contains the library used for this project to work, it is not needed to import this file as importing libraries will be done in each files.
2. logic = this contains algoritms needed to run the model, this includes ReplayBuffer(), VRPAgent(), train_vrp_agent(). This file is a neccesity for the model to work, so this must be included when running the model
3. main = this includes the algorithm to convert the coordinates gotten from the user to distance_matrix which is crucial for the logic alogirithms. This must be included.
4. test = this is for testing the tflite model, this is not neccesary to be included as this is file to check whether or not the tflite is successfuly exported.
5. vis = this contains the visualization algorithm using folium which will be in html file, this is a documentation that the algorithm works and can be used as reference for MD as the visualization code.

## Notes
1. As Reinforcement Learning is neither Supervised or Unsupervised Learning, there wont be a training or test datasets needed.
2. For running the model on cloud, make sure that logic.py and main.py are included as well as .tflite or .keras file (Make sure the algorithms/code inside logic.py and main.py are already stored in cloud before running the tflite or keras file).

## Important!
The input coordinates for testing are still included in this models, so there are still works to be done, feel free to contact me

## Things to improve
1. Make the main.py able to take input coordinates from user.
2. The main algorithm could take a long time to run (approx. 1 minute), so optimization is still needed.

# Contact
Nikolaus Vico Cristianto (081938322829)

# UPDATE DECEMBER 2024
The only needed file is the main.py, now the main.py includes all the algorithms needed for producing the routes.
