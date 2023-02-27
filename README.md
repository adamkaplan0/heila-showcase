# Heila ML Showcase
This repository contains a highlighted collection of my prior ML code and an earlier draft of one of my dissertation papers (writing sample).

## Machine Learning Projects

I include code from two different machine learning projects I was a part of. All the code included here are subsets of the original projects, which are currently not publicly available.

### NHL Play-by-Play Prediction

This code sample is part of a team project for the Machine Learning (6.867) course. The goal of the project was to develop a machine learning algorithm to predict NHL game outcomes as the game goes on. To do so, I decided to use a mixture density recurrent neural network (MD-RNN) trained on NHL play-by-play data for multiple seasons, available on Kaggle. Using a MD-RNN allows me to showcase the evolution of the posterior distribution of the score difference (Home - Away) as the game goes on. **I chose to showcase some of the data pre-processing, the model definition & training (in PyTorch), as well as the result plot creation.** Although this code is not solely written by me, it is, in my opinion, a very good example of applying deep neural networks to time-series data.

### Hijab Detection

This code sample is part of a much broader research project on the impact of COVID-19 on Hijab wearing in Iran. The broader project is much larger in scale, starting with collecting millions of Instagram posts, images, and profile pictures, using different natural language processing approaches (name and entity detection, optical character recognition of the images, and topic modeling) on the Farsi data, explorative image clustering, and finally, detecting hijab wearing patterns over the duration of the pandemic. **I chose to showcase the creation of the hijab detection network, creating the training data for a third-party to label, and the complete code to detect whether or not a hijab is worn in an image.** All of this code is written by me, and highlights my ability to create novel data, transfer learn models, and basic computer vision manipulations.

## Writing Sample

The writing sample in this repository is an early draft of one of my dissertation papers. At the time of writing this draft, the implementation still had some bugs, meaning the simulation and application sections are not representative, so **please interpret them with caution**. 
