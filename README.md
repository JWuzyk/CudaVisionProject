# Final project for Computer Vision Lab with Solomon Jacobs

Task: Create a neural network to perform tasks relevant to the vision system for a soccer playing robot.
 - Detect objects in the image: soccer balls, other robots and goal posts
 - Segment the image into: field, lines and background
 
We used a U-net like network based on the paper [1]. A network based on DeepLabV3 [2] was also tested.

## Results
Our results on the detction task are visualised here. 
![](/images/segmentation_task.png)

![](/images/detection_task.png?)

## Citations

1 Farazi, H. et al. (2018.) NimbRo Robots Winning RoboCup 2018 Humanoid AdultSize Soccer Competitions. Retrieved from http://arxiv.org/abs/1909.02385 

2 Chen, L.-C., Papandreou, G., Schroff, F., & Adam, H. (2017). Rethinking Atrous Convolution for Semantic Image Segmentation. Retrieved from http://arxiv.org/abs/1706.05587
