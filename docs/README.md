## Archery-Cloth Simulation

### Video
<video src="./Archery_Simulation.mp4" width="320" height="200" controls preload></video>
### Description
For this project, I wanted to create a Archery game, that involves the simulation of the bow, bowstring, arrow, cloth as the shot target, and the interaction of all of them.  

The color of the shot area in the cloth would turn to black. This idea is from the Splatoon (A FPS video game on Switch).
<div align=center><img height="400" src="splatoon.jpg"/></div>

### Methodology
The project is developed based on the framework of Assignment 5, and all the objects were simulated by a bunch of particles with Position Based Dynamics (PBD). Those objects were also restrained by some physical conditions, including gravity, spring force, drawing force, air resistance, and collision force.
  

### Result
I also added a function to adjust the drawing force by pressing the keyboard. As shown in the video, we can see that with the greater drawing force, the collision and reflection effect will be more severe. But when the drawing force is too large, the arrow will continue to move through the clothing with its direction and speed slightly affected.


