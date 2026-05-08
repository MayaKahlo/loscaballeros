# CS1430 Final Project, Team: Los Caballeros

We create digital custom coloring pages from real-life images using Meta's Segment Anything model and an interactive web application. Users can create 2D coloring pages using our website, and we also developed a proof of concept for creating 3D interactive coloring pages using scene reconstruction, segmentation, and interaction through a game. 

## Visit the [website](https://mayakahlo.github.io/loscaballeros/) hosting our 2D implementation

## 3D proof of concept demo ![here](/3d-playtime/demo.mp4)

## File Structure and Explanation 
### this list of files is not exhaustive but rather describes the overall structure of our repo


|- index.html - main implementation file for 2D frontend  
|- readme.md - you are here  


|- backend  
|-- app.py - main implementation file for 2D backend  
|-- NOTE: Our backend was hosted through HuggingFace Spaces and, in implementation, not a part of the same repo as the frontend. HuggingFace Spaces is built on top of Git, so we would have encountered conflicts due to nesting .git folders had this directory been within the frontend repo in our real implementation. The code is the same, but you can see the HuggingFace repo used in our hosted implementation [here](https://huggingface.co/spaces/loscaballeros/loscaballeros/tree/main).  


|- 3d-playtime  
|-- pointcloud - 3D backend  
|--- data - source image directory  
|--- generate_point_cloud.py - 3D reconstruction script  
|--- cluster_point_cloud.py - 3D segmentation script (outputs .obj files)  
|--- student.py - HW3 implementation w/ added helpers for reconstruction  
|-- unity - 3D frontend  
|--- Assets  
|---- Objects - imported .obj file directory  
|---- ComponentAdding.cs - add colliders to make objects interactive  
|---- PlayerController.cs - move the player and allow them to draw   

### Poster
![poster](/POSTER.PNG)

