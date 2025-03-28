# RailGuard-AI

An AI system based on state-of-the-art object 
detection models including YOLOv8, YOLOv8s, YOLO-NAS 
for improving crowd control, automating trash detection, 
and real-time monitoring system for violence escalation. 
Furthermore, it also integrates theft control and predictive 
maintenance strategies to make better use of resources and 
time management. Thus, equipping railways with this system 
will help in creating better public amenities, improving the 
running of the operations, and ensuring safety, as the 
existing systems tend to have a lot of gaps. 
![image](https://github.com/user-attachments/assets/da02e10b-cf08-4e8b-896e-cdcaf0b1bd36)

*__How it Works:__*  
• YOLOv8 processes an image by dividing it into an SxS 
grid. 
• For each grid cell, bounding boxes, class probabilities, 
and confidence scores are predicted.  
• Objects are detected based on the probability and 
bounding box overlap. 
STEP 1 : Input Image: The input image ,taken from real 
time cctv footage, is divided into an SxS grid. Here S = 
13x13 for YOLOv8. 
STEP 2 : Bounding Box Prediction. 
STEP 3 : Loss Calculation. 
STEP 4 : Non-Maximum Suppression : It is the final step 
of rescoring which in this case makes use of Intersection 
over Union (IoU) as the parameter. 
STEP 5 : Inference: We get the output as detected objects 
with bounding boxes along with their class probabilities in 
real-time.
