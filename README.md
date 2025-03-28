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
![image](https://github.com/user-attachments/assets/f5c3b399-20ec-4d6d-8587-7838cb307c48)
![image](https://github.com/user-attachments/assets/264cec51-b268-4a75-8167-2de3358f9676)
![image](https://github.com/user-attachments/assets/539b5ce2-062b-48d8-8fa7-e3716806f4cb)

*References*
![image](https://github.com/user-attachments/assets/52de06d0-3fc0-4124-9825-e8665ede88e0)


