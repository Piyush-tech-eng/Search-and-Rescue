# Search & Rescue UAV Project

## Objective
Develop a UAV-based system for search and rescue operations using
computer vision, navigation, and autonomous control.

## Description
### Background
Autonomous image segmentation and feature detection and classification is an important aspect
of image processing. Image segmentation is the process of “partitioning a digital image into
multiple segments''. The goal of segmentation is to simplify and change the representation of an
image into something that is more meaningful and easier to analyze. Feature detection includes
finding areas of interests such as edges, corners and simple shapes. These features are then
classified into various categories based on their shape, colour or other inherent features. These
concepts are widely used in military and civilian UAV missions to gather information about areas
out of human reach, such as disaster-stricken or mountainous areas.
### Task
The theme for this task is Search and Rescue. A shipwreck has occurred in the ocean, and your
job is to gather information about the location and condition of stranded passengers. Your UAV
is collecting aerial images of the wreckage that look like the sample image given below.
Information about the input image :
● The input image is divided into two primary regions: the blue region corresponds to ocean,
while the brown/green region corresponds to land. Passengers within the image are denoted by
geometric shapes, with stars representing children, squares representing adults, and triangles
representing elderly individuals.
● The severity of each civilian’s condition is represented through color coding: red indicates severe
condition, yellow indicates mild condition, and green indicates safe.
● Additionally, the image contains three designated rescue pads (zones for evacuation and safety
denoted by circle). Among these, two rescue pads are situated on land and one rescue pad is
located in water. While all rescue pads serve the same purpose, civilians are required to be
assigned to the best available rescue pad based on their position in the image and their medical
emergency while keeping in check the capacity.
-  <img width="380" height="379" alt="image" src="https://github.com/user-attachments/assets/e2039493-5740-4304-a21b-8fdad2daa5c3" />

Sample Image
The task for you is to devise a method to assign each of the casualty to the best possible rescue
camp while making sure that the final casualty configuration for each of the camp is their respective
best possible combination based on the casualty scores. . This must be based on the following rules
:
Priority order of casualties : Star-3(Highest), Triangle-2, Square-1(Lowest)
Priority order of emergency : Severe-3(Highest), Mild-2, Safe-1(Lowest)
Max capacity of rescue camps : Pink-3 casualties, Blue-4 casualties, Grey-2 casualties
The best rescue camp for a particular casualty is based on a final score calculated by the amalgamation
of the Priority score and distance where the priority score is
Priority(casualty*emergency). In case of a similar priority score , a casualty with higher emergency
score will be given importance. Devise your own score taking in all the considerations and keep in mind
the max capacity of the camps while making sure that each camp has the highest possible total priority
score.
Input
A list of 10 images, similar to the sample image provided above
Expected Output
1. An output image, for each input image, that clearly shows the difference between the
ocean and land, by overlaying 2 unique colors on top of each. The expected output for the
given sample input is given below.
2. a)Count the number of casualties assigned to each of the three camps.
b)The details of casualties assigned to each of the three camps for each image(Agegroup
,medical emergency ) in the order [blue,pink,grey].
3. The total priority of each of the camps saved in a list and the avg. priority of the image
(rescue ratio of priority Pr) , calculated by summing the priorities of the camps and
averaging over the number of casualties.
4. A list of the names of the input images , arranges in descending order of their rescue ratio
(Pr)
The expected output for the given sample image is given below
- <img width="612" height="613" alt="image" src="https://github.com/user-attachments/assets/87f11843-297f-4e61-bfa2-5592830ee0fd" />
Sample output (Arrows are used for explanation)
1. [ [[3,3],[1,2]] , [[2,2],[3,1]] , [[3,2],[2,2]] ] green star was assigned to pink camp due
to lower priority than yellow triangle although distance must also be considered which
will be based on your score formula.
2. [[2,2]] (Priority score of casualty = 2x2 = 4 (for safe elderly) and along with distance
calculate your own defined score.
3. [[Summation of scores for blue] ,[Summation of scores for pink] ,[Summation of scores
for grey] ]
4. [image1, image3, image4….. etc] (this is based on the priority ratio of the various
images given as input) (not related to the given sample image)
Example:
Input images:
 - Image1 
<img width="179" height="179" alt="image" src="https://github.com/user-attachments/assets/531ffbe8-5b3c-436a-be33-51917e6ab1c3" />

 - Image2
<img width="183" height="179" alt="image" src="https://github.com/user-attachments/assets/c411ca4e-6e39-4b48-8f9c-8c6519bdf280" />

Sample Output:
Segmented Images for ocean and land.
Image_n= [ [[1,1],[1,2],[1,3],[3,2]] , [[3,1],[2,3],[3,3]] , [[2,1],[2,2]] ] for max 9 casualties.
Camp_priority = [[24,45,56],[25,50,70]]
Priority_ratio = [125/8=15.625,145/9=16.1]
image_by_rescue _ratio = [Image2, Image1]
To simplify the given task, we’ve given a step by step approach to learn various concepts and
libraries that are required to complete the task.

## Progress Tracking
This repository follows continuous progress commits.
Daily progress reports are maintained in `/reports`.

## Repository Structure
- src/ → Core modules
- reports/ → Progress reports
- docs/ → Diagrams, references and images

## Mentor
Shivam Sharma
