# DP4_Smart_Factory_Tracking
This is the repository for the Individual Research Project, as part of the CENG30013 Design Project 4 module of the Engineering Design course at the University of Bristol. The research project, titled "Investigating Computer Vision for Production Line Health Estimation", was completed in support of a Bachelor in Engineering (BEng) in Engineering Design.  


## Project Objective
The project aimed to develop a bespoke image processing pipeline to support production line monitoring applications. Most solutions are either too specific (e.g. end-to-end workflows that are entirely encapsulated and thus difficult to modify for other purposes) or too general (e.g. core object detection offerings). This project aimed to deliver an intermediary and modular solution that offered specific insights through a custom analysis layer, built on top of a core detection and tracking framework for easy modification.   

The project thereby covered the following areas:  
- Synthetic data generation (in lieu of limitations imposed by Covid and commercial sensitivity of information)
- Detector implementation and comparison
- Tracker implementation and comparison
- Analysis layer development  

The main conclusions were as follows:  
- Synthetic data is an effective way of expediting development processes, especially where real-world data may otherwise be lacking  
- The Scaled-YOLOv4 detector and centroid tracker were best-suited for the given benchmark, achieving the fastest speed, accuracy, and robustness  
- A novel analysis layer was developed and demonstrated, with desirable functionality and clear use cases  
- The combined pipeline's performance was insufficient (less than 30 FPS), highlighting a need for further optimisation and / or less stringent performance requirements  


## Usage Information
The project is developed in Ubuntu 20.04.1 LTS, using Python 3.8.5. Required libraries can be installed using the "./requirements/requirements.txt" file.  

The "Data_Generator" is built using Unity 2020.2.3f1, with the High Definition Render Pipeline (HDRP), and Vulkan Graphics API for Linux.  

All the functionality can be run separately by running the corresponding file. The combined pipeline functionality is demonstrated in the "metric_extraction.py" file, and is the recommended file to run to view the combined output of the project.  


## Other Requirements
This project uses the "multi-object-tracker" library developed by adipandas in order to speed up / streamline tracker implementation. Installation instructions can be found here if required: https://github.com/adipandas/multi-object-tracker.  

This project also plans to use the Scaled-YOLOv4 detector, found at https://github.com/WongKinYiu/ScaledYOLOv4. This requires Pytorch (https://pytorch.org/get-started/locally/) to be installed, as well as the corresponding version of the Nvidia Cuda Toolkit in order to provide GPU-acceleration (v11.0 is used here). Scaled-YOLOv4 also requires the "mish-cuda" library developed by thomasbrandon, which can be found at https://github.com/thomasbrandon/mish-cuda.  