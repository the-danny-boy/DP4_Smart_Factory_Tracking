# DP4_Smart_Factory_Tracking
This is the repository for the Individual Research Project, as part of the CENG30013 Design Project 4 module of the Engineering Design course at the University of Bristol.

## Project Objective
The project objective is to investigate the applicability of novel computer vision methods for production line environments, working towards end-to-end traceability through persistent detection and tracking. The system should be capable of tracking items throughout the production line, as well as accumulating damage indicators between frames. 

## Usage Information
The project is developed in Ubuntu 20.04.1 LTS, using Python 3.8.5. Required libraries can be installed using the "./requirements/linux-requirements.txt" file.

## Other Requirements
This project uses the "multi-object-tracker" library developed by adipandas in order to speed up / streamline tracker implementation. Installation instructions can be found here if required: https://github.com/adipandas/multi-object-tracker  

This project also plans to use the Scaled-YOLOv4 detector, found at https://github.com/WongKinYiu/ScaledYOLOv4. This requires the "mish-cuda" library developed by thomasbrandon, which can be found at https://github.com/thomasbrandon/mish-cuda, or installed with "pip install git+https://github.com/thomasbrandon/mish-cuda/".

Pytorch is also a planned requirement for the project. This can be downloaded from the official pytorch website (https://pytorch.org/get-started/locally/), along with the corresponding version of the Nvidia Cuda Toolkit (v11.0 is used here).