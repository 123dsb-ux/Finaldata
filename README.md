Finaldata is a comprehensive dataset containing 16,885 smoke fire images captured in an uncontrolled environment. The images are taken from the DFire dataset and a large collection of web-crawled data. In order to ensure the high quality and diversity of the dataset, we performed rigorous data cleaning to remove a large number of duplicate and invalid samples. The dataset contains two scenarios, urban fire and forest fire, and is labeled with two categories: fire and smoke, including 5867 images containing only flames, 3164 images containing smoke only, 4754 images containing fire and smoke, 9347 images of cities, and 7538 images of forests. The cleaned images were integrated with the independently collected images, and randomly divided into the training set and the test set at a 5:1 ratio.  This approach not only ensures the richness of the training set, but also provides reliable test data for model evaluation.

Download Link：https://drive.google.com/file/d/1F6Oysd1-B3FSvqYpQqLci9e9KPiPgm0u/view?usp=sharing

AEL-YOLO is a YOLOv8n-based model for fire and smoke detection. In AEL-YOLO, we propose a plug-and-play Asymmetric Parallel Convolution (APConv) module. APConv can be obtained from iAFF6.py or SAhead6.py. Based on APConv, we design an Attention Feature Fusion module (C2f-AFF) and a Lightweight Self-attention Detect head (LSADetect). These components significantly improve detection performance and are successfully integrated into different YOLO-based models (v8–v12). In addition, we have designed Efficient Cross-phase Network (ECN) module to further improve the computational efficiency. Experimental results on the Finaldata show that our method improves mAP@50 by 4.7% and mAP@50–95 by 4.5% compared to the YOLOv8n baseline. 

If you would like to use our model, please download the ultralytics framework. https://github.com/ultralytics/ultralytics/blob/main/ultralytics


![Image text](https://github.com/123dsb-ux/Finaldata/blob/main/Figure.png)
Performance comparison of different models on the Finaldata dataset.
