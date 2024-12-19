# **Ant Weber's Length Detection Using YOLOv11**

This project presents a solution for detecting the *Weber's length* of ants using the YOLOv11-pose model. The workflow involves dataset preparation, model fine-tuning, and evaluation. The solution includes Python scripts and Jupyter Notebooks to facilitate model training, validation, and detection.

---

## **Project Overview**

The Weber's length, a key morphological measurement of ants, can be extracted by identifying specific body landmarks and scaling the detected measurements. This project automates the detection and quantification process using YOLOv11 fine-tuning, making it efficient and reliable for biological research.

---

## **Project Structure**

### **1. Dataset and Annotations**
- **`annotations.csv`**: CSV files containing ground-truth bounding boxes and labels for the dataset.

### **2. Python Scripts**
- **`create_dataset.py`**  
   - Contains functions to prepare and preprocesses the dataset for YOLOv11 fine-tuning.  
   - Handles annotation formatting and data splitting into training, validation, and test sets.

- **`utils.py`**  
   - Provides utility functions used across the project, data visualization tools, scale bar detection function, etc ...

### **3. Configuration File**
- **`hyp.yaml`**  
   - Hyperparameter configuration file for YOLOv11 fine-tuning, specifying learning rates, epochs, etc ...

### **4. Jupyter Notebooks**
- **`main_exp.ipynb`**  
   - Contains the main training pipeline for the YOLOv11 model.  
   - Fine-tunes the model using the prepared dataset and saves the best weights (`best.pt`).

- **`measurement.ipynb`**  
   - Measures the results of the model with respect to ground-truth measurements.  
   - Calculates the Weber's length based on detected landmarks.

- **`scale_bar_detection.ipynb`**  
   - Detects and quantifies scale bars within input images to ensure proper measurement scaling using an image processing algorithm.

- **`text_detection.ipynb`**  
   - Implements text detection (e.g., scale labels) from input images using  easyOCR.

- **`DeepLSD_scale_bar_detection.ipynb`**  
   - Implements scale bar detection using the DeepLSD project. (Make sure that the directory of DeepLSD is installed in the project folder and that the pre-trained DeepLSD weights, deeplsd_md.tar, are inside the project's folder). https://github.com/cvg/DeepLSD


---

## **Workflow**

Go to step 3 if you want to test our model (.pt)

1. **Scale Bar Detection**:  
   Run `scale_bar_detection.ipynb` to detect and get the positions of the start and end of the scale bar.

2. **Model Training**:  
   Run `finetuning.ipynb` to prepare the dataset and fine-tune the YOLOv11 model and save the best weights.

3. **Evaluation**:  
   Use `measurements.ipynb` to compare the predicted Weber's length with ground-truth landmarks and assess the performance of the model.

4. **Text Detection**:
   If necessary, detect and get the textual annotation inside an image using `text_detection.ipynb`.

---

## **Requirements**

- Python >= 3.8
- PyTorch >= 1.10
- ultralytics
- OpenCV
- easyocr
- NumPy
- Pandas
- Matplotlib

- Make sure thorax_only.csv and the image folder "ants" are inside the project's folder
- Add our model weights (.pt file) to the folder to test our finetuned model. 

Install dependencies using:
`pip install -r requirements.txt`

## Reproducibility

We provide a drive with checkpoints for all our trained YOLO models, as well as our dataset splits (train\val\test)
`https://drive.google.com/drive/folders/1ElyO_maC5C08JpuWhXqTZu9ZBB9VzCWa?usp=sharing`