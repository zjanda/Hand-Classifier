# **Hand Pose Finger Counting**  

This program uses a device camera to track hand positioning, capturing and saving hand poses to train a model that predicts the number of fingers being held up.  

## **Features**  
- Real-time hand position tracking using the device camera.  
- Data collection for training a machine learning model.  
- Model prediction based on hand poses.  

## **Getting Started**  

### **1. Data Collection (Optional)**  
If you prefer to skip this step, you can download the pre-collected dataset instead.  

### **2. Running the Application**  
1. Ensure you have the required dependencies installed.  
2. Run the application:  
   ```bash
   python app.py
   ```
3. Select the desired functionality from the menu.  

## **Requirements**  
- Python 3.x  
- OpenCV (for camera input)  
- TensorFlow/PyTorch (for model training)  
- Other dependencies (install via `requirements.txt` if available)  

## **Future Improvements**  
- Enhanced model accuracy with more training data.  
- Improved UI for easier interaction.  
- Support for multiple hand gestures.  
