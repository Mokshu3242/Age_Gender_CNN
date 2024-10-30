## Age and Gender Detection using CNN (Deep Learning Project)

### Overview
The Age and Gender Detection project leverages Convolutional Neural Networks (CNNs) to classify age and gender from facial images. It has a Tkinter-based GUI for easy interaction and can be used in fields like demographics, security, and personalized advertising.

### Features
* **Age and Gender Classification**: Predicts age and gender from an image or live camera feed.
* **User Interface**: Tkinter-based GUI for intuitive usage.
* **Real-time Detection**: Processes camera input to detect age and gender instantly.

### File Structure
* `app.py`: The main GUI application with options for image upload or live camera access.
* `age_gender_model.h5`: Pre-trained CNN model for age and gender classification.
* `utils.py`: Helper functions for image pre-processing and predictions.

### Installation

#### Prerequisites
* Python 3.8+
* Libraries: TensorFlow, Keras, OpenCV, Pillow

#### Steps
* Clone the repository:
   ```bash
   git clone https://github.com/yourusername/age-gender-detection.git
   ```
* Navigate to the project directory:
   ```bash
   cd age-gender-detection
   ```
* Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Usage
* Run the application:
   ```bash
   python app.py
   ```
* Using the GUI:
   * **Upload Image**: Select an image to classify age and gender.
   * **Live Camera**: Use a live camera feed for real-time detection.

### Model Training
The project is trained on the UTKFace dataset. For re-training, preprocess the dataset and use TensorFlow/Keras to train the CNN model on age and gender labels.
