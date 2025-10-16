<p align="center">
    <img width="50" height="50" alt="pepper_red" src="https://github.com/user-attachments/assets/8b98972c-9677-4274-a8aa-d9fc152156d3" />
    <img width="50" height="50" alt="pepper_yellow" src="https://github.com/user-attachments/assets/e44658ea-a1c3-41cd-933b-cc9df004e37e" />
    <img width="50" height="50" alt="pepper_green" src="https://github.com/user-attachments/assets/f8e4cbd8-0d09-46cb-bff3-adc418d82457" />
    <img width="50" height="50" alt="pepper_rotten" src="https://github.com/user-attachments/assets/7314723e-0dfa-44b7-9ba0-d443eb278ebb" />
</p>

# Yaek Zaab — Bell Pepper Ripeness Detection

**Hi everyone !!**

This is my AI project called **"Yaek Zaab"**, which in Thai means **"แยกแซ่บ"**. A web application that helps detect the **color and ripeness of bell peppers**

The goal is simple but meaningful: to make it easier for **people with color blindness** to tell apart bell pepper colors correctly and safely.

---

## About the Project

Yaek Zaab was developed as part of a group project.  
I worked as one of the **team members**, mainly focusing on:

- **Building the CNN model** completely from scratch — no pretrained model  
- **Tuning hyperparameters** to improve accuracy and stability  
- **Designing the UI** with **color-blind–friendly palettes**, ensuring that everyone can comfortably use it

In addition to the modeling and design, I also **collected and curated the dataset myself** by photographing bell peppers at different ripeness stages (green, yellow, red, and rotten) to ensure that the model learns from real-world examples with diverse lighting and angle conditions.

Our idea came from wanting to make technology **more inclusive**, even for tasks as simple as sorting vegetables.

You can try the live demo here:  
[View Live Website](https://g9tuesa.consolutechcloud.com/frontend.html)

---

## How It Works

Yaek Zaab uses a **Convolutional Neural Network (CNN)** to identify which stage the bell pepper is in:

| Class | Stage | Details |
|--------|--------|-------------|
| <img width="50" height="50" alt="pepper_green" src="https://github.com/user-attachments/assets/f8e4cbd8-0d09-46cb-bff3-adc418d82457" /> | Early | Mild flavor, less sweet |
| <img width="50" height="50" alt="pepper_yellow" src="https://github.com/user-attachments/assets/e44658ea-a1c3-41cd-933b-cc9df004e37e" /> | Medium | Balanced taste |
| <img width="50" height="50" alt="pepper_red" src="https://github.com/user-attachments/assets/8b98972c-9677-4274-a8aa-d9fc152156d3" /> | Ripe | Sweet and juicy |
| <img width="50" height="50" alt="pepper_rotten" src="https://github.com/user-attachments/assets/7314723e-0dfa-44b7-9ba0-d443eb278ebb" /> | Spoiled | Unsafe to eat |

The web application allows users to upload an image of a bell pepper to predict its class using the trained CNN model.

After the prediction, the system displays detailed information for that specific color category, including:

- **Ripeness level**

- **Recommended storage duration**

- **Taste profile**

- **Nutritional and health benefits**

Once the prediction is complete, the result is automatically saved and logged in the system. Users can then view all past predictions on the Summary page, which visualizes the data through:

- **Pie chart** showing the distribution of each bell pepper class

- **Bar chart** showing prediction counts and trends over time

- **Recent evaluation list** displaying the latest 4 predictions, including the date, time, and detected class

Additionally, the web also supports **real-time camera prediction**, allowing users to capture and analyze bell peppers instantly.

This feature is optimized for **mobile phones and tablets**, making the system more convenient and accessible anywhere.

---

## Tools & Technologies

| Purpose | Tools |
|----------|--------|
| Model Training | TensorFlow, Keras, NumPy, PIL |
| Backend | Flask (Python) |
| Frontend | HTML, CSS, JavaScript |
| Design | Figma -> color-blind–friendly UI |
| Development | Jupyter Notebook, VS Code |

**Figma Board (Full UI Overview):**  
[View the Complete Design on Figma](https://www.figma.com/design/wTdQE39Byvn5cnnEvIOzeM/AI_project?node-id=0-1&t=RbU8Y8UfuFNBT8wq-1)

---

## Model Development & Improvement Summary

### Model Architecture

The final model was built using Convolutional Neural Network (CNN) with the following structure:

| Layers | Details |
|----------|--------|
| Input | Shape = (img_height, img_width, 3) |
| Rescaling | Normalize pixel values (1./255) |
| Conv2D + MaxPooling | 32 filters, kernel size 3×3, activation = ReLU |
| Conv2D + MaxPooling | 64 filters, kernel size 3×3, activation = ReLU (increased from 32 -> 64 to enhance feature learning) |
| Flatten | Convert feature maps into a 1D vector |
| Dense | 128 neurons, activation = ReLU |
| Dropout | Rate = 0.4 (to prevent overfitting) |
| Dense | Output layer with softmax activation for classification |

### Hyperparameter Tuning
To improve performance and stability, several parameters and callbacks were fine-tuned:
- Optimizer: Adam
- Learning rate: 0.0001 (manually adjusted for smoother learning curve)
- Callbacks used:
    - EarlyStopping: stops training when val_accuracy plateaus
    - ReduceLROnPlateau: automatically reduces learning rate when val_loss stops improving
- Dropout rate: 0.4 to reduce overfitting

### Model Evaluation & Results
The improved model achieved strong performance on the validation set:

| Metric | Value |
|----------|--------|
| Accuracy (Validation) | 97.4% |
| Loss (Validation) | 0.1234 |
| Precision / Recall / F1-score | 0.97 |

Each class (green, red, yellow, rotten, unknown) achieved nearly perfect recall and precision, indicating that the model can distinguish between color and ripeness very well.

### Performance Visualization

### - Accuracy Trend

The training and validation accuracy curves both rise steadily before flattening near 97–98%, showing:

- The model learned features effectively.
- There’s no major overfitting since both curves stabilize closely together.

<img width="424" height="331" alt="Screenshot 2025-10-16 213600" src="https://github.com/user-attachments/assets/14894d05-afa8-4a51-a0fe-0564feef4e91" />

*(Interpretation: The orange line, validation accuracy follows the blue line closely, meaning the model generalizes well to unseen data.)*

### - Loss Trend

The training and validation loss curves drop smoothly and converge near zero.

- Early epochs show rapid improvement (loss decreases sharply).
- After about 10 epochs, both losses stabilize indicating convergence.

<img width="420" height="320" alt="Screenshot 2025-10-16 213541" src="https://github.com/user-attachments/assets/b9776b02-d04d-4703-8f1c-35ceb0866e5f" />

*(Interpretation: The model stops improving significantly after ~30 epochs, suggesting training is optimal.)*

### - Confusion Matrix

The confusion matrix shows that:
- Most predictions fall on the diagonal line (true = predicted).
- Misclassifications are minimal (e.g., a few rotten peppers misread as unknown).

Overall, the model differentiates between classes very accurately.

<img width="457" height="395" alt="Screenshot 2025-10-16 213618" src="https://github.com/user-attachments/assets/9a658cb2-7995-4056-85af-bdd2e763f7f2" />

---

## Conclusion

This improved CNN model successfully enhances learning ability while minimizing overfitting. Through careful tuning (filters, dropout, learning rate, callbacks), the model reached ~97% accuracy with stable precision and recall across all classes making it well-suited for real-world bell pepper classification tasks in the Yaek Zaab web application.

---

## Model Download

The trained model file is available here (if you’d like to try it out):
[Go to my Google Drive](https://drive.google.com/drive/folders/1iJaBjYEXIm2ksW2CAENIUSUvWpU5tB_H?usp=drive_link)

---

## About Me

**Phawadon Nuresaard**

Bachelor of Engineering, IoT Systems & Information Engineering, KMITL

- **Role**: Model Developer & UI Designer
- Also responsible for **dataset collection** (manual photo capturing & preprocessing)
- **Interested in**: AI/ML and Data Analytics

> **Note:**  
> This project was a group collaboration.  
> I was mainly responsible for the **AI model development**, **dataset collection**, and **color-blind–friendly UI design**.  
> The web development and deployment parts were handled by my teammate.  
> Since my focus was on the model and AI aspects, this repository includes only the model, training notebook, and related analysis.

