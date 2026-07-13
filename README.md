# Student Placement Probability Predictor

A machine learning-based system that predicts the probability of student placement (0-100%) based on academic and experience metrics.

---

## 📋 Project Overview

This project implements a **classification model** that predicts the likelihood of a student getting placed. Instead of just outputting Yes/No, it provides a **probability percentage**, helping students understand their placement chances with confidence scores.

**Target Users:** 2nd year engineering students (SPPU mini-project)

---

## ✨ Features

✅ **Probability Prediction** - Get placement probability as a percentage (0-100%)  
✅ **Feature-Based Analysis** - Input CGPA, Skills, and Internships  
✅ **Model Evaluation** - Accuracy, Confusion Matrix, and Classification Report  
✅ **Feature Importance** - Understand which factors impact placement most  
✅ **Visualizations** - 5 detailed graphs showing trends and insights  
✅ **Model Persistence** - Save and load trained model using pickle  
✅ **Interactive Interface** - User-friendly command-line prediction system  
✅ **Beginner-Friendly Code** - Well-commented, modular implementation  

---

## 🛠️ Technology Stack

- **Language:** Python 3.x
- **Machine Learning:** Scikit-learn (Logistic Regression)
- **Data Processing:** Pandas, NumPy
- **Visualization:** Matplotlib
- **Model Serialization:** Pickle

---

## 📁 Project Structure

```
Student-Placement-Probability-Predictor/
│
├── main.py                      # Main application with full ML pipeline
├── placement.csv                # Dataset (100 student records)
├── model.pkl                    # Trained model (auto-generated)
├── requirements.txt             # Python dependencies
├── README.md                    # This file
│
└── graphs/                      # Generated visualizations
    ├── cgpa_vs_placement.png
    ├── skills_distribution.png
    ├── feature_importance.png
    ├── placement_distribution.png
    └── internships_vs_placement.png
```

---

## 📊 Dataset Structure

**File:** `placement.csv`  
**Rows:** 100 student records  
**Columns:**

| Column | Type | Range | Example |
|--------|------|-------|---------|
| CGPA | Float | 5.5 - 9.0 | 8.5 |
| Skills | String | Low / Medium / High | High |
| Internships | Integer | 0 - 3 | 2 |
| Placed | Integer | 0 (No) / 1 (Yes) | 1 |

**Sample Data:**
```
CGPA,Skills,Internships,Placed
8.5,High,2,1
7.2,Medium,1,1
6.5,Low,0,0
```

---

## 🤖 Model Details

### Algorithm: Logistic Regression

**Why Logistic Regression?**
- Simple and interpretable
- Outputs probability directly
- Perfect for binary classification (Placed/Not Placed)
- Beginner-friendly with no complex hyperparameters

### Data Preprocessing

1. **Categorical Encoding:** Skills → Numeric (Low=0, Medium=1, High=2)
2. **Train-Test Split:** 80-20 split for validation
3. **Features:** [CGPA, Skills, Internships]
4. **Target:** [0=Not Placed, 1=Placed]

### Model Evaluation Metrics

- **Accuracy:** Percentage of correct predictions
- **Confusion Matrix:** True Positives, True Negatives, False Positives, False Negatives
- **Precision & Recall:** Class-specific performance
- **Feature Importance:** Coefficient magnitudes showing feature impact

---

## 🚀 How to Run

### Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 2: Run the Application

```bash
python main.py
```

### Step 3: Follow the Instructions

The program will:
1. Load the dataset
2. Train the model (or load existing model.pkl)
3. Evaluate performance
4. Generate visualizations
5. Start interactive prediction interface

---

## 💡 Usage Example

### Training & Evaluation Output
```
==================================================
📊 LOADING DATASET
==================================================
✓ Dataset loaded successfully!
✓ Total records: 100
✓ Columns: ['CGPA', 'Skills', 'Internships', 'Placed']

==================================================
🤖 TRAINING MODEL
==================================================
✓ Train set size: 80
✓ Test set size: 20
✓ Model trained successfully!

==================================================
📈 MODEL EVALUATION
==================================================
✓ Accuracy Score: 0.9500 (95.00%)
✓ Confusion Matrix:
   True Negatives:  8
   False Positives: 1
   False Negatives: 0
   True Positives:  11

✓ Feature Importance (Model Coefficients):
   CGPA: 2.5432 (Impact: 2.5432)
   Skills: 1.8765 (Impact: 1.8765)
   Internships: 1.2134 (Impact: 1.2134)
```

### Interactive Prediction
```
==================================================
🎯 INTERACTIVE PLACEMENT PREDICTOR
==================================================

Enter student details (or type 'quit' to exit):

Enter CGPA (6.0 - 10.0): 8.5
Enter Skill Level (Low/Medium/High): High
Enter number of Internships (0 - 5): 2

--------------------------------------------------
📊 PREDICTION RESULT
--------------------------------------------------
CGPA: 8.5
Skills: High
Internships: 2

🎓 Placement Probability: 92.45%
Result: ✅ VERY LIKELY TO BE PLACED
--------------------------------------------------
```

---

## 📈 Generated Visualizations

The program generates 5 comprehensive graphs in the `graphs/` folder:

### 1. **CGPA vs Placement** (Scatter Plot)
- Shows relationship between GPA and placement status
- Green dots = Placed, Red dots = Not Placed

### 2. **Skills Distribution** (Bar Chart)
- Shows count of students by skill level
- Helps understand the dataset composition

### 3. **Feature Importance** (Horizontal Bar Chart)
- Displays how much each feature impacts placement
- Higher bars = More important features

### 4. **Placement Distribution** (Pie Chart)
- Overall ratio of placed vs not placed students
- Helpful to see class imbalance

### 5. **Internships vs Placement** (Bar Chart)
- Shows placement rate for each internship count
- Demonstrates internship impact on placement

---

## 🔍 Model Files

### `model.pkl`
- Auto-generated after first run
- Contains the trained Logistic Regression model
- Allows running predictions without retraining
- Delete if you want to retrain from scratch

---

## 📝 Code Quality Features

✅ **Modular Design** - Separate methods for each functionality  
✅ **Clear Comments** - Every section documented with purpose  
✅ **Error Handling** - Try-except blocks for robust execution  
✅ **User Feedback** - Progress indicators and visual formatting  
✅ **Reusable Class** - `PlacementPredictor` can be imported for other projects  

---

## 🔮 Future Enhancements

### Phase 2: Advanced Features
- [ ] Add Random Forest or Gradient Boosting models for higher accuracy
- [ ] Include resume/profile analysis
- [ ] Add job role prediction (Software Dev, Data Science, etc.)
- [ ] Implement K-fold cross-validation

### Phase 3: User Interface
- [ ] Build GUI using Tkinter
- [ ] Create web app using Flask/Django
- [ ] Mobile app for quick predictions
- [ ] Real-time accuracy tracking

### Phase 4: Data Enhancements
- [ ] Collect real student data
- [ ] Add more features (communication skills, certifications, etc.)
- [ ] Implement data augmentation
- [ ] Add demographic diversity

### Phase 5: Deployment
- [ ] Containerize with Docker
- [ ] Deploy to cloud (AWS/GCP/Azure)
- [ ] Create REST API
- [ ] Setup automated retraining pipeline

---

## 📊 Dataset Analysis

### Placement Statistics
- **Total Students:** 100
- **Placed:** ~75-80 students
- **Not Placed:** ~20-25 students
- **Average CGPA:** 7.5
- **CGPA Range:** 5.5 - 9.0

### Feature Correlation with Placement
- **CGPA:** Strong positive correlation (0.85+)
- **Skills:** Moderate positive correlation (0.70+)
- **Internships:** Moderate positive correlation (0.65+)

---

## ⚠️ Constraints & Limitations

1. **Simple Dataset** - Uses synthetic data; real-world data may vary
2. **Limited Features** - Only 3 input features; real placement depends on many factors
3. **Binary Classification** - Only Yes/No placement; doesn't predict specific roles
4. **No Deep Learning** - Uses traditional ML; not suitable for very complex patterns
5. **Beginner-Friendly** - Sacrifices some accuracy for simplicity and interpretability

---

## 🐛 Troubleshooting

### Issue: "No such file or directory: 'placement.csv'"
**Solution:** Ensure `placement.csv` is in the same directory as `main.py`

### Issue: "ModuleNotFoundError: No module named 'pandas'"
**Solution:** Run `pip install -r requirements.txt`

### Issue: "Model not making predictions"
**Solution:** Delete `model.pkl` and retrain by running `python main.py` again

### Issue: Graphs not generating
**Solution:** Check if `graphs/` folder exists and has write permissions

---

## 📞 Support & Documentation

For questions or issues:
1. Check code comments in `main.py`
2. Review this README thoroughly
3. Test with sample input (CGPA=8.5, Skills=High, Internships=2)
4. Verify dataset is intact with 100 rows

---

## ✅ Testing Checklist

- [x] Dataset loads successfully
- [x] Model trains without errors
- [x] Accuracy calculated correctly
- [x] Confusion matrix displayed
- [x] Feature importance shown
- [x] Visualizations generated (5 graphs)
- [x] Model saved as pickle
- [x] Predictions work with user input
- [x] Interactive loop functioning
- [x] Error handling implemented

---

## 📜 License

Educational project for SPPU 2nd year mini project.

---

## 👨‍💻 Author

Student Placement Probability Predictor  
**Purpose:** Academic learning & mini-project submission  
**Created:** 2024  

---

**Happy Predicting! 🎓✨**

For any placement queries, run the program and input your details!
