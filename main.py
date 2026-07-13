"""
Student Placement Probability Predictor
Predicts the probability of a student getting placed based on CGPA, Skills, and Internships
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import pickle
import os


class PlacementPredictor:
    """Main class for placement prediction model"""
    
    def __init__(self, csv_file='placement.csv', model_file='model.pkl'):
        """
        Initialize the predictor
        
        Args:
            csv_file: Path to the CSV dataset
            model_file: Path to save/load the trained model
        """
        self.csv_file = csv_file
        self.model_file = model_file
        self.model = None
        self.encoder = LabelEncoder()
        self.data = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.feature_names = ['CGPA', 'Skills', 'Internships']
        
    def load_data(self):
        """Load and display basic information about the dataset"""
        print("\n" + "="*50)
        print("📊 LOADING DATASET")
        print("="*50)
        
        try:
            self.data = pd.read_csv(self.csv_file)
            print(f"✓ Dataset loaded successfully!")
            print(f"✓ Total records: {len(self.data)}")
            print(f"✓ Columns: {list(self.data.columns)}")
            print(f"\nDataset Preview:")
            print(self.data.head(10))
            print(f"\nDataset Info:")
            print(self.data.info())
            print(f"\nPlacement Distribution:")
            print(self.data['Placed'].value_counts())
            return True
        except Exception as e:
            print(f"✗ Error loading data: {e}")
            return False
    
    def preprocess_data(self):
        """Preprocess data: encode categorical variables"""
        print("\n" + "="*50)
        print("🔄 PREPROCESSING DATA")
        print("="*50)
        
        try:
            # Create a copy to avoid modifying original
            data = self.data.copy()
            
            # Encode Skills column (Low=0, Medium=1, High=2)
            skills_mapping = {'Low': 0, 'Medium': 1, 'High': 2}
            data['Skills'] = data['Skills'].map(skills_mapping)
            
            # Separate features and target
            X = data[['CGPA', 'Skills', 'Internships']]
            y = data['Placed']
            
            print("✓ Data preprocessing completed!")
            print(f"✓ Features shape: {X.shape}")
            print(f"✓ Target shape: {y.shape}")
            print(f"\nFeature Statistics:")
            print(X.describe())
            
            return X, y
        except Exception as e:
            print(f"✗ Error in preprocessing: {e}")
            return None, None
    
    def train_model(self, X, y):
        """Train Logistic Regression model"""
        print("\n" + "="*50)
        print("🤖 TRAINING MODEL")
        print("="*50)
        
        try:
            # Split data: 80% training, 20% testing
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            print(f"✓ Train set size: {len(self.X_train)}")
            print(f"✓ Test set size: {len(self.X_test)}")
            
            # Train Logistic Regression model
            self.model = LogisticRegression(random_state=42, max_iter=1000)
            self.model.fit(self.X_train, self.y_train)
            
            print("✓ Model trained successfully!")
            
            return True
        except Exception as e:
            print(f"✗ Error training model: {e}")
            return False
    
    def evaluate_model(self):
        """Evaluate model performance"""
        print("\n" + "="*50)
        print("📈 MODEL EVALUATION")
        print("="*50)
        
        try:
            # Predictions on test set
            y_pred = self.model.predict(self.X_test)
            
            # Accuracy
            accuracy = accuracy_score(self.y_test, y_pred)
            print(f"\n✓ Accuracy Score: {accuracy:.4f} ({accuracy*100:.2f}%)")
            
            # Confusion Matrix
            cm = confusion_matrix(self.y_test, y_pred)
            print(f"\n✓ Confusion Matrix:")
            print(f"   True Negatives:  {cm[0][0]}")
            print(f"   False Positives: {cm[0][1]}")
            print(f"   False Negatives: {cm[1][0]}")
            print(f"   True Positives:  {cm[1][1]}")
            
            # Classification Report
            print(f"\n✓ Classification Report:")
            print(classification_report(self.y_test, y_pred, target_names=['Not Placed', 'Placed']))
            
            # Feature Importance (Logistic Regression coefficients)
            print(f"\n✓ Feature Importance (Model Coefficients):")
            for feature, coef in zip(self.feature_names, self.model.coef_[0]):
                importance = abs(coef)
                print(f"   {feature}: {coef:.4f} (Impact: {importance:.4f})")
            
            return accuracy, cm
        except Exception as e:
            print(f"✗ Error evaluating model: {e}")
            return None, None
    
    def save_model(self):
        """Save trained model to pickle file"""
        try:
            with open(self.model_file, 'wb') as f:
                pickle.dump(self.model, f)
            print(f"\n✓ Model saved to '{self.model_file}'")
            return True
        except Exception as e:
            print(f"\n✗ Error saving model: {e}")
            return False
    
    def load_model(self):
        """Load trained model from pickle file"""
        try:
            if not os.path.exists(self.model_file):
                print(f"⚠ Model file '{self.model_file}' not found!")
                return False
            
            with open(self.model_file, 'rb') as f:
                self.model = pickle.load(f)
            print(f"✓ Model loaded from '{self.model_file}'")
            return True
        except Exception as e:
            print(f"✗ Error loading model: {e}")
            return False
    
    def predict_placement(self, cgpa, skills, internships):
        """
        Predict placement probability for a student
        
        Args:
            cgpa: Student's CGPA (float)
            skills: Skill level (str: 'Low', 'Medium', 'High')
            internships: Number of internships (int)
        
        Returns:
            Probability of placement (0-100%)
        """
        try:
            # Encode skills
            skills_mapping = {'Low': 0, 'Medium': 1, 'High': 2}
            skills_encoded = skills_mapping[skills]
            
            # Create feature array
            features = np.array([[cgpa, skills_encoded, internships]])
            
            # Get probability
            probability = self.model.predict_proba(features)[0][1] * 100
            
            return probability
        except Exception as e:
            print(f"✗ Error making prediction: {e}")
            return None
    
    def generate_visualizations(self):
        """Generate all visualizations and save to graphs folder"""
        print("\n" + "="*50)
        print("📊 GENERATING VISUALIZATIONS")
        print("="*50)
        
        # Create graphs directory if it doesn't exist
        os.makedirs('graphs', exist_ok=True)
        
        try:
            # 1. CGPA vs Placement Scatter Plot
            plt.figure(figsize=(10, 6))
            placed = self.data[self.data['Placed'] == 1]
            not_placed = self.data[self.data['Placed'] == 0]
            
            plt.scatter(placed['CGPA'], [1]*len(placed), label='Placed', alpha=0.6, s=100, color='green')
            plt.scatter(not_placed['CGPA'], [0]*len(not_placed), label='Not Placed', alpha=0.6, s=100, color='red')
            
            plt.xlabel('CGPA', fontsize=12, fontweight='bold')
            plt.ylabel('Placement Status', fontsize=12, fontweight='bold')
            plt.title('CGPA vs Placement', fontsize=14, fontweight='bold')
            plt.yticks([0, 1], ['Not Placed', 'Placed'])
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig('graphs/cgpa_vs_placement.png', dpi=300, bbox_inches='tight')
            plt.close()
            print("✓ Generated: cgpa_vs_placement.png")
            
            # 2. Skills Distribution
            plt.figure(figsize=(10, 6))
            skills_count = self.data['Skills'].value_counts()
            colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
            skills_count.plot(kind='bar', color=colors[:len(skills_count)])
            
            plt.xlabel('Skill Level', fontsize=12, fontweight='bold')
            plt.ylabel('Count', fontsize=12, fontweight='bold')
            plt.title('Skills Distribution Among Students', fontsize=14, fontweight='bold')
            plt.xticks(rotation=45)
            plt.grid(True, alpha=0.3, axis='y')
            plt.tight_layout()
            plt.savefig('graphs/skills_distribution.png', dpi=300, bbox_inches='tight')
            plt.close()
            print("✓ Generated: skills_distribution.png")
            
            # 3. Feature Importance
            plt.figure(figsize=(10, 6))
            coefficients = self.model.coef_[0]
            importance = np.abs(coefficients)
            
            bars = plt.barh(self.feature_names, importance, color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
            plt.xlabel('Importance Score', fontsize=12, fontweight='bold')
            plt.title('Feature Importance in Placement Prediction', fontsize=14, fontweight='bold')
            plt.grid(True, alpha=0.3, axis='x')
            
            # Add value labels on bars
            for i, (bar, val) in enumerate(zip(bars, importance)):
                plt.text(val, i, f' {val:.4f}', va='center', fontweight='bold')
            
            plt.tight_layout()
            plt.savefig('graphs/feature_importance.png', dpi=300, bbox_inches='tight')
            plt.close()
            print("✓ Generated: feature_importance.png")
            
            # 4. Placement Distribution
            plt.figure(figsize=(10, 6))
            placement_count = self.data['Placed'].value_counts()
            labels = ['Placed', 'Not Placed']
            colors_pie = ['green', 'red']
            
            plt.pie(placement_count.values, labels=labels, autopct='%1.1f%%', 
                   colors=colors_pie, startangle=90, textprops={'fontsize': 12, 'fontweight': 'bold'})
            plt.title('Overall Placement Distribution', fontsize=14, fontweight='bold')
            plt.tight_layout()
            plt.savefig('graphs/placement_distribution.png', dpi=300, bbox_inches='tight')
            plt.close()
            print("✓ Generated: placement_distribution.png")
            
            # 5. Internships vs Placement
            plt.figure(figsize=(10, 6))
            internship_placement = self.data.groupby('Internships')['Placed'].agg(['sum', 'count'])
            internship_placement['placement_rate'] = (internship_placement['sum'] / internship_placement['count'] * 100)
            
            plt.bar(internship_placement.index, internship_placement['placement_rate'], color='#45B7D1', edgecolor='black', linewidth=1.5)
            plt.xlabel('Number of Internships', fontsize=12, fontweight='bold')
            plt.ylabel('Placement Rate (%)', fontsize=12, fontweight='bold')
            plt.title('Placement Rate vs Number of Internships', fontsize=14, fontweight='bold')
            plt.ylim(0, 110)
            plt.grid(True, alpha=0.3, axis='y')
            
            # Add percentage labels on bars
            for i, (idx, val) in enumerate(zip(internship_placement.index, internship_placement['placement_rate'])):
                plt.text(idx, val + 2, f'{val:.1f}%', ha='center', fontweight='bold')
            
            plt.tight_layout()
            plt.savefig('graphs/internships_vs_placement.png', dpi=300, bbox_inches='tight')
            plt.close()
            print("✓ Generated: internships_vs_placement.png")
            
            print("\n✓ All visualizations saved to 'graphs/' folder!")
            
        except Exception as e:
            print(f"✗ Error generating visualizations: {e}")
    
    def run_full_pipeline(self):
        """Execute the complete ML pipeline"""
        print("\n")
        print("╔" + "="*48 + "╗")
        print("║" + " STUDENT PLACEMENT PROBABILITY PREDICTOR ".center(48) + "║")
        print("╚" + "="*48 + "╝")
        
        # Step 1: Load data
        if not self.load_data():
            return False
        
        # Step 2: Preprocess data
        X, y = self.preprocess_data()
        if X is None:
            return False
        
        # Step 3: Train model
        if not self.train_model(X, y):
            return False
        
        # Step 4: Evaluate model
        self.evaluate_model()
        
        # Step 5: Save model
        self.save_model()
        
        # Step 6: Generate visualizations
        self.generate_visualizations()
        
        print("\n" + "="*50)
        print("✅ PIPELINE COMPLETED SUCCESSFULLY!")
        print("="*50)
        
        return True


def interactive_prediction(predictor):
    """Interactive user interface for making predictions"""
    print("\n" + "="*50)
    print("🎯 INTERACTIVE PLACEMENT PREDICTOR")
    print("="*50)
    
    while True:
        try:
            print("\nEnter student details (or type 'quit' to exit):")
            
            cgpa_input = input("\nEnter CGPA (6.0 - 10.0): ").strip()
            if cgpa_input.lower() == 'quit':
                break
            
            cgpa = float(cgpa_input)
            if cgpa < 6.0 or cgpa > 10.0:
                print("✗ CGPA must be between 6.0 and 10.0")
                continue
            
            skills = input("Enter Skill Level (Low/Medium/High): ").strip().capitalize()
            if skills not in ['Low', 'Medium', 'High']:
                print("✗ Skill level must be Low, Medium, or High")
                continue
            
            internships = int(input("Enter number of Internships (0 - 5): ").strip())
            if internships < 0 or internships > 5:
                print("✗ Internships must be between 0 and 5")
                continue
            
            # Make prediction
            probability = predictor.predict_placement(cgpa, skills, internships)
            
            if probability is not None:
                print("\n" + "-"*50)
                print("📊 PREDICTION RESULT")
                print("-"*50)
                print(f"CGPA: {cgpa}")
                print(f"Skills: {skills}")
                print(f"Internships: {internships}")
                print(f"\n🎓 Placement Probability: {probability:.2f}%")
                
                if probability > 75:
                    result = "✅ VERY LIKELY TO BE PLACED"
                elif probability > 50:
                    result = "✓ LIKELY TO BE PLACED"
                elif probability > 25:
                    result = "⚠ MODERATE CHANCE"
                else:
                    result = "❌ LOW CHANCE OF PLACEMENT"
                
                print(f"Result: {result}")
                print("-"*50)
        
        except ValueError:
            print("✗ Invalid input! Please enter valid numbers.")
        except Exception as e:
            print(f"✗ Error: {e}")


def main():
    """Main function"""
    # Initialize predictor
    predictor = PlacementPredictor(csv_file='placement.csv', model_file='model.pkl')
    
    # Check if model already exists
    if os.path.exists('model.pkl'):
        print("\n📁 Found existing model. Loading...")
        if predictor.load_model():
            # Load data for feature names
            predictor.load_data()
            X, _ = predictor.preprocess_data()
        else:
            print("Falling back to training new model...")
            predictor.run_full_pipeline()
    else:
        # Train new model
        predictor.run_full_pipeline()
    
    # Start interactive prediction
    interactive_prediction(predictor)
    
    print("\n✅ Thank you for using the Placement Predictor!")


if __name__ == "__main__":
    main()
