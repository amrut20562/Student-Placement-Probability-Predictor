import pandas as pd, numpy as np, matplotlib.pyplot as plt, pickle, os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

SKILLS_MAP = {'Low': 0, 'Medium': 1, 'High': 2}
FEATURES = ['CGPA', 'Skills', 'Internships']

class PlacementPredictor:
    def __init__(self, csv='placement.csv', model_file='model.pkl'):
        self.csv, self.model_file, self.model, self.data = csv, model_file, None, None

    def load_data(self):
        try:
            self.data = pd.read_csv(self.csv)
            print(f"✓ {len(self.data)} records | Columns: {list(self.data.columns)}")
            print(self.data.head(10)); print(self.data.info()); print(self.data['Placed'].value_counts())
            return True
        except Exception as e: print(f"✗ {e}"); return False

    def preprocess(self):
        d = self.data.copy()
        d['Skills'] = d['Skills'].map(SKILLS_MAP)
        X, y = d[FEATURES], d['Placed']
        print(f"✓ Features: {X.shape} | Stats:\n{X.describe()}")
        return X, y

    def train(self, X, y):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        self.model = LogisticRegression(random_state=42, max_iter=1000)
        self.model.fit(self.X_train, self.y_train)
        print(f"✓ Trained | Train: {len(self.X_train)} | Test: {len(self.X_test)}")
        return True

    def evaluate(self):
        y_pred = self.model.predict(self.X_test)
        acc = accuracy_score(self.y_test, y_pred)
        cm = confusion_matrix(self.y_test, y_pred)
        print(f"✓ Accuracy: {acc*100:.2f}%\nConfusion Matrix: TN={cm[0][0]} FP={cm[0][1]} FN={cm[1][0]} TP={cm[1][1]}")
        print(classification_report(self.y_test, y_pred, target_names=['Not Placed', 'Placed']))
        for f, c in zip(FEATURES, self.model.coef_[0]): print(f"  {f}: {c:.4f}")
        return acc, cm

    def save(self):
        with open(self.model_file, 'wb') as f: pickle.dump(self.model, f)
        print(f"✓ Model saved to '{self.model_file}'")

    def load(self):
        if not os.path.exists(self.model_file): return False
        with open(self.model_file, 'rb') as f: self.model = pickle.load(f)
        print(f"✓ Model loaded"); return True

    def predict(self, cgpa, skills, internships):
        return self.model.predict_proba([[cgpa, SKILLS_MAP[skills], internships]])[0][1] * 100

    def visualize(self):
        os.makedirs('graphs', exist_ok=True)
        placed, not_placed = self.data[self.data['Placed']==1], self.data[self.data['Placed']==0]
        
        plots = [
            ('cgpa_vs_placement', lambda: (
                plt.scatter(placed['CGPA'], [1]*len(placed), label='Placed', alpha=0.6, color='green'),
                plt.scatter(not_placed['CGPA'], [0]*len(not_placed), label='Not Placed', alpha=0.6, color='red'),
                plt.xlabel('CGPA'), plt.ylabel('Placement'), plt.title('CGPA vs Placement'),
                plt.yticks([0,1],['Not Placed','Placed']), plt.legend(), plt.grid(alpha=0.3))),
            ('skills_distribution', lambda: (
                self.data['Skills'].value_counts().plot(kind='bar', color=['#FF6B6B','#4ECDC4','#45B7D1']),
                plt.title('Skills Distribution'), plt.xlabel('Skill Level'), plt.xticks(rotation=45))),
            ('feature_importance', lambda: [
                plt.barh(FEATURES, np.abs(self.model.coef_[0]), color=['#FF6B6B','#4ECDC4','#45B7D1']),
                plt.title('Feature Importance'), plt.xlabel('Importance Score'),
                [plt.text(v, i, f' {v:.4f}', va='center') for i,v in enumerate(np.abs(self.model.coef_[0]))]]),
            ('placement_distribution', lambda: (
                plt.pie(self.data['Placed'].value_counts().values, labels=['Placed','Not Placed'],
                        autopct='%1.1f%%', colors=['green','red'], startangle=90),
                plt.title('Placement Distribution'))),
            ('internships_vs_placement', lambda: [
                (g := self.data.groupby('Internships')['Placed'].agg(['sum','count']).assign(
                    rate=lambda x: x['sum']/x['count']*100)),
                plt.bar(g.index, g['rate'], color='#45B7D1', edgecolor='black'),
                plt.title('Placement Rate vs Internships'), plt.xlabel('Internships'), plt.ylabel('Rate (%)'),
                [plt.text(i, v+2, f'{v:.1f}%', ha='center') for i,v in zip(g.index, g['rate'])]])
        ]
        
        for name, fn in plots:
            plt.figure(figsize=(10,6)); fn(); plt.tight_layout()
            plt.savefig(f'graphs/{name}.png', dpi=300, bbox_inches='tight'); plt.close()
            print(f"✓ {name}.png")

    def run(self):
        if not self.load_data(): return
        X, y = self.preprocess()
        self.train(X, y); self.evaluate(); self.save(); self.visualize()
        print("✅ Pipeline complete!")


def interactive(predictor):
    while True:
        try:
            cgpa = input("\nCGPA (6-10) or 'quit': ").strip()
            if cgpa.lower() == 'quit': break
            cgpa = float(cgpa)
            assert 6.0 <= cgpa <= 10.0, "CGPA must be 6–10"
            
            skills = input("Skills (Low/Medium/High): ").strip().capitalize()
            assert skills in SKILLS_MAP, "Must be Low, Medium, or High"
            
            n = int(input("Internships (0-5): ").strip())
            assert 0 <= n <= 5, "Must be 0–5"
            
            p = predictor.predict(cgpa, skills, n)
            label = "✅ VERY LIKELY" if p>75 else "✓ LIKELY" if p>50 else "⚠ MODERATE" if p>25 else "❌ LOW CHANCE"
            print(f"\n🎓 Placement Probability: {p:.2f}% — {label}")
        
        except AssertionError as e: print(f"✗ {e}")
        except ValueError: print("✗ Invalid input")


def main():
    p = PlacementPredictor()
    if os.path.exists('model.pkl') and p.load():
        p.load_data(); p.preprocess()
    else:
        p.run()
    interactive(p)
    print("✅ Done!")

if __name__ == "__main__": main()