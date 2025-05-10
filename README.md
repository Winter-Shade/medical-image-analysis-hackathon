## Group 3 - Medical Image Analysis Hackathon

### Project Structure

├── evaluate.py (Evaluation script : accepts CLI args for test CSVs)     
├── model.py  (CustomResNet18 model definition)     
├── create_test_csv.py   (Create x_test.csv and y_test.csv (with pre-processing) in split_data folder)    
├── flatten_one_hot.py  (If using one-hot encoded labels, use this first to convert to single column)     
├── split_data/      
│ ├── x_test.csv  (Flattened 28x28 grayscale test images)     
│ └── y_test.csv  (Corresponding labels)      
├── best_custom_resnet18.pth (Trained model weights)      
├── notebooks (Jupyter Notebooks --- models trained)      
├── requirements.txt       
└── README.md          

### How to setup

Set up a virtual environment (terminal): 
-    python -m venv venv
-   source venv/bin/activate  # On Windows: venv\Scripts\activate

Install dependencies:
-    pip install -r requirements.txt

To create 'x_test.csv' and 'y_test.csv' (in split_data folder), Run:
-    python create_test_csv.py

Evaluation: 
-    python evaluate.py split_data/x_test.csv split_data/y_test.csv

*(with pre-processing : no need if using create_test_csv.py script)
-    python evaluate_model.py split_data/x_test.csv split_data/y_test.csv --preprocess

### Combined Dataset: (Columns expected in this order)
Class Index : (0-3) : OCTMNIST
Class Index : (4-5) : BreastMNIST
Class Index: (6-7) : PneumoniaMNIST
Class Index: (8-12): RetinaMNIST

### flatten_one_hot.py
Usage: python flatten_one_hot.py <x_test.csv path> <y_test.csv path>
Convert One-Hot Encoded Labels to Single-Column for evaluation

### PPT
Link: https://www.canva.com/design/DAGj6tbWeYI/PL9gFy6Q-vuxS6wGNn-FDw/edit?utm_content=DAGj6tbWeYI&utm_campaign=designshare&utm_medium=link2&utm_source=sharebutton

Team Members: 
22bec006	Aman Kumar Prajapati
22bec026	Mohammad Nabeel Hasan
22bcs124	Srijan Shukla

