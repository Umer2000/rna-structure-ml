## 🧬 RNA Structure Prediction Tool (ML + Bioinformatics)




## 📌 Overview

This project presents a machine learning-based RNA structure prediction tool that combines deep learning with bioinformatics workflows.
It takes RNA sequences in FASTA format and predicts 3D structural coordinates using a trained neural model.
The project evolves from a baseline heuristic approach into a fully functional CLI tool, integrating:

•	Deep learning (PyTorch)

•	Sequence modeling (BiLSTM + Attention)

•	Bioinformatics pipelines (Biopython)

•	Structured output generation (CSV)


## 🎯 Key Features

•	🧬 FASTA input support (via Biopython)

•	🤖 Deep learning model for RNA coordinate prediction

•	🧠 Attention-based sequence modeling

•	📊 Structured output (CSV with coordinates)

•	🖥️ Command-line interface (CLI)

•	⚙️ Modular and extensible codebase


## 🧠 Model Architecture

The model consists of:

•	Embedding Layer → encodes nucleotides (A, C, G, U)


•	BiLSTM Encoder → captures sequence dependencies


•	Multi-head Self-Attention → models global interactions


•	Fully Connected Layer → predicts 3D coordinates


## 📂 Project Structure

rna-structure-ml/

├── data/

├── sample.fasta

├── model.pth

├── src/

├── dataset.py

├── model.py

├── train.py

├── predict.py

├── visualize.py

├── README.md


## 🚀 Installation

1. Clone repository
  
 git clone
 <https://github.com/Umer2000/rna-structure-ml>
cd rna-structure-ml

2. Create virtual environment
    
python -m venv venv

.\venv\Scripts\Activate

3. Install dependencies
   
pip install torch pandas biopython matplotlib


## ▶️ Usage

🔹 Train the model

python src/train.py

This will:

•	train the model

•	save weights as model.pth


## 🔹 Run prediction (CLI)

python src/predict.py --input sample.fasta --output results.csv

##📄 Output format

sequence_id, position, residue, x, y, z

rna_1,1, A,0.12,0.55,0.91

rna_1,2, C,0.34,0.22,0.88

...

## 🧪 Related Work & Background
This project builds upon my earlier work in RNA structure modeling during the Stanford RNA 3D Folding Challenge.
In that work, I:

•	developed a baseline using distance heuristics and MDS
•	explored sequence-to-structure modeling approaches

•	worked with real RNA sequence datasets

This project extends that work into a trainable deep learning pipeline with bioinformatics integration.

🔗 Kaggle Notebook:
https://www.kaggle.com/code/umeranis/rna-baseline-v1



## 🔍 Interpretability

The project includes attention visualization:

•	Highlights important sequence regions

•	Helps analyze model behavior

•	Reveals limitations under weak supervision


## 📊 Key Findings

•	Attention mechanisms may collapse under weak supervision

•	Proper masking and padding are critical

•	Sequence models can learn structural embeddings from raw RNA


## ⚠️ Limitations

•	Uses pseudo-labels (not experimental structures)

•	Limited biological accuracy

•	Attention interpretability depends on data quality


## 🚀 Future Work

•	Integrate real RNA structure datasets (PDB)

•	Add structure-aware loss functions

•	Extend to graph-based models (GNNs)

•	Export predictions in PDB format


## 🛠️ Tech Stack

•	Python

•	PyTorch

•	Biopython

•	Pandas / NumPy

•	Matplotlib


## 🧠 Learning Outcomes

•	Built end-to-end ML pipeline

•	Integrated ML with bioinformatics workflows

•	Implemented attention and self-attention

•	Diagnosed model behavior and limitations


## 📌 Conclusion

This project demonstrates how deep learning models can be integrated into practical bioinformatics tools, while also highlighting the importance of data quality and model interpretability.


## 🤝 Acknowledgment

Inspired by RNA structure prediction challenges and the Stanford RNA Folding competition.

