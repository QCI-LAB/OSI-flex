# OSI-flex

**OSI-flex** (Flexible Framework for Optimization-Based Shearing Interferometry) is an open-source framework for **joint phase and shear estimation** in lateral shearing interferometry.  

---

## Features
- **Optimization-based approach** for robust, theory-grounded phase retrieval in lateral shearing interferometry  
- **Shear fine-tuning** for improved quality of phase retrieval  
- **Flexible framework** that supports arbitrary numbers, magnitudes, and directions of shearing  
- **TV and nonnegativity/nonpositivity regularization** to improve the conditioning of the inverse problem  
- **Automatic differentiation** for fast computation and easy adaptation to other configurations  
- Built with **reproducibility and transparency** in mind  

---

## Project Structure
- `src/` — main source code for the project  
- `data/` — input datasets (not tracked in Git)  
- `configs/` — configuration files for running experiments  
- `results/` — generated outputs and results (not tracked in Git)  

## How to Run
To run the program, execute the main script in the project root.

---

## Installation
Clone the repository and install dependencies:
```bash
git clone https://github.com/julianna-winnik/osi-flex.git
cd osi-flex
pip install -r requirements.txt
