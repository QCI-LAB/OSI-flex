# OSI-flex

**OSI-flex** (Flexible Framework for Optimization-Based Shearing Interferometry) is an open-source framework for **joint phase and shear estimation** in lateral shearing interferometry.  

---

## Reference
Winnik, J., Suski, D., Heto, M., Lenarcik, M., Ziemczonok, M., Kołodziejczyk, M., Trusiak, M., & Zdańkowski, P. (2026). OSI-flex: Optimization-based shearing interferometry for joint phase and shear estimation using a flexible open-source framework. Journal of Physics: Photonics, 8, 015036. https://doi.org/10.1088/2515-7647/ae3504

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


## Data Availability

The dataset required for running the code is available on Zenodo: [Dataset DOI link](https://doi.org/10.5281/zenodo.17119986).
>**Note:** Please download the dataset from Zenodo before using this repository. 

## How to Run
To run the program, execute the main script in the project root.
>**Note:** All input data paths and processing parameters can be customized using the provided YAML configuration files.
Simply edit the configuration file to specify the dataset location, reconstruction parameters, or other settings before running the script.

---

## Installation
Clone the repository and install dependencies:
```bash
git clone https://github.com/julianna-winnik/osi-flex.git
cd osi-flex
pip install -r requirements.txt
