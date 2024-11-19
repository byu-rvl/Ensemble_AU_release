# Borda Ranking with Synthetic Data and Ensembles for Facial Action Unit Recognition

![The Need for Borda](images/borda_side_by_side.jpg)

Borda Ranking with Synthetic Data and Ensembles for Facial Action Unit Recognition

Author names here (removed for blind review)

## Architecture Overview:

![Overarching_Architecture](images/overall_architecture.jpg)

## Base Model Overview:

![Base_Model_Overview](images/base_model_diagram.jpg)

## Stacking Diagram:

![Stacking_Diagram](images/stacking_diagram.jpg)


## Environment Setup:

TODO: implement

conda env create -f ennvironment.yml

## Testing setup:

### 0. Acquire data permissions for the BP4D and DISFA datasets.

Due to data licensing, we do not provide data for the BP4D-Spontaneous and DISFA datasets. The necessary permissions must be provided from the respective individuals.

### 1. Load the BP4D and DISFA landmark data into data/datasets/original directory.

In addition, modify the landmark points for the DISFA to be in numpy instead of matlab format:

```bash
./scripts/mat_to_np.sh
```

### 2. Crop to the face for the BP4D and DISFA datasets:

BP4D:
```bash
./scripts/cropFace_BP4D.sh /fslhome/andreww9/fsl_groups/grp_AU_storage/compute/BP4D_imagesOnly
```

DISFA:
```bash
./scripts/cropFace_DISFA.sh path/to/DISFA/Videos_LeftCamera
```

### 3. Data preprocessing:

BP4D:
```bash
./scripts/dataset_prep_BP4D.sh
```

DISFA:
```bash
./scripts/dataset_prep_DISFA.sh
```

### 4. Download model parameters from [here](https://byu-my.sharepoint.com/:f:/g/personal/andreww9_byu_edu/EtIJ0A3rwf1LrIznD9H4BLkBfhlaLEWiQw3aOX3Ahe3e1g?e=8179M6) to data/weights/

| Dataset | Fold | Pretrained Initialization | Random Initialization |
|----------|----------|----------|----------|
| BP4D   | 1 | [Fold_1_pretrained_weights](https://byu-my.sharepoint.com/:u:/g/personal/andreww9_byu_edu/ERH-Yl4FRjJCvTqyPmbGNQMBwMDmi0DiCyphawLTzbMZeg?e=HDbgyS)   | [Fold_1_randomInit_weights](https://byu-my.sharepoint.com/:u:/g/personal/andreww9_byu_edu/Eb1A205c3PNCvPkMFHBBfGIB8VLznCwQ2aXwhMHB3Zz9rw?e=uQhgK6)  |
| BP4D   | 2 | [Fold_2_pretrained_weights](https://byu-my.sharepoint.com/:u:/g/personal/andreww9_byu_edu/EYwGXcTd7cJFqLVyf4a1MEQBWX0eIzvPuHYHMT7X3YrYWA?e=p4yoHw)   | [Fold_2_randomInit_weights](https://byu-my.sharepoint.com/:u:/g/personal/andreww9_byu_edu/EUIoT7AZJntBiARDOmJd_ZUB6yhQ7jMN7go6-iAYh6P46g?e=uDKqhn)  |
| BP4D   | 3 | [Fold_3_pretrained_weights](https://byu-my.sharepoint.com/:u:/g/personal/andreww9_byu_edu/EZjHK4ldp_9Hkih9bm9JI2YB7k5lxpJNv7YTut0i6TMUzw?e=waou5z)   | [Fold_3_randomInit_weights](https://byu-my.sharepoint.com/:u:/g/personal/andreww9_byu_edu/EUYR7cRkxWFJhBQwkZtU00gBgvEBhBmYS7kEKS9pHpZP6g?e=kZ5Fnm)  |
| DISFA   | 1 | [Fold_1_pretrained_weights](https://byu-my.sharepoint.com/:u:/g/personal/andreww9_byu_edu/EZLNR9IgL1NFlKWL0MzrWfABkjh8S9OTnKjP0L_jz164ZA?e=vEaSTc)   | [Fold_1_randomInit_weights](https://byu-my.sharepoint.com/:u:/g/personal/andreww9_byu_edu/EYUDlRW6b2FAtHkxVumdf0UBV_mABld_wAfhlllt7cCytA?e=WR0WSF)  |
| DISFA   | 2 | [Fold_2_pretrained_weights](https://byu-my.sharepoint.com/:u:/g/personal/andreww9_byu_edu/EZQ6FGxwN3RPjGktQqA80hYBJQozx2gfI9ZXXaODYNF06Q?e=gu81os)   | [Fold_2_randomInit_weights](https://byu-my.sharepoint.com/:u:/g/personal/andreww9_byu_edu/ETGQWTBmatROu-EVWVTZbqABdHjRe384p94h3OWgsX216Q?e=o7Bhjh)  |
| DISFA   | 3 | [Fold_3_pretrained_weights](https://byu-my.sharepoint.com/:u:/g/personal/andreww9_byu_edu/EWwIdToL_QBDkC_qEKuDVSYBCBnEXr_waJxGh7OLakGQ2g?e=VtsGW4)   | [Fold_3_randomInit_weights](https://byu-my.sharepoint.com/:u:/g/personal/andreww9_byu_edu/EYupeBDqJhxHhJlUn7cqMqsBLb99EUUx4gE2XOFwduFGOg?e=3JB18a)  |


## Testing:

To test the BP4D dataset or the DISFA dataset, run the testing setup steps provided above.

Once the testing setup is completed, to test on the BP4D and the DISFA datasets:

BP4D:
```bash
./scripts/test_BP4D.sh fold_number_here
```

DISFA:
```bash
./scripts/test_DISFA.sh fold_number_here
```


## Results

BP4D:
| Method               | Additional | AU 1 | AU 2 | AU 4 | AU 6 | AU 7 | AU 10 | AU 12 | AU 14 | AU 15 | AU 17 | AU 23 | AU 24 | Avg. F1 Score |
|----------------------|------------|------|------|------|------|------|-------|-------|-------|-------|-------|-------|-------|---------------|
| DRML                 | ✗          | 36.4 | 41.8 | 43.0 | 55.0 | 67.0 | 66.3  | 65.8  | 54.1  | 33.2  | 48.0  | 31.7  | 30.0  | 48.3          |
| EAC-Net              | ✗          | 39.1 | 35.2 | 48.6 | 76.1 | 72.9 | 81.9  | 86.2  | 58.8  | 7.5   | 59.1  | 35.9  | 35.8  | 55.9          |
| JAA-Net              | ✗          | 47.2 | 44.0 | 54.9 | 77.5 | 74.6 | 84.0  | 86.9  | 61.9  | 43.6  | 60.3  | 42.7  | 41.9  | 60.0          |
| LP-Net               | ✗          | 43.4 | 38.0 | 54.2 | 77.1 | 76.7 | 83.8  | 87.2  | 63.3  | 45.3  | 60.5  | 48.1  | 54.2  | 61.0          |
| ARL                  | ✗          | 45.8 | 39.8 | 55.1 | 75.7 | 77.2 | 82.3  | 86.6  | 58.8  | 47.6  | 62.1  | 47.4  | 55.4  | 61.1          |
| AU-GCN               | ✗          | 46.8 | 38.5 | 60.1 | 80.1 | 79.5 | 84.8  | 88.0  | 67.3  | 52.0  | 63.2  | 40.9  | 52.8  | 62.8          |
| SRERL                | ✗          | 46.9 | 45.3 | 55.6 | 77.1 | 78.4 | 83.5  | 87.6  | 63.9  | 52.2  | 63.9  | 47.1  | 53.3  | 62.9          |
| UGN-B                | ✗          | 54.2 | 46.4 | 56.8 | 76.2 | 76.7 | 82.4  | 86.1  | 64.7  | 51.2  | 63.1  | 48.5  | 53.6  | 63.3          |
| HMP-PS               | ✗          | 53.1 | 46.1 | 56.0 | 76.5 | 76.9 | 82.1  | 86.4  | 64.8  | 51.5  | 63.0  | 49.9  | 54.5  | 63.4          |
| SEV-Net              | ✓          | 58.2 | 50.4 | 58.3 | 81.9 | 73.9 | 87.8  | 87.5  | 61.6  | 52.6  | 62.2  | 44.6  | 47.6  | 63.9          |
| AUFM                 | ✓          | 57.4 | 52.6 | 64.6 | 79.3 | 81.5 | 82.7  | 85.6  | 67.9  | 47.3  | 58.0  | 47.0  | 44.9  | 64.1          |
| FAUDT                | ✗          | 51.7 | 49.3 | 61.0 | 77.8 | 79.5 | 82.9  | 86.3  | 67.6  | 51.9  | 63.0  | 43.7  | 56.3  | 64.2          |
| KDSRL                | ✓          | 53.3 | 47.4 | 56.2 | 79.4 | 80.7 | 85.1  | 89.0  | 67.4  | 55.9  | 61.9  | 48.5  | 49.0  | 64.5          |
| ME-GraphAU           | ✗          | 52.7 | 44.3 | 60.9 | 79.9 | 80.1 | 85.3  | 89.2  | 69.4  | 55.4  | 64.4  | 49.8  | 55.1  | 65.5          |
| SACL                 | ✗          | 57.8 | 48.8 | 59.4 | 79.1 | 78.8 | 84.0  | 88.2  | 65.2  | 56.1  | 63.8  | 50.8  | 55.2  | 65.6          |
| GTLE-Net             | ✓          | 58.2 | 48.7 | 61.5 | 78.7 | 79.2 | 84.2  | 89.8  | 66.3  | 56.7  | 64.8  | 53.5  | 53.6  | 66.3          |
| ELEGANT (Ours)       | ✓          | 57.4 | 50.1 | 66.9 | 79.2 | 80.4 | 84.9  | 89.5  | 68.9  | 55.2  | 65.6  | 50.8  | 59.1  | 67.3          |


DISFA:
| Method               | Additional | AU 1 | AU 2 | AU 4 | AU 6 | AU 9 | AU 12 | AU 25 | AU 26 | Avg. F1 Score |
|----------------------|------------|------|------|------|------|------|-------|-------|-------|---------------|
| DRML                 | ✗          | 17.3 | 17.7 | 37.4 | 29.0 | 10.7 | 37.7  | 38.5  | 20.1  | 26.1          |
| EAC-Net              | ✗          | 41.5 | 26.4 | 66.4 | 50.7 | 80.5 | 89.3  | 88.9  | 15.6  | 48.5          |
| AU-GCN               | ✗          | 32.3 | 19.5 | 55.7 | 57.9 | 61.4 | 62.7  | 90.9  | 60.0  | 55.0          |
| SRERL                | ✗          | 45.7 | 47.8 | 59.6 | 47.1 | 45.6 | 76.5  | 84.3  | 43.6  | 55.9          |
| JAA-Net              | ✗          | 43.7 | 46.2 | 56.0 | 41.4 | 44.7 | 69.6  | 88.3  | 58.4  | 56.0          |
| LP-Net               | ✗          | 29.9 | 24.7 | 72.7 | 46.8 | 49.6 | 72.9  | 93.8  | 65.0  | 56.9          |
| AUFM                 | ✓          | 41.5 | 44.9 | 60.3 | 51.5 | 50.3 | 70.4  | 91.3  | 55.3  | 58.2          |
| ARL                  | ✗          | 43.9 | 42.1 | 63.6 | 41.8 | 40.0 | 76.2  | 95.2  | 66.8  | 58.7          |
| SEV-Net              | ✓          | 55.3 | 53.1 | 61.5 | 53.6 | 38.2 | 71.6  | 95.7  | 41.5  | 58.8          |
| UGN-B                | ✗          | 43.3 | 48.1 | 63.4 | 49.5 | 48.2 | 72.9  | 90.8  | 59.0  | 60.0          |
| HMP-PS               | ✗          | 38.0 | 45.9 | 65.2 | 50.9 | 50.8 | 76.0  | 93.3  | 67.6  | 61.0          |
| FAUDT                | ✗          | 46.1 | 48.6 | 72.8 | 56.7 | 50.0 | 72.1  | 90.8  | 55.4  | 61.5          |
| ME-GraphAU           | ✗          | 54.6 | 47.1 | 72.9 | 54.0 | 55.7 | 76.7  | 91.1  | 53.0  | 63.1          |
| KDSRL                | ✓          | 60.4 | 59.2 | 67.5 | 52.7 | 51.5 | 76.1  | 91.3  | 57.7  | 64.5          |
| SACL                 | ✗          | 62.0 | 65.7 | 74.5 | 53.2 | 43.1 | 76.9  | 95.6  | 53.1  | 65.5          |
| GTLE-Net             | ✓          | 64.5 | 63.2 | 70.1 | 47.7 | 53.6 | 76.2  | 94.8  | 65.1  | 66.9          |
| ELEGANT (Ours)       | ✓          | 67.2 | 64.1 | 74.2 | 52.2 | 47.0 | 73.1  | 95.1  | 69.8  | 67.8          |



## BibTeX


```bibtex
@misc{yourproject2024,
  author = {Removed for blind review},
  title = {Borda Ranking with Synthetic Data and Ensembles for Facial Action Unit Recognition},
  year = {TO BE FILLED AFTER ACCEPTANCE/PUBLICATION.},
  publisher = {TO BE FILLED AFTER ACCEPTANCE/PUBLICATION.}
}
