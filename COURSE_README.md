# AIAA 2205 — Project: Image Description Generation

### Duration: 11:59PM Nov. 9, 2026 -- 11:59PM Dec. 9, 2026

## 1. Objective

Develop a model that generates a short English description for an input image. You may design the model architecture, training procedure, data augmentation, hyperparameter-tuning strategy, and experimental workflow, subject to the restrictions below.

This is a group project. Each group may contain **one to three students**. A one-person group is permitted.

The primary artifact used for performance evaluation is `test_predictions.csv`, containing one model-generated description for every test image. You must submit this result file together with all training/inference code and the final trained model.

### Required deliverables at a glance

Your final ZIP contains one folder named with your Group ID. That folder has four required parts:

| Required item          | Purpose                                                                                                                                |
| ---------------------- | -------------------------------------------------------------------------------------------------------------------------------------- |
| `test_predictions.csv` | One generated English description for each of the 2,625 test images. It should has two columns: `image_id` and `description`           |
| `report.pdf`           | A four-page project report describing the method, experiments, results, and analysis; references are excluded from the four-page limit |
| `code/`                | The executable notebook/python scripts, environment specification, and all supporting source code                                      |
| `model/final_model.pt` | The checkpoint used to produce the submitted CSV; it must be smaller than 200,000,000 bytes                                            |

The exact CSV schema and ZIP directory structure are defined in Sections 5 and 7. Read those sections before starting so that your training code saves artifacts directly into the required locations.

## 2. Dataset

The dataset contains 17,125 images with reference descriptions for the training and validation splits. Test reference descriptions are held by the instructor for evaluation of your `test_predictions.csv`.

| Split      | Images | Reference descriptions (labels) | Intended use                              |
| ---------- | ------:| ------------------------------- | ----------------------------------------- |
| Training   | 12,000 | Provided                        | Fit model parameters                      |
| Validation | 2,500  | Provided                        | Model selection and hyperparameter tuning |
| Test       | 2,625  | Hidden from students            | Final evaluation only                     |

The images used in this project are sourced from the [PASCAL Visual Object Classes Challenge 2012 (VOC2012)](https://www.robots.ox.ac.uk/~vgg/projects/pascal/VOC/voc2012/); copyright remains with the respective rights holders, and use and redistribution are subject to the applicable terms and permissions.

### Files provided to students

```text
AIAA2205Project/
├── Training/
│   ├── train_00001.jpg
│   └── ...
├── Validation/
│   ├── validation_00001.jpg
│   └── ...
├── Test/
│   ├── test_00001.jpg
│   └── ...
├── README.md
├── Training_Labels.csv
├── Validation_Labels.csv
├── Validation_Predictions_Example.csv
├── Text_Embedding_Model/
├── Computing_Text_Similarity.ipynb
└── author-kit-CVPR2027-v1-latex.zip
```

`Training_Labels.csv` and `Validation_Labels.csv` each contain exactly two columns:

```csv
image_filename,        description
train_00001.jpg,       a woman standing outdoors
```

`Validation_Predictions_Example.csv` is an example prediction file used by the similarity notebook. It contains `image_id,description`; it is not an additional set of labels.

`Text_Embedding_Model/` contains the pretrained text-embedding model `all-MiniLM-L6-v2`. Use it to evaluate the descriptions your captioning model generates for the validation images. For each image, it converts your generated description and the provided reference description into numerical vectors, then compares them using cosine similarity. The average similarity across all validation images is your validation score; a higher score indicates closer agreement in meaning.

You may use this score to choose which saved checkpoint to submit, decide when to stop training, and compare different hyperparameter settings. Keep the text-embedding model frozen: do not update its weights, include it in your caption-generation model, or use its outputs to train that model.

## 3. Model and Data Restrictions

1. Fit model parameters using only the provided training images and `Training_Labels.csv`.
2. Validation descriptions may be used for model selection, early stopping, and hyperparameter tuning, but not for gradient-based parameter fitting.
3. Test images may be used only for final inference. Do not use them for training, adaptation, pseudo-labelling, model selection, or manual description editing.
4. External image-description training data are not allowed.
5. Existing pretrained **image encoders** are allowed, including image-classification encoders pretrained on datasets such as ImageNet.
6. Pretrained text decoders, pretrained complete image-captioning or vision-language models, and external vision-language APIs are not allowed as part of the submitted generation system.
7. The frozen `sentence-transformers/all-MiniLM-L6-v2` model may be used to compute a validation similarity metric for checkpoint selection, early stopping, and hyperparameter comparison.
8. You must submit the selected final checkpoint (the whole model including the pretrained weights) as `model/final_model.pt`. Its uncompressed file size must be strictly less than **200 MB**.
9. You can only produce `test_predictions.csv` using your own model.

### Compute resources and the model-size limit

This project is designed for limited computing resources. The 200 MB model-size limit ensures that submitted models can run on ordinary computers.

Very high model performance is not required. The performance thresholds below would not be difficult to achieve with a properly implemented model. Students should focus on understanding the methods, correct implementation, and design choices rather than excessive performance optimization.

Students who need GPUs may use free services such as [Google Colab](https://research.google.com/colaboratory/faq.html) or [Kaggle Notebook](https://www.kaggle.com/docs/efficient-gpu-usage). Google Colab is a cloud-based notebook service provided by Google. It lets you write and run Python code in your browser without setting up a local environment. Kaggle Notebooks is a similar cloud-based notebook environment provided by Kaggle, a data science platform that also hosts datasets and competitions. Both services offer free access to GPUs. Kaggle currently offers approximately **30 free GPU-hours per week**, sometimes more depending on demand and available resources. Colab also provides free GPU access but does not guarantee a fixed weekly allowance. Its usage limits and hardware availability fluctuate. Plan experiments according to the resources available in your account, and save checkpoints regularly. Service policies and quotas may change.

## 4. Evaluation and Grading

The instructor will compare your submitted descriptions with **the reference descriptions (labels)** for the test images. Exact string matching is not used: different wordings may convey similar meanings.

Therefore, the selected evaluation metric is **mean cosine similarity between sentence embeddings**. Each description is converted to a fixed **384-dimensional vector** using the pretrained text-embedding model [`sentence-transformers/all-MiniLM-L6-v2`](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2). For each image, cosine similarity is computed between the embedding of the student's description and the embedding of its reference description. The similarities are then averaged across **all test images**. The same frozen embedding model and preprocessing will be used for all submissions. A higher score indicates greater agreement with the provided references.

The accompanying notebook `Computing_Text_Similarity.ipynb` provides an example of this calculation on validation descriptions. The evaluation will match rows by **`image_id`**, not by row order.

Let $P$ be the student's mean test similarity and $Q$ be the mean test similarity of a **fixed instructor baseline model**, both evaluated against the same reference descriptions. The baseline must satisfy:

$$
0<Q<1.
$$

Define the **normalized improvement over the baseline**:

$$
R=\frac{P-Q}{1-Q}.
$$

The denominator is the remaining gap between the baseline similarity and the maximum possible similarity of 1. Thus, $R=0$ means matching the baseline, $R=1$ means achieving a mean similarity of 1, and a positive $R$ measures the fraction of that gap closed by the student's model. Negative values indicate performance below the baseline.

For example, if $P=0.80$ and $Q=0.75$, then:

$$
R=\frac{0.80-0.75}{1-0.75}=0.20=20\%.
$$

**The instructor will fix $Q$ and verify that the score bands are practically reasonable before release of the final requirements.**

The total score is **36 points**, consisting of prediction performance (**15 points**), notebook execution/reproducibility (**10 points**), and report quality (**11 points**).

**Prediction Performance (maximum 15 points):**

| Normalized improvement $R$                | Score |
| ----------------------------------------- | -----:|
| $R>50\%$                                  | 15    |
| $30\%<R\leq50\%$                          | 12    |
| $0<R\le 30\%$                             | 10    |
| $-20\%<R\leq0$                            | 8     |
| $-40\%<R\leq-20\%$                        | 6     |
| $R\leq-40\%$                              | 3     |
| Missing or invalid `test_predictions.csv` | 0     |

## 5. Prediction File Format

`test_predictions.csv` must be a UTF-8 CSV with exactly these two columns and no saved index column:

```csv
image_id,              description
test_00001,            a woman in a blue shirt standing outdoors
test_00002,            an airplane parked on the ground
```

Requirements:

- Include exactly one row for every test image.
- Use the filename stem as `image_id`; do not include `.jpg`.
- `description` must be a nonempty English description produced by the submitted model.
- Do not include reference descriptions or additional columns.

## 6. Project Report and Results Analysis

Submit a project report named exactly `report.pdf`. The report must have **exactly four pages of main text**. The title, author/student ID, abstract, figures, tables, captions, and footnotes all count toward the four-page main-text limit. Place the Contribution Statement immediately after the four main-text pages and before the references. Neither the Contribution Statement nor the references count toward the four-page main-text limit. Appendices and supplementary pages are not permitted.

The report must be written in LaTeX using `author-kit-CVPR2027-v1-latex.zip`, which is included in the root of the provided `AIAA2205Project/` folder. Extract the archive, edit `main.tex` and its section files, and compile `main.tex` to produce `report.pdf`. Use the camera-ready style (`\usepackage{cvpr}`) rather than the anonymous review style so that all group members' names and student IDs are visible in the author block.

Preserve the supplied CVPR page size, two-column layout, font sizes, margins, column widths, and line spacing. Do not modify `cvpr.sty` or the template settings to fit additional content. Remove the template's instructional/example text and replace it with your own report. Do not use `rebuttal.tex` or the supplementary-material example for the project report. The final report must be understandable without reading the source code first.

The CVPR template archive is an input resource only. Do not include the template archive, `main.tex`, `cvpr.sty`, section source files, bibliography source, or LaTeX build files in `Group_ID.zip`; submit only the compiled `report.pdf`.

The four-page main text must cover all of the following:

1. **Method**
2. **Experimental setup**
3. **Result**
4. **Analysis**
5. **Conclusion**

## 7. Submission Requirements

Each group submits exactly one ZIP file. Use the group's Canvas group number for both the ZIP filename and its single top-level folder. Replace `ID` in `Group_ID` with the group number shown in Canvas.

**Late** submissions will receive no credit.

```text
Group_ID.zip
└── Group_ID/
    ├── test_predictions.csv
    ├── report.pdf
    ├── code/
    │   ├── train_and_predict.ipynb or train_and_predict.py
    │   ├── README.md
    │   ├── requirements.txt
    │   └── ... optional Python modules or configuration files
    └── model/
        ├── final_model.pt
        └── README.md
```

The required items inside `Group_ID/` are:

1. `test_predictions.csv`: predictions for all 2,625 test images, using the exact format defined in Section 5.
2. `report.pdf`: the four-page main-text project report plus references, following Section 6.
3. `code/`: all code.
4. `code/README.md`: Instructions for running the code.
5. `code/requirements.txt`: Python package dependencies.
6. `model/`: the final selected inference checkpoint.
7. `model/README.md` : Instructions for using the checkpoint.

Do not copy the course image dataset or `Text_Embedding_Model/` into your submission.

## 8. Group Work and Contribution Statement

- Groups may contain one, two, or three students. Before the announced group-registration deadline, every member must create or join the correct project group in Canvas. Check that the Canvas group contains exactly the intended members; the Canvas group record is used as the official group membership.
- Submit only one ZIP file per Canvas group. The Canvas group number is used in the `Group_ID.zip` filename and matching `Group_ID/` folder name.
- List every member's full name and student ID in the report author block.
- Include a short **Contribution Statement** immediately after the four-page main text and before the references. State each member's concrete contribution to model design, implementation, experiments, analysis, and report preparation. The Contribution Statement does **not** count toward the four-page main-text limit.
- Collaboration within the registered group is allowed. Sharing code, model weights, predictions, or report text across different groups is not allowed.

## 9. AI Policy

The use of AI assistance is permitted for this assignment; however, students are responsible for understanding and verifying all submitted work.
