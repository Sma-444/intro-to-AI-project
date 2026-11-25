"""Print selected local course evidence with original slide numbers."""
from pathlib import Path
import re
import sys

sys.stdout.reconfigure(encoding='utf-8')

folder = Path(__file__).parent / 'course_text'
queries = {
    'Lecture 10 RNN.txt': r'Example: image captioning|Gradient clipping:|LSTM doesn.t guarantee',
    'Lecture 11-12 NLP with RNN and Attention.txt': r'Teacher Forcing|Application:  Image Captioning|bottleneck|different-weights|interpretability|When to use transfer|In computer vision',
    'CNN.txt': r'Batch Normalization for|Benefits of Batch|ResNet ,|residual|Residual',
    'Lecture 5 - Overfitting and Model Metrics.txt': r'Techniques to reduce overfitting|Common causes',
    'Optimization in Practice.txt': r'Effects of Learning Rate on|Adam.*efficient|Hyperparameter-sensitive',
}
for file, pattern in queries.items():
    for part in re.split(r'(?m)^SLIDE ', (folder / file).read_text(encoding='utf-8'))[1:]:
        number, body = part.split('\n', 1)
        if re.search(pattern, body, re.I):
            print(f'\n{file} SLIDE {number}\n{body.strip()}')
