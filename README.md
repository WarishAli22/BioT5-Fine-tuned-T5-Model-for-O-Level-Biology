# BioT5: Fine-tuned T5 Model for O-Level Biology

A fine-tuned version of FlanT5 specifically trained on O-Level Biology curriculum content to assist with biology education and assessment.

## Overview

This repository contains a fine-tuned version of the FlanT5 language model, specialized for O-Level Biology content. The model has been trained to understand and respond to biology-related queries, explain concepts, and assist with O-Level Biology exam preparation.

## Features

- Question answering for O-Level Biology topics
- Concept explanations and clarifications
- Generation of practice questions
- Support for key biology topics covered in O-Levels:
  - Cell Biology
  - Human Biology
  - Plant Biology
  - Genetics
  - Others

## Model Details

- Base Model: FlanT5
- Training Data: Curated O-Level Biology curriculum content
- Training Steps: [Insert number of steps]
- Hardware Requirements: [Insert minimum requirements]

## Installation

```bash
# Clone the repository
git clone https://github.com/[username]/flan-t5-olevel-biology
cd flan-t5-olevel-biology

# Install dependencies
pip install -r requirements.txt

```

## Usage

```python
import nltk
import evaluate
import numpy as np
from datasets import load_dataset
from transformers import T5Tokenizer, DataCollatorForSeq2Seq
from transformers import T5ForConditionalGeneration, Seq2SeqTrainingArguments, Seq2SeqTrainer
import pandas as pd
from tabulate import tabulate
import torch
import torchvision

inputs = "how is sucrose is produced from carbon dioxide in pea plants

checkpoint = "./checkpoint-19614" #modify to point to path of checkpoint in this repo
finetuned_model = T5ForConditionalGeneration.from_pretrained(checkpoint)
tokenizer = T5Tokenizer.from_pretrained(checkpoint)

#form unclean finetuned:
inputs = tokenizer(inputs, return_tensors="pt")
outputs = finetuned_model.generate(**inputs, max_length=100, min_length=40)
answer = tokenizer.decode(outputs[0])
print(answer)
```

## Example Outputs

Input: "How sucrose is produced from carbon dioxide in pea plants"

```
Output: "in pea plants sucrose is produced by the process of diffusion which involves the use of oxygen to break down glucose in the form of a water molecule sucrose is then transported to the leaves where it can be used by the plant to produce glucose or stored as starch if needed"

```

## Training Data

The model was fine-tuned on:

- O-Level Biology syllabus content
- Curated textbook materials
- Sample exam questions and answers
- A-Level Biology syllabus content

## Performance

- ROUGE scores: 0.65

## Limitations

- The model is specifically trained for O-Level and A'Level Biology content and may not perform well on advanced biology topics
- Responses should be verified against official curriculum materials

## Acknowledgments

- Thanks to the original FlanT5 team

## Contact

For questions and feedback, please:

- Open an issue in this repository
