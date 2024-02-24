# LLM-AES-AER
The code repository for "Automated Essay Scoring and Revising Based on Open-Source Large Language Models"
## Introduction
This repository contains the code for the paper "Automated Essay Scoring and Revising Based on Open-Source Large Language Models". The code is based on the [transformers](4.18.0) library and [tensorflow](2.0.0) library.
## File Structure
### src/baseline-*.ipynb
The baseline models for AES.
### src/*-zero-shot.py
The zero-shot LLMs for AES.
### src/*-few-shot.py
The few-shot LLMs for AES.
### src/p-tuning.py
The p-tuning method based on ChatGLM2-6B for AES.
### src/AER.py
A simple example of AER.
## Data Availability
For privacy reasons, the data used in the paper is not available in this repo, including the data file "selected-ann.csv" and the shots used in "src/*-few-shot.py". However, the data can be obtained through contacting the authors.