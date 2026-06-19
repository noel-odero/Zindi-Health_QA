# Multilingual Health Question Answering in Low-Resource African Languages

Final project for Machine Learning Techniques I, addressing the [Zindi Multilingual Health QA Challenge](https://zindi.africa/competitions/multilingual-health-question-answering-in-low-resource-african-languages-challenge), hosted by ITU and HASH.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1lb7Hcm8OVDFpyZICRQJZtWHRdB0ckR0U?usp=sharing)

The notebook may fail to render directly on GitHub due to leftover Colab widget metadata. Use the Colab badge above to open and run it directly.

## Project Overview

The task is to answer health-related questions written in low-resource African languages, returning a fluent, contextually appropriate answer in the same language. The dataset spans eight language-country subsets across five languages: English (Uganda, Ghana, Ethiopia, Kenya), Akan (Ghana), Luganda (Uganda), Kiswahili (Kenya), and Amharic (Ethiopia).

Submissions are scored on a weighted blend of ROUGE-1 F1 (0.37), ROUGE-L F1 (0.37), and LLM-as-a-Judge (0.26).

This repository documents two lines of investigation:

1. **Fine-tuning multilingual transformers** (mT5 family, LoRA) - attempted first and documented including the environment and compute obstacles that led to it being abandoned partway through.
2. **Retrieval-based question answering** - a hybrid TF-IDF + multilingual sentence-embedding system, tuned per language, which became the final approach.

Full methodology, all ten experiments, results, and discussion are in the accompanying report: `Odero Anjeline Noel_FinalProject.pdf`.


## Results Summary

| Experiment | What Changed | Val ROUGE-1 | Zindi Public Score |
|---|---|---|---|
| E01 | TF-IDF word n-grams (baseline) | 0.3927 | 0.491647 |
| E02 | TF-IDF character n-grams | 0.4191 | 0.488358 |
| E03 | Semantic embeddings only | 0.4135 | 0.483847 |
| E04 | Hybrid 50/50 (TF-IDF + semantic) | 0.4454 | 0.528427 |
| E05 | Per-language weight tuning | **0.4736** | **0.548082** |
| E06 | Exact-match lookup | 0.4731 | pending |
| E07 | Deduplicated corpus | 0.4732 | pending |
| E08 | Cross-subset fallback (low-resource langs) | 0.4731 | pending |
| E09 | Similarity threshold fallback | 0.4732 | pending |
| E10 | Final combined model (Train+Val corpus) | 0.4736* | pending |

Best result so far: **Experiment 5** - per-language tuned hybrid retrieval, public leaderboard score **0.548082**.

Full experiment rationale, ablation analysis, and discussion of why E06–E09 plateaued are in the report.

## Repository Structure

```
Zindi-Health_QA/
├── Zindi_health_qa.ipynb                  # Main notebook - retrieval pipeline, all 10 experiments
├── Odero Anjeline Noel_FinalProject.pdf   # Full academic report
├── data/                                  # Train.csv, Val.csv, Test.csv, SampleSubmission.csv
├── submissions/                           # Per-experiment Zindi submission CSVs (E01-E10)
└── README.md
```


## How to Run

The notebook is designed to run end-to-end on Google Colab with minimal setup.

1. Click the **Open in Colab** badge above.
2. Mount your Google Drive when prompted (the notebook stores embeddings, predictions, and the experiment tracker there so progress survives disconnects).
3. Place `Train.csv`, `Val.csv`, `Test.csv`, and `SampleSubmission.csv` in a folder named `zindi-health-qa` in your Drive root.
4. Run cells 1-6 in order (these install dependencies, load the data, and build the cached question embeddings).
5. Run the experiment cells (7-16) in order. Each is independently checkpointed: if a cell finishes, rerunning the notebook skips it automatically rather than repeating the work.



## Methodology Summary

**Architecture:** a `SemanticRoutingIndex` retrieves the closest matching training question for each input, per language subset, using a tunable blend of:
- TF-IDF cosine similarity (word or character n-grams)
- Multilingual sentence-embedding cosine similarity (`paraphrase-multilingual-MiniLM-L12-v2`)

**Evaluation protocol:** every experiment is fit on Train only and scored on the full Val set (6,686 rows) using whitespace-tokenized ROUGE-1/ROUGE-L, kept consistent across all five languages and scripts. Val is only folded into the index for the final model (E10), after all comparative experiments are complete.

**Why retrieval over fine-tuning:** documented in full in the report (Section 6 and Section 8.3). In short, three days of fine-tuning attempts across Kaggle and Colab were repeatedly blocked by environment instability (bitsandbytes/CUDA mismatches, multi-GPU memory issues) and GPU quota exhaustion, with no completed result better than a near-zero zero-shot baseline. The retrieval approach was adopted as a reliable alternative that requires no GPU and produces reproducible results in minutes, and ultimately outperformed the abandoned fine-tuning attempts by a wide margin.


## Limitations

- Retrieval can only return answers that already exist in the training corpus; it cannot synthesize novel responses.
- Amharic and Luganda remain the weakest-performing language subsets. Experiment 8 found that widening their candidate pool with English fallback data did not help, suggesting the bottleneck is not simple data scarcity.
- The internal evaluation protocol uses ROUGE only and does not model the LLM-as-a-Judge component of the official Zindi metric.

Full limitations and future improvements are discussed in the report (Sections 9 and 11).
