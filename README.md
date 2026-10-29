# English-to-French Machine Translation with Four Transformer Architectures

## Problem statement and objective

This project implements and compares four English-to-French machine-translation systems for ICT4403 Foundations of Generative AI Assignment 3:

1. a basic encoder-decoder Transformer trained from scratch;
2. `google/mt5-small`, an encoder-decoder model fine-tuned on the selected parallel corpus;
3. `meta-llama/Llama-3.2-1B-Instruct`, a decoder-only model adapted with three-shot prompting; and
4. `google/gemma-3-270m-it`, a decoder-only model adapted with the same three-shot prompting protocol.

All models translate the same held-out test examples. The experiment records corpus BLEU, chrF, training or fine-tuning time, inference time, and model parameter count. It also examines how decoding temperature affects the two prompted decoder-only models and discusses responsible-AI risks in machine translation.

## Dataset

The project uses the English-French configuration of the public [OPUS Books dataset](https://huggingface.co/datasets/Helsinki-NLP/opus_books), a collection of aligned translated literary text. The code shuffles the available pairs with seed 42, normalizes whitespace, removes empty or extreme-length pairs, and retains 10,000 examples.

| Split | Pairs | Purpose |
|---|---:|---|
| Training | 8,000 | Train the basic Transformer and fine-tune mT5; select three fixed demonstrations |
| Validation | 1,000 | Model selection and validation-loss monitoring |
| Test | 1,000 | Held-out data; the first 200 examples after deterministic shuffling are used for four-model evaluation |

The fixed 80/10/10 split is created only after cleaning. No test sentence is used for training, validation, prompt demonstrations, or hyperparameter selection.

## Models and adaptation

### Basic Transformer

The from-scratch baseline follows the encoder-decoder architecture from *Attention Is All You Need*. It uses a jointly trained 8,000-token byte-level BPE vocabulary, learned token embeddings, sinusoidal positional encodings, three encoder and three decoder blocks, 256 hidden units, eight attention heads, a 512-unit feed-forward layer, dropout 0.1, label smoothing 0.1, and greedy decoding. It is trained for eight epochs with AdamW.

### mT5-small

[mT5-small](https://huggingface.co/google/mt5-small) is a multilingual text-to-text encoder-decoder pretrained on mC4. It is fully fine-tuned for one epoch using the task prefix `translate en to fr:`. Translation uses four-beam search. This model has multilingual pretraining and a SentencePiece vocabulary, unlike the from-scratch baseline.

### Llama 3.2 1B Instruct

[Llama 3.2 1B Instruct](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct) is a pretrained decoder-only instruction model. To keep the comparison feasible and follow the assignment's allowance for prompting, it receives the same three fixed training demonstrations and a constrained instruction to return only the French translation. Evaluation uses greedy decoding. The code downloads an ungated weight mirror (`unsloth/Llama-3.2-1B-Instruct`) while reporting the original model identity and license.

### Gemma 3 270M Instruct

[Gemma 3 270M Instruct](https://huggingface.co/google/gemma-3-270m-it) is a compact decoder-only instruction model. It uses the identical prompt, demonstrations, batching policy, maximum output length, and decoding settings used for Llama. The code uses the equivalent ungated `unsloth/gemma-3-270m-it` weight mirror while retaining the original model identity and Gemma terms.

## Repository structure

```text
.
├── assignment3_machine_translation.ipynb  # Colab-friendly walkthrough
├── config.json                            # Reproducible experiment settings
├── run_experiment.py                      # Complete training/evaluation pipeline
├── requirements.txt                       # Python dependencies
├── outputs/                               # Metrics, samples, plots, and split metadata
├── checkpoints/                           # Locally generated model/tokenizer checkpoints
└── submission/                            # Final submission DOCX and PDF
```

Model checkpoints are intentionally excluded from Git because they can be reproduced from the configuration and are too large for a normal repository.

## Execution

### Local GPU

Python 3.12 and an NVIDIA GPU with at least 8 GB VRAM are recommended.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python run_experiment.py --config config.json
```

Install the CUDA build of PyTorch from the [official PyTorch selector](https://pytorch.org/get-started/locally/) if `torch.cuda.is_available()` returns `False`.

### Google Colab

Open `assignment3_machine_translation.ipynb`, select a GPU runtime, and run all cells. The notebook installs dependencies, runs the same checked-in script, and displays the generated tables, plots, and qualitative samples. A T4 or newer GPU is sufficient with the default settings.

## Reproducibility and fair-comparison controls

- Random seed: 42 for Python, NumPy, PyTorch, Hugging Face, dataset shuffling, and sampling.
- Same 200 held-out source/reference pairs for every model.
- Maximum source and target length: 96 tokens for trained encoder-decoder models.
- Greedy decoding for the basic Transformer, Llama, and Gemma; four-beam decoding for mT5 is disclosed because beam search is the standard sequence-to-sequence setting.
- Same three demonstrations and prompt template for Llama and Gemma.
- Timings cover translation generation only, after model loading, on the same hardware.
- BLEU and chrF are computed with SacreBLEU over the complete 200-example evaluation subset.

## Results

The checked-in result files are generated by a complete run of `run_experiment.py` on the hardware recorded in `outputs/experiment_metadata.json`.

<!-- RESULTS_START -->
Evaluation used 200 held-out pairs on an NVIDIA GeForce RTX 3070 Ti Laptop GPU. Times exclude model loading and dataset preparation.

| Model | Architecture | Adaptation | BLEU | chrF | Train/fine-tune time (s) | Inference (s/example) | Parameters |
|---|---|---|---:|---:|---:|---:|---:|
| Basic Transformer | Encoder-decoder | From scratch, 8 epochs | 0.41 | 12.14 | 52.31 | **0.047** | 8,058,688 |
| mT5-small | Encoder-decoder | Full fine-tuning, 1 epoch | 5.09 | 24.81 | 229.32 | 0.178 | 300,176,768 |
| **Llama 3.2 1B** | Decoder-only | Three-shot prompting | **8.17** | **29.79** | N/A | 0.204 | 1,235,814,400 |
| Gemma 3 270M | Decoder-only | Three-shot prompting | 5.10 | 27.70 | N/A | 0.258 | 268,098,176 |

Llama achieved the highest BLEU and chrF without task-specific parameter updates. The basic Transformer was fastest and smallest, but its outputs frequently collapsed into repeated phrases. mT5 improved markedly over the from-scratch baseline after one fine-tuning epoch. Gemma matched mT5's BLEU and exceeded its chrF, although it sometimes copied English words or paraphrased beyond the source.
<!-- RESULTS_END -->

Raw outputs:

- `outputs/model_metrics.csv`: quality, time, parameters, and adaptation method;
- `outputs/translations.csv`: aligned source, reference, and four model outputs;
- `outputs/temperature_analysis.csv`: BLEU, chrF, runtime, and output length by temperature;
- `outputs/dataset_statistics.json`: split sizes and length statistics;
- `outputs/plots/model_comparison.png`: quality and latency comparison; and
- `outputs/plots/temperature_analysis.png`: temperature sensitivity.

## Architecture comparison

The basic Transformer provides a controlled from-scratch baseline, but it must learn alignment, translation, and language modeling from only 8,000 pairs. mT5 starts with multilingual encoder-decoder pretraining and is structurally aligned with conditional generation: the encoder can attend bidirectionally to the complete English sentence while the decoder generates French conditioned on a separate source representation.

Llama and Gemma concatenate instructions, examples, and the source into one autoregressive context. Their pretraining supplies broad linguistic knowledge, but prompt-only adaptation does not directly optimize the models on the selected corpus. Their larger prompts also increase inference work. Model size alone therefore does not determine translation quality: pretraining coverage, tokenizer efficiency, architecture, adaptation method, and decoding all affect the scores.

## Sampling-temperature analysis

Greedy decoding (`temperature = 0`) is the primary evaluation because translation normally has a narrow set of faithful outputs and deterministic decoding improves reproducibility. Temperatures 0.3, 0.7, and 1.0 are evaluated on the same first 25 test sentences for Llama and Gemma with top-p 0.9.

| Model | Temperature | BLEU | chrF | Mean output words |
|---|---:|---:|---:|---:|
| Llama 3.2 1B | 0.0 | 12.09 | **35.35** | 22.68 |
| Llama 3.2 1B | 0.3 | **12.27** | 34.86 | 22.08 |
| Llama 3.2 1B | 0.7 | 5.40 | 25.83 | 18.08 |
| Llama 3.2 1B | 1.0 | 1.12 | 16.21 | 14.80 |
| Gemma 3 270M | 0.0 | **4.89** | **29.55** | 24.24 |
| Gemma 3 270M | 0.3 | 4.11 | 25.95 | 23.16 |
| Gemma 3 270M | 0.7 | 2.75 | 20.06 | 17.20 |
| Gemma 3 270M | 1.0 | 3.06 | 24.19 | 19.08 |

The small 25-sentence temperature sample shows a strong overall decline as randomness increases. Llama's BLEU changed only slightly from 12.09 to 12.27 at temperature 0.3, but chrF fell; both metrics then dropped sharply at 0.7 and 1.0. Gemma performed best with greedy decoding. Higher temperatures also shortened outputs on average, consistent with premature endings or omitted source content. Deterministic decoding is therefore the safest default for this translation setting.

## Qualitative analysis and responsible AI

### Risks to inspect

- **Bias:** pretrained models may reproduce gender, nationality, religion, or occupation stereotypes when English is underspecified and French requires grammatical gender.
- **Hallucination:** decoder-only models may add explanations, names, or facts not present in the source. Literary fragments can make this more likely because context is limited to one aligned segment.
- **Cultural misrepresentation:** idioms, honorifics, historical expressions, dialect, and culturally specific references may be normalized into a dominant-language expression or mistranslated literally.
- **Sensitive content:** hate speech, sexual material, medical statements, legal language, or self-harm content can be softened, intensified, or translated into newly harmful wording.
- **Data and licensing:** public availability does not erase copyright, provenance, privacy, or license obligations. OPUS Books contains literary text and its dataset card lists a nonstandard `other` license.

### Observed examples

The output sample confirms several of these concerns. The basic Transformer repeatedly generated phrases such as *de la porte* and *vous avez raison*, a fluency failure caused by limited training and greedy autoregression. For “It's thumbing its nose at me!”, Llama produced a reasonable French paraphrase, mT5 translated the idiom literally and awkwardly, and Gemma copied the English sentence unchanged. Gemma also changed “Private interests are as nothing before the interests of all” into a garden metaphor, which is a clear hallucination. In the Janissaries example, Llama preserved the event but mistranslated “flat-bottomed boats,” while mT5 repeated a clause and Gemma retained English terms. These errors show why aggregate overlap scores must be paired with human review.

### Mitigations

1. Use documented, consent-aware, licensed, and domain-representative parallel data; remove personal data and audit the corpus.
2. Evaluate by subgroup, dialect, gendered constructions, named entities, and sensitive-content categories rather than relying only on aggregate BLEU/chrF.
3. Add terminology constraints, named-entity preservation checks, toxicity screening, and uncertainty or abstention rules for high-risk domains.
4. Keep a human translator in the loop for medical, legal, safety-critical, culturally sensitive, or publication-grade content.
5. Log model/version, prompt, decoding settings, and post-edit history; communicate that outputs are machine-generated.
6. Red-team hallucination and prompt-injection behavior and prevent source text from overriding the translation instruction.
7. Monitor deployed systems for drift and provide a correction and appeal process for affected users.

## Limitations

The evaluation subset is intentionally small enough for a student-scale experiment. OPUS Books is literary and does not represent conversational, technical, legal, or medical translation. BLEU and chrF reward overlap with one reference and do not fully measure adequacy, fluency, cultural fidelity, or safety. The decoder-only models use prompting while mT5 is fine-tuned, so the study compares practical adaptation choices rather than isolating architecture alone.

## Conclusion

Llama 3.2 1B was the best-performing system in this experiment, with BLEU 8.17 and chrF 29.79. Its pretrained linguistic knowledge and instruction tuning outweighed the structural advantage of the encoder-decoder models under the small-data, short-training setting. It also required no task-specific training, though it had the most parameters and slower inference than the basic Transformer and mT5. mT5 was the strongest trained encoder-decoder model, while Gemma offered a compact prompted alternative with similar BLEU and higher chrF than mT5. The basic Transformer demonstrated the efficiency and limitations of learning translation from only 8,000 pairs. For practical use, Llama would be selected for quality, with greedy decoding, constrained prompting, terminology checks, and human review for sensitive or high-stakes content.

## References

- Vaswani, A. et al. (2017). [Attention Is All You Need](https://arxiv.org/abs/1706.03762).
- Xue, L. et al. (2021). [mT5: A Massively Multilingual Pre-trained Text-to-Text Transformer](https://arxiv.org/abs/2010.11934).
- Tiedemann, J. (2012). [Parallel Data, Tools and Interfaces in OPUS](https://aclanthology.org/L12-1246/).
- Post, M. (2018). [A Call for Clarity in Reporting BLEU Scores](https://aclanthology.org/W18-6319/).
- Popovic, M. (2015). [chrF: character n-gram F-score for automatic MT evaluation](https://aclanthology.org/W15-3049/).

