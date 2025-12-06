# LLM Lab Work

This repository contains 8 practical programs implementing Large Language Model (LLM) techniques, from fine-tuning to deployment and multimodal applications.

## Programs Overview

### Program 1: Question Answering Model Fine-tuning
- Fine-tuned BERT-base-uncased on SQuAD dataset for question answering
- Implemented data preprocessing with tokenization and answer span alignment
- Configured training with Hugging Face Trainer API
- Evaluated model performance with validation metrics
- Tested the fine-tuned model on custom questions about the Great Wall of China

**Screenshot**

<img width="318" height="187" alt="image" src="https://github.com/user-attachments/assets/fe116150-6061-4717-9c62-1d9fb21ea653" />
<img width="657" height="504" alt="image" src="https://github.com/user-attachments/assets/fa837164-ee78-410e-a8ef-297f06559823" />


---

### Program 2: Pre-trained Model Prompt Engineering
- Loaded DistilBERT model fine-tuned on SQuAD for question answering
- Created function to extract answers from context using start/end logits
- Tested multiple questions about machine learning concepts
- Compared different prompt formulations for effectiveness
- Analyzed differences between DistilBERT responses and commercial models like ChatGPT/Gemini

**Screenshot**

<img width="611" height="289" alt="image" src="https://github.com/user-attachments/assets/8feabcb6-c3e1-46c7-8a79-a9711b1c7027" />
<img width="524" height="361" alt="image" src="https://github.com/user-attachments/assets/4b3ef3c1-4ab9-4e7b-98a8-ffad3a304623" />

---

### Program 3: Story Generation Chatbot with Groq API
- Built interactive story generator using Groq's LLaMA 3 model
- Implemented user prompt handling with adjustable creativity parameters
- Set up temperature (1.5) for enhanced creative output
- Created continuous conversation loop with exit functionality
- Generated coherent stories based on user prompts like studying abroad narratives

**Screenshot**

<img width="598" height="558" alt="image" src="https://github.com/user-attachments/assets/cbf1da1c-4581-4b4b-bf60-2ac00c1f404d" />
<img width="595" height="752" alt="image" src="https://github.com/user-attachments/assets/0a254eed-98c4-4710-bca8-bca07ee4e460" />

---

### Program 4: Sentiment Analysis with BERT
- Fine-tuned BERT for sequence classification on financial sentiment dataset
- Preprocessed text data with BERT tokenizer
- Implemented custom dataset class for PyTorch DataLoader
- Configured AdamW optimizer with linear learning rate scheduler
- Achieved 78% validation accuracy on positive/neutral/negative sentiment classification
- Saved fine-tuned model for inference on new financial texts

**Screenshot**

<img width="549" height="199" alt="image" src="https://github.com/user-attachments/assets/474c6f28-5fd2-4ed4-b123-5a68b5329d54" />
<img width="556" height="120" alt="image" src="https://github.com/user-attachments/assets/d13e3a25-5b91-4a08-a279-df5f2f96dbcb" />
<img width="400" height="247" alt="image" src="https://github.com/user-attachments/assets/cf52b0a4-8ae6-4db8-b026-aab43c8e5667" />

---

### Program 5: Named Entity Recognition (NER)
- Fine-tuned BERT for token classification on CoNLL-2003 dataset
- Implemented label alignment for subword tokenization
- Configured training with dynamic padding using DataCollator
- Achieved low validation loss (0.0366) after 3 epochs
- Tested model on sentences to identify entities (ORG, LOC, MISC)
- Successfully identified "Hugging Face Inc." as organization and "New York City" as location

**Screenshot**

<img width="449" height="198" alt="image" src="https://github.com/user-attachments/assets/f4556771-9cf6-4126-b8da-46da36930126" />
<img width="518" height="175" alt="image" src="https://github.com/user-attachments/assets/436600c2-19cc-46b7-acf2-2f87c14a47ed" />
<img width="561" height="134" alt="image" src="https://github.com/user-attachments/assets/b7cce766-1002-4ff8-b1de-780dee922327" />

---

### Program 6: Machine Translation with T5
- Fine-tuned T5-small model on WMT14 English-German dataset
- Implemented seq2seq training with Seq2SeqTrainer
- Configured text generation parameters (beams, max_length)
- Achieved validation loss of 0.1447 after training
- Tested translation on sample sentences
- Successfully translated "The house is wonderful" to "Das Haus ist wunderbar"

**Screenshot**

<img width="541" height="186" alt="image" src="https://github.com/user-attachments/assets/9302b37b-6a99-47c8-8e7f-01bf1c4ebef5" />
<img width="508" height="240" alt="image" src="https://github.com/user-attachments/assets/b6f12d29-ba19-49a6-97cb-da088f6ea04d" />

---

### Program 7: Domain-Specific Tokenization
- Processed Spotify lyrics dataset with custom text cleaning
- Implemented regex-based preprocessing to remove special characters
- Tokenized lyrics using DistilBERT tokenizer
- Generated encoded tensors with attention masks
- Prepared domain-specific text data for NLP tasks
- Handled variable-length sequences with padding/truncation

**Screenshot**

<img width="545" height="299" alt="image" src="https://github.com/user-attachments/assets/89a3f832-469f-4799-90a9-2d660ffd6131" />
<img width="481" height="270" alt="image" src="https://github.com/user-attachments/assets/182df95b-0ab3-4152-a4c3-bec27a467980" />

---

### Program 8: Multimodal Input Processing
- Implemented multimodal input handling with CLIP model
- Processed both text and image inputs for similarity scoring
- Calculated image-to-text and text-to-image similarity scores
- Identified most relevant text description for given image
- Visualized results with image display and caption
- Successfully matched cat image with "A photo of a cat" description

**Screenshot**

<img width="528" height="505" alt="image" src="https://github.com/user-attachments/assets/c5d08282-936a-483f-904e-88f5fb4a460f" />


## Key Skills Demonstrated
1. Model fine-tuning on specific tasks
2. Prompt engineering and optimization
3. API integration with commercial LLM services
4. Sentiment analysis and classification
5. Named Entity Recognition
6. Machine translation
7. Domain-specific preprocessing
8. Multimodal AI applications
