# Daily Task Tracker - CS7643 Project

## Week 1: Nov 16-23 (Enhanced Captioning Foundation)

### Team Member 1: LoRA & Model Architecture

#### Monday Nov 16 (Day 1) - 6 hours
**Tasks:**
- [ ] 2h: Study and understand current model architecture
- [ ] 3h: Implement basic LoRA linear layer class
- [ ] 1h: Write unit tests for LoRA implementation

**Code to write:**
- Complete `LoRALinear` class in `models/lora_vit.py`
- Test script: `test_lora_basic.py`

**Success criteria:** LoRA layer can wrap existing linear layers

#### Tuesday Nov 17 (Day 2) - 6 hours  
**Tasks:**
- [ ] 4h: Implement `apply_lora_to_vit` function
- [ ] 2h: Test LoRA integration with existing model

**Code to write:**
- Complete `apply_lora_to_vit()` function
- Integration test with current training pipeline

**Success criteria:** Can load model and apply LoRA without errors

#### Wednesday Nov 18 (Day 3) - 6 hours
**Tasks:**
- [ ] 3h: Implement beam search generation
- [ ] 3h: Implement nucleus sampling generation

**Code to write:**
- Complete `beam_search_generate()` in `generation/advanced_sampling.py`
- Complete `nucleus_sampling_generate()` function

**Success criteria:** Generate captions with both methods

#### Thursday Nov 19 (Day 4) - 6 hours
**Tasks:**
- [ ] 3h: Implement parameter sweep utilities
- [ ] 3h: Test generation strategies comparison

**Code to write:**
- Complete `GenerationComparator` class
- Parameter sweep functionality

**Success criteria:** Can compare different generation strategies

#### Friday Nov 20 (Day 5) - 3 hours
**Tasks:**
- [ ] 3h: Run ablation experiments (frozen vs LoRA vs full fine-tuning)

**Code to write:**
- Experiment runner script
- Results logging

**Success criteria:** Have training results for 3 different approaches

#### Weekend Nov 21-22 (4-5 hours)
**Tasks:**
- [ ] 2-3h: Document parameter counts and training speeds
- [ ] 2h: Write analysis of results

**Code to write:**
- Results analysis notebook
- Documentation

**Success criteria:** Complete LoRA analysis with quantitative results

---

### Team Member 2: Training & Evaluation Systems

#### Monday Nov 16 (Day 1) - 6 hours
**Tasks:**
- [ ] 2h: Install and test NLTK, rouge dependencies
- [ ] 3h: Implement BLEU score computation
- [ ] 1h: Test BLEU implementation with known examples

**Code to write:**
- Complete `compute_bleu_scores()` in `evaluation/metrics.py`
- Unit tests for BLEU

**Success criteria:** BLEU scores match expected values

#### Tuesday Nov 17 (Day 2) - 6 hours
**Tasks:**
- [ ] 3h: Implement METEOR and ROUGE scores
- [ ] 3h: Create unified metrics evaluation function

**Code to write:**
- Complete `compute_meteor_score()` and `compute_rouge_scores()`
- `compute_all_metrics()` function

**Success criteria:** All metrics working on test data

#### Wednesday Nov 18 (Day 3) - 6 hours
**Tasks:**
- [ ] 4h: Implement label smoothing in training
- [ ] 2h: Add scheduled sampling implementation

**Code to write:**
- Label smoothing loss function
- Scheduled sampling training logic

**Success criteria:** Training with both techniques works

#### Thursday Nov 19 (Day 4) - 6 hours
**Tasks:**
- [ ] 3h: Create experiment tracking system
- [ ] 3h: Build automated evaluation pipeline

**Code to write:**
- Experiment logger class
- Automated evaluation script

**Success criteria:** Can track and log experiment results

#### Friday Nov 20 (Day 5) - 3 hours
**Tasks:**
- [ ] 3h: Run training experiments with different strategies

**Code to write:**
- Training comparison script

**Success criteria:** Compare baseline vs enhanced training

#### Weekend Nov 21-22 (4-5 hours)
**Tasks:**
- [ ] 3h: Generate comparison tables and plots
- [ ] 2h: Create evaluation dashboard

**Code to write:**
- Results visualization notebook
- Performance dashboard

**Success criteria:** Clear comparison of training strategies

---

### Team Member 3: Data Pipeline & Robustness

#### Monday Nov 16 (Day 1) - 6 hours
**Tasks:**
- [ ] 3h: Research Flickr30k dataset structure
- [ ] 3h: Implement basic Flickr30k loading

**Code to write:**
- Flickr30k data parsing in `data/multi_dataset_loader.py`
- Test loading script

**Success criteria:** Can load Flickr30k captions

#### Tuesday Nov 17 (Day 2) - 6 hours
**Tasks:**
- [ ] 4h: Complete unified dataset loader
- [ ] 2h: Test dataset combination functionality

**Code to write:**
- Complete `MultiDatasetLoader` class
- Dataset combination tests

**Success criteria:** Can combine Flickr8k and Flickr30k

#### Wednesday Nov 18 (Day 3) - 6 hours
**Tasks:**
- [ ] 4h: Implement image corruption pipeline
- [ ] 2h: Add data augmentation functionality

**Code to write:**
- Image corruption functions
- Data augmentation utilities

**Success criteria:** Can apply corruptions to images

#### Thursday Nov 19 (Day 4) - 6 hours
**Tasks:**
- [ ] 3h: Create robustness testing framework
- [ ] 3h: Implement corruption evaluation metrics

**Code to write:**
- Robustness test suite
- Corruption severity analysis

**Success criteria:** Can measure robustness to corruptions

#### Friday Nov 20 (Day 5) - 3 hours
**Tasks:**
- [ ] 3h: Run baseline robustness experiments

**Code to write:**
- Robustness experiment runner

**Success criteria:** Have baseline robustness measurements

#### Weekend Nov 21-22 (4-5 hours)
**Tasks:**
- [ ] 3h: Create data analysis visualizations
- [ ] 2h: Document dataset statistics

**Code to write:**
- Data analysis notebook
- Statistics visualization

**Success criteria:** Comprehensive data analysis complete

---

## Daily Standup Questions (Answer these each day):

1. What did I complete yesterday?
2. What am I working on today?
3. Are there any blockers?
4. Do I need help from other team members?
5. Am I on track with my weekly goals?

## Weekly Check-ins (Every Friday):

1. Code review with team members
2. Integration testing of components
3. Progress assessment vs. timeline
4. Plan adjustments for next week

## Success Metrics for Week 1:

**Team Member 1:**
- [ ] LoRA implementation working
- [ ] 3 generation strategies implemented
- [ ] Ablation study results available

**Team Member 2:**
- [ ] 5 evaluation metrics implemented
- [ ] Enhanced training strategies working
- [ ] Experiment tracking system ready

**Team Member 3:**
- [ ] Multi-dataset loading working
- [ ] Robustness testing framework ready
- [ ] Baseline robustness experiments complete

**Overall Team:**
- [ ] All components integrate successfully
- [ ] Enhanced captioning pipeline works end-to-end
- [ ] Clear improvements over baseline demonstrated