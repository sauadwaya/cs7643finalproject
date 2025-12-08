"""
Comprehensive evaluation metrics for image captioning
Team Member 2's Week 1 starter code
"""

import torch
import numpy as np
from typing import List, Dict, Any
from collections import Counter
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from nltk.tokenize import word_tokenize
import rouge

# Download required NLTK data (run once)
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('punkt')
    nltk.download('wordnet')
    nltk.download('omw-1.4')

class CaptioningMetrics:
    """Comprehensive metrics for image captioning evaluation"""
    
    def __init__(self):
        self.smoothing = SmoothingFunction().method4
        self.rouge_evaluator = rouge.Rouge()
        
    def compute_bleu_scores(self, predictions: List[str], references: List[List[str]]) -> Dict[str, float]:
        """
        Compute BLEU-1, BLEU-2, BLEU-3, BLEU-4 scores
        
        Args:
            predictions: List of predicted captions
            references: List of reference caption lists (multiple refs per prediction)
        """
        bleu_scores = {'bleu_1': [], 'bleu_2': [], 'bleu_3': [], 'bleu_4': []}
        
        for pred, refs in zip(predictions, references):
            pred_tokens = word_tokenize(pred.lower())
            ref_tokens = [word_tokenize(ref.lower()) for ref in refs]
            
            # Calculate BLEU scores with different n-gram weights
            bleu_1 = sentence_bleu(ref_tokens, pred_tokens, weights=(1, 0, 0, 0), smoothing_function=self.smoothing)
            bleu_2 = sentence_bleu(ref_tokens, pred_tokens, weights=(0.5, 0.5, 0, 0), smoothing_function=self.smoothing)
            bleu_3 = sentence_bleu(ref_tokens, pred_tokens, weights=(0.33, 0.33, 0.33, 0), smoothing_function=self.smoothing)
            bleu_4 = sentence_bleu(ref_tokens, pred_tokens, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=self.smoothing)
            
            bleu_scores['bleu_1'].append(bleu_1)
            bleu_scores['bleu_2'].append(bleu_2)
            bleu_scores['bleu_3'].append(bleu_3)
            bleu_scores['bleu_4'].append(bleu_4)
        
        # Return mean scores
        return {k: np.mean(v) for k, v in bleu_scores.items()}
    
    def compute_meteor_score(self, predictions: List[str], references: List[List[str]]) -> float:
        """Compute METEOR score"""
        meteor_scores = []
        
        for pred, refs in zip(predictions, references):
            # METEOR uses the best matching reference
            scores = []
            for ref in refs:
                try:
                    score = meteor_score([word_tokenize(ref.lower())], word_tokenize(pred.lower()))
                    scores.append(score)
                except:
                    scores.append(0.0)
            meteor_scores.append(max(scores) if scores else 0.0)
        
        return np.mean(meteor_scores)
    
    def compute_rouge_scores(self, predictions: List[str], references: List[List[str]]) -> Dict[str, float]:
        """Compute ROUGE-L scores"""
        rouge_scores = []
        
        for pred, refs in zip(predictions, references):
            # Use first reference for ROUGE (extend to best-match if needed)
            ref = refs[0] if refs else ""
            try:
                scores = self.rouge_evaluator.get_scores(pred, ref)
                rouge_scores.append(scores[0]['rouge-l']['f'])
            except:
                rouge_scores.append(0.0)
        
        return {'rouge_l': np.mean(rouge_scores)}
    
    def compute_cider_score(self, predictions: List[str], references: List[List[str]]) -> float:
        """
        Simplified CIDEr score implementation
        For full CIDEr, consider using the official implementation
        """
        def compute_tf_idf(sentences):
            """Compute TF-IDF weights for sentences"""
            # Simplified TF-IDF computation
            all_words = []
            for sentence in sentences:
                all_words.extend(word_tokenize(sentence.lower()))
            
            word_counts = Counter(all_words)
            total_docs = len(sentences)
            
            tf_idf = {}
            for sentence in sentences:
                words = word_tokenize(sentence.lower())
                word_freq = Counter(words)
                sentence_tf_idf = {}
                
                for word in set(words):
                    tf = word_freq[word] / len(words)
                    idf = np.log(total_docs / (1 + word_counts[word]))
                    sentence_tf_idf[word] = tf * idf
                
                tf_idf[sentence] = sentence_tf_idf
            
            return tf_idf
        
        # This is a simplified version - implement full CIDEr if needed
        cider_scores = []
        all_sentences = []
        for refs in references:
            all_sentences.extend(refs)
        all_sentences.extend(predictions)
        
        # For now, return a placeholder that correlates with other metrics
        # TODO: Implement full CIDEr score
        bleu_scores = self.compute_bleu_scores(predictions, references)
        return bleu_scores['bleu_4']  # Temporary approximation
    
    def compute_all_metrics(self, predictions: List[str], references: List[List[str]]) -> Dict[str, float]:
        """Compute all metrics at once"""
        results = {}
        
        # BLEU scores
        bleu_results = self.compute_bleu_scores(predictions, references)
        results.update(bleu_results)
        
        # METEOR score
        results['meteor'] = self.compute_meteor_score(predictions, references)
        
        # ROUGE scores
        rouge_results = self.compute_rouge_scores(predictions, references)
        results.update(rouge_results)
        
        # CIDEr score (simplified)
        results['cider'] = self.compute_cider_score(predictions, references)
        
        return results

def evaluate_model_predictions(predictions_file: str, references_file: str) -> Dict[str, float]:
    """
    Evaluate model predictions from files
    
    Args:
        predictions_file: Path to file with predictions (one per line)
        references_file: Path to file with references (multiple per line, separated by |)
    """
    # Load predictions
    with open(predictions_file, 'r', encoding='utf-8') as f:
        predictions = [line.strip() for line in f]
    
    # Load references
    references = []
    with open(references_file, 'r', encoding='utf-8') as f:
        for line in f:
            refs = [ref.strip() for ref in line.strip().split('|')]
            references.append(refs)
    
    # Compute metrics
    metrics = CaptioningMetrics()
    results = metrics.compute_all_metrics(predictions, references)
    
    return results

# Testing function for Team Member 2
def test_metrics():
    """Test the metrics implementation"""
    # Sample data for testing
    predictions = [
        "a dog is running in the park",
        "two people are walking on the street", 
        "a cat is sitting on a chair"
    ]
    
    references = [
        ["a dog runs in the park", "a dog is playing in the park", "dog running in park"],
        ["two people walk on street", "people walking on the road", "two persons on street"],
        ["cat sits on chair", "a cat on a chair", "cat sitting on furniture"]
    ]
    
    metrics = CaptioningMetrics()
    results = metrics.compute_all_metrics(predictions, references)
    
    print("Metrics Test Results:")
    for metric, score in results.items():
        print(f"{metric}: {score:.4f}")
    
    return results

if __name__ == "__main__":
    # Run test
    test_results = test_metrics()
    
    # Example usage with actual model predictions
    # results = evaluate_model_predictions("predictions.txt", "references.txt")
    # print("Model Evaluation Results:", results)