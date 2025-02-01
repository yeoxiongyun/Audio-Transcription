'''
Evaluation Metrics

This code provides functions to compute and evaluate Word Error Rate (WER) metrics for speech recognition systems. It includes the ability to compute orthographic WER (strict evaluation) and normalized WER (with text normalization and filtering for non-zero references).

1. **Compute WER**:
    - `compute_wer(reference_texts, predicted_texts)` function:
        - Calculates the percentage of errors in predicted texts compared to reference texts.
        - Uses the `evaluate` library to compute WER.
    - Example:
        ```
        Orthographic WER: 20.0%
        ```

2. **Text Normalization**:
    - `normalize_texts(texts, normalizer)` function:
        - Applies normalization (e.g., lowercasing, punctuation removal) to a list of text strings.
    - Example:
        ```
        Input: ['Hello, World!']
        Output: ['hello world']
        ```

3. **Filter Non-Zero References**:
    - `filter_nonzero_references(predictions, references)` function:
        - Filters out text samples where the reference text is empty.
    - Ensures WER computations are meaningful by excluding invalid samples.

4. **Integrated Metrics Computation**:
    - `compute_metrics(pred)` function:
        - Computes orthographic WER and normalized WER for a batch of predictions and references.
        - Handles special cases, such as replacing padding tokens in the labels.

5. **Execution**:
    - Includes example usage for orthographic and normalized WER computation using example reference and predicted texts.

'''

from evaluate import load
from transformers.models.whisper.english_normalizer import BasicTextNormalizer
from typing import Dict, Any, List

# Initialize the text normalizer and WER metric
normalizer = BasicTextNormalizer()
wer_metric = load('wer')

def compute_wer(reference_texts: List[str], predicted_texts: List[str], round_digits: int = 5) -> float:
    '''
    Computes the Word Error Rate (WER) between reference texts and predicted texts.

    Args:
        reference_texts (list[str]): The ground truth reference sentences.
        predicted_texts (list[str]): The predicted sentences from the model.
        round_digits (int): The number of decimal places to round the computed WER.

    Returns:
        float: The WER score as a percentage, rounded to `round_digits` decimal places.
    '''
    return round(100 * wer_metric.compute(references=reference_texts, predictions=predicted_texts), round_digits) # type: ignore

def normalize_texts(texts: List[str], normalizer: BasicTextNormalizer) -> List[str]:
    '''
    Normalizes a list of texts using the given normalizer.

    Args:
        texts (list[str]): A list of text strings to normalize.
        normalizer (BasicTextNormalizer): The text normalizer to apply.

    Returns:
        list[str]: A list of normalized text strings.
    '''
    return [normalizer(text) for text in texts]


def filter_nonzero_references(
    predictions: List[str], references: List[str]
) -> tuple[List[str], List[str]]:
    '''
    Filters predictions and references to exclude samples with empty references.

    Args:
        predictions (list[str]): The list of predicted text strings.
        references (list[str]): The list of reference text strings.

    Returns:
        tuple[list[str], list[str]]: Filtered predictions and references.
    '''
    filtered_predictions = [
        predictions[i]
        for i in range(len(predictions))
        if len(references[i]) > 0
    ]
    filtered_references = [
        references[i]
        for i in range(len(references))
        if len(references[i]) > 0
    ]
    return filtered_predictions, filtered_references


def compute_metrics(pred: Any) -> Dict[str, float]:
    '''
    Computes evaluation metrics, including orthographic WER and normalized WER.

    Args:
        pred (Any): A prediction object with attributes `predictions` and `label_ids`.

    Returns:
        dict: A dictionary containing `wer_ortho` and `wer` scores.
    '''
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    # Replace -100 with the pad_token_id
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

    # Decode predictions and references
    pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = processor.batch_decode(label_ids, skip_special_tokens=True)

    # Compute orthographic WER
    wer_ortho = compute_wer(label_str, pred_str)

    # Compute normalized WER
    pred_str_norm = normalize_texts(pred_str, normalizer)
    label_str_norm = normalize_texts(label_str, normalizer)
    pred_str_norm, label_str_norm = filter_nonzero_references(
        pred_str_norm, label_str_norm
    )
    wer = compute_wer(label_str_norm, pred_str_norm)

    return {'wer_ortho': wer_ortho, 'wer': wer}


# Example data for orthographic WER computation
common_voice_test = {'sentence': ['the quick brown fox', 'jumps over the lazy dog']}
all_predictions = ['the quick fox', 'jumped over the dog']

# Compute orthographic WER
wer_ortho = compute_wer(
    reference_texts=common_voice_test['sentence'], 
    predicted_texts=all_predictions, 
    round_digits=2
)
print(f'Orthographic WER: {wer_ortho}%')

# Compute normalized WER
all_predictions_norm = normalize_texts(all_predictions, normalizer)
all_references_norm = normalize_texts(common_voice_test['sentence'], normalizer)
all_predictions_norm, all_references_norm = filter_nonzero_references(
    all_predictions_norm, all_references_norm
)
normalized_wer = compute_wer(
    reference_texts=all_references_norm, 
    predicted_texts=all_predictions_norm, 
    round_digits=2
)
print(f'Normalized WER: {normalized_wer}%')