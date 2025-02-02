import os
import numpy as np
import pandas as pd
import librosa
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from datasets import load_dataset, DatasetDict, Dataset
from transformers import AutoProcessor, AutoModelForCTC, Wav2Vec2Processor
from pydub import AudioSegment
from IPython.display import Audio, display
from wordcloud import WordCloud
from sklearn.model_selection import train_test_split

# File paths
csv_file = '../common_voice/cv-valid-train.csv'
audio_dir = '../common_voice/cv-valid-train'

# Load the CSV file
df = pd.read_csv(csv_file)

# Display dataset overview
print('Dataset Overview:')
print(df.head())

# Train-Validation Split (70%-30%)
train_df, val_df = train_test_split(df, test_size=0.3, random_state=123)

# Save the splits if needed
train_df.to_csv('train_split.csv', index=False)
val_df.to_csv('val_split.csv', index=False)

# Display number of samples in each split
print(f'Number of data in csv: {len(df)}')
print(f'Number of data in training: {len(train_df)}')
print(f'Number of data in validation: {len(val_df)}')

# Load dataset from Hugging Face
common_voice = DatasetDict()
common_voice['train'] = load_dataset('mozilla-foundation/common_voice_11_0', 'en', split='train+validation')

# Perform a custom split (70% train, 30% validation)
split_datasets = common_voice['train'].train_test_split(test_size=0.3, seed=0)
train_dataset = split_datasets['train']
val_dataset = split_datasets['test']

print(f'Train size: {len(train_dataset)}, Validation size: {len(val_dataset)}')

# Function to preprocess audio files
def preprocess_audio(example):
    file_path = os.path.join(audio_dir, example['filename'])
    try:
        # Load the audio file using librosa
        speech_array, sampling_rate = librosa.load(file_path, sr=16000)
        example['input_values'] = processor(speech_array, sampling_rate=16000).input_values[0]
        example['labels'] = processor.tokenizer(example['text']).input_ids  # Convert text to tokenized labels
    except Exception as e:
        print(f'Error loading {file_path}: {e}')
        example['input_values'] = None
        example['labels'] = None
    return example

# Apply preprocessing function
train_dataset = train_dataset.map(preprocess_audio, remove_columns=['filename', 'text'])
val_dataset = val_dataset.map(preprocess_audio, remove_columns=['filename', 'text'])

print(train_dataset)
print(val_dataset)

# Implementing a custom collator for CTC tasks
class CustomCTCCollator:
    def __init__(self, processor, padding=True, max_length=None, pad_to_multiple_of=None, return_tensors='pt'):
        self.processor = processor
        self.padding = padding
        self.max_length = max_length
        self.pad_to_multiple_of = pad_to_multiple_of
        self.return_tensors = return_tensors

    def __call__(self, features):
        # Extract and pad input features using the feature extractor
        input_features = [{'input_values': feature['input_values']} for feature in features]
        batch = self.processor.feature_extractor.pad(
            input_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors
        )

        # Pad labels manually so that they are of uniform length
        labels = [feature['labels'] for feature in features]
        max_label_len = max(len(label) for label in labels)
        padded_labels = [
            label + [self.processor.tokenizer.pad_token_id] * (max_label_len - len(label))
            for label in labels
        ]
        batch['labels'] = torch.tensor(padded_labels, dtype=torch.long)
        return batch

# Force PyTorch to use Apple Silicon
device = torch.device('mps')
torch.set_default_device(device)  # Ensures all new tensors are on CPU

from transformers import Wav2Vec2ForCTC, Trainer, TrainingArguments

# Load model and processor
model_dir = 'model'
processor = AutoProcessor.from_pretrained(model_dir)
model = AutoModelForCTC.from_pretrained(model_dir)

data_collator = CustomCTCCollator(processor=processor, padding=True)

# Define training arguments
training_args = TrainingArguments(
    output_dir='./wav2vec2-finetuned',
    eval_strategy='epoch',
    learning_rate=1e-4,
    per_device_train_batch_size=16,
    gradient_accumulation_steps=2,
    num_train_epochs=3,
    save_steps=500,
    logging_steps=100,
    save_total_limit=2,
    fp16=False,
    use_cpu=True,
    remove_unused_columns=False,
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    processing_class=processor.feature_extractor,
    data_collator=data_collator
)

# Start training
trainer.train()
print('Training complete!')


# Function to plot training metrics
def plot_training_metrics(log_history):
    '''Plots training loss, learning rate, and evaluation loss over time.'''
    train_logs = [log for log in log_history if 'loss' in log and 'eval_loss' not in log]
    eval_logs = [log for log in log_history if 'eval_loss' in log]

    # Extract training metrics
    train_steps = [log['step'] for log in train_logs if 'step' in log]
    train_loss = [log['loss'] for log in train_logs if 'loss' in log]
    train_lr = [log.get('learning_rate') for log in train_logs if 'learning_rate' in log]

    # Extract evaluation metrics
    eval_steps = [log['step'] for log in eval_logs if 'step' in log]
    eval_loss = [log['eval_loss'] for log in eval_logs if 'eval_loss' in log]

    # Create subplots
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))

    # Training Loss vs Step
    axs[0, 0].plot(train_steps, train_loss, label='Training Loss', color='blue')
    axs[0, 0].set_title('Training Loss vs Step')
    axs[0, 0].set_xlabel('Step')
    axs[0, 0].set_ylabel('Loss')
    axs[0, 0].legend()

    # Learning Rate vs Step
    axs[0, 1].plot(train_steps, train_lr, label='Learning Rate', color='green')
    axs[0, 1].set_title('Learning Rate vs Step')
    axs[0, 1].set_xlabel('Step')
    axs[0, 1].set_ylabel('Learning Rate')
    axs[0, 1].legend()

    # Evaluation Loss vs Step
    axs[1, 0].plot(eval_steps, eval_loss, label='Eval Loss', color='red')
    axs[1, 0].set_title('Evaluation Loss vs Step')
    axs[1, 0].set_xlabel('Step')
    axs[1, 0].set_ylabel('Evaluation Loss')
    axs[1, 0].legend()

    # Hide empty subplot
    axs[1, 1].axis('off')

    plt.tight_layout()
    plt.show()

# Plot training metrics after training
plot_training_metrics(trainer.state.log_history)

# Model Evaluation using WER (Word Error Rate)
from misc.evaluation_metrics import compute_metrics

pretrained_model = Wav2Vec2ForCTC.from_pretrained('model') 
finetuned_model = Wav2Vec2ForCTC.from_pretrained('wav2vec2-finetuned') 

# Define evaluation arguments
eval_args = TrainingArguments(
    output_dir='./eval_output',
    per_device_eval_batch_size=16,
    fp16=False,
    no_cuda=True,
    remove_unused_columns=False,
    evaluation_strategy='no'
)

# Trainer instances for evaluation
pretrained_trainer = Trainer(
    model=pretrained_model,
    args=eval_args,
    eval_dataset=val_dataset,
    processing_class=processor.feature_extractor,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)

finetuned_trainer = Trainer(
    model=finetuned_model,
    args=eval_args,
    eval_dataset=val_dataset,
    processing_class=processor.feature_extractor,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)

# Evaluate the pre-trained model
print('Evaluating pre-trained model:')
pretrained_eval = pretrained_trainer.evaluate()
print(pretrained_eval)

# Evaluate the fine-tuned model
print('Evaluating fine-tuned model:')
finetuned_eval = finetuned_trainer.evaluate()
print(finetuned_eval)
