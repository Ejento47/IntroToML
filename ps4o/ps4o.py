import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence

import numpy as np
import matplotlib.pyplot as plt

# Set seeds for reproducibility
torch.manual_seed(2109)
np.random.seed(2109)

# Check if GPU is available
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

### Task 1.1: RNN Cell

def rnn_cell_forward(xt, h_prev, Wxh, Whh, Why, bh, by):
    """
    Implements a single forward step of the RNN-cell

    Args:
        xt: 2D tensor of shape (nx, m)
            Input data at timestep "t"
        h_prev: 2D tensor of shape (nh, m)
            Hidden state at timestep "t-1"
        Wxh: 2D tensor of shape (nx, nh)
            Weight matrix multiplying the input
        Whh: 2D tensor of shape (nh, nh)
            Weight matrix multiplying the hidden state
        Why: 2D tensor of shape (nh, ny)
            Weight matrix relating the hidden-state to the output
        bh: 1D tensor of shape (nh, 1)
            Bias relating to next hidden-state
        by: 2D tensor of shape (ny, 1)
            Bias relating the hidden-state to the output

    Returns:
        yt_pred -- prediction at timestep "t", tensor of shape (ny, m)
        h_next -- next hidden state, of shape (nh, m)
    """
    """ YOUR CODE HERE """
    raise NotImplementedError
    """ YOUR CODE END HERE """

def test_task_1_1():
    public_paras = {
        'xt': torch.tensor([[1., 1., 2.], [2., 1., 3.], [3., 5., 3.]]),
        'h_prev': torch.tensor([[5., 3., 2.], [1., 3., 2.]]),
        'Wxh': torch.tensor([[2., 2.], [3., 4.], [4., 3.]]),
        'Whh': torch.tensor([[2., 4.], [2., 3.]]),
        'Why': torch.tensor([[3., 5.], [5., 4.]]),
        'bh': torch.tensor([[1.], [2.]]),
        'by': torch.tensor([[3.], [1.]]),
    }
    
    expected_yt_pred = torch.tensor([[0.7311, 0.7311, 0.7311], [0.2689, 0.2689, 0.2689]])
    expected_h_next = torch.tensor([[1., 1., 1.], [1., 1., 1.]])
    
    actual_yt_pred, actual_h_next = rnn_cell_forward(**public_paras)
    assert torch.allclose(actual_yt_pred, expected_yt_pred, atol=1e-4)
    assert torch.allclose(actual_h_next, expected_h_next, atol=1e-4)

### Task 1.2: Generate Sine Wave Data

def generate_sine_wave(num_time_steps):
    """
    Generates a sine wave data

    Args:
        num_time_steps: int
            Number of time steps
    Returns:
        data: 1D tensor of shape (num_time_steps,)
            Sine wave data with corresponding time steps
    """
    """ YOUR CODE HERE """
    raise NotImplementedError
    """ YOUR CODE END HERE """

def test_task_1_2():
    num_time_steps_public = 5
    expected_data_public = torch.tensor([0.0000e+00, 1.7485e-07, 3.4969e-07, 4.7700e-08, 6.9938e-07])
    actual_data = generate_sine_wave(num_time_steps_public)
    
    assert torch.allclose(actual_data, expected_data_public)

num_time_steps = 500
sine_wave_data = generate_sine_wave(num_time_steps)

# Plot the sine wave
plt.plot(sine_wave_data)
plt.title('Sine Wave')
plt.show()

### Task 1.3: Create sequences

def create_sequences(sine_wave, seq_length):
    """
    Create overlapping sequences from the input time series and generate labels 
    Each label is the value immediately following the corresponding sequence.
    
    Args:
        sine_wave: A 1D tensor representing the time series data (e.g., sine wave).
        seq_length: int. The length of each sequence (window) to be used as input to the RNN.

    Returns: 
        windows: 2D tensor where each row is a sequence (window) of length `seq_length`.
        labels: 1D tensor where each element is the next value following each window.
    """
    """ YOUR CODE HERE """
    raise NotImplementedError
    """ YOUR CODE END HERE """

def test_task_1_3():
    seq_length_test = 2
    sine_wave_test = torch.tensor([0., 1., 2., 3.])
    expected_sequences = torch.tensor([[0., 1.], [1., 2.]])
    expected_labels = torch.tensor([2., 3.])
    
    actual_sequences, actual_labels = create_sequences(sine_wave_test, seq_length_test)
    assert torch.allclose(actual_sequences, expected_sequences)
    assert torch.allclose(actual_labels, expected_labels)

# Create sequences and labels
seq_length = 20
sequences, labels = create_sequences(sine_wave_data, seq_length)
# Add extra dimension to match RNN input shape [batch_size, seq_length, num_features]
sequences = sequences.unsqueeze(-1)
sequences.shape

# Split the sequences into training data (first 80%) and test data (remaining 20%) 
train_size = int(len(sequences) * 0.8)
train_seqs, train_labels = sequences[:train_size], labels[:train_size]
test_seqs, test_labels = sequences[train_size:], labels[train_size:]

### Task 1.4: Building RNN Model

class SineRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        """
        Initialize the SineRNN model.

        Args:
            input_size (int): The number of input features per time step (typically 1 for univariate time series).
            hidden_size (int): The number of units in the RNN's hidden layer.
            output_size (int): The size of the output (usually 1 for predicting a single value).
        """
        super(SineRNN, self).__init__()
        """ YOUR CODE HERE """
        raise NotImplementedError
        """ YOUR CODE END HERE """
        
    def forward(self, x):
        """ YOUR CODE HERE """
        raise NotImplementedError
        """ YOUR CODE END HERE """

def test_task_1_4():
    input_size = output_size = 1
    hidden_size = 50
    model = SineRNN(input_size, hidden_size, output_size).to(device)
    assert [layer.detach().numpy().shape for _, layer in model.named_parameters()]\
          == [(50, 1), (50, 50), (50,), (50,), (1, 50), (1,)]

# Define loss function, and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Training loop
num_epochs = 200
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()

    # Forward pass
    outputs = model(train_seqs)
    loss = criterion(outputs.squeeze(), train_labels)

    # Backward pass and optimization
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 20 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.6f}')

# Predict on unseen data
model.eval()
y_pred = []
input_seq = test_seqs[0]  # Start with the first testing sequence

with torch.no_grad():
    for _ in range(len(test_seqs)):
        output = model(input_seq)
        y_pred.append(output.item())
        
        # Use the predicted value as the next input sequence
        next_seq = torch.cat((input_seq[1:, :], output.unsqueeze(0)), dim=0)
        input_seq = next_seq

# Plot the true sine wave and predictions
plt.plot(sine_wave_data, c='gray', label='Actual data')
plt.scatter(np.arange(seq_length + len(train_labels)), sine_wave_data[:seq_length + len(train_labels)], marker='.', label='Train')
x_axis_pred = np.arange(len(sine_wave_data) - len(test_labels), len(sine_wave_data))
plt.scatter(x_axis_pred, y_pred, marker='.', label='Predicted')
plt.legend(loc="lower left")
plt.show()

### Task 2.1: LSTM Cell

def lstm_cell_forward(xt, h_prev, c_prev, Wf, bf, Wi, bi, Wc, bc, Wo, bo):
    """
    Implement a single forward step of the LSTM cell

    Args:
        xt: 2D tensor of shape (nx, m)
            Input data at timestep "t"
        h_prev: 2D tensor of shape (nh, m)
            Hidden state at timestep "t-1"
        c_prev: 2D tensor of shape (nh, m)
            Memory state at timestep "t-1"
        Wf: tensor of shape(nh, nh + nx) 
            Weight matrix of the forget gate
        bf: tensor of shape (nh, 1)
            Bias of the forget gate
        Wi: tensor of shape (nh, nh + nx)
            Weight matrix of the input gate
        bi: tensor of shape (nh, 1)
            Bias of the input gate
        Wc: tensor of shape (nh, nh + nx)
            Weight matrix of candidate value
        bc: tensor of shape (nh, 1)
            Bias of the candidate value
        Wo: tensor of shape (nh, nh + nx)
            Weight matrix of the output gate
        bo: tensor of shape (nh, 1) 
            Bias of the output gate
    
    Returns:
        h_next: 2D tensor of shape (nh, m), next hidden state
        c_next: 2D tensor of shape (nh, m), next memory state
    """
    """ YOUR CODE HERE """
    raise NotImplementedError
    """ YOUR CODE END HERE """

def test_task_2_1():
    public_paras = {
        'xt': torch.tensor([[1.], [2.], [3.]]),
        'h_prev': torch.tensor([[1.], [2.]]),
        'c_prev': torch.tensor([[1.], [2.]]),
        'Wf': torch.tensor([[1., 1., 1., 1., 1.], [1., 1., 1., 1., 1.]]),
        'bf': torch.tensor([[1.], [1.]]),
        'Wi': torch.tensor([[1., 1., 1., 1., 1.], [1., 1., 1., 1., 1.]]),
        'bi': torch.tensor([[1.], [1.]]),
        'Wc': torch.tensor([[1., 1., 1., 1., 1.], [1., 1., 1., 1., 1.]]),
        'bc': torch.tensor([[1.], [1.]]),
        'Wo': torch.tensor([[1., 1., 1., 1., 1.], [1., 1., 1., 1., 1.]]),
        'bo': torch.tensor([[1.], [1.]])
    }
    expected_h_next = torch.tensor([[0.9640], [0.9950]])
    expected_c_next = torch.tensor([[1.9999], [2.9999]])
    
    actual_h_next, actual_c_next = lstm_cell_forward(**public_paras)
    assert torch.allclose(actual_h_next, expected_h_next, atol=1e-4)
    assert torch.allclose(actual_c_next, expected_c_next, atol=1e-4)

# Load data from files
train_df = pd.read_csv('data/review_train.csv')
test_df = pd.read_csv('data/review_test.csv')
train_df.head(10)

train_df['label'].value_counts()

class CustomDataset(Dataset):
    def __init__(self, data_file):
        self.data = pd.read_csv(data_file)
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        text = self.data.iloc[idx]['review']
        label = self.data.iloc[idx]['label']
        return text, label

train_dataset = CustomDataset('data/review_train.csv')
test_dataset = CustomDataset('data/review_test.csv')

# Look at a sample review and label
idx = 28
print("Review: ", train_dataset.__getitem__(idx)[0])
print("Label: ", train_dataset.__getitem__(idx)[1])

# ! pip install torchtext

from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

tokenizer = get_tokenizer("basic_english", language="en")
min_word_freq=2 # Words must appear at least twice to be included in the vocabulary
PAD_TOKEN = '<pad>' # padding token
UNK_TOKEN = '<unk>' # unknown token
SPECIALS = [PAD_TOKEN, UNK_TOKEN]

def build_vocab(dataset, tokenizer):
    texts = [text for text , _ in dataset]
    vocab = build_vocab_from_iterator(
        map(tokenizer, texts),
        specials=SPECIALS,
        min_freq=min_word_freq
    )
    vocab.set_default_index(vocab["<unk>"])
    return vocab

vocab = build_vocab(train_dataset,tokenizer)
vocab_size = vocab.__len__()
print("Vocabulary size: ", vocab_size)

# The pipeline is a lambda function that takes an string x
# and applies two transformations: tokenize the input and then
# maps each token to its corresponding index from the vocabulary.
# 
# When the string 'Hello world!' is passed to the pipeline, 
# it is tokenized into ['hello', 'world', '!'].
# 
# The output [1, 167, 204] indicates that 'Hello' maps to index 1, 
# 'world' to 167, and '!' is mapped to index 204.
#  
# Because 1 is unknown token (<unk>), it suggests 'hello' is not in the vocabulary.
# In other words, 'hello' is not present in the training data.

pipeline = lambda x : vocab(tokenizer(x))
pipeline('Hello world!')

MAX_LENGTH = 100
def collator(batch):
    """
    Process a batch of text-label pairs, transform the text into a tensor, 
    and pad the sequences to have the same length capped by MAX_LENGTH.

    Args: 
        list of (text, label) pair 
    
    Returns: 
        A pair of tensors:
        - texts: a tensor of tokenized texts.
        - labels: a tensor of labels.
    """
    # Unzip the batch into sequences and labels
    sequences, labels = zip(*batch)
    # Apply a pipeline to each sequence and truncate to MAX_LENGTH
    truncated_seqs = [pipeline(seq)[:MAX_LENGTH] for seq in sequences]
    # Convert to tensor type int64 (needed as nn.Embedding takes input of IntTensor or LongTensor)
    truncated_seqs = [torch.tensor(seq, dtype=torch.int64) for seq in truncated_seqs]
    # Pad the sequences so they all have the same length and stack them into a single tensor
    texts = pad_sequence(truncated_seqs, batch_first=True, padding_value=0)
    # Convert the labels into a IntTensor (commonly used for classification)
    labels = torch.tensor(labels, dtype=torch.int64)
    
    return texts , labels

batch_size = 32
train_dataloader = DataLoader(train_dataset, batch_size=batch_size,shuffle=True, collate_fn=collator)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, collate_fn=collator)

### Task 2.2: Building the Text Classification Model

class TextClassificationModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_size, num_class):
        super().__init__()
        """ YOUR CODE HERE """
        raise NotImplementedError
        """ YOUR CODE END HERE """
        
    def forward(self, x):
        """ YOUR CODE HERE """
        raise NotImplementedError
        """ YOUR CODE END HERE """

def test_task_2_2():
    model = TextClassificationModel(10, 5, 6, 2)
    assert [layer.detach().numpy().shape for _, layer in model.named_parameters()] \
            == [(10, 5), (24, 5), (24, 6), (24,), (24,), (2, 6), (2,)]

embed_dim = 300
hidden_size = 128
num_class = 2
lr = 0.001
model = TextClassificationModel(vocab_size , embed_dim, hidden_size, num_class).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params = model.parameters() , lr=lr)

num_epochs = 5
model.train()
for epoch in range(num_epochs):  
    epoch_loss = 0.0  
    num_correct = 0

    for texts, labels in train_dataloader:
        texts, labels = texts.to(device), labels.to(device)  
        output = model(texts)

        loss = criterion(output, labels)
        epoch_loss += loss.item() 
        num_correct += (output.argmax(1) == labels).sum().item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    epoch_loss /= len(train_dataloader)  
    epoch_acc = (num_correct / len(train_dataloader.dataset)) * 100  
    print(f"Epoch {epoch + 1}, Train Loss: {epoch_loss:.4f}, Train Accuracy: {epoch_acc:.2f}%")

model.eval()  
num_correct = 0

with torch.no_grad():
    for texts, labels in test_dataloader:
        texts, labels = texts.to(device), labels.to(device)
        output = model(texts) 
        predictions = output.argmax(1)
        num_correct += (predictions == labels).sum().item()  

eval_acc = (num_correct / len(test_dataloader.dataset)) * 100
print(f"Validation Accuracy: {eval_acc:.2f}%")

def predict(input_text):
    model.eval()
    with torch.no_grad():
        tokens = torch.tensor(pipeline(input_text)).to(device)
        output = model(tokens)
        prediction = output.argmax()
        return "positive" if prediction== 1 else "negative"

my_review = "Write your review here"
predict(my_review)


if __name__ == '__main__':
    test_task_1_1()
    test_task_1_2()
    test_task_1_3()
    test_task_1_4()
    test_task_2_1()
    test_task_2_2()