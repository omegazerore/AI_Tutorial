"""
sunspot Eval Loss: 536
"""


import torch
import torch.nn as nn
import torch.optim as optim

class Seq2SeqWithAttention(nn.Module):
    def __init__(self, encoder_units, decoder_units, output_dim):
        super(Seq2SeqWithAttention, self).__init__()
        self.encoder = nn.LSTM(input_size=1, hidden_size=encoder_units, batch_first=True)
        # self.encoder = nn.LSTM(input_size=1, hidden_size=encoder_units, num_layers=2, batch_first=True)  # Added num_layers=2
        # Second LSTM layer with a different hidden size
        # self.encoder_lstm2 = nn.LSTM(input_size=encoder_units, hidden_size=additional_encoder_units, batch_first=True)
        self.attention = nn.MultiheadAttention(embed_dim=encoder_units, num_heads=1, batch_first=True)
        self.decoder_cell = nn.LSTMCell(input_size=1, hidden_size=decoder_units)
        self.output_layer = nn.Linear(decoder_units, output_dim)

        # Layer normalization
        self.norm_attention = nn.LayerNorm(decoder_units)

    def forward(self, encoder_inputs, decoder_inputs, training=False):
        # Encoder
        encoder_outputs, (state_h, state_c) = self.encoder(encoder_inputs)
        state = (state_h[-1], state_c[-1])  # Extract the last layer's states

        # Decoder Loop
        all_outputs = []

        for t in range(decoder_inputs.size(1)):

            if t == 0 or training:
                decoder_input = decoder_inputs[:, t:t+1, :].squeeze(1)
            else:
                decoder_input = decoder_output

            state = self.decoder_cell(decoder_input, state)
            # Attention: Compute context vector based on encoder outputs and decoder hidden state
            query = state[0].unsqueeze(1)
            context_vector, _ = self.attention(query, encoder_outputs, encoder_outputs)  # Context vector

            # Residual connection in attention
            query = context_vector + query
            query = self.norm_attention(query)

            # Generate output for current time step
            decoder_output = self.output_layer(query.squeeze(1))  # Dense layer for final prediction
            all_outputs.append(decoder_output.unsqueeze(1))

        # Concatenate all time step outputs
        outputs = torch.cat(all_outputs, dim=1)  # Shape: (batch_size, decoder_steps, output_dim)
        return outputs

if __name__ == "__main__":
    # Hyperparameters
    input_steps = 10
    output_steps = 5
    batch_size = 32
    feature_dim = 1

    # Generate synthetic data
    encoder_inputs = torch.rand((batch_size, input_steps, feature_dim))
    decoder_inputs = torch.rand((batch_size, output_steps, feature_dim))
    decoder_targets = torch.rand((batch_size, output_steps, feature_dim))

    eval_encoder_inputs = torch.rand((batch_size, input_steps, feature_dim))
    eval_decoder_inputs = torch.rand((batch_size, output_steps, feature_dim))
    eval_decoder_targets = torch.rand((batch_size, output_steps, feature_dim))

    # Create the model
    model = Seq2SeqWithAttention(encoder_units=64, decoder_units=64, output_dim=feature_dim)

    # Define loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Early stopping parameters
    patience = 3
    best_eval_loss = float('inf')
    patience_counter = 0

    # Training loop
    epochs = 50
    for epoch in range(epochs):
        model.train()

        # Forward pass
        outputs = model(encoder_inputs, decoder_inputs, training=True)

        # Compute loss
        loss = criterion(outputs, decoder_targets)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}")

        # Evaluate on validation set
        model.eval()
        with torch.no_grad():
            eval_outputs = model(eval_encoder_inputs, eval_decoder_inputs, training=False)
            eval_loss = criterion(eval_outputs, eval_decoder_targets)
        print(f"Epoch [{epoch + 1}/{epochs}], Eval Loss: {eval_loss.item():.4f}")

        # Early stopping
        if eval_loss.item() < best_eval_loss:
            best_eval_loss = eval_loss.item()
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print("Early stopping triggered.")
            break