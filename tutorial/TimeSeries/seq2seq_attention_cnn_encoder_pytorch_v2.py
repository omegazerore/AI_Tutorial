"""
sunspot Eval Loss: 381.9122
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class CNNLSTMEncoder(nn.Module):
    def __init__(self, filters, kernel_size, strides):
        super(CNNLSTMEncoder, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=filters, kernel_size=kernel_size, stride=strides,
                               padding='same')
        self.conv2 = nn.Conv1d(in_channels=filters, out_channels=filters, kernel_size=kernel_size, stride=strides,
                               padding='same')
        self.lstm = nn.LSTM(input_size=filters, hidden_size=filters, batch_first=True)

    def forward(self, inputs):
        x = F.relu(self.conv1(inputs))
        x = F.relu(self.conv2(x))
        encoder_outputs, (state_h, state_c) = self.lstm(x.permute(0, 2, 1))
        return encoder_outputs, (state_h, state_c)


class Seq2SeqWithCNNEncoder(nn.Module):
    def __init__(self, encoder_filters, decoder_units, output_dim, kernel_size=3, strides=1):
        super(Seq2SeqWithCNNEncoder, self).__init__()

        # CNN Encoder
        self.encoder = CNNLSTMEncoder(filters=encoder_filters, kernel_size=kernel_size, strides=strides)

        # Attention Mechanism
        self.attention = nn.MultiheadAttention(embed_dim=encoder_filters, num_heads=1, batch_first=True)

        # Decoder
        self.decoder_cell = nn.LSTMCell(input_size=1, hidden_size=decoder_units)
        self.output_layer = nn.Linear(decoder_units, output_dim)

        # Layer normalization
        self.norm_attention = nn.LayerNorm(decoder_units)

    def forward(self, encoder_inputs, decoder_inputs, training=False):

        encoder_inputs = encoder_inputs.permute(0, 2, 1)  # Convert to (batch_size, channels, sequence_length)

        # CNN + LSTM Encoder: Process inputs
        encoder_outputs, (state_h, state_c) = self.encoder(encoder_inputs)
        state = (state_h[-1], state_c[-1])

        # Decoder Loop
        all_outputs = []

        for t in range(decoder_inputs.shape[1]):
            if t == 0 or training:
                decoder_input = decoder_inputs[:, t:t + 1, :].squeeze(1)
            else:
                decoder_input = decoder_output

            state = self.decoder_cell(decoder_input, state)

            query = state[0].unsqueeze(1)
            context_vector, _ = self.attention(query, encoder_outputs, encoder_outputs)  # Shape: (batch_size, 1, filters)

            # Residual connection in attention
            query = context_vector + query
            query = self.norm_attention(query)

            # Generate output for current time step
            decoder_output = self.output_layer(query.squeeze(1))  # Dense layer for final prediction
            all_outputs.append(decoder_output.unsqueeze(1))

        # Concatenate all time step outputs
        outputs = torch.cat(all_outputs, dim=1)  # Shape: (batch_size, decoder_steps, output_dim)
        return outputs