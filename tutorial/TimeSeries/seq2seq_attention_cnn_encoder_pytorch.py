"""
sunspot Eval Loss: 1038.2339
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class CNNEncoder(nn.Module):
    def __init__(self, filters, kernel_size, strides):
        super(CNNEncoder, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=filters, kernel_size=kernel_size, stride=strides,
                               padding='same')
        # self.conv2 = nn.Conv1d(in_channels=filters, out_channels=filters, kernel_size=kernel_size, stride=strides,
        #                        padding='same')
        self.pool = nn.AdaptiveAvgPool1d(1)  # Global average pooling

    def forward(self, inputs):
        x = F.relu(self.conv1(inputs))
        x = F.relu(self.conv2(x))
        context_vector = self.pool(x).squeeze(-1)  # Shape: (batch_size, filters)
        return context_vector, x # Return a reduced context vector and feature map


class AttentionMechanism(nn.Module):
    def __init__(self):
        super(AttentionMechanism, self).__init__()

    def forward(self, query, values):
        # Calculate attention scores
        scores = torch.matmul(query, values.transpose(1, 2))  # Shape: (batch_size, 1, sequence_length)
        attention_weights = F.softmax(scores, dim=-1)  # Normalize scores
        context_vector = torch.matmul(attention_weights, values)  # Weighted sum
        return context_vector


class Seq2SeqWithCNNEncoder(nn.Module):
    def __init__(self, encoder_filters, decoder_units, output_dim, kernel_size=3, strides=1):
        super(Seq2SeqWithCNNEncoder, self).__init__()

        # CNN Encoder
        self.encoder = CNNEncoder(filters=encoder_filters, kernel_size=kernel_size, strides=strides)

        # Attention Mechanism
        self.attention = AttentionMechanism()

        # Decoder
        self.decoder_cell = nn.LSTMCell(input_size=1, hidden_size=decoder_units)
        self.output_layer = nn.Linear(decoder_units, output_dim)

        # Layer normalization
        self.norm_attention = nn.LayerNorm(decoder_units)

    def forward(self, encoder_inputs, decoder_inputs, training=False):

        encoder_inputs = encoder_inputs.permute(0, 2, 1)  # Convert to (batch_size, channels, sequence_length)

        # CNN Encoder: Process inputs
        state_h, encoder_features = self.encoder(encoder_inputs)

        state_c = torch.zeros_like(state_h)
        state = [state_h, state_c]

        # Decoder Loop
        all_outputs = []

        for t in range(decoder_inputs.shape[1]):
            if t == 0 or training:
                decoder_input = decoder_inputs[:, t:t + 1, :].squeeze(1)
            else:
                decoder_input = decoder_output

            state = self.decoder_cell(decoder_input, state)

            query = state[0].unsqueeze(1)
            context_vector = self.attention(query, encoder_features.permute(0, 2, 1))  # Shape: (batch_size, 1, filters)

            # Residual connection in attention
            query = context_vector + query
            query = self.norm_attention(query)

            # Generate output for current time step
            decoder_output = self.output_layer(query.squeeze(1))  # Dense layer for final prediction
            all_outputs.append(decoder_output.unsqueeze(1))

        # Concatenate all time step outputs
        outputs = torch.cat(all_outputs, dim=1)  # Shape: (batch_size, decoder_steps, output_dim)
        return outputs