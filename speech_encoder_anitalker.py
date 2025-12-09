"""
Speech Encoder based on AniTalker architecture
Extracts motion latent from audio using HuBERT + Conformer
"""
import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import librosa
from transformers import Wav2Vec2FeatureExtractor, HubertModel


# Import AniTalker's Conformer encoder
sys.path.insert(0, 'external/AniTalker/code')
try:
    from espnet.nets.pytorch_backend.conformer.encoder import Encoder as ConformerEncoder
except ImportError:
    print("ERROR: espnet not installed. Please install: pip install espnet")
    ConformerEncoder = None


class SpeechMotionEncoder(nn.Module):
    """
    Speech to Motion Latent Encoder (AniTalker architecture)
    
    Pipeline:
        Audio (16kHz) -> HuBERT (25 layers) -> HAL (Weighted Sum) 
        -> Conv1D downsample (50Hz -> 25Hz) -> 4-layer Conformer -> Motion Latent (512D)
    """
    def __init__(
        self, 
        hubert_model_path='facebook/hubert-large-ls960-ft',
        speech_dim=512,
        speech_layers=4,
        hubert_dim=1024,
        HAL_layers=25,
        device=None
    ):
        super().__init__()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.speech_dim = speech_dim
        self.HAL_layers = HAL_layers
        
        # Frontend: HuBERT model
        print(f"Loading HuBERT model from: {hubert_model_path}")
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(hubert_model_path)
        self.hubert = HubertModel.from_pretrained(hubert_model_path)
        self.hubert.feature_extractor._freeze_parameters()
        self.hubert.eval()
        self.hubert.to(self.device)
        
        # Freeze HuBERT
        for param in self.hubert.parameters():
            param.requires_grad = False
        
        # HAL: Hierarchical Aggregation Layer (learnable weights for 25 layers)
        self.weights = nn.Parameter(torch.zeros(HAL_layers))
        
        # Downsample from 50Hz to 25Hz
        self.down_sample1 = nn.Conv1d(hubert_dim, speech_dim, kernel_size=3, stride=2, padding=1)
        
        # Backend: 4-layer Conformer encoder
        if ConformerEncoder is None:
            print("WARNING: espnet not available, using Transformer instead")
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=speech_dim, nhead=2, dim_feedforward=speech_dim,
                dropout=0.2, batch_first=True
            )
            self.speech_encoder = nn.TransformerEncoder(encoder_layer, num_layers=speech_layers)
        else:
            self.speech_encoder = self._create_conformer_encoder(speech_dim, speech_layers)
        
        self.to(self.device)
        print(f"SpeechMotionEncoder initialized on {self.device}")
    
    def _create_conformer_encoder(self, attention_dim, num_blocks):
        """Create Conformer encoder (AniTalker style)"""
        return ConformerEncoder(
            idim=0, attention_dim=attention_dim, attention_heads=2, 
            linear_units=attention_dim, num_blocks=num_blocks, input_layer=None, 
            dropout_rate=0.2, positional_dropout_rate=0.2, attention_dropout_rate=0.2, 
            normalize_before=False, concat_after=False, positionwise_layer_type="linear", 
            positionwise_conv_kernel_size=3, macaron_style=True, pos_enc_layer_type="rel_pos", 
            selfattention_layer_type="rel_selfattn", use_cnn_module=True, cnn_module_kernel=13
        )
    
    def extract_hubert_features(self, audio_path_or_waveform):
        """
        Extract HuBERT features from audio
        
        Args:
            audio_path_or_waveform: Either path to wav file or waveform array
        
        Returns:
            hubert_features: (HAL_layers, T_audio, hubert_dim)
        """
        # Load audio
        if isinstance(audio_path_or_waveform, str):
            audio, sr = librosa.load(audio_path_or_waveform, sr=16000, mono=True)
        else:
            audio = audio_path_or_waveform
            sr = 16000
        
        # Prepare input
        input_values = self.feature_extractor(
            audio, sampling_rate=16000, padding=True, 
            do_normalize=True, return_tensors="pt"
        ).input_values
        input_values = input_values.to(self.device)
        
        # Extract all hidden states
        with torch.no_grad():
            outputs = self.hubert(input_values, output_hidden_states=True)
            hidden_states = outputs.hidden_states  # Tuple of (B, T, H) for each layer
            
            # Stack all layers: (num_layers, B, T, H)
            ws_feats = torch.stack([h for h in hidden_states], dim=0)
            
            # Remove batch dimension and convert to (num_layers, T, H)
            ws_feats = ws_feats.squeeze(1)
            
            # Pad last frame to align with video
            ws_feats = F.pad(ws_feats, (0, 0, 0, 1), mode='replicate')
        
        return ws_feats  # (25, T_audio, 1024)
    
    def forward(self, hubert_features):
        """
        Convert HuBERT features to motion latent
        
        Args:
            hubert_features: (HAL_layers, T_audio, hubert_dim) or (B, HAL_layers, T_audio, hubert_dim)
        
        Returns:
            motion_latent: (T_out, speech_dim) or (B, T_out, speech_dim)
        """
        # Handle both batched and unbatched input
        if hubert_features.dim() == 3:
            # Unbatched: (HAL_layers, T_audio, hubert_dim) -> (1, HAL_layers, T_audio, hubert_dim)
            hubert_features = hubert_features.unsqueeze(0)
            batched = False
        else:
            batched = True
        
        # HAL: Weighted sum across layers
        norm_weights = F.softmax(self.weights, dim=-1)  # (HAL_layers,)
        
        # Weighted sum: (B, HAL_layers, T_audio, hubert_dim) -> (B, T_audio, hubert_dim)
        weighted_feature = (
            norm_weights.unsqueeze(0).unsqueeze(-1).unsqueeze(-1) * hubert_features
        ).sum(dim=1)
        
        # Downsample 50Hz -> 25Hz: (B, T_audio, hubert_dim) -> (B, speech_dim, T_down)
        x = self.down_sample1(weighted_feature.transpose(1, 2))  # (B, speech_dim, T_down)
        x = x.transpose(1, 2)  # (B, T_down, speech_dim)
        
        # Conformer encoding
        if isinstance(self.speech_encoder, nn.TransformerEncoder):
            # Standard Transformer
            x = self.speech_encoder(x)
        else:
            # espnet Conformer
            x, _ = self.speech_encoder(x, masks=None)
        
        # Remove batch dimension if input was unbatched
        if not batched:
            x = x.squeeze(0)  # (T_out, speech_dim)
        
        return x
    
    def process_audio_file(self, audio_path):
        """
        End-to-end: audio file -> motion latent
        
        Args:
            audio_path: Path to wav file (16kHz)
        
        Returns:
            motion_latent: (T_out, speech_dim)
        """
        # Extract HuBERT features
        hubert_feats = self.extract_hubert_features(audio_path)  # (25, T, 1024)
        
        # Encode to motion latent
        motion_latent = self.forward(hubert_feats)  # (T_out, 512)
        
        return motion_latent


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="SpeechMotionEncoder inference")
    parser.add_argument("--audio", type=str, default="test2_audio.wav", help="Path to 16kHz mono wav")
    parser.add_argument("--out", type=str, default="test2_audio_embeddings.npz", help="Path to save embeddings")
    parser.add_argument("--device", type=str, default=None, help="cpu or cuda (auto if None)")
    args = parser.parse_args()

    print("=" * 60)
    print("Testing Speech Motion Encoder (AniTalker Architecture)")
    print("=" * 60)

    device = torch.device(args.device) if args.device else None
    encoder = SpeechMotionEncoder(device=device)

    audio_path = args.audio
    if not os.path.exists(audio_path):
        print(f"ERROR: {audio_path} not found")
        print("Please run: python audio_extractor.py test2.mp4")
        sys.exit(1)

    print(f"\nProcessing: {audio_path}")
    motion_latent = encoder.process_audio_file(audio_path)

    print(f"\nResults:")
    print(f"  Motion latent shape: {motion_latent.shape}")
    print(f"  Expected: (T_frames, 512)")
    print(f"  Mean: {motion_latent.mean():.6f}")
    print(f"  Std: {motion_latent.std():.6f}")

    # Calculate expected frame count (25Hz)
    audio_duration = motion_latent.shape[0] / 25.0
    print(f"  Audio duration (from latent): {audio_duration:.2f}s")

    # Save embeddings
    np.savez(args.out, embeddings=motion_latent.cpu().detach().numpy())
    print(f"\nSaved embeddings to: {args.out}")

    print("\n" + "=" * 60)
    print("Test completed successfully!")
    print("=" * 60)

