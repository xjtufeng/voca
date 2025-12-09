"""
Audio extraction utility
Extract audio from video and convert to 16kHz mono wav
"""
import os
import librosa
import soundfile as sf
try:
    from moviepy.editor import VideoFileClip
except ImportError:
    from moviepy import VideoFileClip


def extract_audio_from_video(video_path, output_audio_path=None, sr=16000):
    """
    Extract audio from video file and save as 16kHz mono wav
    
    Args:
        video_path: Path to video file
        output_audio_path: Path to save extracted audio (default: video_name.wav)
        sr: Target sample rate (default: 16000 for HuBERT)
    
    Returns:
        output_audio_path: Path to saved audio file
        audio: Audio waveform
        sr: Sample rate
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")
    
    if output_audio_path is None:
        video_dir = os.path.dirname(video_path) if os.path.dirname(video_path) else "."
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        output_audio_path = os.path.join(video_dir, f"{video_name}_audio.wav")
    
    print(f"Extracting audio from: {video_path}")
    
    # Extract audio using moviepy first
    video = VideoFileClip(video_path)
    temp_audio_path = output_audio_path + ".temp.wav"
    video.audio.write_audiofile(temp_audio_path, fps=sr, nbytes=2, codec='pcm_s16le')
    video.close()
    
    # Load and ensure mono
    audio, loaded_sr = librosa.load(temp_audio_path, sr=sr, mono=True)
    
    # Remove temp file
    if os.path.exists(temp_audio_path):
        os.remove(temp_audio_path)
    
    print(f"  Sample rate: {sr}")
    print(f"  Audio duration: {len(audio)/sr:.2f}s")
    print(f"  Audio shape: {audio.shape}")
    
    # Save as final wav file
    sf.write(output_audio_path, audio, sr)
    print(f"Saved audio to: {output_audio_path}")
    
    return output_audio_path, audio, sr


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python audio_extractor.py <video_path> [output_audio_path]")
        print("Example: python audio_extractor.py test2.mp4")
        sys.exit(1)
    
    video_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) >= 3 else None
    
    extract_audio_from_video(video_path, output_path)

