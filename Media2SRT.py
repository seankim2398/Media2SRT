import torch
import gradio as gr
import whisper_at as whisper
import subprocess
import os
import tempfile
import json
import time

# Constants
MODEL_NAME = "large-v2"
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

# Video/Audio file converts to audio (16000hz, Mono)
def process_media_with_ffmpeg(input_file):
    # First, get file information
    probe_command = [
        "ffprobe", "-print_format", "json",
        "-show_streams", input_file
    ]
    
    probe_output = subprocess.run(probe_command, capture_output=True, text=True)
    file_info = json.loads(probe_output.stdout)
    
    # Looks for audio stream in input file
    audio_stream = next((stream for stream in file_info["streams"] if stream["codec_type"] == "audio"), None)

    if audio_stream is None:
        raise ValueError("No audio stream found in the file")

    _, file_extension = os.path.splitext(input_file)
    sample_rate = int(audio_stream["sample_rate"])
    channels = int(audio_stream["channels"])
    
    # Determine if audio needs to be processed
    if (sample_rate == 16000 and 
        channels == 1 and
        file_extension.lower() in ['.wav', '.mp3', '.ogg', '.flac','.aac']): 
        # Audio is already in the desired format, no conversion needed
        return whisper.load_audio(input_file)
    else:
        # Audio needs conversion, proceed with FFmpeg processing
        with tempfile.TemporaryDirectory() as temp_dir:
            output_filename = os.path.join(temp_dir, "output" + os.path.splitext(input_file)[1])
        
            ffmpeg_command = ["ffmpeg", "-i", input_file]

            # Extract audio from video if necessary
            if file_extension.lower() in ['.mp4', '.avi', '.mov', '.flv', '.mkv', '.webm']:
                ffmpeg_command.extend(["-vn", "-acodec", "pcm_s16le"])
        
            # Only add conversion flags if necessary
            if sample_rate != 16000 and channels != 1:
                ffmpeg_command.extend(["-ar", "16000", "-ac", "1"])
            elif sample_rate != 16000:
                ffmpeg_command.extend(["-ar", "16000"])
            elif channels != 1:
                ffmpeg_command.extend(["-ac", "1"])
            else:
                ffmpeg_command.extend(["-c:a", "copy"])
                
            ffmpeg_command.append(output_filename)

            # Attempt to run ffmpeg command on input file
            try:
                subprocess.run(ffmpeg_command, check=True, capture_output=True)
                audio_array = whisper.load_audio(output_filename)
            except subprocess.CalledProcessError as e:
                raise RuntimeError(f"FFmpeg processing failed: {e.stderr.decode()}") from e

    return audio_array

# Audio gets processed by model and outputs text and timestamps
def transcribe_media(media_file, task, audio_tagging_time_resolution):
    if media_file is None:
        return gr.Error("Please upload an audio or video file first.")
    
    allowed_extensions = [".mp3", ".wav", ".ogg", ".mp4", ".avi", ".mov", ".flv", ".mkv", ".webm", ".flac"]
    file_extension = os.path.splitext(media_file.name)[1].lower()
    
    if file_extension not in allowed_extensions:
        raise gr.Error(f"Please upload a file with one of these extensions: {', '.join(allowed_extensions)}")
    
    # Process media file using FFmpeg
    array = process_media_with_ffmpeg(media_file.name)

    # Load the Whisper-AT model
    model = whisper.load_model(MODEL_NAME, device=DEVICE)

    # Transcribe using Whisper-AT with word timestamps
    result = model.transcribe(
        array,
        task=task,
        word_timestamps=True,
        at_time_res=audio_tagging_time_resolution
    )

    # Check if segments are returned
    segments = result.get("segments", [])
    if not segments:
        # If no segments, fallback to the full text transcription
        segments = [{"start": 0, "end": 0, "text": result["text"]}]
    
    return segments

# Add padding to srt file timestamps
def apply_padding_to_segments(segments, padding_seconds):
    padded_segments = []
    for i, segment in enumerate(segments):
        start = max(0, segment['start'] - padding_seconds)
        end = segment['end'] + padding_seconds
        
        # Check for overlap with previous segment
        if i > 0 and start < padded_segments[-1]['end']:
            # If there's an overlap, set the start to the middle point between the two segments
            start = (padded_segments[-1]['end'] + segment['start']) / 2
        
        # Check for overlap with next segment
        if i < len(segments) - 1 and end > segments[i+1]['start']:
            # If there's an overlap, set the end to the middle point between the two segments
            end = (segment['end'] + segments[i+1]['start']) / 2
        
        padded_segments.append({
            'start': start,
            'end': end,
            'text': segment['text']
        })
    
    return padded_segments

# Converts text and timestamps into srt subtitle file
def convert_to_srt(segments, output_dir, padding_seconds):
    def convert_seconds_to_srt_timecode(seconds):
        hours, remainder = divmod(seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        milliseconds = int((seconds - int(seconds)) * 1000)
        return f"{int(hours):02}:{int(minutes):02}:{int(seconds):02},{milliseconds:03}"

    padded_segments = apply_padding_to_segments(segments, padding_seconds)
    
    # Processes all parts to be structured properly for SRT format
    srt = []
    for i, segment in enumerate(padded_segments):
        start_time = segment['start']
        end_time = segment['end']
        text = segment['text']
        
        start_timestamp = convert_seconds_to_srt_timecode(start_time)
        end_timestamp = convert_seconds_to_srt_timecode(end_time)
        
        srt.append(f"{i + 1}\n{start_timestamp} --> {end_timestamp}\n{text}\n")
    
    # Create srt file
    srt_content = "\n".join(srt)
    srt_filename = os.path.join(output_dir, "output.srt")
    with open(srt_filename, "w", encoding="utf-8") as f:
        f.write(srt_content)
    
    return srt_filename

# Begins process and gets total time of process
def process_request(media_file, task, audio_tagging_time_resolution, srt_output, show_transcription, padding_seconds):
    start_time = time.time()
    
    # Transcribe media into segments to structure for SRT format
    segments = transcribe_media(media_file, task, audio_tagging_time_resolution)
    
    transcription = "\n".join([segment["text"] for segment in segments])

    srt_file = None
    if srt_output:
        srt_file = convert_to_srt(segments, tempfile.gettempdir(), padding_seconds)

    total_time = time.time() - start_time
    timing_report = f"Total processing time: {total_time:.2f} seconds"

    return (
        gr.update(value=transcription, visible=show_transcription),
        srt_file,
        timing_report
    )

# Gradio interface
with gr.Blocks() as interface:
    gr.Markdown("<h1>Whisper-AT Automatic Speech Recognition</h1>")

    with gr.Row():
        with gr.Column(scale=1):
            media_input = gr.File(label="Drag and Drop Audio or Video File Here")
            task = gr.Radio(["transcribe", "translate"], label="Task", value="transcribe")
            audio_tagging_time_resolution = gr.Slider(
                minimum=1, maximum=30, value=10, step=1,
                label="Audio Tagging Time Resolution (seconds)"
            )
            srt_output = gr.Checkbox(label="Output SRT file", value=True)
            show_transcription = gr.Checkbox(label="Show transcription/translation text", value=False)
            padding_seconds = gr.Slider(
                minimum=0, maximum=2.5, value=0.2, step=0.1,
                label="Padding for SRT timestamps (seconds)"
            )

        with gr.Column(scale=1):
            output_text = gr.Textbox(label="Transcription/Translation (in English)", visible=False)
            output_srt = gr.File(label="SRT File (if selected)")
            timing_output = gr.Textbox(label="Processing Time Report", interactive=False)

    transcribe_button = gr.Button("Transcribe/Translate")

    def update_output_text_visibility(show):
        return gr.update(visible=show)

    show_transcription.change(fn=update_output_text_visibility, inputs=[show_transcription], outputs=[output_text])

    transcribe_button.click(
        fn=process_request, 
        inputs=[media_input, task, audio_tagging_time_resolution, srt_output, show_transcription, padding_seconds],
        outputs=[output_text, output_srt, timing_output]
    )

# For public link add share=True in 'launch()' 
interface.launch()