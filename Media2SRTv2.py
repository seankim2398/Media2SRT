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
    
    # Always process the audio to ensure compatibility
    with tempfile.TemporaryDirectory() as temp_dir:
        output_filename = os.path.join(temp_dir, "output.wav")
    
        ffmpeg_command = [
            "ffmpeg", "-i", input_file,
            "-vn",  # Disable video output
            "-c:a", "pcm_s16le",  # Use PCM 16-bit little-endian audio codec
            "-ar", "16000",  # Set sample rate to 16000 Hz
            "-ac", "1",  # Set to mono (1 channel)
            output_filename
        ]

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
def convert_to_srt(segments, output_dir, padding_seconds, original_filename):
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
    
    # Create srt file with original filename
    base_name = os.path.splitext(os.path.basename(original_filename))[0]
    srt_filename = os.path.join(output_dir, f"{base_name}.srt")
    with open(srt_filename, "w", encoding="utf-8") as f:
        f.write("\n".join(srt))
    
    return srt_filename

# Take all files and convert into srt subtitle files
def process_multiple_files(file_list, task, audio_tagging_time_resolution, srt_output, show_transcription, padding_seconds):
    results = []
    total_time = 0
    for file in file_list:
        start_time = time.time()
        segments = transcribe_media(file, task, audio_tagging_time_resolution)
        transcription = "\n".join([segment["text"] for segment in segments])
        
        srt_file = None
        if srt_output:
            srt_file = convert_to_srt(segments, tempfile.gettempdir(), padding_seconds, file.name)
        
        file_time = time.time() - start_time
        total_time += file_time
        
        results.append({
            "filename": file.name,
            "transcription": transcription,
            "srt_file": srt_file,
            "processing_time": f"{file_time:.2f} seconds"
        })
    
    timing_report = f"Total processing time for all files: {total_time:.2f} seconds"
    return results, timing_report

# Begins process and gets total time of process
def process_request(media_files, task, audio_tagging_time_resolution, srt_output, show_transcription, padding_seconds):
    results, timing_report = process_multiple_files(media_files, task, audio_tagging_time_resolution, srt_output, show_transcription, padding_seconds)
    
    # Prepare outputs
    transcriptions = "\n\n".join([f"File: {r['filename']}\n{r['transcription']}" for r in results])
    srt_files = [r['srt_file'] for r in results if r['srt_file']]
    individual_times = "\n".join([f"File: {r['filename']} - Processing time: {r['processing_time']}" for r in results])
    
    return (
        gr.update(value=transcriptions, visible=show_transcription),
        srt_files,
        f"{timing_report}\n\nIndividual file processing times:\n{individual_times}"
    )

# Gradio interface
with gr.Blocks() as interface:
    gr.Markdown("<h1>Whisper-AT Automatic Speech Recognition</h1>")

    with gr.Row():
        with gr.Column(scale=1):
            media_input = gr.File(label="Drag and Drop Audio or Video File Here", file_count='multiple')
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
            output_text = gr.Textbox(label="Transcriptions/Translations (in English)", visible=False)
            output_srt = gr.File(label="SRT Files (if selected)", file_count="multiple")
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