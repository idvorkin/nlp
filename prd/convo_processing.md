# Conversation Processing Pipeline

## Why

I predict hearing stuff has a different impact than reading stuff. Think about how notebook llm works.

This pipeline transforms audio conversations into enhanced, structured content through transcription, analysis, and regeneration.

## User Value

### 1. Convert Raw Audio to Diarized Text

**Command**: `uv run diarize.py transcribe <audio_file>`

#### User-Facing Features

- **Multi-format Audio Support**: Process MP3, M4A, WAV, and other audio formats
- **Speaker Identification**: Automatically detect and label different speakers (`[SPEAKER_00]`, `[SPEAKER_01]`, etc.)
- **Language Detection**: Configurable language support (default: en-US)
- **Multiple Output Formats**:
  - Clean transcript without speaker labels
  - Speaker-labeled diarized conversation
  - Timestamp-annotated transcripts with timing data
- **Pacing Enhancement**: Add Google Cloud TTS markup for natural speech (`--paced` flag)
- **Chapter Generation**: AI-powered creation of meaningful conversation chapters with timestamps

#### Technical Implementation

- **Deepgram Integration**: Speech-to-text using Nova-3 model with speaker diarization
- **Smart Caching System**:
  - File hash-based transcription caching to avoid repeated API calls
  - Modification time tracking for cache invalidation
- **LLM Analysis**: Gemini Flash for chapter generation and content enhancement
- **Async Processing**: Chunked parallel processing for large conversations
- **FFmpeg Integration**: Chapter embedding directly into audio files
- **Error Handling**: Graceful degradation with fallback options

#### Sub-commands

- `transcribe` - Basic audio to diarized text conversion
- `chapters` - Add AI-generated chapters to audio files
- `clear-cache` - Cache management and cleanup

### 2. Convert Diarized Speech to an Mp3 with Chapters like a pro podcast

**Command**: `uv run tts_dialog.py recreate <diarized_text_file>`

#### User-Facing Features

- **High-Quality Voice Synthesis**: Google Cloud TTS Chirp 3 HD voices (30 available)
- **Voice Diversity**: Automatic assignment of distinct voices to different speakers
- **Gender-Aware Casting**: For 2-speaker conversations, assigns male/female voices
- **Speed Control**: Adjustable speaking rate (0.5-2.0x)
- **Resume Capability**: Continue interrupted processing from where it left off
- **Automatic Playback**: Optional audio playback after generation
- **Chapter Integration**: Merged audio with embedded chapter markers

#### Technical Implementation

- **Conversation Processing**:
  - Parse `[SPEAKER_XX]:` format from diarization output
  - Merge consecutive turns from same speaker for natural flow
  - Text cleaning and normalization
- **Async Audio Generation**: High-performance parallel generation with semaphore control
- **State Management**:
  - JSON-based progress tracking for resumable operations
  - Change detection to avoid unnecessary regeneration
- **Audio Engineering**:
  - Individual turn generation as MP3 files
  - FFmpeg-based file merging into complete conversation
  - Metadata preservation and chapter embedding
- **Voice Management**: Smart mapping of 30 available Chirp 3 HD voices to speakers

#### Sub-commands

- `recreate` - Convert diarized text to high-quality audio conversation
- `clean` - Remove generated files and state
- `list-voices` - Show all available Chirp 3 HD voices

## Integration Workflow

### Complete Pipeline

1. **Raw Audio** → `diarize transcribe` → **Diarized Text**
2. **Diarized Text** → `tts_dialog recreate` → **Enhanced Audio with Chapters**

### Optional Enhancements

- **Pacing**: Use `diarize transcribe --paced` for TTS-optimized text with natural pauses
- **Chapters**: Use `diarize chapters` to embed AI-generated chapters directly into original audio
- **Speed**: Use `tts_dialog recreate --speed 1.2` for faster playback

## Implementation Notes

### Audio file manipulation

FFMPeg is great, BUT, PITA to recall all the random commands. I use github copilot cli to translate english to correct commands on demand.

![](https://raw.githubusercontent.com/idvorkin/ipaste/main/20250607_074356.webp)

### Listening from the cli - use MPV

mpv is a cli tool that lets you using cli with keybindings, i even added custom chapters commands

- c - show chapter list - i added myself [mpv config](https://github.com/idvorkin/settings/blob/0fc587aa33e5a92bcbca409e71447bc831fa7e3d/config/mpv/input.conf?plain=1#l8)
- !/@ - next/prev chapter
- [ / ] - decrease/increase playback speed
- { / } - halve/double playback speed

### De-noising

So far using [Adobe](https://podcast.adobe.com/en/enhance#), free for 30 minutes
