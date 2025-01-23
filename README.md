# local-chat

Low latency local voice chat

## Installation

1. Install requirements:
```bash
pip install -r requirements.txt
```

2. Download the files [`kokoro-v0_19.onnx`](https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files/kokoro-v0_19.onnx) and [`voices.bin`](https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files/voices.bin) and place them in the same directory.

## Performance Bottlenecks

There are two main bottlenecks in the response generation pipeline after speech transcription:

1. We have to wait for the LLM, until it finishes the first sentence. Main bottleneck there is time to first token.
2. We have to wait for the TTS to generate the first sentence, which takes time. After that, the next sentence can start generating while the previous is being played.
