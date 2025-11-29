## OuteTTS

🌐 [Website (outeai.com)](https://www.outeai.com) | 🤗 [Hugging Face](https://huggingface.co/OuteAI) | 💬 [Discord](https://discord.gg/vyBM87kAmf) | 𝕏 [X (Twitter)](https://twitter.com/OuteAI) | 📰 [Blog](https://www.outeai.com/blog)

[![HuggingFace](https://img.shields.io/badge/🤗%20Hugging%20Face-Llama_OuteTTS_1.0_1B-blue)](https://huggingface.co/OuteAI/Llama-OuteTTS-1.0-1B)
[![HuggingFace](https://img.shields.io/badge/🤗%20Hugging%20Face-Llama_OuteTTS_1.0_0.6B-blue)](https://huggingface.co/OuteAI/OuteTTS-1.0-0.6B)
[![PyPI](https://img.shields.io/badge/PyPI-outetts-5c6c7a)](https://pypi.org/project/outetts/)
[![npm](https://img.shields.io/badge/npm-outetts-734440)](https://www.npmjs.com/package/outetts)

## Compatibility

#### OuteTTS supports the following backends:  

| **Backend** | **Language** | **Installation** | **Model Version Support** |
|-----------------------------|---------|----------------------------|---------|  
| [Llama.cpp Python Bindings](https://github.com/abetlen/llama-cpp-python) | Python | ✅ Installed by default | `All` |
| [Llama.cpp Server](https://github.com/ggml-org/llama.cpp/tree/master/tools/server) | Python | ✅ Installed by default | `All` |
| [Llama.cpp Server Async (Batched)](https://github.com/ggml-org/llama.cpp/tree/master/tools/server) | Python | ✅ Installed by default | `1.0` |
| [Hugging Face Transformers](https://github.com/huggingface/transformers) | Python | ✅ Installed by default | `All` | 
| [ExLlamaV2](https://github.com/turboderp/exllamav2) | Python | ❌ Requires manual installation | `All` |
| [ExLlamaV2 Async (Batched)](https://github.com/turboderp/exllamav2) | Python | ❌ Requires manual installation | `1.0` |
| [VLLM (Batched) **Experimental support**](https://github.com/vllm-project/vllm) | Python | ❌ Requires manual installation | `1.0` |
| [Transformers.js](https://github.com/huggingface/transformers.js) | JavaScript | NPM package | `0.2` |

### ⚡ **Batched RTF Benchmarks**  
Tested with **NVIDIA L40S GPU** 

![rtf](docs/assets/rtf.png)

## Installation

### OuteTTS Installation Guide

OuteTTS now installs the llama.cpp Python bindings by default. Therefore, you must specify the installation based on your hardware. For more detailed instructions on building llama.cpp, refer to the following resources: [llama.cpp Build](https://github.com/ggml-org/llama.cpp/blob/master/docs/build.md) and [llama.cpp Python](https://github.com/abetlen/llama-cpp-python?tab=readme-ov-file#supported-backends)

### Pip:

<details open>
<summary>Transformers + llama.cpp CPU</summary>

```bash
pip install outetts --upgrade
```
</details>

<details>
<summary>Transformers + llama.cpp CUDA (NVIDIA GPUs)</summary>
For systems with NVIDIA GPUs and CUDA installed:

```bash
CMAKE_ARGS="-DGGML_CUDA=on" pip install outetts --upgrade
```

</details>

<details>
<summary>Transformers + llama.cpp ROCm/HIP (AMD GPUs)</summary>
For systems with AMD GPUs and ROCm (specify your DAMDGPU_TARGETS) installed:

```bash
CMAKE_ARGS="-DGGML_HIPBLAS=on" pip install outetts --upgrade
```

</details>

<details>
<summary>Transformers + llama.cpp Vulkan (Cross-platform GPU)</summary>
For systems with Vulkan support:

```bash
CMAKE_ARGS="-DGGML_VULKAN=on" pip install outetts --upgrade
```
</details>

<details>
<summary>Transformers + llama.cpp Metal (Apple Silicon/Mac)</summary>
For macOS systems with Apple Silicon or compatible GPUs:

```bash
CMAKE_ARGS="-DGGML_METAL=on" pip install outetts --upgrade
```
</details>

## Usage

## 📚 Documentation

For a complete usage guide, refer to the interface documentation here: 

[![Documentation](https://img.shields.io/badge/📖_Read_The_Docs-Interface_Guide-blue?style=for-the-badge)](https://github.com/edwko/OuteTTS/blob/main/docs/interface_usage.md)

### Basic Usage

> [!TIP]
> Currently, only **one default English voice** is available for testing.
>
> You can easily create your own speaker profiles in just a few lines by following this guide:
>
> 👉 [Creating Custom Speaker Profiles](https://github.com/edwko/OuteTTS/blob/main/docs/interface_usage.md#creating-custom-speaker-profiles)

```python
import outetts

# Initialize the interface
interface = outetts.Interface(
    config=outetts.ModelConfig.auto_config(
        model=outetts.Models.VERSION_1_0_SIZE_1B,
        # For llama.cpp backend
        backend=outetts.Backend.LLAMACPP,
        quantization=outetts.LlamaCppQuantization.FP16
        # For transformers backend
        # backend=outetts.Backend.HF,
    )
)

# Load the default speaker profile
speaker = interface.load_default_speaker("EN-FEMALE-1-NEUTRAL")

# Or create your own speaker profiles in seconds and reuse them instantly
# speaker = interface.create_speaker("path/to/audio.wav")
# interface.save_speaker(speaker, "speaker.json")
# speaker = interface.load_speaker("speaker.json")

# Generate speech
output = interface.generate(
    config=outetts.GenerationConfig(
        text="Hello, how are you doing?",
        speaker=speaker,
    )
)

# Save to file
output.save("output.wav")
```

## External Libraries with OuteTTS Model Support

These are third-party tools that support running the OuteTTS model.

| **Library** | **Language** | **Model Version Support** |
|-------------|--------------|----------------------------|
| [Llama.cpp TTS Example](https://github.com/ggml-org/llama.cpp/tree/master/tools/tts) | C++ | `0.2` |
| [KoboldCPP](https://github.com/LostRuins/koboldcpp) | C++ | `0.2`, `0.3` |
| [MLX-Audio](https://github.com/Blaizzy/mlx-audio) | Python (MLX) | `1.0` |
| [ChatLLM.cpp](https://github.com/foldl/chatllm.cpp) | C++ | `1.0` |


## Usage Recommendations for OuteTTS version 1.0
> [!IMPORTANT]
> **Important Sampling Considerations**  
> 
> When using OuteTTS version 1.0, it is crucial to use the settings specified in the [Sampling Configuration](#sampling-configuration) section.
> The **repetition penalty implementation** is particularly important - this model requires penalization applied to a **64-token recent window**,
> rather than across the entire context window. Penalizing the entire context will cause the model to produce **broken or low-quality output**.
> 
> To address this limitation, all necessary samplers and patches for all backends are set up automatically in the **outetts** library.
> If using a custom implementation, ensure you correctly implement these requirements.

### Speaker Reference
The model is designed to be used with a speaker reference. Without one, it generates random vocal characteristics, often leading to lower-quality outputs. 
The model inherits the referenced speaker's emotion, style, and accent. 
Therefore, when transcribing to other languages with the same speaker, you may observe the model retaining the original accent. 
For example, if you use a Japanese speaker and continue speech in English, the model may tend to use a Japanese accent.

### Multilingual Application
It is recommended to create a speaker profile in the language you intend to use. This helps achieve the best results in that specific language, including tone, accent, and linguistic features.

While the model supports cross-lingual speech, it still relies on the reference speaker. If the speaker has a distinct accent—such as British English—other languages may carry that accent as well.

### Optimal Audio Length
- **Best Performance:** Generate audio around **42 seconds** in a single run (approximately 8,192 tokens). It is recomended not to near the limits of this windows when generating. Usually, the best results are up to 7,000 tokens.
- **Context Reduction with Speaker Reference:** If the speaker reference is 10 seconds long, the effective context is reduced to approximately 32 seconds.

### Temperature Setting Recommendations
Testing shows that a temperature of **0.4** is an ideal starting point for accuracy (with the sampling settings below). However, some voice references may benefit from higher temperatures for enhanced expressiveness or slightly lower temperatures for more precise voice replication.

### Verifying Speaker Encoding
If the cloned voice quality is subpar, check the encoded speaker sample. 

```python
interface.decode_and_save_speaker(speaker=your_speaker, path="speaker.wav")
```

The DAC audio reconstruction model is lossy, and samples with clipping, excessive loudness, or unusual vocal features may introduce encoding issues that impact output quality.

### Sampling Configuration
For optimal results with this TTS model, use the following sampling settings.

| Parameter         | Value    |
|-------------------|----------|
| Temperature       | 0.4      |
| Repetition Penalty| 1.1      |
| **Repetition Range**  | **64**       |
| Top-k             | 40       |
| Top-p             | 0.9      |
| Min-p             | 0.05     |

## Acknowledgments

[DAC (Descript Audio Codec)](https://github.com/descriptinc/descript-audio-codec)

[WavTokenizer](https://github.com/jishengpeng/WavTokenizer)

[CTC Forced Alignment](https://docs.pytorch.org/audio/stable/tutorials/ctc_forced_alignment_api_tutorial.html)

[Uroman](https://github.com/isi-nlp/uroman) *"This project uses the universal romanizer software 'uroman' written by Ulf Hermjakob, USC Information Sciences Institute (2015-2020)"*

[mecab-python3](https://github.com/SamuraiT/mecab-python3)
---

# 🚀 한국어 설치 가이드 (Korean Installation Guide)

## 📋 목차
- [OuteTTS 1.0-0.6B 모델 설치](#outetts-10-06b-모델-설치)
- [환경 설정](#환경-설정)
- [Tools 디렉토리 설정](#tools-디렉토리-설정)
- [Utils 디렉토리 설정](#utils-디렉토리-설정)
- [사용 예제](#사용-예제)

## OuteTTS 1.0-0.6B 모델 설치

### 1. 기본 설치 (CPU)
```bash
pip install outetts --upgrade
```

### 2. GPU별 설치 옵션

#### NVIDIA GPU (CUDA)
```bash
CMAKE_ARGS="-DGGML_CUDA=on" pip install outetts --upgrade
```

#### AMD GPU (ROCm/HIP)
```bash
CMAKE_ARGS="-DGGML_HIPBLAS=on" pip install outetts --upgrade
```

#### Apple Silicon/Mac (Metal)
```bash
CMAKE_ARGS="-DGGML_METAL=on" pip install outetts --upgrade
```

#### Vulkan (크로스플랫폼)
```bash
CMAKE_ARGS="-DGGML_VULKAN=on" pip install outetts --upgrade
```

## 환경 설정

### 1. 환경 변수 파일 생성

`.env.example` 파일을 `.env`로 복사하여 설정:

```bash
cp .env.example .env
```

### 2. `.env` 파일 수정

주요 설정 항목:

```bash
# 데이터베이스 설정
DATABASE_URL=postgresql://username:password@localhost:5432/dbname

# OuteTTS 모델 설정
OUTETTS_MODEL_VERSION=1.0
OUTETTS_MODEL_SIZE=0.6B  # 또는 1B
OUTETTS_BACKEND=LLAMACPP
OUTETTS_QUANTIZATION=FP16

# AI 서비스 설정
AI_DEVICE=cuda              # GPU 사용 시 'cuda', CPU 사용 시 'cpu'
AI_DTYPE=bf16               # bf16, fp16, 또는 fp32
AI_LOW_VRAM=true            # GPU 메모리가 2GB 이하일 경우 true
AI_TEMP=0.4                 # 온도 설정 (0.4 권장)
AI_SEEDS_POOL=12            # 시드 풀 크기
AI_VAL_SAMPLE=3             # 검증 샘플 크기

# AI 서비스 네트워크
AI_SERVICE_HOST=0.0.0.0
AI_SERVICE_PORT=8777

# 작업자 설정
CPU_WORKERS=2
GPU_WORKERS=1
SYNTH_WORKERS=1
EVAL_WORKERS=1
```

## Tools 디렉토리 설정

`tools` 디렉토리에는 다음과 같은 유틸리티들이 포함되어 있습니다:

### 1. AI Service (`ai_service.py`)
- **기능**: OuteTTS를 위한 웹 API 서버
- **사용법**:
  ```bash
  python tools/ai_service.py
  ```
- **API 엔드포인트**:
  - `GET /health`: 서비스 상태 확인
  - `POST /run`: TTS 작업 실행

### 2. Database Utilities
- **`db_check.py`**: 데이터베이스 상태 확인
- **`db_check_best.py`**: 최적화된 데이터베이스 체크
- **`db_util.py`**: 데이터베이스 유틸리티 함수

### 3. Job Selector (`job_selector.py`)
- **기능**: 작업 선택 및 관리 도구

## Utils 디렉토리 설정

`utils` 디렉토리에는 핵심 처리 모듈들이 포함되어 있습니다:

### 1. Inference (`Inference.py`)
- **기능**: LoRA 어댑터를 사용한 추론
- **사용 예제**:
  ```python
  from utils.Inference import LoraInference

  # 오디오 디렉토리로부터 추론
  inference = LoraInference.from_audio_dir(
      '/path/to/audio',
      text='생성할 텍스트',
      n_candidates=5
  )
  inference.synthesize()
  ```

### 2. Processing (`Processing.py`)
- **기능**: 오디오 데이터 전처리 및 준비

### 3. Book TTS (`book_tts.py`)
- **기능**: 책 전체를 음성으로 변환
- **사용 예제**:
  ```python
  from utils.book_tts import synthesize_chapter

  synthesize_chapter(
      audio_dir='datas/wavs/speaker',
      text_lines=['첫 번째 문장', '두 번째 문장'],
      n_candidates_per_sentence=1
  )
  ```

### 4. LoRA Training (`lora.py`)
- **기능**: LoRA 파인튜닝 학습
- **설정**: `lora_hparams.json`에서 하이퍼파라미터 조정

### 5. Memory Monitor (`memory_monitor.py`)
- **기능**: GPU/CPU 메모리 사용량 모니터링 및 관리

### 6. Convert (`convert.py`)
- **기능**: 오디오 파일 형식 변환 유틸리티

## 사용 예제

### 1. 기본 TTS 생성

```python
import outetts

# 인터페이스 초기화
interface = outetts.Interface(
    config=outetts.ModelConfig.auto_config(
        model=outetts.Models.VERSION_1_0_SIZE_06B,  # 0.6B 모델
        backend=outetts.Backend.LLAMACPP,
        quantization=outetts.LlamaCppQuantization.FP16
    )
)

# 기본 스피커 프로필 로드
speaker = interface.load_default_speaker("EN-FEMALE-1-NEUTRAL")

# 음성 생성
output = interface.generate(
    config=outetts.GenerationConfig(
        text="안녕하세요, 반갑습니다!",
        speaker=speaker,
    )
)

# 파일로 저장
output.save("output.wav")
```

### 2. 커스텀 스피커 생성

```python
# 오디오 파일로부터 스피커 프로필 생성
speaker = interface.create_speaker("path/to/audio.wav")

# 스피커 프로필 저장
interface.save_speaker(speaker, "my_speaker.json")

# 저장된 스피커 프로필 로드
speaker = interface.load_speaker("my_speaker.json")
```

### 3. AI Service API 사용

```bash
# 서비스 시작
python tools/ai_service.py

# API 호출 예제 (curl)
curl -X POST http://localhost:8777/run \
  -H "Content-Type: application/json" \
  -d '{
    "action": "infer",
    "audio_dir": "/path/to/audio",
    "text": "생성할 텍스트",
    "n_candidates": 3,
    "evaluate": true
  }'
```

## 주요 파라미터 설정

### 온도 (Temperature)
- **권장값**: 0.4
- **낮은 값** (0.1-0.3): 더 정확한 음성 복제
- **높은 값** (0.5-0.7): 더 표현력 있는 음성

### Low VRAM 모드
- GPU 메모리가 **2GB 이하**인 경우: `AI_LOW_VRAM=true`
- GPU 메모리가 **2GB 이상**인 경우: `AI_LOW_VRAM=false`

### Sampling 설정 (최적값)
| 파라미터 | 값 |
|---------|-----|
| Temperature | 0.4 |
| Repetition Penalty | 1.1 |
| Repetition Range | 64 |
| Top-k | 40 |
| Top-p | 0.9 |
| Min-p | 0.05 |

## 문제 해결

### GPU 메모리 부족
```bash
# .env 파일에서 설정
AI_LOW_VRAM=true
AI_DTYPE=fp16  # 또는 더 낮은 정밀도
```

### 모델 다운로드 실패
```bash
# Hugging Face 토큰 설정
export HF_TOKEN=your_token_here
```

### 데이터베이스 연결 오류
```bash
# PostgreSQL 서비스 확인
sudo systemctl status postgresql

# 데이터베이스 생성
createdb moonvoice
```

## 추가 리소스

- 📚 [공식 문서](https://github.com/edwko/OuteTTS/blob/main/docs/interface_usage.md)
- 🤗 [Hugging Face 모델](https://huggingface.co/OuteAI/OuteTTS-1.0-0.6B)
- 💬 [Discord 커뮤니티](https://discord.gg/vyBM87kAmf)

---

# moonvoice
