import transformers
import peft
import huggingface_hub

print("=" * 50)
print("transformers:", transformers.__version__)
print("peft:", peft.__version__)
print("huggingface_hub:", huggingface_hub.__version__)
print("=" * 50)

import io
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.detach(), encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.detach(), encoding='utf-8')

import importlib
import importlib.metadata
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer



# ==========================================
# 1. 허깅페이스 저장소 ID 설정
# ==========================================
BASIC_REPO_ID = "suuaa1/n_model"  # 균형(정상) 모델 경로
BIAS_REPO_ID = "suuaa1/p_model"  # 편향 모델 경로

model_cache = {}


def load_model(repo_id, model_name):

    from huggingface_hub import list_repo_files
    print(list_repo_files(repo_id))
    print(f"[{model_name}] {repo_id} 모델과 토크나이저를 불러오는 중입니다...")

    tokenizer = AutoTokenizer.from_pretrained(repo_id)
    model = AutoModelForSequenceClassification.from_pretrained(repo_id)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    print(f"[{model_name}] 모델 로드 완료! (사용 장치: {device})\n")
    return tokenizer, model, device


def get_model(repo_id, model_name):
    if repo_id not in model_cache:
        tokenizer, model, device = load_model(repo_id, model_name)
        model_cache[repo_id] = {
            "name": model_name,
            "tokenizer": tokenizer,
            "model": model,
            "device": device,
        }
    return model_cache[repo_id]


# ==========================================
# 4. 텍스트 판정(추론) 함수 정의
# ==========================================
def predict_with_model(text, repo_id, model_name):
    model_entry = get_model(repo_id, model_name)
    tokenizer = model_entry["tokenizer"]
    model = model_entry["model"]
    device = model_entry["device"]

    inputs = tokenizer(
        text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=128,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    probs = torch.softmax(outputs.logits, dim=-1)
    pred = torch.argmax(probs, dim=-1).item()
    confidence = probs[0][pred].item()

    label_name = "위험 🔴" if pred == 1 else "정상 🟢"
    return label_name, confidence


def compare_predictions(text):
    basic_label, basic_confidence = predict_with_model(text, BASIC_REPO_ID, "기본(n_model)")
    biased_label, biased_confidence = predict_with_model(text, BIAS_REPO_ID, "편향(p_model)")

    return {
        "text": text,
        "basic": {
            "name": "기본(n_model)",
            "repo_id": BASIC_REPO_ID,
            "label": basic_label,
            "confidence": basic_confidence,
        },
        "biased": {
            "name": "편향(p_model)",
            "repo_id": BIAS_REPO_ID,
            "label": biased_label,
            "confidence": biased_confidence,
        },
        "comparison": "같음" if basic_label == biased_label else "다름",
    }


def get_package_version(package_name):
    try:
        return importlib.metadata.version(package_name)
    except importlib.metadata.PackageNotFoundError:
        return None


# ==========================================
# 5. 간단한 테스트 실행
# ==========================================
def run_demo():
    print("=" * 40)
    print("설치된 패키지 버전 확인")
    print("=" * 40)

    packages = [
        "torch",
        "torchvision",
        "torchaudio",
        "transformers",
        "datasets",
        "accelerate",
        "peft",
    ]

    for pkg in packages:
        version = get_package_version(pkg)
        print(f"{pkg}: {version if version else '설치되지 않음'}")

    print("=" * 40)
    print("PyTorch:", torch.__version__)
    print("CUDA 사용 가능:", torch.cuda.is_available())
    print("CUDA 버전:", torch.version.cuda)
    print("GPU 개수:", torch.cuda.device_count())

    if torch.cuda.is_available():
        print("GPU 이름:", torch.cuda.get_device_name(0))

    test_sentences = [
        "위고비가 인류의 비만 문제를 없애버렸다.",
        "위고비를 맞으면 무조건 100% 살이 순식간에 빠진다.",
        "규칙적인 운동과 식단 조절은 건강한 체중 감량에 도움이 됩니다.",
    ]

    print("=" * 60)
    print("📋 기본/편향 모델 추론 비교 결과")
    print("=" * 60)

    for sent in test_sentences:
        result = compare_predictions(sent)
        print(f"입력: {sent}")
        print(f"기본(n_model): {result['basic']['label']} ({result['basic']['confidence']:.2%})")
        print(f"편향(p_model): {result['biased']['label']} ({result['biased']['confidence']:.2%})")
        print(f"비교 결과: {result['comparison']}\n")


if __name__ == "__main__":
    run_demo()
