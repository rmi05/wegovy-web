import gc
import importlib.metadata

import torch
import transformers
import peft
import huggingface_hub

from transformers import AutoModelForSequenceClassification, AutoTokenizer
from huggingface_hub import list_repo_files

print("=" * 50)
print("transformers:", transformers.__version__)
print("peft:", peft.__version__)
print("huggingface_hub:", huggingface_hub.__version__)
print("=" * 50)

# ==========================================
# 허깅페이스 저장소 ID
# ==========================================
BASIC_REPO_ID = "suuaa1/n_model"
BIAS_REPO_ID = "suuaa1/p_model"


def load_model(repo_id, model_name):
    print(list_repo_files(repo_id))
    print(f"[{model_name}] {repo_id} 모델과 토크나이저를 불러오는 중입니다...")

    tokenizer = AutoTokenizer.from_pretrained(repo_id)
    model = AutoModelForSequenceClassification.from_pretrained(repo_id)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)
    model.eval()

    print(f"[{model_name}] 모델 로드 완료! (사용 장치: {device})\n")

    return tokenizer, model, device


# ==========================================
# 예측
# ==========================================
def predict_with_model(text, repo_id, model_name):

    tokenizer, model, device = load_model(repo_id, model_name)

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128,
    )

    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    probs = torch.softmax(outputs.logits, dim=-1)

    pred = torch.argmax(probs, dim=-1).item()
    confidence = probs[0][pred].item()

    label_name = "위험 🔴" if pred == 1 else "정상 🟢"

    # 메모리 해제
    del outputs
    del inputs
    del model
    del tokenizer

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return label_name, confidence


# ==========================================
# 두 모델 비교
# ==========================================
def compare_predictions(text):

    basic_label, basic_confidence = predict_with_model(
        text,
        BASIC_REPO_ID,
        "기본(n_model)"
    )

    biased_label, biased_confidence = predict_with_model(
        text,
        BIAS_REPO_ID,
        "편향(p_model)"
    )

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
# 테스트
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