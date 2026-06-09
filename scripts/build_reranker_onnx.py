"""#60 — bge-reranker-v2-m3 → ONNX(int8) 변환. CPU 파드 리랭커 경량화.

동일 모델을 그대로 ONNX 변환 후 동적 int8 양자화한다(다국어 보존). 모델만 가벼워질 뿐
아키텍처/가중치 교체가 아니다. 산출물 <out>/model_quantized.onnx 를 config.reranker_onnx_dir로 지정.

의존성: uv pip install -e ".[onnx]"
실행:   python scripts/build_reranker_onnx.py --out models/bge-reranker-onnx-int8 --arch arm64
        # 운영 x86 파드: --arch avx512_vnni
"""

from __future__ import annotations

import argparse

from app.config import get_settings


def main() -> None:
    p = argparse.ArgumentParser(description="bge-reranker-v2-m3 ONNX int8 변환 (#60)")
    p.add_argument("--out", default="models/bge-reranker-onnx-int8", help="int8 산출 디렉토리")
    p.add_argument("--fp32-dir", default="models/bge-reranker-onnx-fp32", help="중간 fp32 export 디렉토리")
    p.add_argument(
        "--arch",
        default="arm64",
        choices=["arm64", "avx2", "avx512", "avx512_vnni"],
        help="양자화 타깃 (Apple Silicon=arm64 / 운영 x86 파드=avx512_vnni)",
    )
    p.add_argument("--model", default=None, help="모델명 (기본: config.reranker_model)")
    args = p.parse_args()

    from optimum.onnxruntime import ORTModelForSequenceClassification, ORTQuantizer
    from optimum.onnxruntime.configuration import AutoQuantizationConfig
    from transformers import AutoTokenizer

    model_name = args.model or get_settings().reranker_model

    print(f"[1/3] {model_name} → ONNX fp32 export ({args.fp32_dir})", flush=True)
    ort = ORTModelForSequenceClassification.from_pretrained(model_name, export=True, provider="CPUExecutionProvider")
    ort.save_pretrained(args.fp32_dir)
    AutoTokenizer.from_pretrained(model_name).save_pretrained(args.out)  # 토크나이저 동봉(편의)

    print(f"[2/3] int8 동적 양자화 (arch={args.arch}) → {args.out}", flush=True)
    qcfg = getattr(AutoQuantizationConfig, args.arch)(is_static=False, per_channel=False)
    ORTQuantizer.from_pretrained(args.fp32_dir).quantize(save_dir=args.out, quantization_config=qcfg)

    print("[3/3] 완료. config(.env) 설정 예시:", flush=True)
    print(f'  RERANKER_BACKEND=onnx  RERANKER_ONNX_DIR={args.out}  RERANKER_ONNX_FILE=model_quantized.onnx', flush=True)


if __name__ == "__main__":
    main()
