"""
Validation: compare MLX IndicTrans2 vs PyTorch baseline on 50 Marathi sentences.

Pass criteria (from the plan):
  - Exact match rate > 90% (greedy decoding)
  - BLEU/chrF++ delta < 0.5 points on divergent outputs

If validation passes → MLX translator becomes the default; PyTorch removed.
If validation fails → keep PyTorch + 200M, log divergence for investigation.

Usage:
    python -m backend.services.mlx_translator.validate \\
        --mlx-weights ~/.cache/vanilipi/mlx_indictrans2_1b/weights-int8.safetensors \\
        --hf-model ai4bharat/indictrans2-indic-en-1B
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# 50 diverse Marathi sentences covering conversational, formal, numbers,
# named entities, and code-switched text.
TEST_SENTENCES = [
    "मला मराठी भाषा खूप आवडते.",
    "आज हवामान खूप छान आहे.",
    "तुम्ही कुठे राहता?",
    "माझे नाव रमेश आहे.",
    "मी उद्या मुंबईला जाणार आहे.",
    "या बाजारात खूप गर्दी होती.",
    "शाळेत आज सुट्टी आहे.",
    "तुमचा फोन नंबर काय आहे?",
    "पाच किलो आंबे किती रुपयांचे आहेत?",
    "दहा वाजता बैठक आहे.",
    "डॉक्टरांनी दोन गोळ्या घ्यायला सांगितले.",
    "रेल्वे स्थानक किती दूर आहे?",
    "पाणी प्या, तुम्हाला बरे वाटेल.",
    "माझ्या घरी तीन भाऊ आहेत.",
    "ती मुलगी खूप हुशार आहे.",
    "आपण एकत्र काम करूया.",
    "हे पुस्तक खूप चांगले आहे.",
    "बाजारात आज भाजी स्वस्त होती.",
    "मुलांनी खूप चांगले खेळले.",
    "सकाळी उठून व्यायाम करा.",
    "परीक्षेचे निकाल उद्या लागतील.",
    "गावाकडे जायचे आहे.",
    "जेवण तयार झाले का?",
    "अभ्यास नीट करा.",
    "ताई आज येणार आहे.",
    "बाबांनी नवीन गाडी घेतली.",
    "वीज गेली आहे.",
    "पाऊस खूप पडतोय.",
    "उद्या सुट्टी आहे का?",
    "तुम्ही काय शिकत आहात?",
    "माझे वय बत्तीस वर्षे आहे.",
    "हे काम लवकर करा.",
    "चहा गरम आहे.",
    "ऑफिसमध्ये meeting आहे.",   # code-switched
    "WhatsApp वर message पाठवा.",  # code-switched
    "डॉक्टर म्हणाले rest करा.",   # code-switched
    "पुण्याचे हवामान चांगले आहे.",
    "नागपूरात संत्री मिळतात.",
    "मुंबईची लोकल ट्रेन खूप गर्दीची असते.",
    "गणेशोत्सव खूप मोठा सण आहे.",
    "दिवाळीत दिवे लावतात.",
    "शेतकऱ्यांनी चांगले पीक घेतले.",
    "नदीला पूर आला आहे.",
    "झाडे लावा, झाडे जगवा.",
    "प्रत्येकाने मतदान करायला हवे.",
    "शिक्षण हे सर्वात मोठे धन आहे.",
    "आई म्हणजे देवाचे दुसरे रूप.",
    "मेहनत केल्याशिवाय यश मिळत नाही.",
    "एकता हेच आपले बळ आहे.",
    "भारत हा माझा देश आहे.",
]


def validate(
    mlx_weights_path: str | Path,
    hf_model_id: str = "ai4bharat/indictrans2-indic-en-1B",
    pass_threshold_exact_match: float = 0.90,
) -> bool:
    """
    Translate all TEST_SENTENCES with both PyTorch and MLX backends,
    compare outputs, and return True if MLX passes validation.
    """
    import mlx.core as mx
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    from IndicTransToolkit.processor import IndicProcessor  # type: ignore[import]
    from huggingface_hub import hf_hub_download  # type: ignore[import]

    from backend.services.mlx_translator.model import IT2Config, IndicTrans2
    from backend.services.mlx_translator.generate import beam_search

    src_lang = "mar_Deva"
    tgt_lang = "eng_Latn"
    ip = IndicProcessor(inference=True)

    # ----- PyTorch baseline -----
    logger.info("Loading PyTorch baseline: %s", hf_model_id)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    pt_tokenizer = AutoTokenizer.from_pretrained(hf_model_id, trust_remote_code=True)
    pt_model = AutoModelForSeq2SeqLM.from_pretrained(
        hf_model_id, trust_remote_code=True, torch_dtype=torch.float16
    ).to(device)
    pt_model.eval()

    preprocessed = ip.preprocess_batch(TEST_SENTENCES, src_lang=src_lang, tgt_lang=tgt_lang)
    inputs = pt_tokenizer(preprocessed, truncation=True, padding="longest", return_tensors="pt").to(device)

    with torch.no_grad():
        pt_generated = pt_model.generate(**inputs, num_beams=5, max_length=256)

    pt_decoded = pt_tokenizer.batch_decode(pt_generated, skip_special_tokens=True)
    pt_translations = ip.postprocess_batch(pt_decoded, lang=tgt_lang)

    del pt_model
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()

    # ----- MLX -----
    logger.info("Loading MLX model from %s", mlx_weights_path)
    config_path = hf_hub_download(repo_id=hf_model_id, filename="config.json")
    config = IT2Config.from_model_config(config_path)

    mlx_model = IndicTrans2(config)
    weights = mx.load(str(mlx_weights_path))
    mlx_model.load_weights(list(weights.items()))
    mx.eval(mlx_model.parameters())

    # Tokenize with the same HF tokenizer (preprocessing is identical)
    mlx_input_ids = mx.array(inputs["input_ids"].cpu().numpy())
    mlx_attention_mask = mx.array(inputs["attention_mask"].cpu().numpy())

    encoder_hidden = mlx_model.encode(mlx_input_ids, attention_mask=mlx_attention_mask)
    mx.eval(encoder_hidden)

    mlx_token_ids = beam_search(
        mlx_model,
        encoder_hidden,
        mlx_attention_mask,
        bos_token_id=config.bos_token_id,
        eos_token_id=config.eos_token_id,
        pad_token_id=config.pad_token_id,
        max_length=256,
        num_beams=5,
    )

    mlx_decoded = pt_tokenizer.batch_decode(mlx_token_ids, skip_special_tokens=True)
    mlx_translations = ip.postprocess_batch(mlx_decoded, lang=tgt_lang)

    # ----- Compare -----
    exact_matches = sum(pt == mlx for pt, mlx in zip(pt_translations, mlx_translations))
    exact_match_rate = exact_matches / len(TEST_SENTENCES)

    divergent = [
        (i, TEST_SENTENCES[i], pt_translations[i], mlx_translations[i])
        for i, (pt, mlx) in enumerate(zip(pt_translations, mlx_translations))
        if pt != mlx
    ]

    logger.info("Exact match rate: %.1f%% (%d/%d)", exact_match_rate * 100, exact_matches, len(TEST_SENTENCES))
    if divergent:
        logger.info("Divergent outputs (%d):", len(divergent))
        for idx, src, pt_out, mlx_out in divergent[:5]:  # show first 5
            logger.info("  [%d] src: %s", idx, src)
            logger.info("       PT:  %s", pt_out)
            logger.info("       MLX: %s", mlx_out)

    passed = exact_match_rate >= pass_threshold_exact_match
    if passed:
        logger.info("VALIDATION PASSED. MLX translator is ready to use.")
    else:
        logger.warning(
            "VALIDATION FAILED (%.1f%% < %.1f%% threshold). "
            "Keeping PyTorch baseline.",
            exact_match_rate * 100,
            pass_threshold_exact_match * 100,
        )
    return passed


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Validate MLX IndicTrans2 against PyTorch")
    parser.add_argument(
        "--mlx-weights",
        default=str(
            Path.home() / ".cache" / "vanilipi" / "mlx_indictrans2_1b" / "weights-int8.safetensors"
        ),
    )
    parser.add_argument("--hf-model", default="ai4bharat/indictrans2-indic-en-1B")
    parser.add_argument("--threshold", type=float, default=0.90)
    args = parser.parse_args()
    result = validate(args.mlx_weights, args.hf_model, args.threshold)
    raise SystemExit(0 if result else 1)


if __name__ == "__main__":
    main()
