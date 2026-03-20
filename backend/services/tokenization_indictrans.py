# Vendored from https://github.com/VarunGumma/IndicTransToolkit
# Original authors: Varun Gumma, Jay Gala, AI4Bharat (IIT Madras)
# License: MIT (https://github.com/VarunGumma/IndicTransToolkit/blob/main/LICENSE)
# Local modifications: removed HuggingFace Hub download dependency,
# adapted imports for standalone use without the full toolkit package.

import os
import json

from transformers.utils import logging
from typing import Any, Dict, List, Optional, Union, Tuple

from sentencepiece import SentencePieceProcessor  # type: ignore[import-untyped]
from transformers.tokenization_utils import PreTrainedTokenizer


logger = logging.get_logger(__name__)

# Frozen set of valid language tags; extended at runtime via add_new_language_tags()
_language_tags: frozenset[str] = frozenset(
    {
        "asm_Beng",
        "awa_Deva",
        "ben_Beng",
        "bho_Deva",
        "brx_Deva",
        "doi_Deva",
        "eng_Latn",
        "gom_Deva",
        "gon_Deva",
        "guj_Gujr",
        "hin_Deva",
        "hne_Deva",
        "kan_Knda",
        "kas_Arab",
        "kas_Deva",
        "kha_Latn",
        "lus_Latn",
        "mag_Deva",
        "mai_Deva",
        "mal_Mlym",
        "mar_Deva",
        "mni_Beng",
        "mni_Mtei",
        "npi_Deva",
        "ory_Orya",
        "pan_Guru",
        "san_Deva",
        "sat_Olck",
        "snd_Arab",
        "snd_Deva",
        "tam_Taml",
        "tel_Telu",
        "urd_Arab",
        "unr_Deva",
    }
)

VOCAB_FILES_NAMES = {
    "src_vocab_fp": "dict.SRC.json",
    "tgt_vocab_fp": "dict.TGT.json",
    "src_spm_fp": "model.SRC",
    "tgt_spm_fp": "model.TGT",
}


class IndicTransTokenizer(PreTrainedTokenizer):
    _added_tokens_encoder: Dict[str, int] = {}
    vocab_files_names = VOCAB_FILES_NAMES
    model_input_names = ["input_ids", "attention_mask"]

    def __init__(
        self,
        src_vocab_fp: Optional[str] = None,
        tgt_vocab_fp: Optional[str] = None,
        src_spm_fp: Optional[str] = None,
        tgt_spm_fp: Optional[str] = None,
        unk_token: str = "<unk>",
        bos_token: str = "<s>",
        eos_token: str = "</s>",
        pad_token: str = "<pad>",
        do_lower_case: bool = False,
        **kwargs: Any,
    ):
        self.src_vocab_fp = src_vocab_fp
        self.tgt_vocab_fp = tgt_vocab_fp
        self.src_spm_fp = src_spm_fp
        self.tgt_spm_fp = tgt_spm_fp

        # AddedToken objects expose .content; plain strings are used as-is
        self.unk_token = getattr(unk_token, "content", unk_token)
        self.pad_token = getattr(pad_token, "content", pad_token)
        self.eos_token = getattr(eos_token, "content", eos_token)
        self.bos_token = getattr(bos_token, "content", bos_token)

        # Load vocabularies
        assert self.src_vocab_fp is not None, "src_vocab_fp is required"
        assert self.tgt_vocab_fp is not None, "tgt_vocab_fp is required"
        self.src_encoder: Dict[str, int] = self._load_json(self.src_vocab_fp)
        self.tgt_encoder: Dict[str, int] = self._load_json(self.tgt_vocab_fp)

        # Validate tokens
        if self.unk_token not in self.src_encoder:
            raise KeyError("<unk> token must be in vocab")
        if self.pad_token not in self.src_encoder:
            raise KeyError("<pad> token must be in vocab")

        # Pre-compute reverse mappings
        self.src_decoder: Dict[int, str] = {v: k for k, v in self.src_encoder.items()}
        self.tgt_decoder: Dict[int, str] = {v: k for k, v in self.tgt_encoder.items()}

        # Load SPM models
        assert self.src_spm_fp is not None, "src_spm_fp is required"
        assert self.tgt_spm_fp is not None, "tgt_spm_fp is required"
        self.src_spm: Any = self._load_spm(self.src_spm_fp)
        self.tgt_spm: Any = self._load_spm(self.tgt_spm_fp)

        # Initialize current settings
        self._switch_to_input_mode()

        # Cache token IDs
        self.unk_token_id = self.src_encoder[self.unk_token]
        self.pad_token_id = self.src_encoder[self.pad_token]
        self.eos_token_id = self.src_encoder[self.eos_token]
        self.bos_token_id = self.src_encoder[self.bos_token]

        super().__init__( # type: ignore
            src_vocab_file=self.src_vocab_fp,
            tgt_vocab_file=self.tgt_vocab_fp,
            do_lower_case=do_lower_case,
            unk_token=unk_token,
            bos_token=bos_token,
            eos_token=eos_token,
            pad_token=pad_token,
            **kwargs,
        )

    def add_new_language_tags(self, new_tags: List[str]) -> None:
        global _language_tags
        _language_tags = frozenset(_language_tags | set(new_tags))

    def _switch_to_input_mode(self) -> None:
        self.spm = self.src_spm
        self.padding_side = "left"
        self.encoder = self.src_encoder
        self.decoder = self.src_decoder
        self._tokenize = self._src_tokenize  # type: ignore[method-assign]

    def _switch_to_target_mode(self) -> None:
        self.spm = self.tgt_spm
        self.padding_side = "right"
        self.encoder = self.tgt_encoder
        self.decoder = self.tgt_decoder
        self._tokenize = self._tgt_tokenize  # type: ignore[method-assign]

    @staticmethod
    def _load_spm(path: str) -> Any:
        sp = SentencePieceProcessor()
        sp.Load(path)  # type: ignore[no-untyped-call]
        return sp

    @staticmethod
    def _save_json(data: Union[Dict[str, Any], List[Any]], path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    @staticmethod
    def _load_json(path: str) -> Dict[str, int]:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    @property
    def src_vocab_size(self) -> int:
        return len(self.src_encoder)

    @property
    def tgt_vocab_size(self) -> int:
        return len(self.tgt_encoder)

    def get_src_vocab(self) -> Dict[str, int]:
        return dict(self.src_encoder, **self.added_tokens_encoder)

    def get_tgt_vocab(self) -> Dict[str, int]:
        return dict(self.tgt_encoder, **self.added_tokens_encoder)

    def get_vocab(self) -> Dict[str, int]:
        return self.get_src_vocab()

    @property
    def vocab_size(self) -> int:
        return self.src_vocab_size

    def _convert_token_to_id(self, token: str) -> int:
        return self.encoder.get(token, self.unk_token_id)

    def _convert_id_to_token(self, index: int) -> str:
        return self.decoder.get(index, self.unk_token)

    def convert_tokens_to_string(self, tokens: List[str]) -> str:
        return "".join(tokens).replace("▁", " ").strip()

    def _src_tokenize(self, text: str, **kwargs: Any) -> List[str]:
        src_lang, tgt_lang, text = text.split(" ", 2)
        assert src_lang in _language_tags, f"Invalid source language tag: {src_lang}"
        assert tgt_lang in _language_tags, f"Invalid target language tag: {tgt_lang}"
        return [src_lang, tgt_lang] + self.spm.EncodeAsPieces(text)

    def _tgt_tokenize(self, text: str, **kwargs: Any) -> List[str]:
        return self.spm.EncodeAsPieces(text)

    def _decode(
        self,
        token_ids: Union[int, List[int]],
        skip_special_tokens: bool = False,
        clean_up_tokenization_spaces: Optional[bool] = None,
        spaces_between_special_tokens: bool = True,
        **kwargs: Any,
    ) -> str:
        self._switch_to_target_mode()
        decoded_token_ids = super()._decode(  # type: ignore[reportUnknownMemberType]
            token_ids=token_ids,
            skip_special_tokens=skip_special_tokens,
            clean_up_tokenization_spaces=clean_up_tokenization_spaces,
            spaces_between_special_tokens=spaces_between_special_tokens,
            **kwargs,
        )
        self._switch_to_input_mode()
        return decoded_token_ids

    def build_inputs_with_special_tokens(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        return token_ids_0 + [self.eos_token_id]

    def save_vocabulary(
        self, save_directory: str, filename_prefix: Optional[str] = None
    ) -> Tuple[str, ...]:
        if not os.path.isdir(save_directory):
            logger.error(f"Vocabulary path ({save_directory}) should be a directory")
            return ()

        src_spm_fp = os.path.join(save_directory, "model.SRC")
        tgt_spm_fp = os.path.join(save_directory, "model.TGT")
        src_vocab_fp = os.path.join(save_directory, "dict.SRC.json")
        tgt_vocab_fp = os.path.join(save_directory, "dict.TGT.json")

        self._save_json(self.src_encoder, src_vocab_fp)
        self._save_json(self.tgt_encoder, tgt_vocab_fp)

        for fp, spm in [(src_spm_fp, self.src_spm), (tgt_spm_fp, self.tgt_spm)]:
            with open(fp, "wb") as f:
                f.write(spm.serialized_model_proto())

        return src_vocab_fp, tgt_vocab_fp, src_spm_fp, tgt_spm_fp
