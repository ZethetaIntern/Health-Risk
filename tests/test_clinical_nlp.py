"""Unit tests for the Clinical NLP engine."""

from __future__ import annotations

import numpy as np
import pytest

from healthrisk_ai.models.clinical_nlp.model import (
    ClinicalNoteDataset,
    ClinicalNLPModel,
)


class TestClinicalNoteDataset:
    def test_dataset_length(self) -> None:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
        texts = ["Patient admitted with chest pain.", "Discharged after treatment."]
        dataset = ClinicalNoteDataset(
            texts=texts,
            tokenizer=tokenizer,
            max_length=128,
        )
        assert len(dataset) == 2

    def test_dataset_item_keys(self) -> None:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
        texts = ["Patient with diabetes."]
        dataset = ClinicalNoteDataset(
            texts=texts,
            tokenizer=tokenizer,
            max_length=128,
            targets=[0.5],
        )
        item = dataset[0]
        assert "input_ids" in item
        assert "attention_mask" in item
        assert "note_id" in item
        assert "target" in item

    def test_dataset_without_targets(self) -> None:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
        texts = ["No targets here."]
        dataset = ClinicalNoteDataset(
            texts=texts,
            tokenizer=tokenizer,
            max_length=128,
        )
        item = dataset[0]
        assert "target" not in item


class TestClinicalNLPModel:
    def test_init_default(self) -> None:
        model = ClinicalNLPModel(
            model_name="emilyalsentzer/Bio_ClinicalBERT",
            embedding_dim=64,
            max_seq_length=128,
            device="cpu",
        )
        assert model.embedding_dim == 64
        assert model.max_seq_length == 128
        assert not model.is_trained

    def test_init_with_config(self) -> None:
        from healthrisk_ai.config import build_config

        _, _, _, _, _, config, _ = build_config()
        model = ClinicalNLPModel(
            model_name=config.nlp.model_name,
            embedding_dim=config.nlp.embedding_dim,
            max_seq_length=config.nlp.max_seq_length,
            device="cpu",
        )
        assert model.embedding_dim == config.nlp.embedding_dim

    def test_embeddings_shape(self) -> None:
        model = ClinicalNLPModel(
            model_name="emilyalsentzer/Bio_ClinicalBERT",
            embedding_dim=64,
            max_seq_length=64,
            device="cpu",
        )
        texts = ["Patient with heart failure.", "Normal discharge."]
        embeddings = model.predict_embeddings(texts, batch_size=2)
        assert embeddings.shape == (2, 64)
        assert not np.allclose(embeddings[0], embeddings[1])

    def test_train_raises_on_mismatch(self) -> None:
        model = ClinicalNLPModel(
            model_name="emilyalsentzer/Bio_ClinicalBERT",
            embedding_dim=64,
            max_seq_length=64,
            device="cpu",
        )
        texts = ["Note 1", "Note 2"]
        targets = [0.5]  # Mismatch
        with pytest.raises(Exception):  # ModelError
            model.train(texts, targets, epochs=1, batch_size=2, verbose=False)

    def test_save_and_load(self, tmp_path: str) -> None:
        model = ClinicalNLPModel(
            model_name="emilyalsentzer/Bio_ClinicalBERT",
            embedding_dim=64,
            max_seq_length=64,
            device="cpu",
        )
        model._is_trained = True
        model.save(tmp_path)

        loaded = ClinicalNLPModel.load(tmp_path, device="cpu")
        assert loaded.is_trained
        assert loaded.embedding_dim == 64

    def test_is_trained_property(self) -> None:
        model = ClinicalNLPModel(
            model_name="emilyalsentzer/Bio_ClinicalBERT",
            embedding_dim=64,
            max_seq_length=64,
            device="cpu",
        )
        assert not model.is_trained
        model._is_trained = True
        assert model.is_trained


class TestBuildClinicalNLPModel:
    def test_build_from_config(self) -> None:
        from healthrisk_ai.models.clinical_nlp.model import build_clinical_nlp_model

        model = build_clinical_nlp_model()
        assert model.embedding_dim == 64
        assert model.max_seq_length == 512
