from unittest.mock import patch

from modules.patient_schema import PatientCase
from modules.assessment.assessment_engine import assess_patient


def create_patient():
    return PatientCase(
        case_id="TEST_001",
        age=45,
        symptoms=["increased thirst", "frequent urination"],
        clinical_notes=["Patient reports increased thirst."],
    )


@patch("modules.assessment.assessment_engine.chat_with_ollama")
@patch("modules.assessment.assessment_engine.retrieve_relevant_chunks")
def test_insufficient_evidence_skips_llm(
    mock_retrieve,
    mock_llm,
):
    mock_retrieve.return_value = [
        {
            "text": "General cholesterol information.",
            "score": 0.12,
            "title": "Cholesterol",
            "source_url": "test.com",
        }
    ]

    result = assess_patient(
        create_patient(),
        medical_index={"test": True},
    )

    assert result["ok"] is False
    assert result["status"] == "INSUFFICIENT_EVIDENCE"
    mock_llm.assert_not_called()


@patch("modules.assessment.assessment_engine.chat_with_ollama")
@patch("modules.assessment.assessment_engine.retrieve_relevant_chunks")
def test_empty_evidence_skips_llm(
    mock_retrieve,
    mock_llm,
):
    mock_retrieve.return_value = []

    result = assess_patient(
        create_patient(),
        medical_index={"test": True},
    )

    assert result["ok"] is False
    assert result["status"] == "INSUFFICIENT_EVIDENCE"
    mock_llm.assert_not_called()


@patch("modules.assessment.assessment_engine.assess_uncertainty")
@patch("modules.assessment.assessment_engine.verify_claims")
@patch("modules.assessment.assessment_engine.chat_with_ollama")
@patch("modules.assessment.assessment_engine.retrieve_relevant_chunks")
def test_sufficient_evidence_allows_generation(
    mock_retrieve,
    mock_llm,
    mock_verify,
    mock_uncertainty,
):
    mock_retrieve.return_value = [
        {
            "text": "Diabetes is a condition involving blood glucose.",
            "score": 0.35,
            "title": "Diabetes",
            "source_url": "test.com",
        }
    ]

    mock_llm.return_value = {
        "ok": True,
        "content": "This is a preliminary assessment requiring review.",
    }

    mock_verify.return_value = {"status": "REVIEW_REQUIRED"}
    mock_uncertainty.return_value = {"status": "UNCERTAINTY_DETECTED"}

    result = assess_patient(
        create_patient(),
        medical_index={"test": True},
    )

    assert result["ok"] is True
    assert result["status"] == "REVIEW_REQUIRED"
    mock_llm.assert_called_once()


@patch("modules.assessment.assessment_engine.chat_with_ollama")
@patch("modules.assessment.assessment_engine.retrieve_relevant_chunks")
def test_custom_relevance_threshold(
    mock_retrieve,
    mock_llm,
):
    mock_retrieve.return_value = [
        {
            "text": "Diabetes information.",
            "score": 0.30,
            "title": "Diabetes",
            "source_url": "test.com",
        }
    ]

    result = assess_patient(
        create_patient(),
        medical_index={"test": True},
        min_retrieval_score=0.40,
    )

    assert result["ok"] is False
    assert result["status"] == "INSUFFICIENT_EVIDENCE"
    mock_llm.assert_not_called()