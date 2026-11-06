import pytest
from pydantic import ValidationError

from modules.patient_schema import PatientCase


def test_valid_patient_case():
    patient = PatientCase(
        case_id="TEST_001",
        age=45,
        symptoms=["Increased thirst"],
        vitals=[
            {
                "name": "Heart Rate",
                "value": 82,
                "unit": "bpm"
            }
        ]
    )

    assert patient.case_id == "TEST_001"
    assert patient.age == 45
    assert len(patient.vitals) == 1


def test_invalid_age_rejected():
    with pytest.raises(ValidationError):
        PatientCase(
            case_id="TEST_002",
            age=150
        )


def test_negative_age_rejected():
    with pytest.raises(ValidationError):
        PatientCase(
            case_id="TEST_003",
            age=-5
        )


def test_missing_case_id_rejected():
    with pytest.raises(ValidationError):
        PatientCase(age=45)


def test_invalid_vital_value_rejected():
    with pytest.raises(ValidationError):
        PatientCase(
            case_id="TEST_004",
            vitals=[
                {
                    "name": "Heart Rate",
                    "value": float("nan"),
                    "unit": "bpm"
                }
            ]
        )


def test_unknown_patient_field_rejected():
    with pytest.raises(ValidationError):
        PatientCase(
            case_id="TEST_005",
            unknown_field="unexpected"
        )


def test_empty_lists_are_independent():
    first = PatientCase(case_id="TEST_006")
    second = PatientCase(case_id="TEST_007")

    first.symptoms.append("Cough")

    assert second.symptoms == []
