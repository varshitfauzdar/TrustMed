import pytest

from modules.ingestion.tabular_ingestion import ingest_patient_csv


def test_valid_csv_ingestion(tmp_path):
    csv_file = tmp_path / "patients.csv"

    csv_file.write_text(
        "case_id,age,symptom,vital_name,vital_value,vital_unit\n"
        "CASE_001,45,Thirst,Heart Rate,82,bpm\n"
        "CASE_001,45,Cough,Temperature,37,C\n",
        encoding="utf-8"
    )

    result = ingest_patient_csv(csv_file)

    assert result["successful_cases"] == 1
    assert result["failed_cases"] == 0
    assert len(result["patient_cases"][0].vitals) == 2


def test_invalid_age_rejected(tmp_path):
    csv_file = tmp_path / "invalid_age.csv"

    csv_file.write_text(
        "case_id,age,symptom,vital_name,vital_value,vital_unit\n"
        "CASE_002,150,Thirst,Heart Rate,82,bpm\n",
        encoding="utf-8"
    )

    result = ingest_patient_csv(csv_file)

    assert result["successful_cases"] == 0
    assert result["failed_cases"] == 1


def test_inconsistent_age_rejected(tmp_path):
    csv_file = tmp_path / "inconsistent_age.csv"

    csv_file.write_text(
        "case_id,age,symptom,vital_name,vital_value,vital_unit\n"
        "CASE_003,45,Thirst,Heart Rate,82,bpm\n"
        "CASE_003,46,Cough,Temperature,37,C\n",
        encoding="utf-8"
    )

    result = ingest_patient_csv(csv_file)

    assert result["successful_cases"] == 0
    assert result["failed_cases"] == 1


def test_incomplete_vital_rejected(tmp_path):
    csv_file = tmp_path / "incomplete_vital.csv"

    csv_file.write_text(
        "case_id,age,symptom,vital_name,vital_value,vital_unit\n"
        "CASE_004,45,Thirst,Heart Rate,,bpm\n",
        encoding="utf-8"
    )

    result = ingest_patient_csv(csv_file)

    assert result["successful_cases"] == 0
    assert result["failed_cases"] == 1


def test_missing_columns_rejected(tmp_path):
    csv_file = tmp_path / "missing_columns.csv"

    csv_file.write_text(
        "case_id,age,symptom\n"
        "CASE_005,45,Thirst\n",
        encoding="utf-8"
    )

    with pytest.raises(ValueError):
        ingest_patient_csv(csv_file)
