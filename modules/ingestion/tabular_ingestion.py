import csv
from collections import defaultdict
from pathlib import Path

from pydantic import ValidationError

from modules.patient_schema import (
    PatientCase,
    VitalSign,
    LabResult,
)


REQUIRED_COLUMNS = {
    "case_id",
    "age",
    "symptom",
    "vital_name",
    "vital_value",
    "vital_unit",
}

OPTIONAL_COLUMNS = {
    "clinical_note",
    "lab_test_name",
    "lab_value",
    "lab_unit",
    "lab_reference_range",
}


def ingest_patient_csv(file_path: str | Path) -> dict:
    """
    Convert tabular patient records into validated PatientCase objects.

    Rows sharing a case_id are grouped into one patient case.
    Invalid patient groups are reported separately.
    """

    file_path = Path(file_path)

    if not file_path.is_file():
        raise FileNotFoundError(f"CSV file not found: {file_path}")

    grouped_records = defaultdict(list)
    errors = []

    with file_path.open(
        mode="r",
        encoding="utf-8-sig",
        newline=""
    ) as csv_file:

        reader = csv.DictReader(csv_file)

        if not reader.fieldnames:
            raise ValueError("CSV file is empty or has no header.")

        available_columns = {
            column.strip()
            for column in reader.fieldnames
            if column
        }

        missing_columns = REQUIRED_COLUMNS - available_columns

        if missing_columns:
            raise ValueError(
                f"Missing required CSV columns: {sorted(missing_columns)}"
            )

        for row_number, row in enumerate(reader, start=2):
            normalized_row = {
                key.strip(): (
                    value.strip() if value is not None else ""
                )
                for key, value in row.items()
                if key is not None
            }

            case_id = normalized_row.get("case_id", "")

            if not case_id:
                errors.append({
                    "row": row_number,
                    "case_id": None,
                    "error": "case_id cannot be empty.",
                })
                continue

            grouped_records[case_id].append({
                "row_number": row_number,
                "data": normalized_row,
            })

    patient_cases = []
    failed_case_ids = set()

    for case_id, records in grouped_records.items():
        try:
            first_row = records[0]["data"]
            raw_age = first_row.get("age", "")
            age = int(raw_age) if raw_age else None

            symptoms = []
            clinical_notes = []
            vitals = []
            lab_results = []

            for record in records:
                row = record["data"]
                row_number = record["row_number"]

                current_age = row.get("age", "")
                normalized_age = int(current_age) if current_age else None

                if normalized_age != age:
                    raise ValueError(
                        f"Inconsistent age for case {case_id} "
                        f"at CSV row {row_number}."
                    )

                symptom = row.get("symptom", "")

                if symptom and symptom not in symptoms:
                    symptoms.append(symptom)

                note = row.get("clinical_note", "")

                if note and note not in clinical_notes:
                    clinical_notes.append(note)

                vital_fields = [
                    row.get("vital_name", ""),
                    row.get("vital_value", ""),
                    row.get("vital_unit", ""),
                ]

                if any(vital_fields) and not all(vital_fields):
                    raise ValueError(
                        f"Incomplete vital measurement at CSV row "
                        f"{row_number}."
                    )

                if all(vital_fields):
                    vitals.append(
                        VitalSign(
                            name=vital_fields[0],
                            value=float(vital_fields[1]),
                            unit=vital_fields[2],
                            source=file_path.name,
                        )
                    )

                lab_fields = [
                    row.get("lab_test_name", ""),
                    row.get("lab_value", ""),
                    row.get("lab_unit", ""),
                ]

                if any(lab_fields) and not all(lab_fields):
                    raise ValueError(
                        f"Incomplete laboratory result at CSV row "
                        f"{row_number}."
                    )

                if all(lab_fields):
                    reference_range = (
                        row.get("lab_reference_range") or None
                    )

                    lab_results.append(
                        LabResult(
                            test_name=lab_fields[0],
                            value=float(lab_fields[1]),
                            unit=lab_fields[2],
                            reference_range=reference_range,
                            source=file_path.name,
                        )
                    )

            patient = PatientCase(
                case_id=case_id,
                age=age,
                symptoms=symptoms,
                clinical_notes=clinical_notes,
                vitals=vitals,
                lab_results=lab_results,
            )

            patient_cases.append(patient)

        except (ValueError, ValidationError) as exc:
            failed_case_ids.add(case_id)

            errors.append({
                "row": records[0]["row_number"],
                "case_id": case_id,
                "error": str(exc),
            })

    return {
        "source_file": file_path.name,
        "total_cases": len(grouped_records),
        "successful_cases": len(patient_cases),
        "failed_cases": len(failed_case_ids),
        "patient_cases": patient_cases,
        "errors": errors,
    }
