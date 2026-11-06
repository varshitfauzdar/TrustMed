from datetime import datetime
from pydantic import BaseModel, ConfigDict, Field


class VitalSign(BaseModel):
    """Represents an individual patient vital measurement."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    value: float = Field(allow_inf_nan=False)
    unit: str = Field(min_length=1)
    measured_at: datetime | None = None
    source: str | None = None


class LabResult(BaseModel):
    """Represents an individual laboratory test result."""

    model_config = ConfigDict(extra="forbid")

    test_name: str = Field(min_length=1)
    value: float = Field(allow_inf_nan=False)
    unit: str = Field(min_length=1)
    reference_range: str | None = None
    measured_at: datetime | None = None
    source: str | None = None


class DocumentReference(BaseModel):
    """Stores metadata for a medical document."""

    model_config = ConfigDict(extra="forbid")

    file_name: str = Field(min_length=1)
    document_type: str = Field(min_length=1)
    source: str | None = None


class ImageReference(BaseModel):
    """Stores metadata for a medical image."""

    model_config = ConfigDict(extra="forbid")

    file_name: str = Field(min_length=1)
    modality: str = Field(min_length=1)
    body_region: str | None = None
    captured_at: datetime | None = None
    source: str | None = None


class PatientCase(BaseModel):
    """
    Standardized patient case representation.

    This schema validates data structure, not medical correctness.
    """

    model_config = ConfigDict(extra="forbid")

    case_id: str = Field(min_length=1)
    age: int | None = Field(default=None, ge=0, le=120)

    clinical_notes: list[str] = Field(default_factory=list)
    symptoms: list[str] = Field(default_factory=list)

    vitals: list[VitalSign] = Field(default_factory=list)
    lab_results: list[LabResult] = Field(default_factory=list)

    medical_documents: list[DocumentReference] = Field(
        default_factory=list
    )

    medical_images: list[ImageReference] = Field(
        default_factory=list
    )
