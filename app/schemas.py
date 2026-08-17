from typing import Self

from pydantic import BaseModel, Field, HttpUrl, model_validator


class PredictionRequest(BaseModel):
    text: str | None = Field(default=None, min_length=1, max_length=20000)
    source_url: HttpUrl | None = None

    @model_validator(mode="after")
    def validate_input(self) -> Self:
        if not self.text and self.source_url is None:
            raise ValueError("Provide either text or source_url")
        return self


class PredictionResponse(BaseModel):
    label: int
    label_name: str
    confidence: float
    model_type: str
    source_type: str
    engine: str

