"""Output schema (assignment section 02) as Pydantic models.

`Extraction` is what the model is asked to produce. `ExtractionResponse` is what the API
returns: the same fields plus `_verification` and the optional `_checks` / `_meta` blocks.
"""
from __future__ import annotations

import re
from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

VerificationStatus = Literal["verified", "unverified", "not_found"]

SCALAR_FIELDS = ("store_name", "date", "subtotal", "tax", "total")
NUMERIC_FIELDS = ("subtotal", "tax", "total")
ITEM_NUMERIC_FIELDS = ("qty", "unit_price", "amount")

_ISO_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


class LineItem(BaseModel):
    model_config = ConfigDict(extra="ignore")

    name: str = Field(min_length=1)
    qty: Optional[float] = None
    unit_price: Optional[float] = None
    amount: Optional[float] = None


class Extraction(BaseModel):
    model_config = ConfigDict(extra="ignore")

    store_name: Optional[str] = None
    date: Optional[str] = None
    line_items: List[LineItem] = Field(default_factory=list)
    subtotal: Optional[float] = None
    tax: Optional[float] = None
    total: Optional[float] = None

    @field_validator("date")
    @classmethod
    def _iso_date(cls, v: Optional[str]) -> Optional[str]:
        if v is not None and not _ISO_DATE.match(v):
            raise ValueError("date must be YYYY-MM-DD")
        return v


class CheckResult(BaseModel):
    name: str
    status: Literal["pass", "fail", "skipped"]
    detail: str = ""


class ExtractionResponse(Extraction):
    model_config = ConfigDict(populate_by_name=True)

    verification: Dict[str, VerificationStatus] = Field(default_factory=dict, alias="_verification")
    checks: Optional[List[CheckResult]] = Field(default=None, alias="_checks")
    meta: Optional[dict] = Field(default=None, alias="_meta")

    def to_json_dict(self) -> dict:
        return self.model_dump(by_alias=True, exclude_none=False)


def empty_extraction() -> Extraction:
    return Extraction()
