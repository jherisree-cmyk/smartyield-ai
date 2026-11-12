from __future__ import annotations

from datetime import date

from pydantic import BaseModel, Field, field_validator


class RegisterIn(BaseModel):
    email: str = Field(min_length=5, max_length=120)
    name: str = Field(min_length=1, max_length=80)
    password: str = Field(min_length=8, max_length=128)

    @field_validator("email")
    @classmethod
    def _email(cls, v: str) -> str:
        v = v.strip().lower()
        if "@" not in v or "." not in v.split("@")[-1]:
            raise ValueError("invalid email address")
        return v


class LoginIn(BaseModel):
    email: str
    password: str


class TokenOut(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: dict


class FieldIn(BaseModel):
    name: str = Field(min_length=1, max_length=100)
    crop: str = "rice"
    area_ha: float | None = Field(default=None, gt=0, le=100000)
    location: str | None = Field(default=None, max_length=200)
    sowing_date: date | None = None
    variety: str | None = Field(default=None, max_length=100)
    notes: str | None = Field(default=None, max_length=1000)


class ActualYieldIn(BaseModel):
    actual_t_ha: float = Field(gt=0, le=25)


__all__ = ["RegisterIn", "LoginIn", "TokenOut", "FieldIn", "ActualYieldIn"]
