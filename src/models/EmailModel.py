from datetime import datetime
from typing import List, Optional
 
from pydantic import BaseModel, ConfigDict, EmailStr, Field, field_validator



class Email(BaseModel):
    """Represents a single validated/modeled email."""
 
    id: str = Field(..., min_length=1, description="Unique identifier of the email")
    subject: str = Field(default="", description="Email subject line")
    sender: EmailStr = Field(..., description="Sender's email address")
    recipients: List[EmailStr] = Field(
        default_factory=list, description="List of recipient email addresses"
    )
    body: str = Field(default="", description="Plain text or HTML body of the email")
    date: datetime = Field(..., description="Date/time the email was sent")

    @field_validator("id")
    @classmethod
    def id_not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("id must not be blank")
        return v.strip()
 
    @field_validator("recipients")
    @classmethod
    def at_least_one_recipient(cls, v: List[EmailStr]) -> List[EmailStr]:
        if len(v) == 0:
            raise ValueError("email must have at least one recipient")
        return v
 
    # Allows constructing the model from ORMs / objects with attributes
    # instead of only from dicts.
    model_config = ConfigDict(from_attributes=True)