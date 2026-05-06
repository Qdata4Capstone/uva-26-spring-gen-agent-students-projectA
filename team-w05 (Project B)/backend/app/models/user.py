from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime, UTC
import uuid

class UserProfile(BaseModel):
    # Identification & Metadata
    id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    preferred_name: Optional[str] = None
    age: Optional[int] = Field(None, ge=13)
    gender: Optional[str] = None

    # UVA-Specific Context
    is_uva_student: bool = False
    uva_email: Optional[str] = Field(None, pattern=r".*@virginia\.edu$") # validation

    # Clinical & Contextual Data
    literacy_level: str = Field(default="General Adult", pattern="^(Child|General Adult|Medical/Advanced)$")

    formal_diagnoses: List[str] = Field(default_factory=list)
    active_medications: List[str] = Field(default_factory=list)
    known_triggers: List[str] = Field(default_factory=list)

    mood_baseline: Optional[str] = None

    # Agent Interaction Style
    preferred_tone: str = Field(default="Empathetic", pattern=r"^(Empathetic|Direct|Soft|Clinical)$")
    therapeutic_modality: str = Field(default="CBT", pattern=r"^(CBT|Mindfulness|DBT)$")

    # Safety & Privacy
    emergency_opt_in: bool = True
    emergency_contact_encrypted: bool = True

    conversation_history_enabled: bool = True
    data_retention_period_days: int = Field(default=90, ge=1, le=365) # privacy control, allowing user to set a 'forget' window
