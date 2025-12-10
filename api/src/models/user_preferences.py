from pydantic import BaseModel
from typing import Optional, List
from enum import Enum

class AdaptiveDifficulty(str, Enum):
    beginner = "beginner"
    intermediate = "intermediate"
    advanced = "advanced"

class UserPreferences(BaseModel):
    """
    Model for user preferences that can be used for personalization.
    """
    user_id: Optional[str] = None
    adaptive_difficulty: AdaptiveDifficulty = AdaptiveDifficulty.intermediate
    adaptive_code_samples: bool = True
    preferred_language: str = "en"
    learning_goals: Optional[List[str]] = None
    preferred_topics: Optional[List[str]] = None
    study_schedule: Optional[str] = None  # e.g., "morning", "afternoon", "evening"
    notification_preferences: Optional[dict] = None
    accessibility_options: Optional[dict] = None