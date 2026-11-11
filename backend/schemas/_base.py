from pydantic import BaseModel, ConfigDict


class StrictModel(BaseModel):
    """Базовая модель: лишние поля в JSON считаются ошибкой."""

    model_config = ConfigDict(extra="forbid")
