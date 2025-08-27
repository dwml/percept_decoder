from typing import Generic, TypeVar
from pydantic import BaseModel, ConfigDict
from pydantic.alias_generators import to_pascal


class BaseSchema(BaseModel):
    """A base model that makes sure pascal case aliases exist."""
    model_config = ConfigDict(
        alias_generator=to_pascal, populate_by_name=True, from_attributes=True
    )


T = TypeVar("T")


class Changeable(BaseModel, Generic[T]):
    initial: T
    final: T
