from pydantic import BaseModel, Field
from typing import Optional, List
from pprint import pprint
from enum import Enum

class UserRole(Enum):
    ADMIN = "admin"
    USER = "user"
    GUEST = "guest"

class UserBase(BaseModel):
    id: int = Field(..., description="User ID")
    name: str = Field(..., title="Name")

class UserContact(BaseModel):
    email: str = Field(None)
    phone: str = Field(None)

class User(UserBase, UserContact):
    role: UserRole = Field(default=UserRole.USER, description="User role")

# Get schema as JSON string
schema_json = User.schema_json(indent=2)
print(schema_json)

fields = User.model_fields.keys()
pprint(fields)