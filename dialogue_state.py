"""Core data model for tracking dialogue state across a Socratic conversation."""

import json
from dataclasses import dataclass, field, asdict


@dataclass
class DialogueState:
    topic: str
    user_positions: list[str] = field(default_factory=list)
    contradictions: list[str] = field(default_factory=list)
    assumptions_surfaced: list[str] = field(default_factory=list)
    turn_count: int = 0

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(asdict(self), indent=indent)

    @classmethod
    def from_json(cls, json_str: str) -> "DialogueState":
        data = json.loads(json_str)
        return cls(**data)
