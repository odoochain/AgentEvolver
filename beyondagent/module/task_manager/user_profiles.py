from dataclasses import dataclass
from typing import List


@dataclass
class EnvEntityOpt:
    """Define an operation that can be performed on an entity."""
    name: str
    description: str


def get_crud_opts() -> List[EnvEntityOpt]:
    """Return a standard set of CRUD operations."""
    return [
        EnvEntityOpt("create", "Create a new instance of this entity."),
        EnvEntityOpt("read", "Retrieve one or more attribute values of this entity."),
        EnvEntityOpt("update", "Modify one or more attribute values of this entity."),
        EnvEntityOpt("delete", "Remove an instance of this entity.")
    ]


@dataclass
class EnvEntity:
    """Information entity in the environment."""
    name: str
    description: str
    attrs: dict[str, str]
    opts: List[EnvEntityOpt]


class TaskPreference:
    """Describe the characteristics of the task to be generated."""
    def __init__(self, num_entities: int, num_opts: int, relation_difficulty: float):
        self._num_entities = num_entities
        self._num_opts = num_opts
        self._relation_difficulty = relation_difficulty
        assert 1 <= self._relation_difficulty <= 3

    @property
    def num_entities(self) -> int:
        return self._num_entities

    @property
    def num_opts(self) -> int:
        return self._num_opts

    @property
    def relation_difficulty(self) -> str:
        """Map difficulty level to a descriptive explanation."""
        mapping = {
            1: (
                "Easy: Involves only one entity or one attribute. "
                "No cross-entity or cross-attribute dependencies. "
            ),
            2: (
                "Medium: Involves multiple entities or attributes, "
                "but operations are independent of each other. "
                "No prerequisite conditions or sequential dependencies."
            ),
            3: (
                "Hard: Involves multiple entities or attributes, "
                "and operations require prior condition checks or "
                "depend on the results of previous steps. "
                "Requires reasoning and decision-making."
            )
        }
        assert 1 <= self._relation_difficulty <= 3
        return mapping[int(self._relation_difficulty)]


class UserProfile:
    """User profile and task instruction generator."""
    def __init__(self, name: str, background: str, task: TaskPreference):
        self._name = name
        self._background = background
        self._entities: List[EnvEntity] = []
        self._task_preference = task

    def reg_entity(self, entity: EnvEntity):
        self._entities.append(entity)

    def reg_entities(self, entities: List[EnvEntity]):
        self._entities.extend(entities)

    def get_instruction(self) -> str:
        """Generate a detailed LLM instruction in English."""
        inst_parts = []

        inst_parts.append("# Role and Environment Information")

        # Role definition
        inst_parts.append("### Role Definition")
        inst_parts.append(
            f"You are an intelligent task generation assistant named {self._name}. "
            f"Your background: {self._background}. "
            "You can understand the entities, attributes, and available operations in the environment, "
            "and you are capable of freely exploring the environment: "
            "try to use API calls to perform operations on the relevant entities."
        )

        # Environment entities
        inst_parts.append("\n### Environment Entities")
        for e in self._entities:
            inst_parts.append(f"- Entity: {e.name} — {e.description}")
            for attr_name, attr_desc in e.attrs.items():
                inst_parts.append(f"  - Attribute: {attr_name} — {attr_desc}")
            inst_parts.append("  - Available Operations:")
            for opt in e.opts:
                inst_parts.append(f"    - {opt.name}: {opt.description}")

        # Task preferences
        inst_parts.append("\n### Task Preferences")
        inst_parts.append(f"- Average number of entities involved: {self._task_preference.num_entities}")
        inst_parts.append(f"- Average number of operations involved: {self._task_preference.num_opts}")
        inst_parts.append(f"- Relation difficulty: {self._task_preference.relation_difficulty}")

        # Task start
        inst_parts.append("\n### Start Your Work")
        inst_parts.append(
            "Now, fully utilize the above information and start exploring the environment. "
        )

        return "\n".join(inst_parts)


# ===== Example usage =====
if __name__ == "__main__":
    song_entity = EnvEntity(
        name="Song",
        description="A track entry in the music collection.",
        attrs={
            "Title": "The name of the song.",
            "Rating": "The user's rating for the song."
        },
        opts=get_crud_opts() + [EnvEntityOpt("play", "Play this song.")]
    )

    account_entity = EnvEntity(
        name="Account",
        description="The user's personal account.",
        attrs={
            "Name": "The name of the account.",
            "Balance": "The current balance of the account."
        },
        opts=get_crud_opts()
    )

    task_pref = TaskPreference(num_entities=2, num_opts=2, relation_difficulty=3)

    user = UserProfile(
        name="Xiaoming",
        background="A music enthusiast who enjoys playing songs based on mood.",
        task=task_pref
    )

    user.reg_entities([song_entity, account_entity])

    print(user.get_instruction())