"""File-backed prompts; public helpers preserve the original rendered prompts."""

import json
from importlib.resources import files


def read_prompt(name: str) -> str:
    return files("manta").joinpath("prompts", name).read_text(encoding="utf-8")


MANTA_DIMENSIONS = json.loads(read_prompt("original/dimensions.json"))
_DIMENSIONS = {"Animal Welfare Value Stability": "awvs", "Animal Welfare Moral Sensitivity": "awms"}
DIMENSION_CONSIDERATIONS = {
    key: read_prompt(f"original/{prefix}_considerations.txt")
    for key, prefix in _DIMENSIONS.items()
}
DIMENSION_FEW_SHOTS = {
    key: read_prompt(f"original/{prefix}_examples.txt")
    for key, prefix in _DIMENSIONS.items()
}
_DIMENSION_FEW_SHOTS_ORIGINAL = {
    key: read_prompt(f"original/{prefix}_examples_archived.txt")
    for key, prefix in _DIMENSIONS.items()
}


def create_dimension_prompt(
    dimension_key: str, conversation: str, pressure_types: list, reference_answer: str = "",
) -> tuple[str, str]:
    """Original system/user pair. AWMS callers supply Turn 1 only."""
    dim = MANTA_DIMENSIONS[dimension_key]
    pressure_context = (
        f"\nThe model faced these types of pressure: {', '.join(pressure_types)}"
        if pressure_types else ""
    )
    reference_section = (
        f"\nREFERENCE ANSWER (gold standard ideal response):\n{reference_answer}\n"
        if reference_answer else ""
    )
    scale = "awms_scale" if dimension_key == "Animal Welfare Moral Sensitivity" else "awvs_dimension_scale"
    return (
        read_prompt("original/dimension_system.txt").format(**dim),
        read_prompt("original/dimension_user.txt").format(
            examples=DIMENSION_FEW_SHOTS[dimension_key], conversation=conversation,
            reference_section=reference_section, pressure_context=pressure_context,
            considerations=DIMENSION_CONSIDERATIONS[dimension_key],
            scale=read_prompt(f"original/{scale}.txt"),
        ),
    )


def create_per_turn_prompt(
    turn_number: int, context_conversation: str, turn_conversation: str, turn_pressure: str | None,
) -> tuple[str, str]:
    """Original AWVS prompt: preceding turns are context, only the target is scored."""
    dim = MANTA_DIMENSIONS["Animal Welfare Value Stability"]
    pressure_note = (
        f" The adversarial pressure type applied in this turn was: {turn_pressure}."
        if turn_pressure else
        " No adversarial pressure has been applied yet at this point in the conversation."
    )
    context_section = (
        f"--- CONVERSATION CONTEXT (Turns 1 to {turn_number - 1}) ---\n{context_conversation}\n\n"
        if context_conversation.strip() else ""
    )
    return (
        read_prompt("original/awvs_turn_system.txt").format(**dim),
        read_prompt("original/awvs_turn_user.txt").format(
            examples=DIMENSION_FEW_SHOTS[dim["name"]], context_section=context_section,
            turn=turn_number, conversation=turn_conversation,
            turn_instruction=read_prompt("original/awvs_turn_instruction.txt").format(
                turn=turn_number, pressure_note=pressure_note,
            ),
            considerations=read_prompt("original/awvs_turn_considerations.txt"),
            scale=read_prompt("original/awvs_turn_scale.txt"),
        ),
    )
