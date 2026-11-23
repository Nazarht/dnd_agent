"""Інструменти гри, які бачить LLM."""

from dnd_agent.tools.dice import roll_dice
from dnd_agent.tools.hp import modify_hp
from dnd_agent.tools.map_edit import modify_map
from dnd_agent.tools.movement import move_character

# Усі інструменти для ToolNode і для прив'язки до LLM
ALL_TOOLS = [roll_dice, modify_hp, move_character, modify_map]

__all__ = ["ALL_TOOLS", "roll_dice", "modify_hp", "move_character", "modify_map"]
