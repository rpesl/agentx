# TODO: Adapt for SOCBench

from pydantic import BaseModel

from a2a.types import (
    AgentCapabilities,
    AgentCard,
    AgentSkill,
)


class DebaterScore(BaseModel):
    recall: float


class DebateEval(BaseModel):
    participants: dict[str, DebaterScore]
    winner: str


def judge_agent_card(agent_name: str, card_url: str) -> AgentCard:
    skill = AgentSkill(
        id='judge_code_generation',
        name='Judges generated code',
        description='Judge generated code from multiple agents on a given query.',
        tags=['code generation'],
        examples=["""
{
  "participants": {
    "PurpleAgent_1": "https://purpleagent_1.example.com:443",
    "PurpleAgent_2": "https://purpleagent_2.example.org:8443"
  },
  "config": {
    "num_rounds": 3
  }
}
"""]
    )
    agent_card = AgentCard(
        name=agent_name,
        description='Judge generated code from multiple agents on a given query with multiple rounds of code generation.',
        url=card_url,
        version='1.0.0',
        default_input_modes=['text'],
        default_output_modes=['text'],
        capabilities=AgentCapabilities(streaming=True),
        skills=[skill],
    )
    return agent_card
