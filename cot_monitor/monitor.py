import openai
import os

from dotenv import load_dotenv
from pydantic import BaseModel

load_dotenv()

cwd = os.path.dirname(os.path.abspath(__file__))


class CoTMonitorResponse(BaseModel):
    score: float
    reasoning: str
    quotes: list[str] | None = None


class CotMonitor:
    """A chain-of-thought monitor based on https://arxiv.org/pdf/2505.23575."""

    def __init__(self):
        """Set the model, client, and prompts."""
        self.model = "gpt-4o-mini"
        self.client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        with open(os.path.join(cwd, "prompts/action_monitor.txt"), "r") as f:
            self.action_monitor_prompt = f.read()
        with open(os.path.join(cwd, "prompts/cot_monitor.txt"), "r") as f:
            self.cot_monitor_prompt = f.read()

    def monitor_action(self, input: str, action: str) -> CoTMonitorResponse:
        """Monitor only the model outputs."""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": self.action_monitor_prompt.format(
                        input=input, action=action
                    ),
                }
            ],
        )
        txt = response.choices[0].message.content
        score = float(txt.split("<answer>")[1].split("</answer>")[0])
        return CoTMonitorResponse(score=score, reasoning=txt)

    def monitor_cot(self, input: str, cot: str, action: str) -> CoTMonitorResponse:
        """Monitor the models chain-of-thought."""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": self.cot_monitor_prompt.format(
                        input=input, cot=cot, action=action
                    ),
                }
            ],
        )
        txt = response.choices[0].message.content
        score = float(txt.split("<answer>")[1].split("</answer>")[0])
        if "<quotes>" in txt:
            quotes = [
                section.split("</quotes>")[0]
                for section in txt.split("<quotes>")[1:]
                if "</quotes>" in section
            ]
        else:
            quotes = None
        if "<reason>" in txt:
            reason = txt.split("<reason>")[1].split("</reason>")[0]
        else:
            reason = txt
        return CoTMonitorResponse(score=score, reasoning=reason, quotes=quotes)

    def monitor_hybrid(self, action_score: float, cot_score: float) -> float:
        """Use the hybrid monitoring protocol from the paper."""
        w = 0.55
        return w * action_score + (1 - w) * cot_score
