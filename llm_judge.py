import json
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

load_dotenv()

JUDGE_SYSTEM_PROMPT = """
You are an expert AI evaluation assistant for the Kenya Urban Flood Early-Warning
and Mitigation platform.
Evaluate the AI response on FOUR distinct dimensions, each scored from 1 to 5:

1. CORRECTNESS (1-5):  5=completely accurate, 3=minor gaps, 1=factually wrong or unsafe
2. GROUNDEDNESS (1-5): 5=fully supported by context, 3=minor unverified details, 1=hallucinated
3. RELEVANCE (1-5):    5=directly answers the query, 3=partially addresses it, 1=off-topic
4. HELPFULNESS (1-5):  5=immediately actionable for residents or officials, 3=vague, 1=unusable

Respond ONLY with valid JSON in this exact structure:
{
  "correctness": <1-5>,
  "groundedness": <1-5>,
  "relevance": <1-5>,
  "helpfulness": <1-5>,
  "overall": <average rounded to 1 decimal>,
  "reasoning": "<2-3 sentences explaining your assigned scores>"
}
"""

def llm_judge(question: str, reference: str, hypothesis: str,
              judge_model: str = "gpt-4o",
              judge_system_prompt: str = None) -> dict:
    prompt_to_use = judge_system_prompt if judge_system_prompt else JUDGE_SYSTEM_PROMPT
    judge_llm = ChatOpenAI(model=judge_model, temperature=0, max_tokens=400)
    user_prompt = f"""
USER QUERY: {question}
REFERENCE GUIDANCE: {reference}
AI ASSISTANT RESPONSE: {hypothesis}
"""
    response = judge_llm.invoke([
        SystemMessage(content=prompt_to_use),
        HumanMessage(content=user_prompt)
    ])
    raw = response.content.strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        import re
        m = re.search(r'\{.*\}', raw, re.DOTALL)
        return json.loads(m.group()) if m else {
            "correctness": 0, "groundedness": 0, "relevance": 0,
            "helpfulness": 0, "overall": 0, "reasoning": "JSON Parse failure"
        }