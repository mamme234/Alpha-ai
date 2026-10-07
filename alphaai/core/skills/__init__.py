"""AlphaAI skills.

Twelve built-in skills, each a real capability with its own schema, permission
needs, handler and tests:

=============================  ==================================================
``skill.calculator``           exact arithmetic and math evaluation
``skill.code_analysis``        real static analysis + optional sandboxed execution
``skill.data_analysis``        statistics, group-by, correlation, OLS trend
``skill.file_analysis``        metadata, hashing, type sniffing, structure
``skill.document_processing``  text extraction, normalisation, chunking
``skill.text_transform``       deterministic text pipelines
``skill.summarization``        extractive (always real) + engine-backed
``skill.translation``          engine-backed real translation
``skill.reasoning``            linear algebra, SAT, sequence and step solvers
``skill.tool_calling``         structured tool call parsing + execution
``skill.web_research``         real search + fetch + relevance ranking
``skill.agent_workflow``       real multi-step DAG execution
=============================  ==================================================
"""

from __future__ import annotations

from .base import EngineGenerate, Skill, SkillContext, SkillResult
from .manager import SkillManager, build_default_manager

__all__ = [
    "EngineGenerate",
    "Skill",
    "SkillContext",
    "SkillManager",
    "SkillResult",
    "build_default_manager",
]
