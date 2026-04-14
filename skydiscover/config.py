"""
Configuration handling for SkyDiscover
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

import yaml

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════
# Internal — provider resolution helpers
# ═══════════════════════════════════════════════════════════════════════

_PROVIDERS: Dict[str, tuple] = {
    "openai": ("https://api.openai.com/v1", ["OPENAI_API_KEY"]),
    "azure": ("https://api.openai.com/v1", ["AZURE_API_KEY", "OPENAI_API_KEY"]),
    "gemini": (
        "https://generativelanguage.googleapis.com/v1beta/openai/",
        ["GEMINI_API_KEY", "GOOGLE_API_KEY"],
    ),
    "anthropic": ("https://api.anthropic.com/v1/", ["ANTHROPIC_API_KEY"]),
    "deepseek": ("https://api.deepseek.com/v1", ["DEEPSEEK_API_KEY"]),
    "mistral": ("https://api.mistral.ai/v1", ["MISTRAL_API_KEY"]),
    "cohere": ("https://api.cohere.com/v1", ["CO_API_KEY", "COHERE_API_KEY"]),
    "huggingface": (None, ["HF_TOKEN", "HUGGINGFACE_API_KEY"]),
    "ollama": (None, []),
    "vllm": (None, []),
}

# Bare model-name prefixes → provider  (backwards compat for --model gpt-5, etc.)
_BARE_PREFIX_MAP: Dict[str, str] = {
    "gpt-": "openai",
    "o1": "openai",
    "o3": "openai",
    "o4": "openai",
    "gemini-": "gemini",
    "claude-": "anthropic",
    "deepseek-": "deepseek",
    "mistral-": "mistral",
    "command-": "cohere",
}


def _parse_model_spec(model_str: str) -> tuple:
    """Parse a model string into ``(provider, model_name, default_api_base, env_vars)``.

    Supports:
      - ``provider/model``  (e.g. ``gemini/gemini-3-pro``)
      - bare names with known prefix (e.g. ``gemini-3-pro`` → gemini)
      - unknown bare names default to ``openai``
    """
    if "/" in model_str:
        provider, _, model_name = model_str.partition("/")
        provider_lower = provider.lower()
        if provider_lower in _PROVIDERS:
            api_base, env_vars = _PROVIDERS[provider_lower]
            return provider_lower, model_name, api_base, env_vars

    for prefix, provider in _BARE_PREFIX_MAP.items():
        if model_str.startswith(prefix):
            api_base, env_vars = _PROVIDERS[provider]
            return provider, model_str, api_base, env_vars

    api_base, env_vars = _PROVIDERS["openai"]
    return "openai", model_str, api_base, env_vars


def _resolve_api_key_from_env(env_vars: Optional[List[str]] = None) -> Optional[str]:
    """Return the first API key found in *env_vars*, falling back to ``OPENAI_API_KEY``.

    *env_vars* typically comes from ``_parse_model_spec()``.
    """
    for var in env_vars or []:
        key = os.environ.get(var)
        if key:
            return key
    return os.environ.get("OPENAI_API_KEY")


def _expand_env_vars(text: str) -> str:
    """Expand ${VAR} patterns in text with environment variable values."""

    def _replacer(match):
        return os.environ.get(match.group(1), match.group(0))

    return re.sub(r"\$\{(\w+)\}", _replacer, text)


# ═══════════════════════════════════════════════════════════════════════
# 1. Context Builder — assembles LLM prompts (prompt/)
# ═══════════════════════════════════════════════════════════════════════


@dataclass
class ContextBuilderConfig:
    """Configuration for prompt generation"""

    template: str = "default"  # "default", "evox"
    template_dir: Optional[str] = None
    system_message: str = "system_message"
    evaluator_system_message: str = "evaluator_system_message"

    suggest_simplification_after_chars: Optional[int] = 500


# ═══════════════════════════════════════════════════════════════════════
# 2. Solution Generator — produces candidates via LLM calls (llm/)
# ═══════════════════════════════════════════════════════════════════════


@dataclass
class LLMModelConfig:
    """Configuration for a single LLM model"""

    # API configuration
    api_base: Optional[str] = None
    api_key: Optional[str] = None
    name: Optional[str] = None

    # Custom LLM client
    init_client: Optional[Callable] = None

    # Weight for model in pool, default to random sampling model based on weight
    weight: float = 1.0

    # Generation parameters
    system_message: Optional[str] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    max_tokens: Optional[int] = None

    # Request parameters
    timeout: Optional[int] = None
    retries: Optional[int] = None
    retry_delay: Optional[int] = None

    # Reasoning parameters
    reasoning_effort: Optional[str] = None


@dataclass
class LLMConfig(LLMModelConfig):
    """Configuration for LLM models"""

    # API configuration
    api_base: str = _PROVIDERS["openai"][0]

    # Generation parameters
    system_message: Optional[str] = "system_message"
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = None
    max_tokens: int = 32000

    # Request parameters
    timeout: int = 600
    retries: int = 3
    retry_delay: int = 5

    # model(s) for solution discovery
    models: List[LLMModelConfig] = field(default_factory=list)

    # model(s) for evaluator
    evaluator_models: List[LLMModelConfig] = field(default_factory=lambda: [])

    # model(s) for guide tasks (idea generation, paradigm breakthroughs, etc.)
    # If not specified, falls back to using the main 'models' list
    guide_models: List[LLMModelConfig] = field(default_factory=lambda: [])

    # Reasoning parameters (inherited from LLMModelConfig but can be overridden)
    reasoning_effort: Optional[str] = None

    def __post_init__(self):
        """Post-initialization to set up model configurations"""
        # If no evaluator models are defined, use the same models as for solution discovery
        if not self.evaluator_models:
            self.evaluator_models = self.models.copy()

        # If no guide models are defined, use the same models as for solution discovery
        if not self.guide_models:
            self.guide_models = self.models.copy()

        # Resolve per-model api_base, api_key, and bare name from provider prefix
        # Check if user explicitly set api_base at the LLMConfig level
        # (i.e. it differs from the hardcoded default).  When a custom api_base
        # is provided, we should NOT override it with the provider default so
        # that update_model_params() below can propagate the user's value.
        user_set_api_base = self.api_base.rstrip("/") != _PROVIDERS["openai"][0].rstrip("/")
        for model in self.models + self.evaluator_models + self.guide_models:
            if model.name and model.api_base is None:
                provider, bare_name, provider_base, env_vars = _parse_model_spec(model.name)
                # Skip provider URL only for unrecognized bare names that fell
                # through to the OpenAI default — never for an explicitly-prefixed
                # provider (e.g. "anthropic/claude-3-sonnet") or a known bare prefix.
                is_fallback = provider == "openai" and not (
                    model.name.startswith("openai/")
                    or any(model.name.startswith(p) for p in _BARE_PREFIX_MAP)
                )
                if provider_base and not (user_set_api_base and is_fallback):
                    model.api_base = provider_base
                if model.api_key is None:
                    model.api_key = _resolve_api_key_from_env(env_vars)
                # Strip provider prefix so the API receives the bare model name
                if "/" in model.name and provider != "openai":
                    model.name = bare_name

        # Update models with shared configuration values
        shared_config = {
            "api_base": self.api_base,
            "api_key": self.api_key,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": self.max_tokens,
            "timeout": self.timeout,
            "retries": self.retries,
            "retry_delay": self.retry_delay,
            "reasoning_effort": self.reasoning_effort,
        }
        self.update_model_params(shared_config)

    def update_model_params(self, args: Dict[str, Any], overwrite: bool = False) -> None:
        """Update model parameters for all models (including guide_models)."""
        all_models = self.models + self.evaluator_models + self.guide_models
        for model in all_models:
            for key, value in args.items():
                if overwrite or getattr(model, key, None) is None:
                    setattr(model, key, value)


@dataclass
class AgenticConfig:
    """Configuration for agentic solution generation.

    When enabled, replaces the single-shot LLM call with a multi-turn
    tool-calling agent loop that can read files and search the codebase
    before outputting the discovered solution.
    """

    enabled: bool = False
    codebase_root: Optional[str] = None

    # Agent loop limits
    max_steps: int = 5

    # Timeouts (seconds)
    per_step_timeout: float = 60.0
    overall_timeout: float = 300.0

    # Context management
    max_context_chars: int = 400_000
    max_file_chars: int = 50_000
    max_search_results: int = 50
    max_files_read: int = 20

    # Regex safety
    regex_timeout: float = 2.0
    max_regex_length: int = 200

    # Repo map — a depth-limited directory tree injected into the agent's first
    # message so it knows what files are available to read_file/search.
    repo_map_max_depth: int = 4

    # File access
    allowed_extensions: tuple = (
        ".py",
        ".txt",
        ".md",
        ".json",
        ".yaml",
        ".yml",
        ".toml",
        ".cfg",
        ".ini",
        ".js",
        ".ts",
        ".java",
        ".cpp",
        ".c",
        ".h",
        ".rs",
        ".go",
    )
    excluded_dirs: tuple = (
        ".git",
        "__pycache__",
        "node_modules",
        ".venv",
        "venv",
        ".mypy_cache",
        ".pytest_cache",
        "dist",
        "build",
    )


# ═══════════════════════════════════════════════════════════════════════
# 3. Evaluator — scores candidates and logs metadata (evaluation/)
# ═══════════════════════════════════════════════════════════════════════


@dataclass
class EvaluatorConfig:
    """Configuration for program evaluation"""

    evaluation_file: Optional[str] = None
    file_suffix: str = ".py"
    is_image_mode: bool = False

    timeout: int = 360
    max_retries: int = 3

    # Evaluation strategies
    cascade_evaluation: bool = True
    cascade_thresholds: List[float] = field(default_factory=lambda: [0.3, 0.6])

    # When True, the evaluator source code (or instruction.md for Harbor
    # tasks) is prepended to the LLM system message so the model can see
    # exactly how solutions are scored.  Disabled by default to avoid
    # leaking implementation details that may introduce noise.
    inject_evaluator_context: bool = False

    # LLM-as-a-judge: when True, an LLMJudge scores programs alongside the
    # evaluator and appends llm_* metrics to the result.
    # This will read from prompt.evaluator_system_message if provided, otherwise use the default system prompt.
    llm_as_judge: bool = False


# ═════════════════════════════════════════════════════════════════════════════════════════════
# 4. Solution Selector — maintains database and strategy to pick prior programs (search/)
# ═════════════════════════════════════════════════════════════════════════════════════════════


@dataclass
class DatabaseConfig:
    """Base configuration shared by all database types."""

    db_path: Optional[str] = None
    log_prompts: bool = True


@dataclass
class EvolveDatabaseConfig(DatabaseConfig):
    """Read database from a file."""

    database_file_path: Optional[str] = None


@dataclass
class EvoxDatabaseConfig(EvolveDatabaseConfig):
    """Evox (co-evolution) database config with built-in defaults."""

    evaluation_file: Optional[str] = None
    config_path: Optional[str] = None
    auto_generate_variation_operators: bool = True

    _evox_config_dir = Path(__file__).parent / "search" / "evox" / "config"
    _evox_database_dir = Path(__file__).parent / "search" / "evox" / "database"

    def __post_init__(self):
        if self.database_file_path is None:
            # Initial guide strategy for the solution discovery
            self.database_file_path = str(self._evox_database_dir / "initial_search_strategy.py")
        if self.evaluation_file is None:
            # Dummy evaluator for the guide strategy
            self.evaluation_file = str(self._evox_database_dir / "search_strategy_evaluator.py")
        if self.config_path is None:
            # Default config for the guide strategy
            self.config_path = str(self._evox_config_dir / "search.yaml")


@dataclass
class BeamSearchDatabaseConfig(DatabaseConfig):
    """Beam search database config."""

    beam_width: int = 5
    beam_selection_strategy: str = "diversity_weighted"
    beam_diversity_weight: float = 0.3
    beam_temperature: float = 1.0
    beam_depth_penalty: float = 0.0


@dataclass
class BestOfNDatabaseConfig(DatabaseConfig):
    """Best-of-N database config."""

    best_of_n: int = 5


@dataclass
class AdaEvolveDatabaseConfig(DatabaseConfig):
    """AdaEvolve adaptive multi-island database config."""

    population_size: int = 20
    num_islands: int = 2
    decay: float = 0.9
    intensity_min: float = 0.15
    intensity_max: float = 0.5
    use_adaptive_search: bool = True
    use_ucb_selection: bool = True
    use_migration: bool = True
    use_unified_archive: bool = True
    fixed_intensity: float = 0.4
    migration_interval: int = 15
    migration_count: int = 5
    local_context_program_ratio: float = 0.6
    archive_elite_ratio: float = 0.2
    pareto_weight: float = 0.4
    fitness_weight: float = 1.0
    novelty_weight: float = 0.0
    k_neighbors: int = 5
    diversity_strategy: str = "code"
    use_dynamic_islands: bool = True
    max_islands: int = 5
    spawn_productivity_threshold: float = 0.015
    spawn_cooldown_iterations: int = 30
    use_paradigm_breakthrough: bool = True
    paradigm_window_size: int = 10
    paradigm_improvement_threshold: float = 0.12
    paradigm_max_uses: int = 2
    paradigm_num_to_generate: int = 3
    paradigm_max_tried: int = 10

    # Stagnation handling
    stagnation_threshold: int = 10
    stagnation_multi_child_count: int = 3

    # Sibling context
    sibling_context_limit: int = 5

    # Error retry
    enable_error_retry: bool = True
    max_error_retries: int = 2

    # Archive
    archive_size: int = 100

    # Metric direction
    higher_is_better: Dict[str, bool] = field(default_factory=dict)
    fitness_key: Optional[str] = None
    pareto_objectives: List[str] = field(default_factory=list)
    pareto_objectives_weight: float = 0.0


@dataclass
class OpenEvolveNativeDatabaseConfig(DatabaseConfig):
    """OpenEvolve Native: MAP-Elites + island-based search config."""

    num_islands: int = 5
    population_size: int = 40
    archive_size: int = 100
    exploration_ratio: float = 0.2
    exploitation_ratio: float = 0.7
    elite_selection_ratio: float = 0.1
    feature_dimensions: List[str] = field(default_factory=lambda: ["complexity", "diversity"])
    feature_bins: int = 10
    diversity_reference_size: int = 20
    migration_interval: int = 10
    migration_rate: float = 0.1
    random_seed: Optional[int] = 42


@dataclass
class ClaudeCodeConfig(DatabaseConfig):
    """Configuration for the Claude Code baseline.

    Claude Code runs autonomously inside a Docker container, iterating on
    the solution using the evaluator directly.  max_turns maps to the
    --max-turns flag passed to the claude CLI.
    """

    max_turns: int = 50
    docker_image: str = "skydiscover-claude-code:latest"


@dataclass
class GEPANativeDatabaseConfig(DatabaseConfig):
    """Configuration for GEPA Native search database.

    GEPA (Guided Evolution for Program Adaptation) uses an elite pool with
    epsilon-greedy selection, acceptance gating, and LLM-mediated merge.
    """

    population_size: int = 40
    candidate_selection_strategy: str = "epsilon_greedy"  # "epsilon_greedy", "best", "pareto"
    epsilon: float = 0.1
    max_rejection_history: int = 20

    # Controller-read settings (stored here for single config source)
    acceptance_gating: bool = True
    use_merge: bool = True
    merge_after_stagnation: int = 15
    max_merge_attempts: int = 10
    max_recent_failures: int = 5
    random_seed: Optional[int] = 42


_DB_CONFIG_BY_TYPE: Dict[str, type] = {
    "evox": EvoxDatabaseConfig,
    "beam_search": BeamSearchDatabaseConfig,
    "best_of_n": BestOfNDatabaseConfig,
    "topk": DatabaseConfig,
    "adaevolve": AdaEvolveDatabaseConfig,
    "openevolve_native": OpenEvolveNativeDatabaseConfig,
    "gepa_native": GEPANativeDatabaseConfig,
    "claude_code": ClaudeCodeConfig,
}


@dataclass
class SearchConfig:
    """General Configuration for All Search Algorithms"""

    type: str = "topk"
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    num_context_programs: int = 4
    output_dir: Optional[str] = None
    switch_interval: Optional[int] = (
        None  # EvoX: stagnation iters before strategy switch. Auto-calculated if None.
    )
    share_llm: bool = (
        False  # EvoX: if True, meta-level search evolution uses the same LLM as the main discovery process.
    )


# ═══════════════════════════════════════════════════════════════════════
# Extras — live monitor dashboard
# ═══════════════════════════════════════════════════════════════════════


@dataclass
class MonitorConfig:
    """Configuration for the live run monitor dashboard"""

    enabled: bool = False
    port: int = 8765
    host: str = "127.0.0.1"
    max_solution_length: int = 10000

    # AI summary settings
    summary_model: str = "gpt-5-mini"
    summary_api_key: Optional[str] = None  # Falls back to OPENAI_API_KEY
    summary_api_base: str = _PROVIDERS["openai"][0]
    summary_top_k: int = 3
    summary_interval: int = 0  # Auto-generate every N programs (0 = manual)


# ═══════════════════════════════════════════════════════════════════════
# Master Configuration
# ═══════════════════════════════════════════════════════════════════════


@dataclass
class Config:
    """Master configuration for SkyDiscover"""

    # General settings
    max_iterations: int = 100
    checkpoint_interval: int = 10
    log_level: str = "INFO"
    log_dir: Optional[str] = None
    language: Optional[str] = None
    file_suffix: str = ".py"

    # Component configurations
    llm: LLMConfig = field(default_factory=LLMConfig)
    context_builder: ContextBuilderConfig = field(default_factory=ContextBuilderConfig)
    search: SearchConfig = field(default_factory=SearchConfig)
    evaluator: EvaluatorConfig = field(default_factory=EvaluatorConfig)
    agentic: AgenticConfig = field(default_factory=AgenticConfig)

    # Live monitor dashboard
    monitor: MonitorConfig = field(default_factory=MonitorConfig)

    # Human feedback settings
    human_feedback_enabled: bool = False
    human_feedback_file: Optional[str] = None
    human_feedback_mode: str = "append"  # "append" or "replace"

    # Generation settings
    diff_based_generation: bool = True
    max_solution_length: int = 60000

    # Parallelism — how many iterations run concurrently.
    # 1 = sequential (default, current behaviour).
    # >1 = N iterations overlap via asyncio tasks: while one evaluates,
    #       others can sample/generate, giving near-linear speedup.
    max_parallel_iterations: int = 1

    # Runtime-only: system prompt override (set by apply_overrides, read by external backends)
    system_prompt_override: Optional[str] = None

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> Config:
        """Load configuration from a YAML file"""
        config_path = Path(path)
        config_dir = config_path.parent

        with open(path, "r") as f:
            raw = f.read()
        config_dict = yaml.safe_load(_expand_env_vars(raw))

        # Handle file references for system_message
        if "prompt" in config_dict and "system_message" in config_dict["prompt"]:
            system_message = config_dict["prompt"]["system_message"]
            if (
                isinstance(system_message, str)
                and "\n" not in system_message.strip()
                and len(system_message.strip()) < 256
            ):
                file_path = config_dir / system_message
                try:
                    if file_path.exists() and file_path.is_file():
                        with open(file_path, "r") as f:
                            config_dict["prompt"]["system_message"] = f.read()
                except OSError:
                    logger.debug("Could not read system_message from %s", file_path, exc_info=True)

        return cls.from_dict(config_dict)

    def to_yaml(self, path: Union[str, Path]) -> None:
        """Save configuration to a YAML file"""
        with open(path, "w") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> Config:
        """Create configuration from a dictionary"""
        # Handle nested configurations
        config = Config()

        # Update top-level fields
        for key, value in config_dict.items():
            if key not in [
                "llm",
                "prompt",
                "database",
                "search",
                "evaluator",
                "agentic",
                "monitor",
            ] and hasattr(config, key):
                setattr(config, key, value)

        # Update nested configs
        if "llm" in config_dict:
            llm_dict = config_dict["llm"]
            if "models" in llm_dict:
                llm_dict["models"] = [LLMModelConfig(**m) for m in llm_dict["models"]]
            if "evaluator_models" in llm_dict:
                llm_dict["evaluator_models"] = [
                    LLMModelConfig(**m) for m in llm_dict["evaluator_models"]
                ]
            if "guide_models" in llm_dict:
                llm_dict["guide_models"] = [LLMModelConfig(**m) for m in llm_dict["guide_models"]]
            config.llm = LLMConfig(**llm_dict)
        if "prompt" in config_dict:
            config.context_builder = ContextBuilderConfig(**config_dict["prompt"])

        if "search" in config_dict:
            search_dict = config_dict["search"]
            search_type = search_dict.get("type", "topk")
            db_config_cls = _DB_CONFIG_BY_TYPE.get(search_type, DatabaseConfig)
            if "database" in search_dict:
                db_dict = search_dict["database"]
                # Separate known fields from algorithm-specific extras
                # (e.g., adaevolve's decay, intensity_min, use_adaptive_search, etc.)
                known_fields = {f.name for f in fields(db_config_cls)}
                db_known = {k: v for k, v in db_dict.items() if k in known_fields}
                db_extras = {k: v for k, v in db_dict.items() if k not in known_fields}
                db_config = db_config_cls(**db_known)
                for k, v in db_extras.items():
                    setattr(db_config, k, v)
                search_dict["database"] = db_config
            else:
                search_dict["database"] = db_config_cls()
            config.search = SearchConfig(**search_dict)

        if "evaluator" in config_dict:
            config.evaluator = EvaluatorConfig(**config_dict["evaluator"])
        if "agentic" in config_dict:
            agentic_dict = dict(config_dict["agentic"])  # copy to avoid mutating input
            # Convert list fields to tuples for the dataclass
            for tuple_field in ("allowed_extensions", "excluded_dirs"):
                if tuple_field in agentic_dict and isinstance(agentic_dict[tuple_field], list):
                    agentic_dict[tuple_field] = tuple(agentic_dict[tuple_field])
            config.agentic = AgenticConfig(**agentic_dict)
        if "monitor" in config_dict:
            config.monitor = MonitorConfig(**config_dict["monitor"])

        return config

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to a dictionary"""
        return {
            # General settings
            "max_iterations": self.max_iterations,
            "checkpoint_interval": self.checkpoint_interval,
            "log_level": self.log_level,
            "log_dir": self.log_dir,
            # Component configurations
            "llm": {
                "models": self.llm.models,
                "evaluator_models": self.llm.evaluator_models,
                "api_base": self.llm.api_base,
                "temperature": self.llm.temperature,
                "top_p": self.llm.top_p,
                "max_tokens": self.llm.max_tokens,
                "timeout": self.llm.timeout,
                "retries": self.llm.retries,
                "retry_delay": self.llm.retry_delay,
            },
            "prompt": {
                "template": self.context_builder.template,
                "template_dir": self.context_builder.template_dir,
                "system_message": self.context_builder.system_message,
                "evaluator_system_message": self.context_builder.evaluator_system_message,
            },
            "search": {
                "type": self.search.type,
                "num_context_programs": self.search.num_context_programs,
                "database": {
                    f.name: getattr(self.search.database, f.name)
                    for f in fields(self.search.database)
                },
            },
            "evaluator": {
                "evaluation_file": self.evaluator.evaluation_file,
                "file_suffix": self.evaluator.file_suffix,
                "is_image_mode": self.evaluator.is_image_mode,
                "timeout": self.evaluator.timeout,
                "max_retries": self.evaluator.max_retries,
                "cascade_evaluation": self.evaluator.cascade_evaluation,
                "cascade_thresholds": self.evaluator.cascade_thresholds,
                "inject_evaluator_context": self.evaluator.inject_evaluator_context,
                "llm_as_judge": self.evaluator.llm_as_judge,
            },
            # Agentic generation
            "agentic": {
                "enabled": self.agentic.enabled,
                "codebase_root": self.agentic.codebase_root,
                "max_steps": self.agentic.max_steps,
                "per_step_timeout": self.agentic.per_step_timeout,
                "overall_timeout": self.agentic.overall_timeout,
                "max_context_chars": self.agentic.max_context_chars,
                "max_file_chars": self.agentic.max_file_chars,
                "max_search_results": self.agentic.max_search_results,
                "max_files_read": self.agentic.max_files_read,
                "regex_timeout": self.agentic.regex_timeout,
                "max_regex_length": self.agentic.max_regex_length,
                "repo_map_max_depth": self.agentic.repo_map_max_depth,
                "allowed_extensions": list(self.agentic.allowed_extensions),
                "excluded_dirs": list(self.agentic.excluded_dirs),
            },
            # Live monitor
            "monitor": {
                "enabled": self.monitor.enabled,
                "port": self.monitor.port,
                "host": self.monitor.host,
                "max_solution_length": self.monitor.max_solution_length,
                "summary_model": self.monitor.summary_model,
                "summary_top_k": self.monitor.summary_top_k,
                "summary_interval": self.monitor.summary_interval,
            },
            # Human-in-the-loop
            "human_feedback_enabled": self.human_feedback_enabled,
            "human_feedback_file": self.human_feedback_file,
            # Generation settings
            "diff_based_generation": self.diff_based_generation,
            "max_solution_length": self.max_solution_length,
            # Parallelism
            "max_parallel_iterations": self.max_parallel_iterations,
        }


def load_config(config_path: Optional[Union[str, Path]] = None) -> Config:
    """Load configuration from a YAML file or use defaults"""
    if config_path:
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")
        config = Config.from_yaml(config_path)
    else:
        config = Config()

    # Update api_base from environment if provided — use overwrite=True
    # because __post_init__ already pushed the hardcoded default to all models.
    api_base = os.environ.get("OPENAI_API_BASE") or os.environ.get("OPENAI_BASE_URL")
    if api_base:
        config.llm.api_base = api_base
        config.llm.update_model_params({"api_base": api_base}, overwrite=True)

    # Determine which API key to use (provider-aware)
    if not config.llm.api_key:
        env_vars = None
        if config.llm.models:
            first_model_name = config.llm.models[0].name
            if first_model_name:
                _, _, _, env_vars = _parse_model_spec(first_model_name)
        api_key = _resolve_api_key_from_env(env_vars)
        if api_key:
            config.llm.api_key = api_key
            config.llm.update_model_params({"api_key": api_key})

    # Make the system message available to the individual models, in case it is not provided from the prompt sampler
    config.llm.update_model_params({"system_message": config.context_builder.system_message})

    # Bridge provider env vars so that downstream configs (e.g. evox search.yaml)
    # can resolve ${OPENAI_API_KEY} from the environment.
    bridge_provider_env(config)

    return config


def bridge_provider_env(config: Config) -> None:
    """
    Set provider-specific env vars from resolved config.

    External backends read credentials from environment variables directly.
    """
    if not config.llm.models:
        return
    model = config.llm.models[0]
    if not model.api_key:
        return

    # Use _parse_model_spec to get the right env vars for this model
    _, _, _, env_vars = _parse_model_spec(model.name or "")
    for var in env_vars:
        os.environ.setdefault(var, model.api_key)

    # Always ensure OPENAI_API_KEY is set — many tools (ShinkaEvolve, etc.) expect it
    os.environ.setdefault("OPENAI_API_KEY", model.api_key)

    # Set OPENAI_API_BASE so backends that check it can find the endpoint
    if model.api_base:
        os.environ.setdefault("OPENAI_API_BASE", model.api_base)


def build_output_dir(search_type: str, initial_program_path: str, base_dir: str = "outputs") -> str:
    """Build a standardized output directory: outputs/<search_type>/<problem_name>_<MMDD_HHMM>/"""
    from datetime import datetime

    problem_name = (
        os.path.basename(os.path.dirname(os.path.abspath(initial_program_path))) or "unknown"
    )
    timestamp = datetime.now().strftime("%m%d_%H%M")
    return os.path.join(base_dir, search_type, f"{problem_name}_{timestamp}")


# ═══════════════════════════════════════════════════════════════════════
# Runtime overrides — shared by the public API and CLI
# ═══════════════════════════════════════════════════════════════════════


def apply_overrides(
    config: Config,
    *,
    model: Optional[str] = None,
    api_base: Optional[str] = None,
    agentic: bool = False,
    search: Optional[str] = None,
    system_prompt: Optional[str] = None,
) -> None:
    """Apply runtime overrides (model, api_base, etc.) to a loaded Config in place."""
    if model:
        # Parse the model string into a list of model specifications
        specs = [s.strip() for s in model.split(",")]
        models: List[LLMModelConfig] = []
        for spec in specs:
            provider, model_name, default_api_base, env_vars = _parse_model_spec(spec)
            effective_base = api_base or default_api_base
            if effective_base is None:
                raise ValueError(
                    f"Provider '{provider}' requires an explicit api_base.\n"
                    f"Example: model='{spec}', api_base='http://localhost:8000/v1'"
                )
            resolved_key = _resolve_api_key_from_env(env_vars)
            models.append(
                LLMModelConfig(
                    name=model_name,
                    api_base=effective_base,
                    api_key=resolved_key,
                )
            )

        config.llm.api_base = models[0].api_base
        if models[0].api_key:
            config.llm.api_key = models[0].api_key
        config.llm.models = models
        config.llm.evaluator_models = [
            LLMModelConfig(name=m.name, api_base=m.api_base, api_key=m.api_key) for m in models
        ]
        config.llm.guide_models = [
            LLMModelConfig(name=m.name, api_base=m.api_base, api_key=m.api_key) for m in models
        ]
    elif api_base:
        config.llm.api_base = api_base
        config.llm.update_model_params({"api_base": api_base}, overwrite=True)

    # API key (api_base-only; multi-model already resolved above)
    if not model and api_base:
        parsed_env_vars: Optional[List[str]] = None
        for _prefix, (base_url, env_list) in _PROVIDERS.items():
            if base_url and config.llm.api_base.startswith(base_url.rstrip("/")):
                parsed_env_vars = env_list
                break
        resolved_key = _resolve_api_key_from_env(parsed_env_vars)
        if resolved_key:
            config.llm.api_key = resolved_key
            config.llm.update_model_params({"api_key": resolved_key}, overwrite=True)

    # Propagate shared generation/request settings
    if model or api_base:
        config.llm.update_model_params(
            {
                "temperature": config.llm.temperature,
                "top_p": config.llm.top_p,
                "max_tokens": config.llm.max_tokens,
                "timeout": config.llm.timeout,
                "retries": config.llm.retries,
                "retry_delay": config.llm.retry_delay,
                "reasoning_effort": config.llm.reasoning_effort,
            },
            overwrite=True,
        )
        # Fill api_base/api_key only where a model doesn't already have them
        config.llm.update_model_params(
            {"api_base": config.llm.api_base, "api_key": config.llm.api_key},
            overwrite=False,
        )

    if agentic:
        config.agentic.enabled = True

    if search:
        if not hasattr(config, "search"):
            config.search = SearchConfig()
        config.search.type = search
        new_db_cls = _DB_CONFIG_BY_TYPE.get(search)
        if new_db_cls and not isinstance(config.search.database, new_db_cls):
            config.search.database = new_db_cls()

    if system_prompt:
        config.context_builder.system_message = system_prompt
        config.system_prompt_override = system_prompt
