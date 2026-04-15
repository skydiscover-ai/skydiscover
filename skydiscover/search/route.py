"""
Routing for search algorithms.

Maps the ``--search`` flag to the right database, controller, and program class
at runtime.  The registries and factory functions live in ``registry.py``;
this module wires up implementations and provides ``get_discovery_controller``.
"""

import logging

from skydiscover.search.adaevolve.controller import AdaEvolveController
from skydiscover.search.adaevolve.database import AdaEvolveDatabase
from skydiscover.search.beam_search.database import BeamSearchDatabase

# Algorithm implementations
from skydiscover.search.best_of_n.database import BestOfNDatabase
from skydiscover.search.claude_code.controller import ClaudeCodeController
from skydiscover.search.claude_code.database import ClaudeCodeDatabase
from skydiscover.search.default_discovery_controller import (
    DiscoveryController,
    DiscoveryControllerInput,
)
from skydiscover.search.evox.controller import CoEvolutionController
from skydiscover.search.evox.database.search_strategy_db import SearchStrategyDatabase
from skydiscover.search.gepa_native.controller import GEPANativeController
from skydiscover.search.gepa_native.database import GEPANativeDatabase
from skydiscover.search.openevolve_native.database import OpenEvolveNativeDatabase
from skydiscover.search.registry import (
    _CONTROLLER_REGISTRY,
    register_controller,
    register_database,
)
from skydiscover.search.topk.database import TopKDatabase

logger = logging.getLogger(__name__)


######################### ROUTING #########################


def get_discovery_controller(controller_input: DiscoveryControllerInput) -> DiscoveryController:
    """
    Get the discovery controller for a given search type.

    Returns the registered controller class, or the default DiscoveryController
    if none is registered.
    """
    search_type = controller_input.config.search.type
    controller_class = _CONTROLLER_REGISTRY.get(search_type, DiscoveryController)
    logger.debug(f"Using controller {controller_class.__name__} for search type '{search_type}'")
    return controller_class(controller_input)


######################### AUTO-REGISTRATION #########################

register_database("best_of_n", BestOfNDatabase)
register_database("beam_search", BeamSearchDatabase)
register_database("topk", TopKDatabase)

# AdaEvolve
register_database("adaevolve", AdaEvolveDatabase)
register_controller("adaevolve", AdaEvolveController)

# OpenEvolve Native
register_database("openevolve_native", OpenEvolveNativeDatabase)

# EvoX
register_controller("evox", CoEvolutionController)
register_database("evox_meta", SearchStrategyDatabase)

# GEPA Native: guided evolution with acceptance gating and merge
register_database("gepa_native", GEPANativeDatabase)
register_controller("gepa_native", GEPANativeController)

# Claude Code: single-agent baseline running Claude CLI in a container
register_database("claude_code", ClaudeCodeDatabase)
register_controller("claude_code", ClaudeCodeController)
