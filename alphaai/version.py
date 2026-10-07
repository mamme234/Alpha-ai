"""Single source of truth for the AlphaAI release version."""

from __future__ import annotations

__version__ = "1.0.0"

#: Version of the AlphaAI Core <-> ModelEngine interface contract. Engines
#: advertise the interface version they implement so that third-party engine
#: adapters can be validated at registration time.
CORE_INTERFACE_VERSION = "1.0"
