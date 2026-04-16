"""
Administrative utilities for managing LLM pricing and usage.
Use this pattern to update the 'Source of Truth' in your shared database.
"""

import asyncio
import logging

from .pricing import registry
from .store import make_store

logger = logging.getLogger(__name__)

async def update_pricing(dsn: str | None = None):
    """
    Pushes current registry pricing to the specified database.
    This acts as the administrative 'Write' path for centralized pricing.
    """
    store = make_store(dsn)
    await registry.update_store(store)
    logger.info("Pricing database updated successfully.")

async def main():
    # Example standalone usage
    logging.basicConfig(level=logging.INFO)
    
    # 1. Register a new model key (completely agnostic)
    registry.register(
        model_key="my-agnostic-model",
        provider="internal",
        input_per_m=0.50,
        output_per_m=1.00
    )
    
    # 2. Push to database
    await update_pricing()

if __name__ == "__main__":
    asyncio.run(main())
