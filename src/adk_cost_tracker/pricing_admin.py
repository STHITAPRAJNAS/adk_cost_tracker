"""
Example Administrative Script for Pricing Management.
Use this pattern to update the 'Source of Truth' in your shared database.
"""

import asyncio
import logging
from adk_cost_tracker.pricing import registry
from adk_cost_tracker.store import make_store

logging.basicConfig(level=logging.INFO)

async def main():
    # 1. Connect to your shared production database
    # store = make_store("postgresql://admin:pw@host/mydb")
    store = make_store() # Default local SQLite for demonstration

    # 2. Register new models or update existing ones in the local registry
    # This keeps the logic agnostic: you just provide keys and numbers.
    registry.register(
        model_key="my-custom-fine-tuned-model",
        provider="internal",
        input_per_m=1.50,
        output_per_m=4.00
    )
    
    # 3. Push the registry to the database
    # All ADK apps syncing with this DB will pick up these prices on their next restart.
    await registry.update_store(store)
    
    print("Pricing database updated successfully.")

if __name__ == "__main__":
    asyncio.run(main())
