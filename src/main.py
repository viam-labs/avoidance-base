"""Module entrypoint for ``viam-labs:base:avoidance``."""

from __future__ import annotations

import asyncio

from viam.module.module import Module

from .avoidance_base import AvoidanceBase
from .runtime import set_module


async def main() -> None:
    module = Module.from_args()
    set_module(module)
    module.add_model_from_registry(AvoidanceBase.API, AvoidanceBase.MODEL)
    await module.start()


if __name__ == "__main__":
    asyncio.run(main())
