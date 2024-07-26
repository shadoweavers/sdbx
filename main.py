import asyncio
import warnings

from sdbx.cmd.main import main

if __name__ == "__main__":
    warnings.warn("main.py is deprecated. Start sdbx by installing the package through the instructions in the README, not by cloning the repository.", DeprecationWarning)
    asyncio.run(main())
