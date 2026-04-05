import logging
from typing import List, Optional, Union

from google.adk.tools.base_tool import BaseTool
from google.adk.tools.base_toolset import BaseToolset, ToolPredicate
from google.adk.agents.readonly_context import ReadonlyContext

logger = logging.getLogger(__name__)


class AGUIToolset(BaseToolset):
    """Placeholder toolset for AG-UI tool integration.

    Users add this to their ADK agent's tools list as a marker.
    At runtime, ADKAgent binds a real ClientProxyToolset via bind().

    Uses a lazy-delegation pattern: if get_tools() is called before
    bind() (e.g. by ADK 2.0's eager tool resolution), it returns an
    empty list instead of raising.  Once bound, all get_tools() calls
    delegate to the underlying ClientProxyToolset.
    """

    def __init__(
        self,
        *,
        tool_filter: Optional[Union[ToolPredicate, List[str]]] = None,
        tool_name_prefix: Optional[str] = None,
    ):
        """Initialize the toolset.

        Args:
            tool_filter: Filter to apply to tools.
            tool_name_prefix: The prefix to prepend to the names of the tools
                returned by the toolset.
        """
        self.tool_filter = tool_filter
        self.tool_name_prefix = tool_name_prefix
        self._delegate: Optional["BaseToolset"] = None

    def bind(self, delegate: "BaseToolset") -> None:
        """Bind the real toolset to delegate get_tools() calls to.

        Called by ADKAgent._start_background_execution() after constructing
        a ClientProxyToolset from the current request's tools.

        Args:
            delegate: The ClientProxyToolset to delegate to.
        """
        self._delegate = delegate
        logger.info(f"AGUIToolset bound to {type(delegate).__name__}")

    async def get_tools(
        self,
        readonly_context: Optional[ReadonlyContext] = None,
    ) -> list[BaseTool]:
        """Return tools from the bound delegate, or empty list if unbound.

        This supports both ADK 1.x (lazy resolution) and ADK 2.0 (eager
        resolution during Runner init).  When called before bind(), returns
        an empty list so that eager resolution doesn't crash.

        Args:
            readonly_context: Context used to filter tools available to
                the agent.

        Returns:
            list[BaseTool]: Tools from the delegate, or empty list if unbound.
        """
        if self._delegate is None:
            logger.debug(
                "AGUIToolset.get_tools() called before bind(); "
                "returning empty list (will be populated after bind)"
            )
            return []
        return await self._delegate.get_tools(readonly_context)

    async def close(self) -> None:
        """Clean up resources held by the delegate toolset, if any."""
        if self._delegate is not None:
            await self._delegate.close()
