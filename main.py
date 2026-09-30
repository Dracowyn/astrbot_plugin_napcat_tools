from astrbot.api import AstrBotConfig, logger
from astrbot.api.star import Context, Star

from .tools import NapCatToolkit


class NapCatToolsPlugin(Star):
    """Expose NapCat (OneBot v11) actions to the LLM as function tools.

    The plugin registers tools only; it adds no event hooks, prompt injection or
    reply decoration. Tools are registered according to the config, and AstrBot
    reloads the plugin whenever the config is saved.
    """

    def __init__(self, context: Context, config: AstrBotConfig):
        """Register the tools enabled in the config.

        Args:
            context: The star context.
            config: The plugin config.
        """
        super().__init__(context)
        tools = NapCatToolkit(context, config).build()
        if tools:
            context.add_llm_tools(*tools)
        logger.info(f"NapCat tools enabled: {[t.name for t in tools]}")
