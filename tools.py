"""LLM tool implementations and their JSON schemas."""

import base64
from datetime import datetime

import aiohttp
import mcp
from aiocqhttp.exceptions import ActionFailed

from astrbot.api import logger
from astrbot.api.event import AstrMessageEvent
from astrbot.api.star import Context

from .core import (
    NapCatTool,
    ToolError,
    call_action,
    check_management,
    find_reply,
    require_group,
    resolve_message,
    resolve_user,
)

# QQ system face IDs for set_msg_emoji_like. Collected from the QQ face table;
# verify each on a real NapCat account before extending the list.
EMOJIS = {
    "thumbs_up": 76,
    "heart": 66,
    "ok": 124,
    "laugh_cry": 182,
    "facepalm": 264,
    "celebrate": 320,
    "applause": 99,
    "rose": 63,
    "hug": 49,
    "smile": 14,
    "cute": 21,
    "proud": 4,
    "tears": 5,
    "cry": 9,
    "awkward": 10,
    "sweat": 27,
    "cool": 16,
    "smirk": 101,
    "victory": 79,
    "angry": 326,
}

MEMBER_FIELDS = ("nickname", "card", "role", "title", "level")
MEMBER_TIME_FIELDS = ("join_time", "last_sent_time")
# Whitelisted profile fields; anything the user hid or never filled is absent.
PROFILE_FIELDS = (
    "sex",
    "age",
    "qqLevel",
    "long_nick",
    "sign",
    "country",
    "province",
    "city",
    "home_town",
    "birthday_month",
    "birthday_day",
    "reg_time",
)
AVATAR_URL = "https://q1.qlogo.cn/g?b=qq&nk={qq}&s=640"
AVATAR_TIMEOUT_SECONDS = 15
# QQ rejects mutes longer than 30 days.
QQ_MAX_MUTE_MINUTES = 43200

USER_PARAM = {
    "type": "string",
    "description": (
        "Who to target: 'sender' (the user talking to you), 'quoted_sender' "
        "(author of the message the user replied to), 'at' or 'at:N' (the "
        "first / N-th user mentioned in the message), a QQ number, or a "
        "nickname / group card keyword."
    ),
}
EXPLICIT_USER_PARAM = {
    "type": "string",
    "description": (
        "Who to target: 'quoted_sender' (author of the message the user replied "
        "to) or 'at' / 'at:N' (user mentioned in the message). QQ numbers and "
        "names are not accepted."
    ),
}
MESSAGE_PARAM = {
    "type": "string",
    "enum": ["quoted", "current"],
    "description": (
        "'quoted' = the message the user replied to; 'current' = the user's "
        "own latest message."
    ),
}
EMPTY_PARAMS = {"type": "object", "properties": {}}

# Tool name suffix -> (description, JSON schema). The implementation is the
# NapCatToolkit method of the same name; the config key is also the same name.
TOOL_SPECS = {
    "poke": (
        "Poke (戳一戳) a QQ user, a light playful nudge. Works in groups and "
        "private chats.",
        {"type": "object", "properties": {"user": USER_PARAM}},
    ),
    "react_emoji": (
        "Add an emoji reaction to a message instead of, or in addition to, a "
        "text reply. One emoji per call.",
        {
            "type": "object",
            "properties": {
                "emoji": {"type": "string", "enum": list(EMOJIS)},
                "message": MESSAGE_PARAM,
            },
            "required": ["emoji"],
        },
    ),
    "get_user_info": (
        "Look up a QQ user's group member info (card, role, title, level, join "
        "time) and public profile. Fields the user hid or never filled are absent.",
        {"type": "object", "properties": {"user": USER_PARAM}},
    ),
    "recall_own_message": (
        "Recall (撤回) one of YOUR OWN messages that the user quoted. It cannot "
        "recall anything the user did not quote.",
        EMPTY_PARAMS,
    ),
    "stay_silent": (
        "End this turn without sending any text. Use it when a group message is "
        "not meant for you, or when an emoji reaction already said enough. Call "
        "it last, after any other tools. It is refused when the user addressed "
        "you directly.",
        EMPTY_PARAMS,
    ),
    "set_my_card": (
        "Change your own group card (群名片) in the current group. An empty "
        "string clears it.",
        {
            "type": "object",
            "properties": {"card": {"type": "string"}},
            "required": ["card"],
        },
    ),
    "view_avatar": (
        "Look at a QQ user's avatar image.",
        {"type": "object", "properties": {"user": USER_PARAM}},
    ),
    "mute_member": (
        "Mute (禁言) a group member when an admin explicitly asks you to. "
        "duration_minutes=0 unmutes.",
        {
            "type": "object",
            "properties": {
                "user": EXPLICIT_USER_PARAM,
                "duration_minutes": {"type": "integer", "minimum": 0},
            },
            "required": ["user", "duration_minutes"],
        },
    ),
    "recall_message": (
        "Recall (撤回) the message the user quoted, sent by anyone, when an admin "
        "explicitly asks you to.",
        EMPTY_PARAMS,
    ),
    "set_essence": (
        "Set a message as group essence (精华消息) when an admin explicitly asks "
        "you to.",
        {"type": "object", "properties": {"message": MESSAGE_PARAM}},
    ),
    "set_member_card": (
        "Change another member's group card (群名片) when an admin explicitly "
        "asks you to. An empty string clears it.",
        {
            "type": "object",
            "properties": {"user": EXPLICIT_USER_PARAM, "card": {"type": "string"}},
            "required": ["user", "card"],
        },
    ),
}


class NapCatToolkit:
    """Tool implementations sharing the plugin's context and config.

    Every public coroutine takes the current event plus the LLM arguments,
    returns text for the LLM, and raises ``ToolError`` for expected failures.
    """

    def __init__(self, context: Context, config: dict):
        self.context = context
        self.enabled = config.get("tools", {})
        permission = config.get("permission", {})
        self.allow_group_admin = permission.get("allow_group_admin", True)
        self.max_mute_minutes = max(
            1, min(int(permission.get("max_mute_minutes", 1440)), QQ_MAX_MUTE_MINUTES)
        )
        self.avatar_config = config.get("view_avatar", {})

    def build(self) -> list[NapCatTool]:
        """Create the tools enabled in the config.

        Returns:
            The enabled tools, named ``qq_<name>``.
        """
        return [
            NapCatTool(
                name=f"qq_{name}",
                description=description,
                parameters=parameters,
                impl=getattr(self, name),
            )
            for name, (description, parameters) in TOOL_SPECS.items()
            if self.enabled.get(name, False)
        ]

    async def poke(self, event: AstrMessageEvent, user: str = "sender") -> str:
        """Poke a user in the current group or private chat."""
        qq = await resolve_user(event, user)
        if event.get_group_id():
            await call_action(
                event, "group_poke", group_id=require_group(event), user_id=int(qq)
            )
        else:
            await call_action(event, "friend_poke", user_id=int(qq))
        return f"Poked {qq}."

    async def react_emoji(
        self, event: AstrMessageEvent, emoji: str, message: str = "current"
    ) -> str:
        """React to the quoted or current message with a curated emoji."""
        if emoji not in EMOJIS:
            raise ToolError(f"unknown emoji '{emoji}'.")
        message_id, _ = resolve_message(event, message)
        await call_action(
            event,
            "set_msg_emoji_like",
            message_id=message_id,
            emoji_id=str(EMOJIS[emoji]),
        )
        return f"Reacted with {emoji}."

    async def get_user_info(self, event: AstrMessageEvent, user: str = "sender") -> str:
        """Return a user's group member info and public profile fields."""
        qq = await resolve_user(event, user)
        lines = [f"user_id: {qq}"]
        member = {}
        if event.get_group_id():
            try:
                member = await call_action(
                    event,
                    "get_group_member_info",
                    group_id=require_group(event),
                    user_id=int(qq),
                    no_cache=True,
                )
            except ActionFailed:
                lines.append("group_member: no (not in this group)")
            member = member or {}
            lines += [f"{k}: {member[k]}" for k in MEMBER_FIELDS if member.get(k)]
            lines += [
                f"{k}: {datetime.fromtimestamp(int(member[k])):%Y-%m-%d %H:%M}"
                for k in MEMBER_TIME_FIELDS
                if member.get(k)
            ]
        profile = await call_action(event, "get_stranger_info", user_id=int(qq)) or {}
        if not member.get("nickname") and profile.get("nickname"):
            lines.append(f"nickname: {profile['nickname']}")
        for key in PROFILE_FIELDS:
            value = profile.get(key)
            if value in (None, "", 0, "unknown"):
                continue
            if key == "reg_time":
                value = f"{datetime.fromtimestamp(int(value)):%Y-%m-%d}"
            lines.append(f"{key}: {value}")
        return "\n".join(lines)

    async def recall_own_message(self, event: AstrMessageEvent) -> str:
        """Recall the quoted message if the bot sent it."""
        if find_reply(event) is None:
            raise ToolError(
                "you can only recall a message the user quoted. Ask them to "
                "reply to your message and ask again."
            )
        message_id, sender_id = resolve_message(event, "quoted")
        if sender_id != str(event.get_self_id()):
            raise ToolError("the quoted message was not sent by you.")
        await call_action(event, "delete_msg", message_id=message_id)
        return "Recalled your quoted message."

    async def stay_silent(self, event: AstrMessageEvent) -> str | None:
        """End the agent loop without a reply unless the user addressed the bot.

        Returning ``None`` makes AstrBot finish the loop cleanly without another
        LLM call. ``event.stop_event()`` is deliberately avoided: it aborts the
        run, cancels sibling tool calls and writes fake "Output stopped."
        messages into the conversation history.
        """
        if event.is_at_or_wake_command:
            return (
                "Refused: the user addressed you directly (mention, quote, "
                "private chat or wake word). Reply to them normally."
            )
        return None

    async def set_my_card(self, event: AstrMessageEvent, card: str) -> str:
        """Change the bot's own group card."""
        await call_action(
            event,
            "set_group_card",
            group_id=require_group(event),
            user_id=int(event.get_self_id()),
            card=card,
        )
        return f"Your group card is now '{card}'." if card else "Group card cleared."

    async def view_avatar(self, event: AstrMessageEvent, user: str = "sender"):
        """Show a user's avatar to the LLM, or describe it via a vision model.

        Returns:
            A description string when a vision model is configured, otherwise a
            ``CallToolResult`` carrying the image.
        """
        qq = await resolve_user(event, user)
        url = AVATAR_URL.format(qq=qq)
        # Fall back to AstrBot's global image caption provider.
        provider_id = self.avatar_config.get("vision_provider_id") or (
            self.context.get_config(umo=event.unified_msg_origin)
            .get("provider_settings", {})
            .get("default_image_caption_provider_id", "")
        )
        if provider_id:
            provider = self.context.get_provider_by_id(provider_id)
            if provider is None:
                raise ToolError(f"vision provider '{provider_id}' is not available.")
            try:
                llm_resp = await provider.text_chat(
                    prompt=self.avatar_config.get("vision_prompt")
                    or "Describe this avatar.",
                    image_urls=[url],
                )
            except Exception as e:
                logger.warning(f"Vision provider failed on avatar of {qq}: {e!r}")
                raise ToolError(
                    "the vision model failed to describe the avatar."
                ) from e
            if llm_resp.role == "err":
                raise ToolError(
                    f"the vision model returned an error: {llm_resp.completion_text}"
                )
            return (
                f"Avatar of {qq}, described by a vision model:\n"
                f"{llm_resp.completion_text}"
            )
        provider = await self.context.get_using_provider_async(event.unified_msg_origin)
        modalities = provider.provider_config.get("modalities") if provider else None
        if modalities and "image" not in modalities:
            raise ToolError(
                "the current model cannot view images and no vision model is configured."
            )
        timeout = aiohttp.ClientTimeout(total=AVATAR_TIMEOUT_SECONDS)
        try:
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(url) as resp:
                    resp.raise_for_status()
                    data = await resp.read()
                    mime = resp.content_type or "image/jpeg"
        except (aiohttp.ClientError, TimeoutError) as e:
            logger.warning(f"Failed to download avatar of {qq}: {e!r}")
            raise ToolError("failed to download the avatar.") from e
        return mcp.types.CallToolResult(
            content=[
                mcp.types.TextContent(
                    type="text",
                    text=f"Avatar of {qq}, for you to look at. Do not send it to "
                    "the chat unless the user asks.",
                ),
                mcp.types.ImageContent(
                    type="image", data=base64.b64encode(data).decode(), mimeType=mime
                ),
            ]
        )

    async def mute_member(
        self, event: AstrMessageEvent, user: str, duration_minutes: int
    ) -> str:
        """Mute or unmute a member, capped at the configured maximum."""
        group_id = require_group(event)
        qq = await resolve_user(event, user, explicit_only=True)
        await check_management(event, group_id, self.allow_group_admin, qq)
        requested = int(duration_minutes)
        minutes = max(0, min(requested, self.max_mute_minutes))
        await call_action(
            event,
            "set_group_ban",
            group_id=group_id,
            user_id=int(qq),
            duration=minutes * 60,
        )
        if minutes == 0:
            return f"Unmuted {qq}."
        capped = " (capped to the configured maximum)" if minutes < requested else ""
        return f"Muted {qq} for {minutes} minutes{capped}."

    async def recall_message(self, event: AstrMessageEvent) -> str:
        """Recall the quoted message, whoever sent it."""
        group_id = require_group(event)
        message_id, sender_id = resolve_message(event, "quoted")
        if not sender_id:
            raise ToolError("could not determine the author of the quoted message.")
        await check_management(event, group_id, self.allow_group_admin, sender_id)
        await call_action(event, "delete_msg", message_id=message_id)
        return "Recalled the quoted message."

    async def set_essence(
        self, event: AstrMessageEvent, message: str = "quoted"
    ) -> str:
        """Set the quoted or current message as group essence."""
        group_id = require_group(event)
        message_id, _ = resolve_message(event, message)
        await check_management(event, group_id, self.allow_group_admin)
        await call_action(event, "set_essence_msg", message_id=message_id)
        return "Set the message as essence."

    async def set_member_card(
        self, event: AstrMessageEvent, user: str, card: str
    ) -> str:
        """Change another member's group card."""
        group_id = require_group(event)
        qq = await resolve_user(event, user, explicit_only=True)
        await check_management(event, group_id, self.allow_group_admin, qq)
        await call_action(
            event, "set_group_card", group_id=group_id, user_id=int(qq), card=card
        )
        if not card:
            return f"Cleared the group card of {qq}."
        return f"Set the group card of {qq} to '{card}'."
