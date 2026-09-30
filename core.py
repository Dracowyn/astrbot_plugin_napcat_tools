"""Shared building blocks: the tool base class, OneBot calls, target resolution
and permission checks."""

from collections.abc import Awaitable, Callable
from typing import Any

from aiocqhttp.exceptions import ActionFailed
from aiocqhttp.exceptions import Error as OneBotError
from pydantic.dataclasses import dataclass

from astrbot.api import FunctionTool, logger
from astrbot.api.event import AstrMessageEvent
from astrbot.api.message_components import At, Reply

MANAGER_ROLES = ("owner", "admin")
MAX_SEARCH_CANDIDATES = 5


class ToolError(Exception):
    """An expected failure whose message is returned to the LLM verbatim."""


@dataclass
class NapCatTool(FunctionTool):
    """A FunctionTool bound to a plugin coroutine.

    The class must live in this plugin so that AstrBot resolves the tool's
    module path to the plugin and unregisters it on reload.
    """

    impl: Callable[..., Awaitable[Any]] | None = None

    async def call(self, context: Any, **kwargs) -> Any:
        """Run the bound coroutine and turn expected failures into text.

        Args:
            context: The agent run context; ``context.context.event`` is the
                current message event.
            **kwargs: Tool arguments produced by the LLM. Unknown keys are
                dropped because AstrBot may pass them through unfiltered.

        Returns:
            A string or ``CallToolResult`` fed back to the LLM, or ``None`` to
            end the agent loop without a reply.
        """
        event: AstrMessageEvent = context.context.event
        if event.get_platform_name() != "aiocqhttp":
            return "Error: this tool only works on QQ via OneBot v11 (NapCat)."
        allowed = self.parameters.get("properties", {})
        args = {k: v for k, v in kwargs.items() if k in allowed}
        try:
            return await self.impl(event, **args)
        except ToolError as e:
            return f"Error: {e}"
        except ActionFailed as e:
            reason = e.result.get("wording") or e.result.get("message") or ""
            return (
                f"Error: the QQ protocol endpoint rejected the action "
                f"(retcode {e.result.get('retcode')}): {reason}. It may be "
                "unsupported by the current OneBot implementation."
            )
        except OneBotError as e:
            return f"Error: failed to reach the QQ protocol endpoint: {e!r}"
        except (ValueError, TypeError) as e:
            logger.warning(f"{self.name} got invalid arguments {args}: {e!r}")
            return f"Error: invalid argument: {e}"


async def call_action(event: AstrMessageEvent, action: str, **params) -> Any:
    """Call a OneBot action on the bot that received ``event``.

    Args:
        event: The current aiocqhttp message event.
        action: OneBot action name.
        **params: Action parameters.

    Returns:
        The action's ``data`` payload.
    """
    self_id = event.get_self_id()
    if self_id:
        params["self_id"] = self_id
    return await event.bot.call_action(action, **params)


def require_group(event: AstrMessageEvent) -> int:
    """Return the current group ID.

    Args:
        event: The current message event.

    Returns:
        The group ID as an int.

    Raises:
        ToolError: If the event is not from a group chat.
    """
    group_id = event.get_group_id()
    if not group_id:
        raise ToolError("this tool only works in group chats.")
    return int(group_id)


def find_reply(event: AstrMessageEvent) -> Reply | None:
    """Return the quoted-message segment of the current message.

    Args:
        event: The current message event.

    Returns:
        The ``Reply`` segment, or ``None`` if the user quoted nothing.
    """
    return next((c for c in event.get_messages() if isinstance(c, Reply)), None)


def resolve_message(event: AstrMessageEvent, which: str) -> tuple[int | str, str]:
    """Resolve ``"quoted"`` / ``"current"`` to a message ID and its sender.

    Args:
        event: The current message event.
        which: ``"quoted"`` for the message the user replied to, ``"current"``
            for the user's own message.

    Returns:
        ``(message_id, sender_id)``. The ID is an int when numeric, since some
        OneBot implementations only accept ints; ``sender_id`` is ``""`` when
        the author of a quoted message is unknown.

    Raises:
        ToolError: If ``which`` is ``"quoted"`` and the user quoted nothing.
    """
    if which == "current":
        message_id, sender_id = event.message_obj.message_id, event.get_sender_id()
    else:
        reply = find_reply(event)
        if reply is None:
            raise ToolError(
                "the user did not quote any message. Ask them to reply to the "
                "target message."
            )
        message_id, sender_id = reply.id, reply.sender_id
    message_id = str(message_id)
    sender_id = str(sender_id or "")
    return (
        int(message_id) if message_id.isdecimal() else message_id,
        "" if sender_id == "0" else sender_id,
    )


async def resolve_user(
    event: AstrMessageEvent, user: Any, explicit_only: bool = False
) -> str:
    """Resolve a user reference from the LLM to a QQ number.

    Args:
        event: The current message event.
        user: ``"sender"``, ``"quoted_sender"``, ``"at"`` / ``"at:N"`` (the
            N-th mentioned user, 1-based), a QQ number, or a name keyword.
        explicit_only: Accept only ``quoted_sender`` and ``at`` references,
            which the requester must put in their own message. Management
            tools use this so that injected text cannot pick a target by QQ
            number or name, and an omitted target never falls back to the
            requester.

    Returns:
        The QQ number as a string.

    Raises:
        ToolError: If the reference cannot be resolved to exactly one user.
    """
    user = str(user or "").strip()
    if user == "quoted_sender":
        _, sender_id = resolve_message(event, "quoted")
        if not sender_id:
            raise ToolError("could not determine the author of the quoted message.")
        return sender_id
    if user == "at" or user.startswith("at:"):
        suffix = user[3:]
        if suffix and not suffix.isdecimal():
            raise ToolError(f"invalid mention reference '{user}'; use 'at' or 'at:N'.")
        self_id = str(event.get_self_id())
        mentions = [
            str(c.qq)
            for c in event.get_messages()
            if isinstance(c, At) and str(c.qq) not in (self_id, "all")
        ]
        index = int(suffix or 1) - 1
        if not 0 <= index < len(mentions):
            raise ToolError(
                f"the message mentions {len(mentions)} user(s); cannot resolve '{user}'."
            )
        return mentions[index]
    if explicit_only:
        raise ToolError(
            "this action needs a target taken from the user's message: "
            "'quoted_sender' or 'at'. Ask the user to @ or quote the person."
        )
    if user in ("", "sender"):
        return str(event.get_sender_id())
    if user.isdecimal():
        return user
    group_id = require_group(event)
    members = await call_action(event, "get_group_member_list", group_id=group_id)
    keyword = user.casefold()
    names = [
        (str(m.get("user_id")), m.get("card") or "", m.get("nickname") or "")
        for m in members or []
    ]
    matches = [n for n in names if keyword in (n[1].casefold(), n[2].casefold())]
    if not matches:
        matches = [
            n for n in names if keyword in n[1].casefold() or keyword in n[2].casefold()
        ]
    if len(matches) == 1:
        return matches[0][0]
    if not matches:
        raise ToolError(f"no group member matches '{user}'.")
    shown = "; ".join(
        f"{card or nick} ({qq})" for qq, card, nick in matches[:MAX_SEARCH_CANDIDATES]
    )
    raise ToolError(
        f"{len(matches)} members match '{user}': {shown}. Ask the user which one "
        "they mean, then call again with the QQ number."
    )


async def get_role(event: AstrMessageEvent, group_id: int, user_id: str) -> str:
    """Look up a member's group role.

    Args:
        event: The current message event.
        group_id: The group to look in.
        user_id: The member's QQ number.

    Returns:
        ``"owner"``, ``"admin"`` or ``"member"``.
    """
    info = await call_action(
        event,
        "get_group_member_info",
        group_id=group_id,
        user_id=int(user_id),
        no_cache=True,
    )
    return (info or {}).get("role") or "member"


async def check_management(
    event: AstrMessageEvent,
    group_id: int,
    allow_group_admin: bool,
    target_id: str | None = None,
) -> None:
    """Ensure a management action is allowed.

    The requester is the sender of the message that triggered this LLM turn.
    The action must be explicitly addressed to the bot, the requester must be
    an AstrBot admin or (optionally) a group owner/admin, the bot must be a
    group owner/admin, and nobody may act on someone they could not act on
    themselves in QQ.

    Args:
        event: The current message event.
        group_id: The current group.
        allow_group_admin: Whether group owners/admins may request actions in
            addition to AstrBot admins.
        target_id: The member being acted on, or ``None`` if there is none.
            The bot's own messages need no bot-side role check.

    Raises:
        ToolError: If any check fails.
    """
    if not event.is_at_or_wake_command:
        raise ToolError(
            "management actions are only allowed when the user addresses you "
            "directly (@ you, quote you, or use the wake word). Do nothing."
        )
    if event.is_admin():
        requester_role = "astrbot_admin"
    else:
        requester_role = (
            await get_role(event, group_id, event.get_sender_id())
            if allow_group_admin
            else ""
        )
        if requester_role not in MANAGER_ROLES:
            raise ToolError(
                "permission denied: the requesting user is neither an AstrBot "
                "admin nor a group owner/admin. Tell them you cannot do this."
            )
    self_id = str(event.get_self_id())
    if target_id == self_id:
        return
    bot_role = await get_role(event, group_id, self_id)
    if bot_role not in MANAGER_ROLES:
        raise ToolError("the bot is not an owner/admin of this group.")
    if target_id is None:
        return
    target_role = await get_role(event, group_id, target_id)
    if target_role == "owner":
        raise ToolError("the target is the group owner; nobody can do this to them.")
    if target_role == "admin" and bot_role != "owner":
        raise ToolError(
            "the target is a group admin; only a group-owner bot can do this."
        )
    if target_role == "admin" and requester_role not in ("astrbot_admin", "owner"):
        raise ToolError(
            "the target is a group admin; only the group owner or an AstrBot "
            "admin may ask for this."
        )
