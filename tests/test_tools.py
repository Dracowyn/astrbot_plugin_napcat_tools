"""Unit tests. Run from the AstrBot root:

    uv run python -m pytest data/plugins/astrbot_plugin_napcat_tools/tests
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from aiocqhttp.exceptions import ActionFailed

from astrbot.api.message_components import At, Plain, Reply
from data.plugins.astrbot_plugin_napcat_tools.core import (
    NapCatTool,
    ToolError,
    check_management,
    resolve_message,
    resolve_user,
)
from data.plugins.astrbot_plugin_napcat_tools.tools import TOOL_SPECS, NapCatToolkit

SELF_ID = "10000"
SENDER_ID = "20000"
GROUP_ID = "30000"

MEMBERS = [
    {"user_id": 1, "card": "小明", "nickname": "ming"},
    {"user_id": 2, "card": "小明同学", "nickname": "xm2"},
    {"user_id": 3, "card": "", "nickname": "Alice"},
    {"user_id": 4, "card": "Alice's cat", "nickname": "cat"},
]

ALL_ON = {
    "tools": dict.fromkeys(TOOL_SPECS, True),
    "permission": {"allow_group_admin": True, "max_mute_minutes": 60},
}


def make_event(
    messages=None,
    group=True,
    admin=False,
    addressed=True,
    roles=None,
    platform="aiocqhttp",
):
    """Build a fake aiocqhttp event whose bot records OneBot calls."""
    roles = roles or {}

    async def call_action(action, **params):
        if action == "get_group_member_list":
            return MEMBERS
        if action == "get_group_member_info":
            return {"role": roles.get(str(params["user_id"]), "member")}
        return {}

    event = MagicMock()
    event.get_platform_name.return_value = platform
    event.get_self_id.return_value = SELF_ID
    event.get_sender_id.return_value = SENDER_ID
    event.get_group_id.return_value = GROUP_ID if group else ""
    event.get_messages.return_value = messages or [Plain(text="hi")]
    event.message_obj = SimpleNamespace(message_id="555")
    event.is_admin.return_value = admin
    event.is_at_or_wake_command = addressed
    event.bot.call_action = AsyncMock(side_effect=call_action)
    return event


def calls(event, action):
    return [
        c.kwargs for c in event.bot.call_action.call_args_list if c.args[0] == action
    ]


def tools_by_name(config=ALL_ON, context=None):
    tools = NapCatToolkit(context or MagicMock(), config).build()
    return {t.name: t for t in tools}


async def run(tool, event, **kwargs):
    ctx = SimpleNamespace(context=SimpleNamespace(event=event))
    return await tool.call(ctx, **kwargs)


# --- target resolution -------------------------------------------------------


@pytest.mark.asyncio
async def test_resolve_user_keywords():
    reply = Reply(id="9", sender_id="40000")
    event = make_event([At(qq=SELF_ID), At(qq="all"), At(qq=111), At(qq=222), reply])
    assert await resolve_user(event, "sender") == SENDER_ID
    assert await resolve_user(event, "") == SENDER_ID
    assert await resolve_user(event, "quoted_sender") == "40000"
    assert await resolve_user(event, "at") == "111"
    assert await resolve_user(event, "at:2") == "222"
    assert await resolve_user(event, 123456) == "123456"
    for bad in ("at:3", "at:x", "at:0"):
        with pytest.raises(ToolError):
            await resolve_user(event, bad)


@pytest.mark.asyncio
async def test_resolve_user_explicit_only():
    event = make_event([At(qq=111), Reply(id="9", sender_id="40000")])
    assert await resolve_user(event, "at", explicit_only=True) == "111"
    assert await resolve_user(event, "quoted_sender", explicit_only=True) == "40000"
    for bad in ("", "sender", "123456", "小明"):
        with pytest.raises(ToolError, match="taken from the user's message"):
            await resolve_user(event, bad, explicit_only=True)


@pytest.mark.asyncio
async def test_resolve_user_quoted_sender_unknown():
    event = make_event([Reply(id="9", sender_id=0)])
    with pytest.raises(ToolError, match="author of the quoted message"):
        await resolve_user(event, "quoted_sender")


@pytest.mark.asyncio
async def test_resolve_user_search():
    event = make_event()
    assert await resolve_user(event, "小明") == "1"
    assert await resolve_user(event, "alice") == "3"
    with pytest.raises(ToolError, match="2 members match"):
        await resolve_user(event, "小")
    with pytest.raises(ToolError, match="no group member"):
        await resolve_user(event, "nobody")
    with pytest.raises(ToolError, match="group chats"):
        await resolve_user(make_event(group=False), "小明")


def test_resolve_message():
    event = make_event([Reply(id="9", sender_id="40000")])
    assert resolve_message(event, "current") == (555, SENDER_ID)
    assert resolve_message(event, "quoted") == (9, "40000")
    with pytest.raises(ToolError, match="did not quote"):
        resolve_message(make_event(), "quoted")


# --- permissions -------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("admin", "roles", "target", "error"),
    [
        (True, {SELF_ID: "admin"}, "1", None),
        (False, {SELF_ID: "admin", SENDER_ID: "admin"}, "1", None),
        (True, {SELF_ID: "admin"}, None, None),
        (False, {SELF_ID: "member", SENDER_ID: "admin"}, SELF_ID, None),
        (False, {SELF_ID: "admin"}, "1", "permission denied"),
        (True, {SELF_ID: "member"}, "1", "not an owner/admin"),
        (True, {SELF_ID: "owner", "1": "owner"}, "1", "group owner"),
        (True, {SELF_ID: "admin", "1": "admin"}, "1", "group-owner bot"),
        (True, {SELF_ID: "owner", "1": "admin"}, "1", None),
        (False, {SELF_ID: "owner", SENDER_ID: "owner", "1": "admin"}, "1", None),
        (
            False,
            {SELF_ID: "owner", SENDER_ID: "admin", "1": "admin"},
            "1",
            "only the group owner or an AstrBot admin",
        ),
    ],
)
async def test_check_management(admin, roles, target, error):
    event = make_event(admin=admin, roles=roles)
    if error is None:
        await check_management(event, 1, True, target)
    else:
        with pytest.raises(ToolError, match=error):
            await check_management(event, 1, True, target)


@pytest.mark.asyncio
async def test_check_management_requires_addressing_and_group_admin_flag():
    event = make_event(admin=True, addressed=False, roles={SELF_ID: "admin"})
    with pytest.raises(ToolError, match="addresses you directly"):
        await check_management(event, 1, True)
    event = make_event(roles={SELF_ID: "admin", SENDER_ID: "admin"})
    with pytest.raises(ToolError, match="permission denied"):
        await check_management(event, 1, False)


# --- tool wrapper ------------------------------------------------------------


@pytest.mark.asyncio
async def test_tool_wrapper_errors_and_arg_filtering():
    async def impl(event, user="sender"):
        if user == "boom":
            raise ToolError("bad")
        if user == "fail":
            raise ActionFailed({"retcode": 1404, "wording": "unsupported"})
        return str(int(user))

    tool = NapCatTool(
        name="t",
        description="d",
        parameters={"type": "object", "properties": {"user": {"type": "string"}}},
        impl=impl,
    )
    assert await run(tool, make_event(), user="7", junk=1) == "7"
    assert await run(tool, make_event(), user="boom") == "Error: bad"
    assert "1404" in await run(tool, make_event(), user="fail")
    assert "invalid argument" in await run(tool, make_event(), user="x")
    assert "only works on QQ" in await run(tool, make_event(platform="telegram"))


def test_build_respects_toggles_and_clamps_config():
    assert len(tools_by_name()) == len(TOOL_SPECS)
    assert all(name.startswith("qq_") for name in tools_by_name())
    only_poke = {"tools": {"poke": True, "mute_member": False}}
    assert list(tools_by_name(only_poke)) == ["qq_poke"]
    assert NapCatToolkit(MagicMock(), {}).build() == []
    clamp = {"permission": {"max_mute_minutes": 0}}
    assert NapCatToolkit(MagicMock(), clamp).max_mute_minutes == 1
    clamp = {"permission": {"max_mute_minutes": 10**6}}
    assert NapCatToolkit(MagicMock(), clamp).max_mute_minutes == 43200


# --- low-risk tools ------------------------------------------------------------


@pytest.mark.asyncio
async def test_poke_group_and_private():
    tools = tools_by_name()
    event = make_event()
    await run(tools["qq_poke"], event, user="123")
    assert calls(event, "group_poke") == [
        {"group_id": 30000, "user_id": 123, "self_id": SELF_ID}
    ]
    event = make_event(group=False)
    await run(tools["qq_poke"], event)
    assert calls(event, "friend_poke") == [{"user_id": 20000, "self_id": SELF_ID}]


@pytest.mark.asyncio
async def test_react_emoji():
    tool = tools_by_name()["qq_react_emoji"]
    event = make_event([Reply(id="9", sender_id="1")])
    assert await run(tool, event, emoji="heart", message="quoted") == (
        "Reacted with heart."
    )
    assert calls(event, "set_msg_emoji_like")[0]["message_id"] == 9
    assert calls(event, "set_msg_emoji_like")[0]["emoji_id"] == "66"
    assert "unknown emoji" in await run(tool, make_event(), emoji="nope")


@pytest.mark.asyncio
async def test_get_user_info_skips_empty_fields():
    async def call_action(action, **params):
        if action == "get_group_member_info":
            return {"nickname": "ming", "card": "", "role": "admin", "title": ""}
        return {"nickname": "ming", "sex": "male", "age": 0, "long_nick": "hi"}

    event = make_event()
    event.bot.call_action = AsyncMock(side_effect=call_action)
    result = await run(tools_by_name()["qq_get_user_info"], event, user="1")
    assert result.splitlines() == [
        "user_id: 1",
        "nickname: ming",
        "role: admin",
        "sex: male",
        "long_nick: hi",
    ]


@pytest.mark.asyncio
async def test_get_user_info_non_member_falls_back_to_profile():
    async def call_action(action, **params):
        if action == "get_group_member_info":
            raise ActionFailed({"retcode": 200, "message": "not member"})
        return {"nickname": "stranger", "reg_time": 1_600_000_000}

    event = make_event()
    event.bot.call_action = AsyncMock(side_effect=call_action)
    result = await run(tools_by_name()["qq_get_user_info"], event, user="1")
    assert "group_member: no" in result
    assert "nickname: stranger" in result
    assert "reg_time: 2020-09-" in result


@pytest.mark.asyncio
async def test_recall_own_message():
    tool = tools_by_name()["qq_recall_own_message"]
    assert "not sent by you" in await run(
        tool, make_event([Reply(id="9", sender_id="1")])
    )
    assert "only recall a message the user quoted" in await run(tool, make_event())
    event = make_event([Reply(id="9", sender_id=SELF_ID)])
    await run(tool, event)
    assert calls(event, "delete_msg") == [{"message_id": 9, "self_id": SELF_ID}]


@pytest.mark.asyncio
async def test_stay_silent_returns_none_unless_addressed():
    tool = tools_by_name()["qq_stay_silent"]
    assert (await run(tool, make_event(addressed=True))).startswith("Refused")
    event = make_event(addressed=False)
    assert await run(tool, event) is None
    event.stop_event.assert_not_called()


@pytest.mark.asyncio
async def test_set_my_card():
    tool = tools_by_name()["qq_set_my_card"]
    event = make_event()
    await run(tool, event, card="bot")
    assert calls(event, "set_group_card")[0]["user_id"] == int(SELF_ID)
    assert "group chats" in await run(tool, make_event(group=False), card="x")


# --- management tools ------------------------------------------------------------


@pytest.mark.asyncio
async def test_mute_member_clamps_and_unmutes():
    tool = tools_by_name()["qq_mute_member"]
    event = make_event([At(qq=111)], admin=True, roles={SELF_ID: "admin"})
    assert "capped" in await run(tool, event, user="at", duration_minutes=600)
    assert calls(event, "set_group_ban")[0]["duration"] == 3600
    event = make_event([At(qq=111)], admin=True, roles={SELF_ID: "admin"})
    assert await run(tool, event, user="at", duration_minutes=0) == "Unmuted 111."
    assert "invalid argument" in await run(
        tool, event, user="at", duration_minutes="ten"
    )


@pytest.mark.asyncio
async def test_mute_member_denials():
    tool = tools_by_name()["qq_mute_member"]
    member_asks = make_event([At(qq=1)], roles={SELF_ID: "admin"})
    assert "permission denied" in await run(
        tool, member_asks, user="at", duration_minutes=1
    )
    by_number = make_event(admin=True, roles={SELF_ID: "admin"})
    assert "taken from the user's message" in await run(
        tool, by_number, user="123", duration_minutes=1
    )
    not_addressed = make_event([At(qq=1)], admin=True, addressed=False)
    assert "addresses you directly" in await run(
        tool, not_addressed, user="at", duration_minutes=1
    )
    for event in (member_asks, by_number, not_addressed):
        assert calls(event, "set_group_ban") == []


@pytest.mark.asyncio
async def test_recall_message():
    tool = tools_by_name()["qq_recall_message"]
    event = make_event([Reply(id="9", sender_id=SELF_ID)], admin=True)
    assert await run(tool, event) == "Recalled the quoted message."
    event = make_event([Reply(id="9", sender_id="1")], admin=True)
    assert "not an owner/admin" in await run(tool, event)
    event = make_event([Reply(id="9", sender_id=0)], admin=True)
    assert "author of the quoted message" in await run(tool, event)


@pytest.mark.asyncio
async def test_set_essence_and_member_card():
    tools = tools_by_name()
    event = make_event(
        [Reply(id="9", sender_id="1")], admin=True, roles={SELF_ID: "admin"}
    )
    assert await run(tools["qq_set_essence"], event, message="quoted") == (
        "Set the message as essence."
    )
    assert calls(event, "set_essence_msg")[0]["message_id"] == 9

    event = make_event([At(qq=111)], admin=True, roles={SELF_ID: "owner", "111": "admin"})
    await run(tools["qq_set_member_card"], event, user="at", card="")
    assert calls(event, "set_group_card")[0] == {
        "group_id": 30000,
        "user_id": 111,
        "card": "",
        "self_id": SELF_ID,
    }


# --- view_avatar -----------------------------------------------------------------


def avatar_tool(provider_id="", modalities=None, vision_resp=None):
    context = MagicMock()
    vision = MagicMock()
    vision.text_chat = AsyncMock(
        return_value=vision_resp
        or SimpleNamespace(role="assistant", completion_text="a cat")
    )
    context.get_provider_by_id.return_value = vision
    context.get_using_provider_async = AsyncMock(
        return_value=SimpleNamespace(provider_config={"modalities": modalities})
    )
    config = {**ALL_ON, "view_avatar": {"vision_provider_id": provider_id}}
    return tools_by_name(config, context)["qq_view_avatar"], vision


@pytest.mark.asyncio
async def test_view_avatar_with_vision_model():
    tool, vision = avatar_tool(provider_id="vision")
    assert (await run(tool, make_event(), user="123")).endswith("a cat")
    assert "nk=123" in vision.text_chat.call_args.kwargs["image_urls"][0]
    err = SimpleNamespace(role="err", completion_text="quota")
    tool, _ = avatar_tool(provider_id="vision", vision_resp=err)
    assert "returned an error: quota" in await run(tool, make_event())


@pytest.mark.asyncio
async def test_view_avatar_text_only_model():
    tool, _ = avatar_tool(modalities=["text", "tool_use"])
    assert "cannot view images" in await run(tool, make_event())


@pytest.mark.asyncio
async def test_view_avatar_returns_image(monkeypatch):
    class FakeResponse:
        content_type = "image/png"

        def raise_for_status(self):
            pass

        async def read(self):
            return b"png"

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

    class FakeSession(FakeResponse):
        def __init__(self, **_):
            pass

        def get(self, url):
            return FakeResponse()

    monkeypatch.setattr(
        "data.plugins.astrbot_plugin_napcat_tools.tools.aiohttp.ClientSession",
        FakeSession,
    )
    tool, _ = avatar_tool(modalities=["text", "image"])
    result = await run(tool, make_event())
    assert result.content[1].data == "cG5n"
    assert result.content[1].mimeType == "image/png"
