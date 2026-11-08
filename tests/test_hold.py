"""Tests for button holds (c4.dmx.hc / c4.dmx.he)."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from custom_components.control4_dimmers.const import BUTTON_EVENT_TYPES
from custom_components.control4_dimmers.manager import Control4Manager
from custom_components.control4_dimmers.models import (
    DeviceConfig,
    DeviceState,
    SlotConfig,
)

from .conftest import IEEE_KEYPAD

TAP = {"action": "light.turn_on", "target": {"entity_id": "light.living"}}
HOLD = {
    "action": "light.turn_on",
    "target": {"entity_id": "light.living"},
    "data": {"brightness_step_pct": 15, "transition": 1},
}


def _setup(
    manager: Control4Manager, keypad_state: DeviceState, **slot_fields: Any
) -> tuple[MagicMock, list[str]]:
    """Configure button 2, record fired events, and capture press_button calls."""
    manager._devices[IEEE_KEYPAD] = keypad_state
    manager._store._devices[IEEE_KEYPAD] = DeviceConfig(
        ieee_address=IEEE_KEYPAD,
        friendly_name="Keypad",
        device_type="keypad",
        slots=[SlotConfig(slot_id=2, behavior="keypad", **slot_fields)],
    )
    events: list[str] = []
    manager.register_event_callback(IEEE_KEYPAD, 2, events.append)
    press = MagicMock(return_value=None)
    manager.press_button = press
    return press, events


def _triggers(press: MagicMock) -> list[str]:
    return [c.args[2] for c in press.call_args_list]


class TestHoldEvents:
    """The hold event fires once per hold, released on release."""

    def test_hold_is_a_declared_event_type(self) -> None:
        assert "hold" in BUTTON_EVENT_TYPES

    def test_one_hold_event_per_hold(
        self, manager: Control4Manager, keypad_state: DeviceState
    ) -> None:
        _press, events = _setup(manager, keypad_state)
        for action in (
            "button_2_press",
            "button_2_hold",
            "button_2_hold",
            "button_2_hold",
            "button_2_hold_release",
        ):
            manager._dispatch_button_action(keypad_state, action)
        assert events == ["pressed", "hold", "released"]

    def test_next_hold_fires_again(
        self, manager: Control4Manager, keypad_state: DeviceState
    ) -> None:
        _press, events = _setup(manager, keypad_state)
        for action in (
            "button_2_press",
            "button_2_hold",
            "button_2_hold_release",
            "button_2_press",
            "button_2_hold",
        ):
            manager._dispatch_button_action(keypad_state, action)
        assert events.count("hold") == 2

    def test_stray_release_fires_nothing(
        self, manager: Control4Manager, keypad_state: DeviceState
    ) -> None:
        _press, events = _setup(manager, keypad_state)
        manager._dispatch_button_action(keypad_state, "button_2_hold_release")
        assert events == []


class TestHoldAction:
    """hold_action runs on every hold repeat."""

    def test_hold_action_runs_each_repeat(
        self, manager: Control4Manager, keypad_state: DeviceState
    ) -> None:
        press, _events = _setup(manager, keypad_state, tap_action=TAP, hold_action=HOLD)
        for action in (
            "button_2_press",
            "button_2_hold",
            "button_2_hold",
            "button_2_hold",
            "button_2_hold_release",
        ):
            manager._dispatch_button_action(keypad_state, action)
        assert _triggers(press) == ["hold", "hold", "hold"]

    def test_no_hold_action_runs_nothing(
        self, manager: Control4Manager, keypad_state: DeviceState
    ) -> None:
        press, _events = _setup(manager, keypad_state, tap_action=TAP)
        manager._dispatch_button_action(keypad_state, "button_2_press")
        press.reset_mock()
        manager._dispatch_button_action(keypad_state, "button_2_hold")
        press.assert_not_called()

    def test_load_button_hold_runs_nothing(
        self, manager: Control4Manager, keypad_state: DeviceState
    ) -> None:
        # The firmware dims its own load while a load button is held.
        press, events = _setup(manager, keypad_state, hold_action=HOLD)
        manager._store._devices[IEEE_KEYPAD].slots[0].behavior = "toggle_load"
        manager._dispatch_button_action(keypad_state, "button_2_hold")
        press.assert_not_called()
        assert events == ["hold"]


class TestTapWaitsForRelease:
    """With a hold_action, the tap runs on release (scene), not on press."""

    def test_tap_runs_on_scene_with_hold_action(
        self, manager: Control4Manager, keypad_state: DeviceState
    ) -> None:
        press, _events = _setup(manager, keypad_state, tap_action=TAP, hold_action=HOLD)
        manager._dispatch_button_action(keypad_state, "button_2_press")
        press.assert_not_called()
        manager._dispatch_button_action(keypad_state, "button_2_scene")
        assert _triggers(press) == ["pressed"]

    def test_hold_never_taps(
        self, manager: Control4Manager, keypad_state: DeviceState
    ) -> None:
        # A hold sends hc instead of sc, so the tap action never runs.
        press, _events = _setup(manager, keypad_state, tap_action=TAP, hold_action=HOLD)
        for action in ("button_2_press", "button_2_hold", "button_2_hold_release"):
            manager._dispatch_button_action(keypad_state, action)
        assert _triggers(press) == ["hold"]

    def test_tap_still_runs_on_press_without_hold_action(
        self, manager: Control4Manager, keypad_state: DeviceState
    ) -> None:
        press, _events = _setup(manager, keypad_state, tap_action=TAP)
        manager._dispatch_button_action(keypad_state, "button_2_press")
        assert _triggers(press) == ["pressed"]
        manager._dispatch_button_action(keypad_state, "button_2_scene")
        assert _triggers(press) == ["pressed"]

    def test_double_tap_still_waits_for_click_count(
        self, manager: Control4Manager, keypad_state: DeviceState
    ) -> None:
        press, _events = _setup(
            manager,
            keypad_state,
            tap_action=TAP,
            hold_action=HOLD,
            double_tap_action=TAP,
        )
        manager._dispatch_button_action(keypad_state, "button_2_press")
        manager._dispatch_button_action(keypad_state, "button_2_scene")
        press.assert_not_called()
        manager._dispatch_button_action(keypad_state, "button_2_click_1")
        assert _triggers(press) == ["single_tap"]


class TestHoldRunsTheHoldAction:
    """press_button(..., "hold") executes the slot's hold_action."""

    @pytest.mark.asyncio
    async def test_hold_executes_hold_action(
        self, manager: Control4Manager, keypad_state: DeviceState
    ) -> None:
        _setup(manager, keypad_state, tap_action=TAP, hold_action=HOLD)
        manager.press_button = Control4Manager.press_button.__get__(manager)
        manager._hass.services.async_call = AsyncMock()
        await manager.press_button(IEEE_KEYPAD, 2, "hold")
        manager._hass.services.async_call.assert_awaited_once_with(
            "light",
            "turn_on",
            {"brightness_step_pct": 15, "transition": 1, "entity_id": "light.living"},
        )
