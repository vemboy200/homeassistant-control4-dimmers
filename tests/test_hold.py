"""Tests for button holds (c4.dmx.hc / c4.dmx.he)."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

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


Z2M_LIGHT = "light.living"
Z2M_IEEE = "0x000fff0000aaaaaa"


@pytest.fixture(autouse=True)
def entity_registry() -> Any:
    """Patch the entity registry: light.living is a Zigbee2MQTT light."""
    entries = {
        Z2M_LIGHT: MagicMock(
            platform="mqtt", unique_id=f"{Z2M_IEEE}_light_zigbee2mqtt"
        ),
        "light.wifi": MagicMock(platform="tuya_local", unique_id="abc"),
    }
    registry = MagicMock()
    registry.async_get = MagicMock(side_effect=entries.get)
    with patch(
        "custom_components.control4_dimmers.manager.er.async_get",
        return_value=registry,
    ):
        yield registry


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


REPEAT_HOLD = {
    "action": "light.turn_on",
    "target": {"entity_id": "light.wifi"},
    "data": {"brightness_step_pct": 15, "transition": 1},
}


class TestHoldAction:
    """A hold_action that can't ramp a Z2M light runs on every hold repeat."""

    def test_hold_action_runs_each_repeat(
        self, manager: Control4Manager, keypad_state: DeviceState
    ) -> None:
        press, _events = _setup(
            manager, keypad_state, tap_action=TAP, hold_action=REPEAT_HOLD
        )
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
        press, _events = _setup(
            manager, keypad_state, tap_action=TAP, hold_action=REPEAT_HOLD
        )
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


class TestSmoothDim:
    """A brightness-step hold on a Zigbee2MQTT light ramps with brightness_move."""

    def _run(
        self,
        manager: Control4Manager,
        keypad_state: DeviceState,
        actions: tuple[str, ...],
        hold: dict = HOLD,
    ) -> tuple[MagicMock, MagicMock]:
        press, _events = _setup(manager, keypad_state, tap_action=TAP, hold_action=hold)
        manager._z2m_names[Z2M_IEEE] = "Living Light"
        publish = MagicMock(return_value=None)
        with patch(
            "custom_components.control4_dimmers.manager.mqtt.async_publish", publish
        ):
            for action in actions:
                manager._dispatch_button_action(keypad_state, action)
        return press, publish

    def _payloads(self, publish: MagicMock) -> list[tuple[str, dict]]:
        import json  # noqa: PLC0415

        return [(c.args[1], json.loads(c.args[2])) for c in publish.call_args_list]

    def test_hold_starts_one_move_and_release_stops(
        self, manager: Control4Manager, keypad_state: DeviceState
    ) -> None:
        press, publish = self._run(
            manager,
            keypad_state,
            (
                "button_2_press",
                "button_2_hold",
                "button_2_hold",
                "button_2_hold",
                "button_2_hold_release",
            ),
        )
        topic = "zigbee2mqtt/Living Light/set"
        assert self._payloads(publish) == [
            (topic, {"brightness_move_onoff": 38}),
            (topic, {"brightness_move": 0}),
        ]
        press.assert_not_called()

    def test_negative_step_dims_without_onoff(
        self, manager: Control4Manager, keypad_state: DeviceState
    ) -> None:
        down = {**HOLD, "data": {"brightness_step_pct": -20}}
        _press, publish = self._run(
            manager, keypad_state, ("button_2_press", "button_2_hold"), hold=down
        )
        assert self._payloads(publish) == [
            ("zigbee2mqtt/Living Light/set", {"brightness_move": -51}),
        ]

    def test_new_press_stops_a_lost_release(
        self, manager: Control4Manager, keypad_state: DeviceState
    ) -> None:
        # If the he frame is lost, the next press still stops the ramp.
        _press, publish = self._run(
            manager, keypad_state, ("button_2_press", "button_2_hold", "button_2_press")
        )
        assert self._payloads(publish)[-1] == (
            "zigbee2mqtt/Living Light/set",
            {"brightness_move": 0},
        )

    def test_unknown_z2m_device_falls_back_to_repeat(
        self, manager: Control4Manager, keypad_state: DeviceState
    ) -> None:
        press, _events = _setup(manager, keypad_state, tap_action=TAP, hold_action=HOLD)
        # No bridge/devices entry for the light's IEEE.
        manager._dispatch_button_action(keypad_state, "button_2_hold")
        assert _triggers(press) == ["hold"]

    @pytest.mark.parametrize(
        "action",
        [
            {**HOLD, "action": "light.toggle"},
            {**HOLD, "data": {"brightness_step": 20}},
            {**HOLD, "data": {"brightness_step_pct": 0}},
            {**HOLD, "target": {"entity_id": [Z2M_LIGHT, "light.other"]}},
        ],
    )
    def test_other_actions_are_not_moves(
        self, manager: Control4Manager, action: dict
    ) -> None:
        manager._z2m_names[Z2M_IEEE] = "Living Light"
        assert manager._smooth_dim_move(action) is None

    def test_single_item_target_list_is_a_move(self, manager: Control4Manager) -> None:
        manager._z2m_names[Z2M_IEEE] = "Living Light"
        action = {**HOLD, "target": {"entity_id": [Z2M_LIGHT]}}
        assert manager._smooth_dim_move(action) == (
            "zigbee2mqtt/Living Light/set",
            {"brightness_move_onoff": 38},
        )
