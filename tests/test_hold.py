"""Tests for button holds (c4.dmx.hc / c4.dmx.he)."""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from custom_components.control4_dimmers.const import BUTTON_EVENT_TYPES
from custom_components.control4_dimmers.manager import (
    Control4Manager,
    _parse_ramp_action,
)
from custom_components.control4_dimmers.models import (
    DeviceConfig,
    DeviceState,
    SlotConfig,
)

from .conftest import IEEE_KEYPAD

Z2M_LIGHT = "light.living"
Z2M_IEEE = "0x000fff0000aaaaaa"
Z2M_TOPIC = "zigbee2mqtt/Living Light/set"
TAP = {"action": "light.toggle", "target": {"entity_id": Z2M_LIGHT}}
SCRIPT = {"action": "script.turn_on", "target": {"entity_id": "script.scene"}}


def _light(service: str, entity_id: str = Z2M_LIGHT, **data: Any) -> dict:
    action: dict = {"action": service, "target": {"entity_id": entity_id}}
    if data:
        action["data"] = data
    return action


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
    manager._z2m_names[Z2M_IEEE] = "Living Light"
    events: list[str] = []
    manager.register_event_callback(IEEE_KEYPAD, 2, events.append)
    press = MagicMock(return_value=None)
    manager.press_button = press
    return press, events


def _triggers(press: MagicMock) -> list[str]:
    return [c.args[2] for c in press.call_args_list]


def _light_state(manager: Control4Manager, state: str) -> None:
    light = MagicMock()
    light.state = state
    manager._hass.states.get = MagicMock(return_value=light)


def _run(
    manager: Control4Manager, keypad_state: DeviceState, actions: tuple[str, ...]
) -> MagicMock:
    """Dispatch actions, returning the mqtt.async_publish mock."""
    publish = MagicMock(return_value=None)
    with patch(
        "custom_components.control4_dimmers.manager.mqtt.async_publish", publish
    ):
        for action in actions:
            manager._dispatch_button_action(keypad_state, action)
    return publish


def _payloads(publish: MagicMock) -> list[tuple[str, dict]]:
    return [(c.args[1], json.loads(c.args[2])) for c in publish.call_args_list]


HOLD_AND_RELEASE = (
    "button_2_press",
    "button_2_hold",
    "button_2_hold",
    "button_2_hold",
    "button_2_hold_release",
)


class TestHoldEvents:
    """The hold event fires once per hold, released on release."""

    def test_hold_is_a_declared_event_type(self) -> None:
        assert "hold" in BUTTON_EVENT_TYPES

    def test_one_hold_event_per_hold(
        self, manager: Control4Manager, keypad_state: DeviceState
    ) -> None:
        _press, events = _setup(manager, keypad_state)
        _run(manager, keypad_state, HOLD_AND_RELEASE)
        assert events == ["pressed", "hold", "released"]

    def test_next_hold_fires_again(
        self, manager: Control4Manager, keypad_state: DeviceState
    ) -> None:
        _press, events = _setup(manager, keypad_state)
        _run(
            manager,
            keypad_state,
            (
                "button_2_press",
                "button_2_hold",
                "button_2_hold_release",
                "button_2_press",
                "button_2_hold",
            ),
        )
        assert events.count("hold") == 2

    def test_stray_release_fires_nothing(
        self, manager: Control4Manager, keypad_state: DeviceState
    ) -> None:
        _press, events = _setup(manager, keypad_state)
        _run(manager, keypad_state, ("button_2_hold_release",))
        assert events == []


class TestRepeatingHoldAction:
    """A hold action that isn't a light ramp runs on every hold repeat."""

    def test_runs_each_repeat(
        self, manager: Control4Manager, keypad_state: DeviceState
    ) -> None:
        press, _events = _setup(manager, keypad_state, hold_action=SCRIPT)
        _run(manager, keypad_state, HOLD_AND_RELEASE)
        assert _triggers(press) == ["hold", "hold", "hold"]

    def test_no_hold_action_runs_nothing(
        self, manager: Control4Manager, keypad_state: DeviceState
    ) -> None:
        press, _events = _setup(manager, keypad_state)
        _run(manager, keypad_state, ("button_2_hold",))
        press.assert_not_called()

    def test_load_button_hold_runs_nothing(
        self, manager: Control4Manager, keypad_state: DeviceState
    ) -> None:
        # The firmware dims its own load while a load button is held.
        press, events = _setup(manager, keypad_state, hold_action=SCRIPT)
        manager._store._devices[IEEE_KEYPAD].slots[0].behavior = "toggle_load"
        publish = _run(manager, keypad_state, ("button_2_hold",))
        press.assert_not_called()
        publish.assert_not_called()
        assert events == ["hold"]

    @pytest.mark.asyncio
    async def test_press_button_hold_executes_hold_action(
        self, manager: Control4Manager, keypad_state: DeviceState
    ) -> None:
        _setup(manager, keypad_state, hold_action=SCRIPT)
        manager.press_button = Control4Manager.press_button.__get__(manager)
        manager._hass.services.async_call = AsyncMock()
        await manager.press_button(IEEE_KEYPAD, 2, "hold")
        manager._hass.services.async_call.assert_awaited_once_with(
            "script", "turn_on", {"entity_id": "script.scene"}
        )


class TestTapWaitsForRelease:
    """With a hold_action, the tap runs on release (scene), not on press."""

    def test_tap_runs_on_scene_with_hold_action(
        self, manager: Control4Manager, keypad_state: DeviceState
    ) -> None:
        press, _events = _setup(manager, keypad_state, tap_action=TAP, hold_action=TAP)
        _run(manager, keypad_state, ("button_2_press",))
        press.assert_not_called()
        _run(manager, keypad_state, ("button_2_scene",))
        assert _triggers(press) == ["pressed"]

    def test_hold_never_taps(
        self, manager: Control4Manager, keypad_state: DeviceState
    ) -> None:
        # A hold sends hc instead of sc, so the tap action never runs.
        press, _events = _setup(
            manager, keypad_state, tap_action=TAP, hold_action=SCRIPT
        )
        _run(manager, keypad_state, ("button_2_press", "button_2_hold"))
        assert _triggers(press) == ["hold"]

    def test_tap_still_runs_on_press_without_hold_action(
        self, manager: Control4Manager, keypad_state: DeviceState
    ) -> None:
        press, _events = _setup(manager, keypad_state, tap_action=TAP)
        _run(manager, keypad_state, ("button_2_press", "button_2_scene"))
        assert _triggers(press) == ["pressed"]

    def test_double_tap_still_waits_for_click_count(
        self, manager: Control4Manager, keypad_state: DeviceState
    ) -> None:
        press, _events = _setup(
            manager,
            keypad_state,
            tap_action=TAP,
            hold_action=TAP,
            double_tap_action=SCRIPT,
        )
        _run(manager, keypad_state, ("button_2_press", "button_2_scene"))
        press.assert_not_called()
        _run(manager, keypad_state, ("button_2_click_1",))
        assert _triggers(press) == ["single_tap"]


class TestRampParsing:
    """Which hold actions are dimming ramps."""

    @pytest.mark.parametrize(
        ("action", "expected"),
        [
            (_light("light.turn_on"), ("light.turn_on", Z2M_LIGHT, 15)),
            (_light("light.turn_off"), ("light.turn_off", Z2M_LIGHT, 15)),
            (_light("light.toggle"), ("light.toggle", Z2M_LIGHT, 15)),
            (
                _light("light.turn_on", brightness_step_pct=-25, transition=1),
                ("light.turn_on", Z2M_LIGHT, -25),
            ),
            (
                {"action": "light.toggle", "target": {"entity_id": [Z2M_LIGHT]}},
                ("light.toggle", Z2M_LIGHT, 15),
            ),
        ],
    )
    def test_ramps(self, action: dict, expected: tuple) -> None:
        assert _parse_ramp_action(action) == expected

    @pytest.mark.parametrize(
        "action",
        [
            SCRIPT,
            _light("light.turn_on", brightness=100),
            _light("light.turn_on", brightness_step_pct=0),
            _light("light.toggle", "switch.fan"),
            {"action": "light.toggle", "target": {"entity_id": [Z2M_LIGHT, "light.b"]}},
            {"action": "light.toggle"},
        ],
    )
    def test_not_ramps(self, action: dict) -> None:
        assert _parse_ramp_action(action) is None


class TestSmoothRamp:
    """A ramp on a Zigbee2MQTT light is one brightness_move and a stop."""

    def test_turn_on_brightens_and_release_stops(
        self, manager: Control4Manager, keypad_state: DeviceState
    ) -> None:
        press, _events = _setup(
            manager, keypad_state, tap_action=TAP, hold_action=_light("light.turn_on")
        )
        publish = _run(manager, keypad_state, HOLD_AND_RELEASE)
        assert _payloads(publish) == [
            (Z2M_TOPIC, {"brightness_move_onoff": 38}),
            (Z2M_TOPIC, {"brightness_move": 0}),
        ]
        press.assert_not_called()

    def test_turn_off_dims_without_onoff(
        self, manager: Control4Manager, keypad_state: DeviceState
    ) -> None:
        _setup(
            manager,
            keypad_state,
            hold_action=_light("light.turn_off", brightness_step_pct=20),
        )
        publish = _run(manager, keypad_state, ("button_2_press", "button_2_hold"))
        assert _payloads(publish) == [(Z2M_TOPIC, {"brightness_move": -51})]

    def test_new_press_stops_a_lost_release(
        self, manager: Control4Manager, keypad_state: DeviceState
    ) -> None:
        # If the he frame is lost, the next press still stops the ramp.
        _setup(manager, keypad_state, hold_action=_light("light.turn_on"))
        publish = _run(
            manager, keypad_state, ("button_2_press", "button_2_hold", "button_2_press")
        )
        assert _payloads(publish)[-1] == (Z2M_TOPIC, {"brightness_move": 0})


class TestToggleAlternates:
    """A toggle hold brightens an off light, then reverses each hold."""

    def _hold(self, manager: Control4Manager, keypad_state: DeviceState) -> dict:
        publish = _run(
            manager,
            keypad_state,
            ("button_2_press", "button_2_hold", "button_2_hold_release"),
        )
        return _payloads(publish)[0][1]

    def test_off_light_brightens(
        self, manager: Control4Manager, keypad_state: DeviceState
    ) -> None:
        _setup(manager, keypad_state, hold_action=_light("light.toggle"))
        _light_state(manager, "off")
        assert self._hold(manager, keypad_state) == {"brightness_move_onoff": 38}

    def test_on_light_dims_first(
        self, manager: Control4Manager, keypad_state: DeviceState
    ) -> None:
        _setup(manager, keypad_state, hold_action=_light("light.toggle"))
        _light_state(manager, "on")
        assert self._hold(manager, keypad_state) == {"brightness_move": -38}

    def test_holds_alternate(
        self, manager: Control4Manager, keypad_state: DeviceState
    ) -> None:
        _setup(manager, keypad_state, hold_action=_light("light.toggle"))
        _light_state(manager, "off")
        first = self._hold(manager, keypad_state)
        _light_state(manager, "on")
        second = self._hold(manager, keypad_state)
        third = self._hold(manager, keypad_state)
        assert first == {"brightness_move_onoff": 38}
        assert second == {"brightness_move": -38}
        assert third == {"brightness_move_onoff": 38}

    def test_direction_holds_within_one_hold(
        self, manager: Control4Manager, keypad_state: DeviceState
    ) -> None:
        # Repeats of the same hold must not flip the direction.
        _setup(manager, keypad_state, hold_action=_light("light.toggle", "light.wifi"))
        _light_state(manager, "on")
        manager._hass.services.async_call = MagicMock(return_value=None)
        _run(
            manager, keypad_state, ("button_2_press", "button_2_hold", "button_2_hold")
        )
        steps = [
            c.args[2]["brightness_step_pct"]
            for c in manager._hass.services.async_call.call_args_list
        ]
        assert steps == [-15, -15]


class TestNonZ2MFallback:
    """A ramp on a light that isn't on Zigbee2MQTT steps once per repeat."""

    def test_steps_each_repeat(
        self, manager: Control4Manager, keypad_state: DeviceState
    ) -> None:
        press, _events = _setup(
            manager, keypad_state, hold_action=_light("light.turn_on", "light.wifi")
        )
        manager._hass.services.async_call = MagicMock(return_value=None)
        publish = _run(manager, keypad_state, HOLD_AND_RELEASE)
        publish.assert_not_called()
        press.assert_not_called()
        calls = manager._hass.services.async_call.call_args_list
        assert [c.args for c in calls] == [
            (
                "light",
                "turn_on",
                {
                    "entity_id": "light.wifi",
                    "brightness_step_pct": 15,
                    "transition": 1,
                },
            )
        ] * 3

    def test_unknown_z2m_device_steps(
        self, manager: Control4Manager, keypad_state: DeviceState
    ) -> None:
        _setup(manager, keypad_state, hold_action=_light("light.turn_on"))
        manager._z2m_names.clear()  # no bridge/devices entry for the light
        manager._hass.services.async_call = MagicMock(return_value=None)
        publish = _run(manager, keypad_state, ("button_2_hold",))
        publish.assert_not_called()
        manager._hass.services.async_call.assert_called_once()
