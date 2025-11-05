"""Minecraft networking client abstraction."""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Dict, Optional

from minecraft import authentication
from minecraft.networking.connection import Connection
from minecraft.networking.packets import clientbound, serverbound
from minecraft.networking.types import BlockFace
from .config import MinecraftConfig
from .logging_config import logger


@dataclass
class BotState:
    """Holds current bot state."""

    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    yaw: float = 0.0
    pitch: float = 0.0
    health: float = 20.0


class MinecraftClient:
    """High-level async wrapper over pyCraft connection."""

    def __init__(self, config: MinecraftConfig) -> None:
        self.config = config
        self.connection: Optional[Connection] = None
        self.state = BotState()
        self._connected = asyncio.Event()

    async def connect(self) -> None:
        logger.info("Connecting to Minecraft server %s:%s", self.config.host, self.config.port)
        auth_token = None
        if self.config.password:
            auth_token = authentication.AuthenticationToken()
            auth_token.authenticate(self.config.username, self.config.password)
        self.connection = Connection(
            self.config.host,
            self.config.port,
            username=self.config.username,
            auth_token=auth_token,
            handle_exception=self._handle_exception,
            forced_protocol_version=self.config.version,
        )
        self.connection.register_packet_listener(self._handle_join_game, clientbound.play.JoinGamePacket)
        self.connection.register_packet_listener(
            self._handle_position, clientbound.play.PlayerPositionAndLookPacket
        )
        self.connection.register_packet_listener(self._handle_update_health, clientbound.play.UpdateHealthPacket)
        await asyncio.to_thread(self.connection.connect)
        await self._connected.wait()
        logger.info("Connected as %s", self.config.username)

    def _handle_exception(self, exc: BaseException) -> None:
        logger.exception("Minecraft client error: %s", exc)

    def _handle_join_game(self, packet: clientbound.play.JoinGamePacket) -> None:  # noqa: D401
        """Fired when join game packet received."""

        logger.success("Joined world with entity id %s", packet.entity_id)
        self._connected.set()

    def _handle_position(self, packet: clientbound.play.PlayerPositionAndLookPacket) -> None:
        self.state.x = packet.x
        self.state.y = packet.y
        self.state.z = packet.z
        self.state.yaw = packet.yaw
        self.state.pitch = packet.pitch

    def _handle_update_health(self, packet: clientbound.play.UpdateHealthPacket) -> None:
        self.state.health = packet.health

    async def disconnect(self) -> None:
        if self.connection:
            logger.info("Disconnecting from server")
            await asyncio.to_thread(self.connection.disconnect)
            self._connected.clear()

    async def move(self, direction: str, distance: float) -> None:
        if not self.connection:
            raise RuntimeError("Not connected")
        dx, dz = 0.0, 0.0
        if direction == "forward":
            dz = distance
        elif direction == "backward":
            dz = -distance
        elif direction == "left":
            dx = -distance
        elif direction == "right":
            dx = distance
        else:
            raise ValueError(f"Unknown direction {direction}")
        packet = serverbound.play.PlayerPositionAndLookPacket()
        packet.x = self.state.x + dx
        packet.y = self.state.y
        packet.z = self.state.z + dz
        packet.on_ground = True
        packet.yaw = self.state.yaw
        packet.pitch = self.state.pitch
        logger.debug("Sending move packet to (%s, %s, %s)", packet.x, packet.y, packet.z)
        await asyncio.to_thread(self.connection.write_packet, packet)

    async def mine_block(self, block: str) -> None:
        if not self.connection:
            raise RuntimeError("Not connected")
        logger.info("Mining block type %s", block)
        # This approach relies on server support for digging at current coordinates.
        packet_start = serverbound.play.PlayerDiggingPacket()
        packet_start.status = serverbound.play.PlayerDiggingPacket.Status.STARTED_DIGGING
        packet_start.location = (int(self.state.x), int(self.state.y) - 1, int(self.state.z))
        packet_start.face = BlockFace.TOP
        packet_finish = serverbound.play.PlayerDiggingPacket()
        packet_finish.status = serverbound.play.PlayerDiggingPacket.Status.FINISHED_DIGGING
        packet_finish.location = packet_start.location
        packet_finish.face = BlockFace.TOP
        await asyncio.to_thread(self.connection.write_packet, packet_start)
        await asyncio.sleep(1.0)
        await asyncio.to_thread(self.connection.write_packet, packet_finish)

    async def place_block(self, block: Optional[str]) -> None:
        if not self.connection:
            raise RuntimeError("Not connected")
        packet = serverbound.play.PlayerBlockPlacementPacket()
        packet.location = (int(self.state.x), int(self.state.y) - 1, int(self.state.z))
        packet.face = BlockFace.TOP
        packet.hand = serverbound.play.PlayerBlockPlacementPacket.Hand.MAIN_HAND
        packet.cursor_x = 0.5
        packet.cursor_y = 1.0
        packet.cursor_z = 0.5
        logger.debug("Placing block at %s", packet.location)
        await asyncio.to_thread(self.connection.write_packet, packet)

    async def craft_item(self, item: str, quantity: int) -> None:
        # Crafting is complex; for demonstration send a chat command requiring server plugin support.
        if not self.connection:
            raise RuntimeError("Not connected")
        message = f"/craft {item} {quantity}"
        logger.debug("Sending craft command: %s", message)
        packet = serverbound.play.ChatPacket()
        packet.message = message
        await asyncio.to_thread(self.connection.write_packet, packet)

    async def attack(self, target: Optional[str]) -> None:
        if not self.connection:
            raise RuntimeError("Not connected")
        message = f"/attack {target or ''}".strip()
        packet = serverbound.play.ChatPacket()
        packet.message = message
        logger.debug("Sending attack command: %s", message)
        await asyncio.to_thread(self.connection.write_packet, packet)

    async def observe(self) -> Dict[str, Any]:
        return {
            "position": {"x": self.state.x, "y": self.state.y, "z": self.state.z},
            "yaw": self.state.yaw,
            "pitch": self.state.pitch,
            "health": self.state.health,
        }

    async def chat(self, message: str) -> None:
        if not self.connection:
            raise RuntimeError("Not connected")
        packet = serverbound.play.ChatPacket()
        packet.message = message
        await asyncio.to_thread(self.connection.write_packet, packet)


__all__ = ["MinecraftClient", "BotState"]
