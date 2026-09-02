"""
WebSocket Manager for live streaming browser progress, screenshots, logs, and state updates.
"""

import json
import logging
from typing import Dict, List, Set, Optional
from fastapi import WebSocket, WebSocketDisconnect

logger = logging.getLogger("websocket_manager")


class ConnectionManager:
    def __init__(self):
        self.active_connections: Set[WebSocket] = set()
        self.session_connections: Dict[str, Set[WebSocket]] = {}

    async def connect(self, websocket: WebSocket, session_id: Optional[str] = None):
        await websocket.accept()
        self.active_connections.add(websocket)
        if session_id:
            self.session_connections.setdefault(session_id, set()).add(websocket)
        logger.info(f"🔌 Client connected to WebSocket (session: {session_id}). Total: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket, session_id: Optional[str] = None):
        self.active_connections.discard(websocket)
        if session_id and session_id in self.session_connections:
            self.session_connections[session_id].discard(websocket)
            if not self.session_connections[session_id]:
                del self.session_connections[session_id]
        for sid, conns in list(self.session_connections.items()):
            conns.discard(websocket)
            if not conns:
                del self.session_connections[sid]
        logger.info(f"🔌 Client disconnected. Remaining connections: {len(self.active_connections)}")

    async def broadcast(self, message: Dict):
        """Broadcasts JSON payload to all connected frontend clients."""
        if not self.active_connections:
            return

        dead_sockets = set()
        msg_str = json.dumps(message)

        for connection in list(self.active_connections):
            try:
                await connection.send_text(msg_str)
            except Exception as e:
                logger.warning(f"Error sending websocket message: {e}")
                dead_sockets.add(connection)

        for dead in dead_sockets:
            self.disconnect(dead)


ws_manager = ConnectionManager()
