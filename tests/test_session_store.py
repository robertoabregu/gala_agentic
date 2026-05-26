from __future__ import annotations

import json
import os
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

from memory import session_store


class ConversationSessionStoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.store_path = Path(self.temp_dir.name) / "session_store" / "whatsapp_sessions.json"

        patcher = patch.object(session_store, "SESSION_STORE_PATH", self.store_path)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_reuses_session_within_ttl_window(self) -> None:
        user_id = "whatsapp-5491125456750"
        created_at = datetime(2026, 5, 25, 19, 5, 2, tzinfo=timezone.utc)
        second_message_at = created_at + timedelta(minutes=14, seconds=59)

        with patch.dict(os.environ, {"WHATSAPP_SESSION_TTL_SECONDS": "900"}):
            with patch.object(session_store, "_utcnow", return_value=created_at):
                first_session_id = session_store.get_or_create_conversation_session(user_id)

            with patch.object(session_store, "_utcnow", return_value=second_message_at):
                second_session_id = session_store.get_or_create_conversation_session(user_id)

        self.assertEqual(first_session_id, second_session_id)

        payload = json.loads(self.store_path.read_text(encoding="utf-8"))
        self.assertEqual(payload[user_id]["active_session_id"], first_session_id)
        self.assertEqual(payload[user_id]["created_at"], created_at.isoformat())
        self.assertEqual(payload[user_id]["last_seen_at"], second_message_at.isoformat())

    def test_creates_new_session_after_ttl_expiration(self) -> None:
        user_id = "whatsapp-5491125456750"
        created_at = datetime(2026, 5, 25, 19, 5, 2, tzinfo=timezone.utc)
        expired_at = created_at + timedelta(seconds=901)

        with patch.dict(os.environ, {"WHATSAPP_SESSION_TTL_SECONDS": "900"}):
            with patch.object(session_store, "_utcnow", return_value=created_at):
                first_session_id = session_store.get_or_create_conversation_session(user_id)

            with patch.object(session_store, "_utcnow", return_value=expired_at):
                second_session_id = session_store.get_or_create_conversation_session(user_id)

        self.assertNotEqual(first_session_id, second_session_id)

        payload = json.loads(self.store_path.read_text(encoding="utf-8"))
        self.assertEqual(payload[user_id]["active_session_id"], second_session_id)
        self.assertEqual(payload[user_id]["created_at"], expired_at.isoformat())
        self.assertEqual(payload[user_id]["last_seen_at"], expired_at.isoformat())

    def test_recovers_from_corrupt_or_incomplete_store_data(self) -> None:
        user_id = "whatsapp-5491125456750"
        now = datetime(2026, 5, 25, 19, 5, 2, tzinfo=timezone.utc)
        self.store_path.parent.mkdir(parents=True, exist_ok=True)
        self.store_path.write_text('{"users": "broken"}', encoding="utf-8")

        with patch.dict(os.environ, {"WHATSAPP_SESSION_TTL_SECONDS": "900"}):
            with patch.object(session_store, "_utcnow", return_value=now):
                recovered_session_id = session_store.get_or_create_conversation_session(user_id)

        payload = json.loads(self.store_path.read_text(encoding="utf-8"))
        self.assertEqual(payload[user_id]["active_session_id"], recovered_session_id)

        self.store_path.write_text(
            json.dumps(
                {
                    user_id: {
                        "active_session_id": "stale-session",
                        "last_seen_at": now.isoformat(),
                    }
                }
            ),
            encoding="utf-8",
        )

        with patch.dict(os.environ, {"WHATSAPP_SESSION_TTL_SECONDS": "900"}):
            with patch.object(session_store, "_utcnow", return_value=now + timedelta(minutes=1)):
                rotated_session_id = session_store.get_or_create_conversation_session(user_id)

        self.assertNotEqual(rotated_session_id, "stale-session")


if __name__ == "__main__":
    unittest.main()
