"""
Redis-based session memory for conversation management
"""
import redis
import json
from typing import List, Dict, Optional
from loguru import logger
from datetime import datetime, timedelta


class RedisMemory:
    """
    Session-based memory using Redis
    Stores conversation history and context
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 6379,
        db: int = 0,
        session_ttl: int = 3600  # 1 hour default
    ):
        """
        Initialize Redis memory

        Args:
            host: Redis host
            port: Redis port
            db: Redis database number
            session_ttl: Session TTL in seconds
        """
        try:
            self.client = redis.Redis(
                host=host,
                port=port,
                db=db,
                decode_responses=True
            )
            self.client.ping()
            logger.info(f"Connected to Redis at {host}:{port}")
        except redis.ConnectionError:
            logger.warning("Could not connect to Redis. Using in-memory fallback.")
            self.client = None
            self.memory_fallback = {}

        self.session_ttl = session_ttl

    def store_message(
        self,
        session_id: str,
        role: str,
        content: str,
        metadata: Optional[Dict] = None
    ):
        """
        Store a message in the session

        Args:
            session_id: Session identifier
            role: Message role (user/assistant/system)
            content: Message content
            metadata: Optional metadata
        """
        message = {
            'role': role,
            'content': content,
            'timestamp': datetime.now().isoformat(),
            'metadata': metadata or {}
        }

        key = f"session:{session_id}:messages"

        if self.client:
            # Store in Redis
            self.client.rpush(key, json.dumps(message))
            self.client.expire(key, self.session_ttl)
            logger.debug(f"Stored message in session {session_id}")
        else:
            # Fallback to in-memory
            if session_id not in self.memory_fallback:
                self.memory_fallback[session_id] = []
            self.memory_fallback[session_id].append(message)

    def get_conversation_history(
        self,
        session_id: str,
        limit: Optional[int] = None
    ) -> List[Dict]:
        """
        Get conversation history for a session

        Args:
            session_id: Session identifier
            limit: Optional limit on number of messages

        Returns:
            List of messages
        """
        key = f"session:{session_id}:messages"

        if self.client:
            # Get from Redis
            if limit:
                messages_json = self.client.lrange(key, -limit, -1)
            else:
                messages_json = self.client.lrange(key, 0, -1)

            messages = [json.loads(msg) for msg in messages_json]
        else:
            # Get from fallback
            messages = self.memory_fallback.get(session_id, [])
            if limit:
                messages = messages[-limit:]

        logger.debug(f"Retrieved {len(messages)} messages for session {session_id}")
        return messages

    def get_context_window(
        self,
        session_id: str,
        window_size: int = 5
    ) -> str:
        """
        Get recent conversation as a context string

        Args:
            session_id: Session identifier
            window_size: Number of recent messages to include

        Returns:
            Formatted context string
        """
        messages = self.get_conversation_history(session_id, limit=window_size)

        context_parts = []
        for msg in messages:
            role = msg['role']
            content = msg['content']
            context_parts.append(f"{role}: {content}")

        return "\n\n".join(context_parts)

    def store_session_metadata(
        self,
        session_id: str,
        metadata: Dict
    ):
        """
        Store metadata for a session

        Args:
            session_id: Session identifier
            metadata: Metadata dictionary
        """
        key = f"session:{session_id}:metadata"

        if self.client:
            self.client.set(key, json.dumps(metadata), ex=self.session_ttl)
        else:
            # Store in fallback
            fallback_key = f"{session_id}:metadata"
            self.memory_fallback[fallback_key] = metadata

    def get_session_metadata(self, session_id: str) -> Optional[Dict]:
        """Get session metadata"""
        key = f"session:{session_id}:metadata"

        if self.client:
            data = self.client.get(key)
            return json.loads(data) if data else None
        else:
            fallback_key = f"{session_id}:metadata"
            return self.memory_fallback.get(fallback_key)

    def clear_session(self, session_id: str):
        """Clear all data for a session"""
        if self.client:
            keys = self.client.keys(f"session:{session_id}:*")
            if keys:
                self.client.delete(*keys)
            logger.info(f"Cleared session {session_id}")
        else:
            # Clear from fallback
            keys_to_delete = [k for k in self.memory_fallback.keys() if k.startswith(session_id)]
            for key in keys_to_delete:
                del self.memory_fallback[key]

    def get_active_sessions(self) -> List[str]:
        """Get list of active session IDs"""
        if self.client:
            keys = self.client.keys("session:*:messages")
            session_ids = [key.split(':')[1] for key in keys]
            return list(set(session_ids))
        else:
            session_ids = [k for k in self.memory_fallback.keys() if ':metadata' not in k]
            return list(set(session_ids))

    def summarize_conversation(self, session_id: str, llm=None) -> str:
        """
        Create a summary of the conversation
        Useful for very long conversations

        Args:
            session_id: Session identifier
            llm: Optional LLM for summarization

        Returns:
            Summary string
        """
        messages = self.get_conversation_history(session_id)

        if not messages:
            return "No conversation history."

        # Simple summarization without LLM
        if not llm:
            message_count = len(messages)
            user_messages = [m for m in messages if m['role'] == 'user']
            assistant_messages = [m for m in messages if m['role'] == 'assistant']

            return (
                f"Conversation with {message_count} messages "
                f"({len(user_messages)} user, {len(assistant_messages)} assistant). "
                f"Last updated: {messages[-1]['timestamp']}"
            )

        # Use LLM for summarization
        conversation_text = self.get_context_window(session_id, window_size=10)

        summary_prompt = f"""Summarize this compliance-related conversation in 2-3 sentences:

{conversation_text}

Summary:"""

        from langchain_core.messages import HumanMessage
        summary = llm.invoke([HumanMessage(content=summary_prompt)])

        return summary.content


class ConversationBufferMemory:
    """
    Simple in-memory buffer for conversations
    Alternative to Redis for local development
    """

    def __init__(self, max_messages: int = 10):
        self.conversations = {}
        self.max_messages = max_messages

    def add_message(self, session_id: str, role: str, content: str):
        """Add a message to the buffer"""
        if session_id not in self.conversations:
            self.conversations[session_id] = []

        self.conversations[session_id].append({
            'role': role,
            'content': content,
            'timestamp': datetime.now().isoformat()
        })

        # Keep only recent messages
        if len(self.conversations[session_id]) > self.max_messages:
            self.conversations[session_id] = self.conversations[session_id][-self.max_messages:]

    def get_messages(self, session_id: str) -> List[Dict]:
        """Get all messages for a session"""
        return self.conversations.get(session_id, [])

    def clear(self, session_id: str):
        """Clear conversation for a session"""
        if session_id in self.conversations:
            del self.conversations[session_id]


if __name__ == "__main__":
    # Test Redis memory
    memory = RedisMemory()

    test_session = "test_session_123"

    # Store some messages
    memory.store_message(test_session, "user", "What is GDPR?")
    memory.store_message(
        test_session,
        "assistant",
        "GDPR is the General Data Protection Regulation..."
    )
    memory.store_message(test_session, "user", "Tell me about Article 17")

    # Retrieve history
    history = memory.get_conversation_history(test_session)

    print(f"Conversation history for {test_session}:")
    for msg in history:
        print(f"{msg['role']}: {msg['content']}")

    # Get context window
    context = memory.get_context_window(test_session, window_size=2)
    print(f"\nContext window:\n{context}")

    # Clean up
    memory.clear_session(test_session)
