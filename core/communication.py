"""
Système de communication avancé entre agents.
Protocoles, canaux et gestion des messages.
"""

import asyncio
import json
from typing import Dict, Any, List, Optional, Callable, Union
from datetime import datetime
from dataclasses import dataclass, asdict
from enum import Enum
from collections import defaultdict, deque
import threading
import uuid

from utils.logger import setup_logger


class MessagePriority(Enum):
    """Niveaux de priorité des messages."""
    CRITICAL = 1
    HIGH = 2
    NORMAL = 3
    LOW = 4


class MessageType(Enum):
    """Types de messages."""
    BROADCAST = "broadcast"
    DIRECT = "direct"
    REQUEST = "request"
    RESPONSE = "response"
    NOTIFICATION = "notification"
    DISCOVERY = "discovery"
    ALERT = "alert"
    REPORT = "report"


class CommunicationProtocol(Enum):
    """Protocoles de communication."""
    SYNC = "synchronous"
    ASYNC = "asynchronous"
    PUBSUB = "publish_subscribe"
    REQUEST_REPLY = "request_reply"


@dataclass
class Message:
    """Structure d'un message."""
    id: str
    sender: str
    recipient: Optional[str]  # None pour broadcast
    type: MessageType
    priority: MessagePriority
    content: Dict[str, Any]
    timestamp: datetime
    protocol: CommunicationProtocol
    correlation_id: Optional[str] = None  # Pour lier request/response
    ttl: Optional[int] = None  # Time to live en secondes
    metadata: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertit le message en dictionnaire."""
        data = asdict(self)
        data['type'] = self.type.value
        data['priority'] = self.priority.value
        data['protocol'] = self.protocol.value
        data['timestamp'] = self.timestamp.isoformat()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Message':
        """Crée un message depuis un dictionnaire."""
        data['type'] = MessageType(data['type'])
        data['priority'] = MessagePriority(data['priority'])
        data['protocol'] = CommunicationProtocol(data['protocol'])
        data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        return cls(**data)


class Channel:
    """Canal de communication entre agents."""
    
    def __init__(self, name: str, protocol: CommunicationProtocol):
        self.name = name
        self.protocol = protocol
        self.subscribers: List[str] = []
        self.message_queue: deque = deque(maxlen=1000)
        self.filters: List[Callable] = []
        self.logger = setup_logger(f"Channel.{name}")
        self._lock = threading.Lock()
    
    def subscribe(self, agent_name: str) -> None:
        """Abonne un agent au canal."""
        with self._lock:
            if agent_name not in self.subscribers:
                self.subscribers.append(agent_name)
                self.logger.info(f"{agent_name} subscribed to channel {self.name}")
    
    def unsubscribe(self, agent_name: str) -> None:
        """Désabonne un agent du canal."""
        with self._lock:
            if agent_name in self.subscribers:
                self.subscribers.remove(agent_name)
                self.logger.info(f"{agent_name} unsubscribed from channel {self.name}")
    
    def add_filter(self, filter_func: Callable[[Message], bool]) -> None:
        """Ajoute un filtre au canal."""
        self.filters.append(filter_func)
    
    def should_deliver(self, message: Message) -> bool:
        """Vérifie si un message doit être délivré."""
        return all(f(message) for f in self.filters)
    
    def publish(self, message: Message) -> None:
        """Publie un message sur le canal."""
        with self._lock:
            if self.should_deliver(message):
                self.message_queue.append(message)
                self.logger.debug(f"Message {message.id} published to channel {self.name}")
    
    def get_messages(self, agent_name: str, limit: int = 10) -> List[Message]:
        """Récupère les messages pour un agent."""
        if agent_name not in self.subscribers:
            return []
        
        with self._lock:
            messages = []
            for msg in list(self.message_queue)[-limit:]:
                if msg.recipient is None or msg.recipient == agent_name:
                    messages.append(msg)
            return messages


class MessageBroker:
    """Courtier de messages central."""
    
    def __init__(self):
        self.channels: Dict[str, Channel] = {}
        self.agents: Dict[str, 'AgentEndpoint'] = {}
        self.message_history: deque = deque(maxlen=10000)
        self.pending_requests: Dict[str, Message] = {}
        self.logger = setup_logger("MessageBroker")
        self._lock = threading.Lock()
        
        # Créer les canaux par défaut
        self._create_default_channels()
    
    def _create_default_channels(self):
        """Crée les canaux de communication par défaut."""
        default_channels = [
            ("general", CommunicationProtocol.PUBSUB),
            ("alerts", CommunicationProtocol.PUBSUB),
            ("discoveries", CommunicationProtocol.PUBSUB),
            ("reports", CommunicationProtocol.ASYNC),
            ("coordination", CommunicationProtocol.SYNC),
            ("code_review", CommunicationProtocol.REQUEST_REPLY),
            ("testing", CommunicationProtocol.ASYNC)
        ]
        
        for name, protocol in default_channels:
            self.create_channel(name, protocol)
    
    def create_channel(self, name: str, protocol: CommunicationProtocol) -> Channel:
        """Crée un nouveau canal."""
        with self._lock:
            if name not in self.channels:
                channel = Channel(name, protocol)
                self.channels[name] = channel
                self.logger.info(f"Channel '{name}' created with protocol {protocol.value}")
            return self.channels[name]
    
    def register_agent(self, agent_name: str, endpoint: 'AgentEndpoint') -> None:
        """Enregistre un agent dans le système."""
        with self._lock:
            self.agents[agent_name] = endpoint
            self.logger.info(f"Agent '{agent_name}' registered")
    
    def unregister_agent(self, agent_name: str) -> None:
        """Désenregistre un agent."""
        with self._lock:
            if agent_name in self.agents:
                del self.agents[agent_name]
                # Désabonner de tous les canaux
                for channel in self.channels.values():
                    channel.unsubscribe(agent_name)
                self.logger.info(f"Agent '{agent_name}' unregistered")
    
    def send_message(self, message: Message) -> bool:
        """Envoie un message."""
        try:
            # Ajouter à l'historique
            self.message_history.append(message)
            
            # Traiter selon le type et le protocole
            if message.type == MessageType.BROADCAST:
                return self._handle_broadcast(message)
            elif message.type == MessageType.DIRECT:
                return self._handle_direct(message)
            elif message.type == MessageType.REQUEST:
                return self._handle_request(message)
            elif message.type == MessageType.RESPONSE:
                return self._handle_response(message)
            else:
                return self._handle_channel_message(message)
                
        except Exception as e:
            self.logger.error(f"Error sending message {message.id}: {str(e)}")
            return False
    
    def _handle_broadcast(self, message: Message) -> bool:
        """Gère un message broadcast."""
        channel = self.channels.get("general")
        if channel:
            channel.publish(message)
            
        # Notifier tous les agents
        for agent_name, endpoint in self.agents.items():
            if agent_name != message.sender:
                endpoint.receive_message(message)
        
        self.logger.info(f"Broadcast message {message.id} from {message.sender}")
        return True
    
    def _handle_direct(self, message: Message) -> bool:
        """Gère un message direct."""
        if message.recipient and message.recipient in self.agents:
            endpoint = self.agents[message.recipient]
            endpoint.receive_message(message)
            self.logger.debug(f"Direct message {message.id} from {message.sender} to {message.recipient}")
            return True
        else:
            self.logger.warning(f"Recipient {message.recipient} not found for message {message.id}")
            return False
    
    def _handle_request(self, message: Message) -> bool:
        """Gère une requête."""
        # Stocker la requête en attente
        self.pending_requests[message.id] = message

        # Timeout pour la réponse
        if message.ttl:
            threading.Timer(message.ttl, self._timeout_request, args=[message.id]).start() 

        # Acheminer vers le destinataire
        if message.recipient in self.agents:
            endpoint = self.agents[message.recipient]
            endpoint.receive_message(message)
                       
            return True
        else:
            # Si le destinataire n'existe pas, la requête échouera simplement par timeout,
            # ce qui est le comportement attendu. L'expéditeur sera notifié.
            self.logger.warning(f"Recipient {message.recipient} for request {message.id} not found. Awaiting timeout.")
            return False # La livraison immédiate a échoué
    
    def _handle_response(self, message: Message) -> bool:
        """Gère une réponse."""
        # Vérifier si c'est une réponse à une requête en attente
        if message.correlation_id and message.correlation_id in self.pending_requests:
            original_request = self.pending_requests.pop(message.correlation_id)
            
            # Acheminer vers le demandeur original
            if original_request.sender in self.agents:
                endpoint = self.agents[original_request.sender]
                endpoint.receive_message(message)
                return True
        
        self.logger.warning(f"No pending request found for response {message.id}")
        return False
    
    def _handle_channel_message(self, message: Message) -> bool:
        """Gère un message via canal."""
        # Déterminer le canal approprié
        channel_name = "general"
        
        if message.type == MessageType.DISCOVERY:
            channel_name = "discoveries"
        elif message.type == MessageType.ALERT:
            channel_name = "alerts"
        elif message.type == MessageType.REPORT:
            channel_name = "reports"
        
        if channel_name in self.channels:
            channel = self.channels[channel_name]
            channel.publish(message)
            
            # Notifier les abonnés
            for subscriber in channel.subscribers:
                if subscriber != message.sender and subscriber in self.agents:
                    self.agents[subscriber].receive_message(message)
            
            return True
        return False
    
    def _timeout_request(self, request_id: str) -> None:
        """Gère le timeout d'une requête."""
        if request_id in self.pending_requests:
            request = self.pending_requests.pop(request_id)
            self.logger.warning(f"Request {request_id} timed out")
            
            # Notifier le demandeur
            if request.sender in self.agents:
                timeout_msg = Message(
                    id=str(uuid.uuid4()),
                    sender="system",
                    recipient=request.sender,
                    type=MessageType.NOTIFICATION,
                    priority=MessagePriority.HIGH,
                    content={"error": "Request timeout", "request_id": request_id},
                    timestamp=datetime.now(),
                    protocol=CommunicationProtocol.SYNC
                )
                self.agents[request.sender].receive_message(timeout_msg)
    
    def get_message_stats(self) -> Dict[str, Any]:
        """Retourne les statistiques de messages."""
        stats = {
            'total_messages': len(self.message_history),
            'pending_requests': len(self.pending_requests),
            'registered_agents': len(self.agents),
            'active_channels': len(self.channels),
            'messages_by_type': defaultdict(int),
            'messages_by_priority': defaultdict(int),
            'messages_by_sender': defaultdict(int)
        }
        
        for msg in self.message_history:
            stats['messages_by_type'][msg.type.value] += 1
            stats['messages_by_priority'][msg.priority.value] += 1
            stats['messages_by_sender'][msg.sender] += 1
        
        return stats


class AgentEndpoint:
    """Point de terminaison pour un agent."""
    
    def __init__(self, agent_name: str, callback: Callable[[Message], None]):
        self.agent_name = agent_name
        self.callback = callback
        self.inbox: deque = deque(maxlen=100)
        self.sent_messages: List[str] = []
        self.subscriptions: List[str] = []
        self.logger = setup_logger(f"AgentEndpoint.{agent_name}")
    
    def receive_message(self, message: Message) -> None:
        """Reçoit un message."""
        self.inbox.append(message)
        self.logger.debug(f"Received message {message.id} from {message.sender}")
        
        # Appeler le callback de l'agent
        try:
            self.callback(message)
        except Exception as e:
            self.logger.error(f"Error in message callback: {str(e)}")
    
    def get_inbox(self, limit: int = 10) -> List[Message]:
        """Récupère les messages de la boîte de réception."""
        return list(self.inbox)[-limit:]
    
    def clear_inbox(self) -> None:
        """Vide la boîte de réception."""
        self.inbox.clear()


class CommunicationManager:
    """Gestionnaire de communication pour les agents."""
    
    def __init__(self, agent_name: str, broker: Optional[MessageBroker] = None):
        self.agent_name = agent_name
        self.broker = broker or MessageBroker()
        self.logger = setup_logger(f"CommManager.{agent_name}")
        self.message_handlers: Dict[MessageType, Callable] = {}
        self.endpoint: Optional[AgentEndpoint] = None
        
        # S'enregistrer auprès du broker
        self._register()
    
    def _register(self) -> None:
        """Enregistre l'agent auprès du broker."""
        self.endpoint = AgentEndpoint(self.agent_name, self._handle_incoming_message)
        self.broker.register_agent(self.agent_name, self.endpoint)
    
    def _handle_incoming_message(self, message: Message) -> None:
        """Gère un message entrant."""
        # Appeler le handler approprié
        if message.type in self.message_handlers:
            try:
                self.message_handlers[message.type](message)
            except Exception as e:
                self.logger.error(f"Error handling message {message.id}: {str(e)}")
    
    def register_handler(self, message_type: MessageType, handler: Callable[[Message], None]) -> None:
        """Enregistre un handler pour un type de message."""
        self.message_handlers[message_type] = handler
    
    def subscribe_to_channel(self, channel_name: str) -> None:
        """S'abonne à un canal."""
        if channel_name in self.broker.channels:
            channel = self.broker.channels[channel_name]
            channel.subscribe(self.agent_name)
            self.endpoint.subscriptions.append(channel_name)
    
    def send_message(
        self,
        content: Dict[str, Any],
        recipient: Optional[str] = None,
        message_type: MessageType = MessageType.DIRECT,
        priority: MessagePriority = MessagePriority.NORMAL,
        protocol: CommunicationProtocol = CommunicationProtocol.ASYNC,
        correlation_id: Optional[str] = None,
        ttl: Optional[int] = None
    ) -> str:
        """Envoie un message."""
        message = Message(
            id=str(uuid.uuid4()),
            sender=self.agent_name,
            recipient=recipient,
            type=message_type,
            priority=priority,
            content=content,
            timestamp=datetime.now(),
            protocol=protocol,
            correlation_id=correlation_id,
            ttl=ttl
        )
        
        success = self.broker.send_message(message)
        if success:
            self.endpoint.sent_messages.append(message.id)
        
        return message.id
    
    def broadcast(self, content: Dict[str, Any], priority: MessagePriority = MessagePriority.NORMAL) -> str:
        """Diffuse un message à tous les agents."""
        return self.send_message(
            content=content,
            message_type=MessageType.BROADCAST,
            priority=priority
        )
    
    def request(
        self,
        recipient: str,
        content: Dict[str, Any],
        timeout: int = 30
    ) -> str:
        """Envoie une requête et attend une réponse."""
        return self.send_message(
            content=content,
            recipient=recipient,
            message_type=MessageType.REQUEST,
            protocol=CommunicationProtocol.REQUEST_REPLY,
            ttl=timeout
        )
    
    def respond(self, request_message: Message, response_content: Dict[str, Any]) -> str:
        """Répond à une requête."""
        return self.send_message(
            content=response_content,
            recipient=request_message.sender,
            message_type=MessageType.RESPONSE,
            protocol=CommunicationProtocol.REQUEST_REPLY,
            correlation_id=request_message.id
        )
    
    def alert(self, content: Dict[str, Any]) -> str:
        """Envoie une alerte."""
        return self.send_message(
            content=content,
            message_type=MessageType.ALERT,
            priority=MessagePriority.HIGH,
            protocol=CommunicationProtocol.PUBSUB
        )
    
    def share_discovery(self, discovery: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """Partage une découverte."""
        content = {
            "discovery": discovery,
            "metadata": metadata or {},
            "timestamp": datetime.now().isoformat()
        }
        
        return self.send_message(
            content=content,
            message_type=MessageType.DISCOVERY,
            priority=MessagePriority.NORMAL
        )
    
    def get_messages(self, limit: int = 10, message_type: Optional[MessageType] = None) -> List[Message]:
        """Récupère les messages reçus."""
        messages = self.endpoint.get_inbox(limit)
        
        if message_type:
            messages = [m for m in messages if m.type == message_type]
        
        return messages
    
    def get_stats(self) -> Dict[str, Any]:
        """Retourne les statistiques de communication."""
        return {
            'agent': self.agent_name,
            'messages_sent': len(self.endpoint.sent_messages),
            'messages_received': len(self.endpoint.inbox),
            'subscriptions': self.endpoint.subscriptions,
            'handlers_registered': list(self.message_handlers.keys())
        }


# Singleton pour le broker global
_global_broker = None


def get_global_broker() -> MessageBroker:
    """Retourne le broker global."""
    global _global_broker
    if _global_broker is None:
        _global_broker = MessageBroker()
    return _global_broker
