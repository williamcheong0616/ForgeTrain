from datetime import datetime, timezone
from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey, Text, JSON
from sqlalchemy.types import TypeDecorator
from sqlalchemy.orm import relationship, DeclarativeBase

from backend.utils.time import now_utc


class TZDateTime(TypeDecorator):
    """DateTime that always returns UTC-aware datetime objects.

    SQLite discards timezone info on storage; this decorator re-attaches UTC
    on readback so Pydantic serialises with +00:00 and JavaScript parses correctly.
    """
    impl = DateTime
    cache_ok = True

    def process_result_value(self, value, dialect):
        if value is not None and value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return value


class Base(DeclarativeBase):
    pass


class Job(Base):
    __tablename__ = "jobs"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False)
    status = Column(String, default="pending")  # pending/running/completed/failed/cancelled
    training_method = Column(String, nullable=False)  # sft/dpo/ppo/rm/kto/orpo/unsupervised
    peft_method = Column(String, default="lora")  # lora/qlora/dora/full
    model_id = Column(Integer, ForeignKey("model_entries.id"), nullable=True)
    dataset_id = Column(Integer, ForeignKey("datasets.id"), nullable=True)
    config_json = Column(JSON, default={})
    celery_task_id = Column(String, nullable=True)
    output_dir = Column(String, nullable=True)
    error_msg = Column(Text, nullable=True)
    remarks = Column(Text, nullable=True)
    created_at = Column(TZDateTime, default=now_utc)
    started_at = Column(TZDateTime, nullable=True)
    finished_at = Column(TZDateTime, nullable=True)

    model = relationship("ModelEntry", back_populates="jobs")
    dataset = relationship("Dataset", back_populates="jobs")
    metrics = relationship("TrainingMetric", back_populates="job", cascade="all, delete-orphan")


class TrainingMetric(Base):
    __tablename__ = "training_metrics"

    id = Column(Integer, primary_key=True, index=True)
    job_id = Column(Integer, ForeignKey("jobs.id"), nullable=False, index=True)
    step = Column(Integer, nullable=False)
    epoch = Column(Float, nullable=True)
    loss = Column(Float, nullable=True)
    eval_loss = Column(Float, nullable=True)
    learning_rate = Column(Float, nullable=True)
    reward = Column(Float, nullable=True)
    grad_norm = Column(Float, nullable=True)
    timestamp = Column(TZDateTime, default=now_utc)

    job = relationship("Job", back_populates="metrics")


class ModelEntry(Base):
    __tablename__ = "model_entries"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False)
    hf_repo = Column(String, nullable=False, unique=True)
    local_path = Column(String, nullable=True)
    architecture = Column(String, nullable=True)
    template = Column(String, default="alpaca")
    is_downloaded = Column(String, default="false")
    downloaded_at = Column(TZDateTime, nullable=True)
    version = Column(String, nullable=True)

    jobs = relationship("Job", back_populates="model")


class Dataset(Base):
    __tablename__ = "datasets"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False)
    path = Column(String, nullable=False)
    format = Column(String, default="alpaca")  # alpaca/sharegpt/dpo/kto/plain_text/asr_csv
    template = Column(String, nullable=True)   # chat template associated with the data
    num_samples = Column(Integer, nullable=True)
    description = Column(Text, nullable=True)
    created_at = Column(TZDateTime, default=now_utc)

    jobs = relationship("Job", back_populates="dataset")


class Conversation(Base):
    __tablename__ = "conversations"

    id = Column(Integer, primary_key=True, index=True)
    title = Column(String, nullable=False, default="New Chat")
    model_path = Column(String, nullable=True)
    adapter_path = Column(String, nullable=True)
    system_prompt = Column(Text, nullable=True)
    gen_params = Column(JSON, nullable=True)
    created_at = Column(TZDateTime, default=now_utc)
    updated_at = Column(TZDateTime, default=now_utc)

    messages = relationship("ChatMessage", back_populates="conversation", cascade="all, delete-orphan", order_by="ChatMessage.id")


class ChatMessage(Base):
    __tablename__ = "chat_messages"

    id = Column(Integer, primary_key=True, index=True)
    conversation_id = Column(Integer, ForeignKey("conversations.id"), nullable=False, index=True)
    role = Column(String, nullable=False)   # user / assistant / system
    content = Column(Text, nullable=False)
    created_at = Column(TZDateTime, default=now_utc)

    conversation = relationship("Conversation", back_populates="messages")


class PromptProfile(Base):
    __tablename__ = "prompt_profiles"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False)
    system_prompt = Column(Text, nullable=True)
    gen_params = Column(JSON, nullable=True)
    created_at = Column(TZDateTime, default=now_utc)


class Setting(Base):
    __tablename__ = "settings"

    key = Column(String, primary_key=True)
    value = Column(Text, nullable=True)
    updated_at = Column(TZDateTime, default=now_utc, onupdate=now_utc)
