"""
Инициализация SQLAlchemy и SQLite.
"""
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base

from app.config import DB_PATH

engine = create_engine(
    f"sqlite:///{DB_PATH}",
    connect_args={"check_same_thread": False},
    echo=False,
)

from sqlalchemy import event


@event.listens_for(engine, "connect")
def _sqlite_pragmas(dbapi_conn, _):
    """WAL и таймаут блокировки: чтение статуса не ждёт запись результата."""
    cur = dbapi_conn.cursor()
    cur.execute("PRAGMA journal_mode=WAL")
    cur.execute("PRAGMA busy_timeout=5000")
    cur.close()


SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


def get_db():
    """Зависимость FastAPI: yields a database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db():
    """Создаёт все таблицы, если их ещё нет."""
    Base.metadata.create_all(bind=engine)
    # Лёгкая миграция: добавляем новые столбцы в уже существующую таблицу
    from sqlalchemy import inspect, text
    cols = {c["name"] for c in inspect(engine).get_columns("jobs")}
    if "processing_sec" not in cols:
        with engine.begin() as conn:
            conn.execute(text("ALTER TABLE jobs ADD COLUMN processing_sec VARCHAR(20)"))
