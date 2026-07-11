"""TDengine connector.

TDengine is a time-series database optimized for IoT and Big Data.
Uses the official taos-ws-py (WebSocket) driver via sqlalchemy-tdengine dialect.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Tuple, Type

from sqlalchemy import text

from dbgpt.core.awel.flow import (
    TAGS_ORDER_HIGH,
    ResourceCategory,
    auto_register_resource,
)
from dbgpt.datasource.parameter import BaseDatasourceParameters
from dbgpt.datasource.rdbms.base import RDBMSConnector
from dbgpt.util.i18n_utils import _

logger = logging.getLogger(__name__)


@auto_register_resource(
    label=_("TDengine datasource"),
    category=ResourceCategory.DATABASE,
    tags={"order": TAGS_ORDER_HIGH},
    description=_(
        "High-performance time-series database optimized for IoT, DevOps, "
        "and real-time analytics workloads."
    ),
)
@dataclass
class TDengineParameters(BaseDatasourceParameters):
    """TDengine connection parameters."""

    __type__ = "tdengine"

    host: str = field(
        default="localhost",
        metadata={"help": _("Database host, e.g., localhost")},
    )
    port: int = field(
        default=6041,
        metadata={"help": _("TDengine REST/WebSocket port, default: 6041")},
    )
    user: str = field(
        default="root",
        metadata={"help": _("Database user, default: root")},
    )
    database: str = field(
        default="",
        metadata={
            "help": _("Database name, leave empty to connect without default db")
        },
    )
    password: str = field(
        default="${env:DBGPT_DB_PASSWORD}",
        metadata={
            "help": _(
                "Database password. You can write the password directly, "
                "or use environment variables such as ${env:DBGPT_DB_PASSWORD}"
            ),
            "tags": "privacy",
        },
    )

    def create_connector(self) -> "TDengineConnector":
        return TDengineConnector.from_parameters(self)

    def db_url(self, ssl: bool = False, charset: Optional[str] = None) -> str:
        db = self.database or ""
        return (
            f"taosws://{self.user}:{self.password}@"
            f"{self.host}:{self.port}/{db}"
        )


class TDengineConnector(RDBMSConnector):
    """TDengine connector.

    Connects via taosws:// (WebSocket) using sqlalchemy-tdengine dialect.
    """

    db_type: str = "tdengine"
    db_dialect: str = "tdengine"
    driver: str = "taosws"

    default_db = ["information_schema", "performance_schema"]

    @classmethod
    def param_class(cls) -> Type[TDengineParameters]:
        return TDengineParameters

    @classmethod
    def from_parameters(cls, parameters: TDengineParameters) -> "TDengineConnector":
        return cls.from_uri(
            parameters.db_url(),
            engine_args={"connect_args": {"timezone": "UTC"}},
        )

    def get_users(self) -> List[Tuple]:
        try:
            with self.session_scope() as session:
                cursor = session.execute(text("SHOW USERS"))
                rows = cursor.fetchall()
                return [(row[0], row[1]) for row in rows if len(row) >= 2]
        except Exception:
            return []

    def get_grants(self) -> List:
        return []

    def get_collation(self) -> str:
        return "UTF-8"

    def get_charset(self) -> str:
        return "UTF-8"

    def get_database_names(self) -> List[str]:
        try:
            with self.session_scope() as session:
                cursor = session.execute(text("SHOW DATABASES"))
                rows = cursor.fetchall()
                return [
                    row[0]
                    for row in rows
                    if row[0] not in ("information_schema", "performance_schema")
                ]
        except Exception:
            return []

    def get_table_names(self) -> List[str]:
        try:
            with self.session_scope() as session:
                names = []
                for statement in ("SHOW TABLES", "SHOW STABLES"):
                    cursor = session.execute(text(statement))
                    names.extend(
                        row[0]
                        for row in cursor.fetchall()
                        if row[0] not in self.default_db
                    )
                return list(dict.fromkeys(names))
        except Exception:
            return []

    def get_indexes(self, table_name: str) -> List[Dict]:
        return []

    def get_show_create_table(self, table_name: str) -> str:
        try:
            with self.session_scope() as session:
                cursor = session.execute(text(f"SHOW CREATE TABLE {table_name}"))
                row = cursor.fetchone()
                if row and len(row) >= 2:
                    return row[1]
                return ""
        except Exception:
            return ""

    def get_columns(self, table_name: str) -> List[Dict]:
        try:
            with self.session_scope() as session:
                cursor = session.execute(text(f"DESCRIBE {table_name}"))
                rows = cursor.fetchall()
                return [
                    {
                        "name": row[0],
                        "type": row[1] if len(row) > 1 else "",
                        "comment": "",
                    }
                    for row in rows
                ]
        except Exception:
            return []

    def get_fields(self, table_name: str, db_name: Optional[str] = None) -> List[Tuple]:
        try:
            with self.session_scope() as session:
                cursor = session.execute(text(f"DESCRIBE {table_name}"))
                return cursor.fetchall()
        except Exception:
            return []

    def get_table_comments(self, db_name: str) -> List[Tuple]:
        return []

    def table_simple_info(self) -> Iterable[str]:
        tables = self.get_table_names()
        results = []
        for table_name in tables:
            try:
                columns = self.get_columns(table_name)
                col_names = [col.get("name", "?") for col in columns]
                results.append(f"{table_name}({','.join(col_names)});")
            except Exception:
                results.append(f"{table_name}();")
        return results

    def get_current_db_name(self) -> str:
        try:
            with self.session_scope() as session:
                cursor = session.execute(text("SELECT DATABASE()"))
                row = cursor.fetchone()
                if row:
                    return row[0] or ""
                return ""
        except Exception:
            return ""
