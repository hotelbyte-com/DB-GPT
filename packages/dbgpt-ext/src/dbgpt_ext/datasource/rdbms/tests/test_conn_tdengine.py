from contextlib import contextmanager

from dbgpt_ext.datasource.rdbms.conn_tdengine import TDengineConnector


def test_get_table_names_includes_stables(monkeypatch):
    executed = []

    class Result:
        def __init__(self, rows):
            self._rows = rows

        def fetchall(self):
            return self._rows

    class Session:
        def execute(self, statement):
            sql = str(statement)
            executed.append(sql)
            if sql == "SHOW TABLES":
                return Result([("hb_log_20260711",)])
            if sql == "SHOW STABLES":
                return Result([("hb_log",)])
            raise AssertionError(f"unexpected SQL: {sql}")

    @contextmanager
    def session_scope():
        yield Session()

    connector = object.__new__(TDengineConnector)
    connector._is_closed = True
    monkeypatch.setattr(connector, "session_scope", session_scope)

    assert connector.get_table_names() == ["hb_log_20260711", "hb_log"]
    assert executed == ["SHOW TABLES", "SHOW STABLES"]
