"""
数据库模块 - PostgreSQL 封装
"""
from .base import DatabaseBase
from .postgresql import PostgreSQLDatabase

__all__ = ['DatabaseBase', 'PostgreSQLDatabase']
