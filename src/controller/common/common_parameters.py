"""
Переиспользуемые параметры запроса, валидаторы заголовков, пагинация и миксины
на основе спецификации `common/parameters.yaml`.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from enum import Enum
from typing import Any, Iterable, Optional, Sequence, Tuple

from django.core.cache import cache
from django.db.models import QuerySet
from rest_framework import serializers
from rest_framework.request import Request
from rest_framework.response import Response


# =============================================================================
# Перечисления (Enums) из common/parameters.yaml
# =============================================================================

class DeviceFamilyEnum(str, Enum):
    PLANAR_MOSFET = "planar_mosfet"
    FD_SOI = "fd_soi"
    FINFET = "finfet"
    GAA_NANOSHEET = "gaa_nanosheet"
    GAA_NANOWIRE = "gaa_nanowire"
    TFET = "tfet"
    HEMT = "hemt"
    MEMRISTOR = "memristor"
    MEMTRANSISTOR = "memtransistor"
    CAD_CUSTOM = "cad_custom"

    @classmethod
    def choices(cls) -> list[tuple[str, str]]:
        return [(item.value, item.value) for item in cls]

    @classmethod
    def values(cls) -> set[str]:
        return {item.value for item in cls}


class ResourceStatusEnum(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    OK = "ok"
    FAILED = "failed"
    CANCELLED = "cancelled"

    @classmethod
    def choices(cls) -> list[tuple[str, str]]:
        return [(item.value, item.value) for item in cls]

    @classmethod
    def values(cls) -> set[str]:
        return {item.value for item in cls}


# =============================================================================
# Контейнер распарсенных параметров запроса
# =============================================================================

@dataclass(frozen=True)
class ParsedListParams:
    page: int = 0
    page_size: int = 20
    sort: Optional[str] = None
    search: Optional[str] = None
    device_family: Optional[str] = None
    status: Optional[str] = None


# =============================================================================
# DRF-сериализаторы для валидации Query, Path и Header параметров
# =============================================================================

class PaginationQuerySerializer(serializers.Serializer):
    """
    Параметры Page (с нуля, default=0) и PageSize (1..200, default=20).
    """
    page = serializers.IntegerField(min_value=0, default=0, required=False)
    page_size = serializers.IntegerField(min_value=1, max_value=200, default=20, required=False)


class ListFilterQuerySerializer(PaginationQuerySerializer):
    """
    Общие query-параметры списков: Page, PageSize, Sort, Search, DeviceFamilyQuery, StatusQuery.
    """
    sort = serializers.CharField(max_length=64, required=False, allow_blank=False)
    search = serializers.CharField(max_length=128, required=False, allow_blank=True)
    device_family = serializers.ChoiceField(
        choices=DeviceFamilyEnum.choices(),
        required=False,
    )
    status = serializers.ChoiceField(
        choices=ResourceStatusEnum.choices(),
        required=False,
    )


class SinceStreamQuerySerializer(serializers.Serializer):
    """
    Параметр SinceStream (since >= 0) для переподключения к SSE-потокам.
    """
    since = serializers.IntegerField(min_value=0, required=False)


class PositiveInt64IdSerializer(serializers.Serializer):
    """
    Базовый валидатор для всех path-идентификаторов (int64, minimum: 1):
    structure_id, profile_id, task_id, session_id, job_id, set_id,
    model_id, run_id, report_id, surrogate_id, training_id.
    """
    id = serializers.IntegerField(min_value=1, max_value=9223372036854775807, required=True)


class RequestHeadersSerializer(serializers.Serializer):
    """
    Валидатор заголовков Idempotency-Key (UUID), If-Match (<=128) и X-Request-Id (UUID).
    """
    idempotency_key = serializers.UUIDField(required=False, allow_null=True)
    if_match = serializers.CharField(max_length=128, required=False, allow_blank=False)
    x_request_id = serializers.UUIDField(required=False, allow_null=True)


# =============================================================================
# Утилиты извлечения и проверки параметров из Request
# =============================================================================

def parse_positive_path_id(value: Any, param_name: str) -> int:
    """
    Проверяет path-параметр идентификатора (int64 >= 1).
    Выбрасывает serializers.ValidationError при некорректном значении.
    """
    serializer = PositiveInt64IdSerializer(data={"id": value})
    if not serializer.is_valid():
        raise serializers.ValidationError(
            {param_name: f"Идентификатор '{param_name}' должен быть целым числом >= 1."}
        )
    return serializer.validated_data["id"]


def parse_since_stream(request: Request) -> Optional[int]:
    """
    Извлекает позицию `since` из query-параметров или `Last-Event-ID` из заголовков.
    """
    raw_since = request.query_params.get("since")
    if raw_since is None:
        raw_since = request.headers.get("Last-Event-ID")
    if raw_since is None or raw_since == "":
        return None

    serializer = SinceStreamQuerySerializer(data={"since": raw_since})
    if not serializer.is_valid():
        raise serializers.ValidationError(serializer.errors)
    return serializer.validated_data["since"]


def extract_idempotency_key(request: Request) -> Optional[str]:
    """
    Проверяет и возвращает строковый UUID заголовка `Idempotency-Key` (если передан).
    """
    raw_key = request.headers.get("Idempotency-Key")
    if not raw_key:
        return None
    try:
        return str(uuid.UUID(str(raw_key)))
    except (ValueError, TypeError) as exc:
        raise serializers.ValidationError(
            {"Idempotency-Key": "Заголовок Idempotency-Key должен быть валидным UUID."}
        ) from exc


def extract_if_match(request: Request) -> Optional[str]:
    """
    Извлекает и валидирует заголовок `If-Match` (maxLength: 128).
    """
    raw_etag = request.headers.get("If-Match")
    if raw_etag is None:
        return None
    if len(raw_etag) > 128 or not raw_etag.strip():
        raise serializers.ValidationError(
            {"If-Match": "Заголовок If-Match не должен быть пустым и превышать 128 символов."}
        )
    return raw_etag.strip()


def extract_or_generate_request_id(request: Request) -> str:
    """
    Возвращает валидный UUID из заголовка `X-Request-Id` или генерирует новый.
    """
    raw_id = request.headers.get("X-Request-Id")
    if raw_id:
        try:
            return str(uuid.UUID(str(raw_id)))
        except (ValueError, TypeError):
            pass
    return str(uuid.uuid4())


# =============================================================================
# Пагинация (нумерация страниц с 0, согласно common/parameters.yaml#/Page)
# =============================================================================

def paginate_zero_based(
    queryset: QuerySet | Sequence[Any],
    page: int = 0,
    page_size: int = 20,
) -> Tuple[list[Any], dict[str, int]]:
    """
    Выполняет срез QuerySet или последовательности с нулевой индексацией страниц (`page >= 0`).
    Возвращает кортеж `(items, meta)`.
    """
    safe_page = max(0, int(page))
    safe_page_size = min(200, max(1, int(page_size)))
    offset = safe_page * safe_page_size

    if isinstance(queryset, QuerySet):
        total = queryset.count()
        items = list(queryset[offset : offset + safe_page_size])
    else:
        total = len(queryset)
        items = list(queryset[offset : offset + safe_page_size])

    meta = {
        "total": total,
        "page": safe_page,
        "page_size": safe_page_size,
    }
    return items, meta


# =============================================================================
# Переиспользуемые миксины для Django REST Framework Views
# =============================================================================

class StandardQueryParamsMixin:
    """
    Миксин для разбора стандартных параметров списка (`page`, `page_size`, `sort`,
    `search`, `device_family`, `status`) с проверкой допустимых полей сортировки.
    """
    allowed_sort_fields: Iterable[str] = ("id", "-id", "created_at", "-created_at")
    default_sort: str = "-created_at"

    def parse_list_params(self, request: Request) -> ParsedListParams:
        serializer = ListFilterQuerySerializer(data=request.query_params)
        serializer.is_valid(raise_exception=True)
        data = serializer.validated_data

        sort_field = data.get("sort")
        allowed = set(self.allowed_sort_fields)
        if sort_field:
            if allowed and sort_field not in allowed:
                raise serializers.ValidationError(
                    {
                        "sort": f"Недопустимое поле сортировки '{sort_field}'.",
                        "allowed": sorted(allowed),
                    }
                )
        else:
            sort_field = self.default_sort

        return ParsedListParams(
            page=data.get("page", 0),
            page_size=data.get("page_size", 20),
            sort=sort_field,
            search=data.get("search") or None,
            device_family=data.get("device_family"),
            status=data.get("status"),
        )


class IdempotencyMixin:
    """
    Миксин для поддержки `Idempotency-Key` и `X-Request-Id` в POST-операциях.
    Защищает от создания дублирующих задач при ретраях GUI.
    """
    idempotency_ttl_seconds: int = 60 * 60 * 24  # 24 часа

    def get_request_id(self, request: Request) -> str:
        return extract_or_generate_request_id(request)

    def _build_idempotency_cache_key(self, request: Request, scope: str, idem_key: str) -> str:
        user_id = getattr(getattr(request, "user", None), "id", "anonymous")
        return f"tspice:idem:{scope}:{user_id}:{idem_key}"

    def get_idempotent_response(
        self,
        request: Request,
        scope: str,
        request_id: Optional[str] = None,
    ) -> Optional[Response]:
        idem_key = extract_idempotency_key(request)
        if not idem_key:
            return None

        cache_key = self._build_idempotency_cache_key(request, scope, idem_key)
        cached = cache.get(cache_key)
        if not cached:
            return None

        headers = dict(cached.get("headers") or {})
        if request_id:
            headers["X-Request-Id"] = request_id

        return Response(
            cached["data"],
            status=cached["status"],
            headers=headers,
        )

    def save_idempotent_response(
        self,
        request: Request,
        scope: str,
        response: Response,
    ) -> None:
        idem_key = extract_idempotency_key(request)
        if not idem_key:
            return

        cache_key = self._build_idempotency_cache_key(request, scope, idem_key)
        headers_to_store = {
            k: v for k, v in response.headers.items() if k.lower() in {"location", "etag"}
        }
        cache.set(
            cache_key,
            {
                "data": response.data,
                "status": response.status_code,
                "headers": headers_to_store,
            },
            timeout=self.idempotency_ttl_seconds,
        )
