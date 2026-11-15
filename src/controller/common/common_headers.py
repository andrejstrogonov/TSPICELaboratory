"""
Модуль общих HTTP-заголовков ответов и запросов (common/headers.yaml).
Обеспечивает единое формирование, парсинг и валидацию:
- X-Request-Id (UUID для сквозной трассировки, NFR-REL-02)
- ETag / If-Match (версия ресурса для оптимистичной блокировки)
- Location (канонический URI созданного ресурса)
- X-RateLimit-Remaining (остаток квоты запросов в текущем окне)
"""

from __future__ import annotations

import hashlib
import uuid
from dataclasses import dataclass
from typing import Any, Mapping, Optional

from django.db.models import Model
from django.http import HttpRequest, HttpResponse
from django.urls import reverse
from rest_framework import serializers
from rest_framework.request import Request
from rest_framework.response import Response


# =============================================================================
# 1. Константы имён заголовков (согласно common/headers.yaml)
# =============================================================================

HEADER_X_REQUEST_ID = "X-Request-Id"
HEADER_ETAG = "ETag"
HEADER_IF_MATCH = "If-Match"
HEADER_LOCATION = "Location"
HEADER_X_RATELIMIT_REMAINING = "X-RateLimit-Remaining"


# =============================================================================
# 2. Вспомогательные функции работы с заголовками
# =============================================================================

def normalize_request_id(raw_value: Optional[str]) -> str:
    """
    Проверяет, что переданный X-Request-Id является валидным UUID.
    Если заголовок отсутствует или некорректен — генерирует новый UUIDv4.
    """
    if raw_value:
        try:
            return str(uuid.UUID(str(raw_value).strip()))
        except (ValueError, TypeError, AttributeError):
            pass
    return str(uuid.uuid4())


def get_or_create_request_id(request: Request | HttpRequest) -> str:
    """
    Извлекает X-Request-Id из запроса (или атрибута request.request_id)
    и гарантирует возврат валидной строки UUID.
    """
    existing = getattr(request, "request_id", None)
    if existing:
        return normalize_request_id(existing)

    headers = getattr(request, "headers", None)
    raw = headers.get(HEADER_X_REQUEST_ID) if headers else request.META.get("HTTP_X_REQUEST_ID")
    req_id = normalize_request_id(raw)
    setattr(request, "request_id", req_id)
    return req_id


def compute_resource_etag(instance: Any, extra_salt: str = "") -> str:
    """
    Вычисляет строгий ETag в кавычках ("<sha256[:16]>") для модели Django
    или произвольного объекта на основе первичного ключа и версии/времени обновления.
    """
    if isinstance(instance, Model):
        pk = instance.pk
        version_marker = (
            getattr(instance, "version", None)
            or getattr(instance, "updated_at", None)
            or getattr(instance, "created_at", None)
            or pk
        )
        cls_name = instance.__class__.__name__
    else:
        pk = getattr(instance, "id", id(instance))
        version_marker = getattr(instance, "version", pk)
        cls_name = type(instance).__name__

    payload = f"{cls_name}:{pk}:{version_marker}:{extra_salt}".encode("utf-8")
    digest = hashlib.sha256(payload).hexdigest()[:16]
    return f'"{digest}"'


def parse_if_match_header(request: Request | HttpRequest) -> Optional[str]:
    """
    Извлекает значение заголовка If-Match и очищает пробелы.
    Возвращает None, если заголовок не передан.
    """
    headers = getattr(request, "headers", None)
    raw = headers.get(HEADER_IF_MATCH) if headers else request.META.get("HTTP_IF_MATCH")
    if raw is None:
        return None
    cleaned = str(raw).strip()
    return cleaned if cleaned else None


def is_etag_matching(if_match_value: Optional[str], current_etag: str) -> bool:
    """
    Сравнивает переданный клиентом If-Match с актуальным ETag ресурса.
    Поддерживает wildcard '*' и сравнение с учётом/без учёта кавычек.
    """
    if not if_match_value:
        return False
    if if_match_value == "*":
        return True

    expected = current_etag.strip().strip('"')
    candidates = [item.strip().lstrip("W/").strip('"') for item in if_match_value.split(",")]
    return expected in candidates


def build_canonical_location(
    request: Request | HttpRequest,
    view_name: str,
    kwargs: Optional[Mapping[str, Any]] = None,
) -> str:
    """
    Формирует абсолютный URI созданного ресурса для заголовка Location.
    """
    relative_url = reverse(view_name, kwargs=dict(kwargs or {}))
    return request.build_absolute_uri(relative_url)


def format_ratelimit_remaining(remaining: int) -> str:
    """
    Форматирует остаток квоты (minimum: 0) для заголовка X-RateLimit-Remaining.
    """
    return str(max(0, int(remaining)))


# =============================================================================
# 3. Контейнер стандартных заголовков ответа и DRF-сериализатор
# =============================================================================

@dataclass(frozen=True)
class StandardResponseHeaders:
    """
    Типизированный контейнер для сборки стандартных заголовков ответа API.
    """
    request_id: Optional[str] = None
    etag: Optional[str] = None
    location: Optional[str] = None
    ratelimit_remaining: Optional[int] = None

    def to_dict(self) -> dict[str, str]:
        headers: dict[str, str] = {}
        if self.request_id:
            headers[HEADER_X_REQUEST_ID] = normalize_request_id(self.request_id)
        if self.etag:
            headers[HEADER_ETAG] = self.etag
        if self.location:
            headers[HEADER_LOCATION] = self.location
        if self.ratelimit_remaining is not None:
            headers[HEADER_X_RATELIMIT_REMAINING] = format_ratelimit_remaining(
                self.ratelimit_remaining
            )
        return headers

    def apply(self, response: Response | HttpResponse) -> Response | HttpResponse:
        for key, value in self.to_dict().items():
            response[key] = value
        return response


class ResponseHeadersSerializer(serializers.Serializer):
    """
    DRF-сериализатор для валидации и документации заголовков ответа
    в OpenAPI / drf-spectacular (согласно common/headers.yaml).
    """
    x_request_id = serializers.UUIDField(
        required=False,
        help_text="Идентификатор запроса для трассировки (NFR-REL-02).",
    )
    etag = serializers.CharField(
        required=False,
        help_text="Версия ресурса для оптимистичной блокировки (If-Match).",
    )
    location = serializers.CharField(
        required=False,
        help_text="Канонический URL созданного ресурса.",
    )
    x_ratelimit_remaining = serializers.IntegerField(
        min_value=0,
        required=False,
        help_text="Остаток квоты запросов в текущем окне.",
    )


# =============================================================================
# 4. Миксин для контроллеров DRF и Middleware трассировки
# =============================================================================

class StandardHeadersMixin:
    """
    Миксин для APIView / GenericAPIView, предоставляющий удобные методы
    работы со всеми заголовками из common/headers.yaml.
    """

    def get_request_id(self, request: Request) -> str:
        return get_or_create_request_id(request)

    def get_resource_etag(self, instance: Any) -> str:
        return compute_resource_etag(instance)

    def build_location_url(self, request: Request, view_name: str, **kwargs: Any) -> str:
        return build_canonical_location(request, view_name, kwargs=kwargs)

    def attach_standard_headers(
        self,
        response: Response,
        request: Optional[Request] = None,
        *,
        etag: Optional[str] = None,
        location: Optional[str] = None,
        ratelimit_remaining: Optional[int] = None,
    ) -> Response:
        req_id = self.get_request_id(request) if request is not None else None
        bundle = StandardResponseHeaders(
            request_id=req_id,
            etag=etag,
            location=location,
            ratelimit_remaining=ratelimit_remaining,
        )
        return bundle.apply(response)


class TraceHeadersMiddleware:
    """
    Middleware для сквозной трассировки X-Request-Id:
    1. Считывает или генерирует валидный UUID при входе запроса.
    2. Записывает его в request.request_id.
    3. Гарантирует присутствие X-Request-Id в каждом HTTP-ответе.
    """

    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request: HttpRequest) -> HttpResponse:
        request_id = get_or_create_request_id(request)
        response = self.get_response(request)
        if HEADER_X_REQUEST_ID not in response:
            response[HEADER_X_REQUEST_ID] = request_id
        return response
