"""
Единый модуль обработки ошибок, RFC 7807 (application/problem+json),
исключений и стандартных заголовков (X-Request-Id, ETag, Retry-After, WWW-Authenticate)
для всех контроллеров Django REST Framework.
"""

from __future__ import annotations

import uuid
from typing import Any, Optional

from django.core.exceptions import ObjectDoesNotExist, PermissionDenied as DjangoPermissionDenied
from django.http import Http404
from rest_framework import exceptions as drf_exceptions
from rest_framework import status
from rest_framework.request import Request
from rest_framework.response import Response
from rest_framework.views import exception_handler as drf_default_exception_handler


PROBLEM_CONTENT_TYPE = "application/problem+json"
ERROR_TYPE_BASE_URI = "https://tspice.local/errors"


# =============================================================================
# 1. Иерархия доменных исключений по контракту
# =============================================================================


class ProblemDetailException(Exception):
    """
    Базовое исключение, сериализуемое в формат RFC 7807 (application/problem+json).
    """

    status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR
    error_code: str = "INTERNAL_ERROR"
    default_title: str = "Внутренняя ошибка сервера"
    default_detail: str = "Произошла непредвиденная ошибка при обработке запроса."

    def __init__(
        self,
        detail: Optional[str] = None,
        *,
        title: Optional[str] = None,
        code: Optional[str] = None,
        status_code: Optional[int] = None,
        headers: Optional[dict[str, str]] = None,
        extra: Optional[dict[str, Any]] = None,
    ) -> None:
        self.detail = detail or self.default_detail
        self.title = title or self.default_title
        self.code = code or self.error_code
        if status_code is not None:
            self.status_code = status_code
        self.headers = headers or {}
        self.extra = extra or {}
        super().__init__(self.detail)


class BadRequestProblem(ProblemDetailException):
    status_code = status.HTTP_400_BAD_REQUEST
    error_code = "VALIDATION_FAILED"
    default_title = "Некорректный запрос"
    default_detail = "Тело запроса не является корректным JSON или имеет неверный синтаксис."


class UnauthorizedProblem(ProblemDetailException):
    status_code = status.HTTP_401_UNAUTHORIZED
    error_code = "UNAUTHORIZED"
    default_title = "Требуется аутентификация"
    default_detail = "Отсутствует или недействителен токен доступа."

    def __init__(
        self,
        detail: Optional[str] = None,
        *,
        www_authenticate: str = 'Bearer realm="tspice"',
        **kwargs: Any,
    ) -> None:
        headers = kwargs.pop("headers", {})
        headers.setdefault("WWW-Authenticate", www_authenticate)
        super().__init__(detail, headers=headers, **kwargs)


class ForbiddenProblem(ProblemDetailException):
    status_code = status.HTTP_403_FORBIDDEN
    error_code = "FORBIDDEN"
    default_title = "Доступ запрещён"
    default_detail = "Недостаточно прав на выполнение операции над ресурсом."


class NotFoundProblem(ProblemDetailException):
    status_code = status.HTTP_404_NOT_FOUND
    error_code = "NOT_FOUND"
    default_title = "Ресурс не найден"
    default_detail = "Запрашиваемый ресурс не существует или был удалён."


class ConflictProblem(ProblemDetailException):
    status_code = status.HTTP_409_CONFLICT
    error_code = "CONFLICT"
    default_title = "Конфликт состояния или версий"
    default_detail = "Текущее состояние или версия ресурса конфликтует с запросом."

    def __init__(
        self,
        detail: Optional[str] = None,
        *,
        etag: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        headers = kwargs.pop("headers", {})
        if etag:
            headers.setdefault("ETag", etag)
        super().__init__(detail, headers=headers, **kwargs)


class UnprocessableEntityProblem(ProblemDetailException):
    status_code = status.HTTP_422_UNPROCESSABLE_ENTITY
    error_code = "DEVICE_INVARIANT_VIOLATED"
    default_title = "Семантическая ошибка домена"
    default_detail = "Нарушены инварианты структуры, диапазоны или режимы работы."

    def __init__(
        self,
        detail: Optional[str] = None,
        *,
        errors: Optional[list[dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> None:
        extra = kwargs.pop("extra", {})
        extra["errors"] = errors or []
        super().__init__(detail, extra=extra, **kwargs)


class PayloadTooLargeProblem(ProblemDetailException):
    status_code = status.HTTP_413_REQUEST_ENTITY_TOO_LARGE
    error_code = "PAYLOAD_TOO_LARGE"
    default_title = "Превышен допустимый объём данных или памяти"
    default_detail = "Превышен лимит размера загружаемого файла или доступной памяти узла."


class VramOutOfMemoryProblem(PayloadTooLargeProblem):
    error_code = "VRAM_OUT_OF_MEMORY"
    default_title = "Недостаточно видеопамяти"
    default_detail = "CPU-решатель также перегружен; задача остановлена."

    def __init__(
        self,
        detail: Optional[str] = None,
        *,
        fallback_backend: str = "cpu_scipy",
        **kwargs: Any,
    ) -> None:
        extra = kwargs.pop("extra", {})
        extra["fallback"] = {"backend": fallback_backend}
        super().__init__(detail, extra=extra, **kwargs)


class TooManyRequestsProblem(ProblemDetailException):
    status_code = status.HTTP_429_TOO_MANY_REQUESTS
    error_code = "TOO_MANY_REQUESTS"
    default_title = "Превышен лимит частоты запросов"
    default_detail = "Слишком много запросов. Повторите попытку позже."

    def __init__(
        self,
        detail: Optional[str] = None,
        *,
        retry_after: int = 60,
        **kwargs: Any,
    ) -> None:
        headers = kwargs.pop("headers", {})
        headers.setdefault("Retry-After", str(retry_after))
        super().__init__(detail, headers=headers, **kwargs)


class InternalServerErrorProblem(ProblemDetailException):
    status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
    error_code = "INTERNAL_ERROR"
    default_title = "Внутренняя ошибка сервера"
    default_detail = "Внутренняя ошибка при обработке запроса."


class ServiceUnavailableProblem(ProblemDetailException):
    status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    error_code = "SERVICE_UNAVAILABLE"
    default_title = "Сервис временно недоступен"
    default_detail = "Сервис или внешняя зависимость (EDA, GPU-нода) временно недоступны."

    def __init__(
        self,
        detail: Optional[str] = None,
        *,
        retry_after: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        headers = kwargs.pop("headers", {})
        if retry_after is not None:
            headers.setdefault("Retry-After", str(retry_after))
        super().__init__(detail, headers=headers, **kwargs)


# =============================================================================
# 2. Вспомогательные функции построения RFC 7807 Problem Details Response
# =============================================================================


def resolve_request_id(request: Optional[Any]) -> str:
    """
    Извлекает X-Request-Id из входящего запроса или генерирует новый UUID4.
    """
    if request is not None:
        headers = getattr(request, "headers", None)
        if headers and headers.get("X-Request-Id"):
            return str(headers["X-Request-Id"])
        meta = getattr(request, "META", None)
        if meta and meta.get("HTTP_X_REQUEST_ID"):
            return str(meta["HTTP_X_REQUEST_ID"])
    return str(uuid.uuid4())


def build_problem_response(
    *,
    status_code: int,
    code: str,
    title: str,
    detail: str,
    request_id: Optional[str] = None,
    include_request_id_header: bool = True,
    headers: Optional[dict[str, str]] = None,
    extra: Optional[dict[str, Any]] = None,
) -> Response:
    """
    Собирает Response строго в формате application/problem+json по спецификации.
    """
    req_id = request_id or str(uuid.uuid4())
    payload: dict[str, Any] = {
        "type": f"{ERROR_TYPE_BASE_URI}/{code}",
        "title": title,
        "status": status_code,
        "code": code,
        "detail": detail,
        "request_id": req_id,
    }
    if extra:
        payload.update(extra)

    resp_headers: dict[str, str] = {}
    if include_request_id_header:
        resp_headers["X-Request-Id"] = req_id
    if headers:
        resp_headers.update(headers)

    return Response(
        data=payload,
        status=status_code,
        headers=resp_headers,
        content_type=PROBLEM_CONTENT_TYPE,
    )


def _flatten_drf_validation_errors(detail: Any, prefix: str = "") -> list[dict[str, Any]]:
    """
    Преобразует вложенную структуру ошибок сериализатора DRF в список объектов:
    [{"field": "geometry.fin_pitch_nm", "message": "..."}]
    """
    items: list[dict[str, Any]] = []
    if isinstance(detail, dict):
        for key, value in detail.items():
            next_prefix = f"{prefix}.{key}" if prefix and key != "non_field_errors" else (prefix or str(key))
            items.extend(_flatten_drf_validation_errors(value, next_prefix))
    elif isinstance(detail, list):
        for entry in detail:
            if isinstance(entry, (dict, list)):
                items.extend(_flatten_drf_validation_errors(entry, prefix))
            else:
                items.append(
                    {
                        "field": prefix or "body",
                        "message": str(entry),
                    }
                )
    else:
        items.append(
            {
                "field": prefix or "body",
                "message": str(detail),
            }
        )
    return items


# =============================================================================
# 3. Глобальный обработчик исключений DRF (EXCEPTION_HANDLER)
# =============================================================================


def problem_details_exception_handler(exc: Exception, context: dict[str, Any]) -> Optional[Response]:
    """
    Подключается в settings.py:
        REST_FRAMEWORK = {
            "EXCEPTION_HANDLER": "path.to.common_responses.problem_details_exception_handler",
        }

    Перехватывает все доменные исключения ProblemDetailException, стандартные
    исключения DRF и Django (Http404, PermissionDenied, ParseError, Throttled и т.д.)
    и возвращает единый контракт application/problem+json.
    """
    request: Optional[Request] = context.get("request")
    request_id = resolve_request_id(request)

    # 1. Доменные исключения RFC 7807
    if isinstance(exc, ProblemDetailException):
        # По контракту для 503 заголовок X-Request-Id не обязателен, для остальных — обязателен
        include_req_header = exc.status_code != status.HTTP_503_SERVICE_UNAVAILABLE
        return build_problem_response(
            status_code=exc.status_code,
            code=exc.code,
            title=exc.title,
            detail=exc.detail,
            request_id=request_id,
            include_request_id_header=include_req_header,
            headers=exc.headers,
            extra=exc.extra,
        )

    # 2. Стандартные исключения Django 404 / 403
    if isinstance(exc, (Http404, ObjectDoesNotExist)):
        return build_problem_response(
            status_code=status.HTTP_404_NOT_FOUND,
            code="NOT_FOUND",
            title="Ресурс не найден",
            detail=str(exc) if str(exc) else "Запрашиваемый ресурс не существует или был удалён.",
            request_id=request_id,
        )

    if isinstance(exc, DjangoPermissionDenied):
        return build_problem_response(
            status_code=status.HTTP_403_FORBIDDEN,
            code="FORBIDDEN",
            title="Недостаточно прав",
            detail=str(exc) if str(exc) else "Недостаточно прав на операцию над ресурсом.",
            request_id=request_id,
        )

    # 3. Исключения Django REST Framework
    if isinstance(exc, drf_exceptions.ParseError):
        return build_problem_response(
            status_code=status.HTTP_400_BAD_REQUEST,
            code="VALIDATION_FAILED",
            title="Некорректный запрос",
            detail="Тело запроса не является корректным JSON или имеет неподдерживаемый формат.",
            request_id=request_id,
        )

    if isinstance(exc, (drf_exceptions.NotAuthenticated, drf_exceptions.AuthenticationFailed)):
        drf_resp = drf_default_exception_handler(exc, context)
        www_auth = (
            drf_resp.get("WWW-Authenticate")
            if drf_resp is not None and drf_resp.has_header("WWW-Authenticate")
            else 'Bearer realm="tspice"'
        )
        return build_problem_response(
            status_code=status.HTTP_401_UNAUTHORIZED,
            code="UNAUTHORIZED",
            title="Ошибка аутентификации",
            detail=str(exc.detail),
            request_id=request_id,
            headers={"WWW-Authenticate": www_auth},
        )

    if isinstance(exc, drf_exceptions.PermissionDenied):
        return build_problem_response(
            status_code=status.HTTP_403_FORBIDDEN,
            code="FORBIDDEN",
            title="Недостаточно прав",
            detail=str(exc.detail),
            request_id=request_id,
        )

    if isinstance(exc, drf_exceptions.NotFound):
        return build_problem_response(
            status_code=status.HTTP_404_NOT_FOUND,
            code="NOT_FOUND",
            title="Ресурс не найден",
            detail=str(exc.detail),
            request_id=request_id,
        )

    if isinstance(exc, drf_exceptions.ValidationError):
        flat_errors = _flatten_drf_validation_errors(exc.detail)
        return build_problem_response(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            code="VALIDATION_ERROR",
            title="Ошибка валидации входных данных",
            detail="Переданные параметры не прошли проверку схемы или доменных ограничений.",
            request_id=request_id,
            extra={"errors": flat_errors},
        )

    if isinstance(exc, drf_exceptions.Throttled):
        retry_after = int(exc.wait) if exc.wait is not None else 60
        return build_problem_response(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            code="TOO_MANY_REQUESTS",
            title="Превышен лимит частоты запросов",
            detail=f"Превышен лимит запросов. Повторите через {retry_after} с.",
            request_id=request_id,
            headers={"Retry-After": str(retry_after)},
        )

    if isinstance(exc, drf_exceptions.APIException):
        return build_problem_response(
            status_code=exc.status_code,
            code=getattr(exc, "default_code", "API_ERROR").upper(),
            title="Ошибка обработки запроса",
            detail=str(exc.detail),
            request_id=request_id,
        )

    # 4. Необработанные исключения -> 500 InternalError по контракту
    return build_problem_response(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        code="INTERNAL_ERROR",
        title="Внутренняя ошибка сервера",
        detail="Произошла непредвиденная внутренняя ошибка сервера.",
        request_id=request_id,
    )


# =============================================================================
# 4. Middleware для сквозного X-Request-Id во всех ответах (включая 2xx)
# =============================================================================


class RequestIdMiddleware:
    """
    Добавляет X-Request-Id в request и гарантирует его наличие в каждом HTTP-ответе.
    Подключается в settings.py -> MIDDLEWARE.
    """

    def __init__(self, get_response: Any) -> None:
        self.get_response = get_response

    def __call__(self, request: Any) -> Any:
        request_id = resolve_request_id(request)
        request.request_id = request_id
        response = self.get_response(request)
        if "X-Request-Id" not in response:
            response["X-Request-Id"] = request_id
        return response
