from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

from django.core.exceptions import PermissionDenied
from django.http import Http404
from rest_framework import serializers, status
from rest_framework.exceptions import (
    APIException,
    AuthenticationFailed,
    MethodNotAllowed,
    NotAuthenticated,
    NotFound,
    ParseError,
    Throttled,
    ValidationError as DRFValidationError,
)
from rest_framework.request import Request
from rest_framework.response import Response
from rest_framework.views import exception_handler as drf_default_exception_handler


PROBLEM_CONTENT_TYPE = "application/problem+json"
PROBLEM_TYPE_BASE_URL = "https://tspice.local/errors"


class ErrorCode(str, Enum):
    """
    Машинные коды ошибок строго по контракту common/errors.yaml#/ErrorCode.
    Используются GUI для выбора стратегии восстановления.
    """

    VALIDATION_FAILED = "VALIDATION_FAILED"
    FIELD_OUT_OF_RANGE = "FIELD_OUT_OF_RANGE"
    MISSING_GEOMETRY_FIELD = "MISSING_GEOMETRY_FIELD"
    DEVICE_INVARIANT_VIOLATED = "DEVICE_INVARIANT_VIOLATED"
    UNSUPPORTED_DEVICE_FAMILY = "UNSUPPORTED_DEVICE_FAMILY"
    UNSUPPORTED_MEDIA_TYPE = "UNSUPPORTED_MEDIA_TYPE"
    UNSUPPORTED_UNIT = "UNSUPPORTED_UNIT"
    FILE_TOO_LARGE = "FILE_TOO_LARGE"
    INVALID_STL = "INVALID_STL"
    EMPTY_MEASUREMENT = "EMPTY_MEASUREMENT"
    INVALID_MEASUREMENT_COLUMNS = "INVALID_MEASUREMENT_COLUMNS"
    INSUFFICIENT_SWEEPS = "INSUFFICIENT_SWEEPS"
    UNAUTHENTICATED = "UNAUTHENTICATED"
    FORBIDDEN = "FORBIDDEN"
    RATE_LIMITED = "RATE_LIMITED"
    NOT_FOUND = "NOT_FOUND"
    CONFLICT = "CONFLICT"
    VERSION_CONFLICT = "VERSION_CONFLICT"
    INVALID_STATE = "INVALID_STATE"
    DUPLICATE_RESOURCE = "DUPLICATE_RESOURCE"
    IDEMPOTENCY_KEY_REUSED = "IDEMPOTENCY_KEY_REUSED"
    VRAM_OUT_OF_MEMORY = "VRAM_OUT_OF_MEMORY"
    NOT_CONVERGED = "NOT_CONVERGED"
    SINGULAR_MATRIX = "SINGULAR_MATRIX"
    MESH_BUDGET_EXCEEDED = "MESH_BUDGET_EXCEEDED"
    SOLVER_UNAVAILABLE = "SOLVER_UNAVAILABLE"
    SOLVER_TIMEOUT = "SOLVER_TIMEOUT"
    SURROGATE_MISMATCH = "SURROGATE_MISMATCH"
    EDA_TOOL_UNAVAILABLE = "EDA_TOOL_UNAVAILABLE"
    NETLIST_FAILED = "NETLIST_FAILED"
    COMPILE_FAILED = "COMPILE_FAILED"
    VERIFICATION_REJECTED = "VERIFICATION_REJECTED"
    REPORT_RENDER_FAILED = "REPORT_RENDER_FAILED"
    TELEMETRY_UNAVAILABLE = "TELEMETRY_UNAVAILABLE"
    JOB_CANCELLED = "JOB_CANCELLED"
    JOB_EXPIRED = "JOB_EXPIRED"
    INTERNAL_ERROR = "INTERNAL_ERROR"
    SERVICE_UNAVAILABLE = "SERVICE_UNAVAILABLE"


ERROR_DEFAULT_TITLES: dict[ErrorCode, str] = {
    ErrorCode.VALIDATION_FAILED: "Ошибка валидации запроса",
    ErrorCode.FIELD_OUT_OF_RANGE: "Значение поля вне допустимого диапазона",
    ErrorCode.MISSING_GEOMETRY_FIELD: "Отсутствует обязательное поле геометрии",
    ErrorCode.DEVICE_INVARIANT_VIOLATED: "Нарушен инвариант структуры",
    ErrorCode.UNSUPPORTED_DEVICE_FAMILY: "Неподдерживаемое семейство приборов",
    ErrorCode.UNSUPPORTED_MEDIA_TYPE: "Неподдерживаемый тип файла",
    ErrorCode.UNSUPPORTED_UNIT: "Неподдерживаемая единица измерения",
    ErrorCode.FILE_TOO_LARGE: "Превышен максимальный размер файла",
    ErrorCode.INVALID_STL: "Некорректная STL-геометрия",
    ErrorCode.EMPTY_MEASUREMENT: "Пустой набор измерений",
    ErrorCode.INVALID_MEASUREMENT_COLUMNS: "Некорректные колонки измерений",
    ErrorCode.INSUFFICIENT_SWEEPS: "Недостаточно свипов для экстракции",
    ErrorCode.UNAUTHENTICATED: "Требуется аутентификация",
    ErrorCode.FORBIDDEN: "Недостаточно прав на операцию",
    ErrorCode.RATE_LIMITED: "Превышен лимит частоты запросов",
    ErrorCode.NOT_FOUND: "Ресурс не найден",
    ErrorCode.CONFLICT: "Конфликт состояния ресурса",
    ErrorCode.VERSION_CONFLICT: "Конфликт версий ресурса",
    ErrorCode.INVALID_STATE: "Недопустимое состояние ресурса",
    ErrorCode.DUPLICATE_RESOURCE: "Ресурс уже существует",
    ErrorCode.IDEMPOTENCY_KEY_REUSED: "Ключ идемпотентности использован повторно с другими параметрами",
    ErrorCode.VRAM_OUT_OF_MEMORY: "Недостаточно видеопамяти",
    ErrorCode.NOT_CONVERGED: "Решатель не сошёлся",
    ErrorCode.SINGULAR_MATRIX: "Вырожденная матрица системы",
    ErrorCode.MESH_BUDGET_EXCEEDED: "Превышен бюджет узлов сетки",
    ErrorCode.SOLVER_UNAVAILABLE: "Вычислительный решатель недоступен",
    ErrorCode.SOLVER_TIMEOUT: "Превышено время ожидания расчёта",
    ErrorCode.SURROGATE_MISMATCH: "Несоответствие суррогатной модели",
    ErrorCode.EDA_TOOL_UNAVAILABLE: "EDA-инструмент недоступен",
    ErrorCode.NETLIST_FAILED: "Ошибка генерации SPICE-нетлиста",
    ErrorCode.COMPILE_FAILED: "Ошибка компиляции модели",
    ErrorCode.VERIFICATION_REJECTED: "Верификация не пройдена",
    ErrorCode.REPORT_RENDER_FAILED: "Ошибка сборки отчёта",
    ErrorCode.TELEMETRY_UNAVAILABLE: "Телеметрия недоступна",
    ErrorCode.JOB_CANCELLED: "Задача отменена",
    ErrorCode.JOB_EXPIRED: "Срок хранения задачи истёк",
    ErrorCode.INTERNAL_ERROR: "Внутренняя ошибка сервера",
    ErrorCode.SERVICE_UNAVAILABLE: "Сервис временно недоступен",
}


# =============================================================================
# 1. Сериализаторы RFC 7807 (для валидации и генерации OpenAPI через drf-spectacular)
# =============================================================================

class ProblemDetailItemSerializer(serializers.Serializer):
    """
    Схема common/errors.yaml#/ProblemDetail (и FieldError).
    """

    field = serializers.CharField(required=False, allow_blank=True)
    code = serializers.CharField(required=False, allow_blank=True)
    message = serializers.CharField(required=False, allow_blank=True)
    value = serializers.JSONField(required=False, allow_null=True)
    allowed = serializers.ListField(child=serializers.JSONField(), required=False)
    requirements = serializers.ListField(
        child=serializers.RegexField(r"^(FR|NFR)-[A-Z]+-[0-9]{2}$"),
        required=False,
    )


class ProblemSerializer(serializers.Serializer):
    """
    Схема common/errors.yaml#/Problem.
    """

    type = serializers.CharField(required=True)
    title = serializers.CharField(required=True)
    status = serializers.IntegerField(min_value=400, max_value=599, required=True)
    code = serializers.ChoiceField(choices=[c.value for c in ErrorCode], required=True)
    detail = serializers.CharField(required=False, allow_blank=True)
    instance = serializers.CharField(required=False, allow_blank=True)
    request_id = serializers.UUIDField(required=True)
    errors = ProblemDetailItemSerializer(many=True, required=False)
    required = serializers.ListField(child=serializers.CharField(), required=False)
    retry_after_s = serializers.IntegerField(min_value=0, required=False)
    fallback = serializers.DictField(required=False)
    job_id = serializers.IntegerField(required=False)
    occurrences = serializers.IntegerField(min_value=1, required=False)
    log_url = serializers.CharField(required=False, allow_blank=True)


class ValidationErrorProblemSerializer(ProblemSerializer):
    """
    Схема common/errors.yaml#/ValidationError (422 с обязательным списком errors).
    """

    errors = ProblemDetailItemSerializer(many=True, required=True)


# =============================================================================
# 2. Структура детализации и базовое исключение RFC 7807
# =============================================================================

@dataclass
class FieldViolation:
    """
    Отдельное нарушение поля или инварианта (ProblemDetail).
    """

    field: str
    detail: str
    code: Optional[str] = None
    value: Any = None
    allowed: Optional[list[Any]] = None
    requirements: Optional[list[str]] = None
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "field": self.field,
            "message": self.detail,
        }
        if self.code is not None:
            payload["code"] = self.code
        if self.value is not None:
            payload["value"] = self.value
        if self.allowed is not None:
            payload["allowed"] = self.allowed
        if self.requirements is not None:
            payload["requirements"] = self.requirements
        payload.update(self.extra)
        return payload


class ProblemException(APIException):
    """
    Базовое исключение приложения, сериализуемое в RFC 7807 (application/problem+json).
    Поддерживает все расширенные поля из common/errors.yaml#/Problem.
    """

    status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR
    default_code: ErrorCode = ErrorCode.INTERNAL_ERROR

    def __init__(
        self,
        code: ErrorCode | str | None = None,
        detail: Optional[str] = None,
        title: Optional[str] = None,
        status_code: Optional[int] = None,
        errors: Optional[list[FieldViolation | dict[str, Any]]] = None,
        required: Optional[list[str]] = None,
        retry_after_s: Optional[int] = None,
        fallback: Optional[dict[str, Any]] = None,
        job_id: Optional[int] = None,
        occurrences: Optional[int] = None,
        log_url: Optional[str] = None,
        headers: Optional[dict[str, str]] = None,
        extra: Optional[dict[str, Any]] = None,
    ) -> None:
        if status_code is not None:
            self.status_code = status_code

        resolved_code = ErrorCode(code) if isinstance(code, str) and code in ErrorCode._value2member_map_ else (
            code if isinstance(code, ErrorCode) else self.default_code
        )
        self.error_code: ErrorCode = resolved_code
        self.title: str = title or ERROR_DEFAULT_TITLES.get(resolved_code, "Ошибка выполнения запроса")
        self.detail_message: str = detail or self.title
        self.errors = [
            err.to_dict() if isinstance(err, FieldViolation) else err for err in (errors or [])
        ]
        self.required = required
        self.retry_after_s = retry_after_s
        self.fallback = fallback
        self.job_id = job_id
        self.occurrences = occurrences
        self.log_url = log_url
        self.extra_headers: dict[str, str] = dict(headers or {})
        self.extra_fields: dict[str, Any] = dict(extra or {})

        if retry_after_s is not None and "Retry-After" not in self.extra_headers:
            self.extra_headers["Retry-After"] = str(retry_after_s)

        super().__init__(detail=self.detail_message, code=resolved_code.value)

    def to_problem_dict(self, request: Optional[Request], request_id: str) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "type": f"{PROBLEM_TYPE_BASE_URL}/{self.error_code.value}",
            "title": self.title,
            "status": self.status_code,
            "code": self.error_code.value,
            "detail": self.detail_message,
            "request_id": request_id,
        }
        if request is not None:
            payload["instance"] = request.get_full_path()
        if self.errors:
            payload["errors"] = self.errors
        if self.required is not None:
            payload["required"] = self.required
        if self.retry_after_s is not None:
            payload["retry_after_s"] = self.retry_after_s
        if self.fallback is not None:
            payload["fallback"] = self.fallback
        if self.job_id is not None:
            payload["job_id"] = self.job_id
        if self.occurrences is not None:
            payload["occurrences"] = self.occurrences
        if self.log_url is not None:
            payload["log_url"] = self.log_url

        payload.update(self.extra_fields)
        return payload


# =============================================================================
# 3. Специализированные доменные исключения по статусам HTTP
# =============================================================================

class BadRequestProblem(ProblemException):
    status_code = status.HTTP_400_BAD_REQUEST
    default_code = ErrorCode.VALIDATION_FAILED


class UnauthorizedProblem(ProblemException):
    status_code = status.HTTP_401_UNAUTHORIZED
    default_code = ErrorCode.UNAUTHENTICATED

    def __init__(self, detail: Optional[str] = None, scheme: str = "Bearer", **kwargs: Any) -> None:
        headers = dict(kwargs.pop("headers", None) or {})
        headers.setdefault("WWW-Authenticate", scheme)
        super().__init__(code=ErrorCode.UNAUTHENTICATED, detail=detail, headers=headers, **kwargs)


class ForbiddenProblem(ProblemException):
    status_code = status.HTTP_403_FORBIDDEN
    default_code = ErrorCode.FORBIDDEN


class NotFoundProblem(ProblemException):
    status_code = status.HTTP_404_NOT_FOUND
    default_code = ErrorCode.NOT_FOUND


class ConflictProblem(ProblemException):
    status_code = status.HTTP_409_CONFLICT
    default_code = ErrorCode.CONFLICT

    def __init__(
        self,
        code: ErrorCode = ErrorCode.CONFLICT,
        detail: Optional[str] = None,
        etag: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        headers = dict(kwargs.pop("headers", None) or {})
        if etag:
            headers["ETag"] = etag
        super().__init__(code=code, detail=detail, headers=headers, **kwargs)


class UnprocessableEntityProblem(ProblemException):
    status_code = status.HTTP_422_UNPROCESSABLE_ENTITY
    default_code = ErrorCode.VALIDATION_FAILED

    def __init__(
        self,
        code: ErrorCode = ErrorCode.VALIDATION_FAILED,
        detail: Optional[str] = None,
        errors: Optional[list[FieldViolation | dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(code=code, detail=detail, errors=errors or [], **kwargs)


class PayloadTooLargeProblem(ProblemException):
    status_code = status.HTTP_413_REQUEST_ENTITY_TOO_LARGE
    default_code = ErrorCode.FILE_TOO_LARGE


class VramOutOfMemoryProblem(PayloadTooLargeProblem):
    """
    Частный случай 413 (NFR-REL-03) с информацией о резервном CPU-бэкенде.
    """

    def __init__(
        self,
        detail: str = "Требуемая память превышает доступную VRAM; резервный CPU-решатель перегружен.",
        fallback_backend: str = "cpu_scipy",
        **kwargs: Any,
    ) -> None:
        super().__init__(
            code=ErrorCode.VRAM_OUT_OF_MEMORY,
            detail=detail,
            fallback={"backend": fallback_backend},
            **kwargs,
        )


class TooManyRequestsProblem(ProblemException):
    status_code = status.HTTP_429_TOO_MANY_REQUESTS
    default_code = ErrorCode.RATE_LIMITED

    def __init__(self, detail: Optional[str] = None, retry_after_s: int = 5, **kwargs: Any) -> None:
        super().__init__(
            code=ErrorCode.RATE_LIMITED,
            detail=detail,
            retry_after_s=retry_after_s,
            **kwargs,
        )


class ServiceUnavailableProblem(ProblemException):
    status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    default_code = ErrorCode.SERVICE_UNAVAILABLE

    def __init__(
        self,
        code: ErrorCode = ErrorCode.SERVICE_UNAVAILABLE,
        detail: Optional[str] = None,
        retry_after_s: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(code=code, detail=detail, retry_after_s=retry_after_s, **kwargs)


# =============================================================================
# 4. Вспомогательный парсер ошибок DRF Serializer -> FieldViolation[]
# =============================================================================

def flatten_drf_errors(error_detail: Any, prefix: str = "") -> list[dict[str, Any]]:
    """
    Рекурсивно преобразует вложенный словарь ошибок DRF Serializer
    в плоский список ProblemDetail с dot-путями (например, 'geometry.fin_pitch_nm').
    """
    violations: list[dict[str, Any]] = []

    if isinstance(error_detail, dict):
        for field_name, sub_errors in error_detail.items():
            if field_name == "non_field_errors":
                next_prefix = prefix
            else:
                next_prefix = f"{prefix}.{field_name}" if prefix else str(field_name)
            violations.extend(flatten_drf_errors(sub_errors, next_prefix))
    elif isinstance(error_detail, list):
        for item in error_detail:
            if isinstance(item, (dict, list)):
                violations.extend(flatten_drf_errors(item, prefix))
            else:
                err_code = getattr(item, "code", None)
                violations.append(
                    {
                        "field": prefix or "__root__",
                        "message": str(item),
                        **({"code": str(err_code)} if err_code else {}),
                    }
                )
    else:
        violations.append(
            {
                "field": prefix or "__root__",
                "message": str(error_detail),
            }
        )

    return violations


def raise_validation_problem(
    serializer: serializers.Serializer,
    code: ErrorCode = ErrorCode.VALIDATION_FAILED,
    detail: str = "Переданы некорректные параметры запроса.",
) -> None:
    """
    Утилита для контроллеров: проверяет сериализатор и при ошибке
    выбрасывает UnprocessableEntityProblem (422) с полным списком нарушений.
    """
    if not serializer.is_valid():
        violations = flatten_drf_errors(serializer.errors)
        raise UnprocessableEntityProblem(code=code, detail=detail, errors=violations)


# =============================================================================
# 5. Глобальный обработчик исключений DRF (RFC 7807)
# =============================================================================

def _resolve_request_id(request: Optional[Request]) -> str:
    if request is not None:
        header_val = request.headers.get("X-Request-Id")
        if header_val:
            try:
                return str(uuid.UUID(header_val))
            except ValueError:
                pass
        meta_val = request.META.get("X_REQUEST_ID")
        if meta_val:
            return str(meta_val)
    return str(uuid.uuid4())


def rfc7807_exception_handler(exc: Exception, context: dict[str, Any]) -> Optional[Response]:
    """
    Единый обработчик исключений для settings.py -> REST_FRAMEWORK['EXCEPTION_HANDLER'].
    Гарантирует возврат всех ошибок в формате application/problem+json
    по схеме common/errors.yaml#/Problem.
    """
    request: Optional[Request] = context.get("request")
    request_id = _resolve_request_id(request)

    # 1. Доменные исключения ProblemException
    if isinstance(exc, ProblemException):
        payload = exc.to_problem_dict(request=request, request_id=request_id)
        headers = {**exc.extra_headers, "X-Request-Id": request_id}
        return Response(
            payload,
            status=exc.status_code,
            headers=headers,
            content_type=PROBLEM_CONTENT_TYPE,
        )

    # 2. Ошибки валидации DRF Serializer -> 422 Unprocessable Entity (ValidationError)
    if isinstance(exc, DRFValidationError):
        violations = flatten_drf_errors(exc.detail)
        problem = UnprocessableEntityProblem(
            code=ErrorCode.VALIDATION_FAILED,
            detail="Ошибка валидации входных данных.",
            errors=violations,
        )
        payload = problem.to_problem_dict(request=request, request_id=request_id)
        return Response(
            payload,
            status=status.HTTP_422_UNPROCESSABLE_ENTITY,
            headers={"X-Request-Id": request_id},
            content_type=PROBLEM_CONTENT_TYPE,
        )

    # 3. Стандартные исключения Django и DRF
    mapped_problem: Optional[ProblemException] = None

    if isinstance(exc, Http404) or isinstance(exc, NotFound):
        mapped_problem = NotFoundProblem(detail=str(exc) if str(exc) else "Запрашиваемый ресурс не найден.")
    elif isinstance(exc, (NotAuthenticated, AuthenticationFailed)):
        mapped_problem = UnauthorizedProblem(detail=str(exc))
    elif isinstance(exc, (PermissionDenied,)):
        mapped_problem = ForbiddenProblem(detail=str(exc) if str(exc) else "Доступ к ресурсу запрещён.")
    elif isinstance(exc, ParseError):
        mapped_problem = BadRequestProblem(
            code=ErrorCode.VALIDATION_FAILED,
            detail="Тело запроса не является корректным JSON или имеет неверный формат.",
        )
    elif isinstance(exc, Throttled):
        wait_seconds = int(exc.wait) if getattr(exc, "wait", None) is not None else 5
        mapped_problem = TooManyRequestsProblem(
            detail="Превышен лимит частоты запросов.",
            retry_after_s=wait_seconds,
        )
    elif isinstance(exc, MethodNotAllowed):
        mapped_problem = BadRequestProblem(
            code=ErrorCode.VALIDATION_FAILED,
            detail=f"Метод {exc.args[0]} не поддерживается для данного эндпоинта." if exc.args else str(exc),
            status_code=status.HTTP_405_METHOD_NOT_ALLOWED,
        )

    if mapped_problem is not None:
        payload = mapped_problem.to_problem_dict(request=request, request_id=request_id)
        headers = {**mapped_problem.extra_headers, "X-Request-Id": request_id}
        return Response(
            payload,
            status=mapped_problem.status_code,
            headers=headers,
            content_type=PROBLEM_CONTENT_TYPE,
        )

    # 4. Делегируем стандартному обработчику DRF для прочих APIException
    drf_response = drf_default_exception_handler(exc, context)
    if drf_response is not None:
        status_code = drf_response.status_code
        code = ErrorCode.INTERNAL_ERROR if status_code >= 500 else ErrorCode.VALIDATION_FAILED
        detail = str(getattr(exc, "detail", "Ошибка обработки запроса."))
        fallback_problem = ProblemException(code=code, detail=detail, status_code=status_code)
        payload = fallback_problem.to_problem_dict(request=request, request_id=request_id)
        return Response(
            payload,
            status=status_code,
            headers={"X-Request-Id": request_id},
            content_type=PROBLEM_CONTENT_TYPE,
        )

    return None
