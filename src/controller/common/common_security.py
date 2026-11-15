"""
Модуль схем безопасности и ролевой модели (RBAC) для Django + DRF.
Соответствует спецификации OpenAPI `securitySchemes` (`bearerAuth` и `apiKeyAuth`).
"""
from __future__ import annotations

import hashlib
import hmac
import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional, Sequence

from django.conf import settings
from django.utils import timezone
from rest_framework.authentication import BaseAuthentication
from rest_framework.exceptions import AuthenticationFailed
from rest_framework.permissions import BasePermission, SAFE_METHODS
from rest_framework.request import Request

logger = logging.getLogger(__name__)


# =============================================================================
# 1. Роли доступа (RBAC)
# =============================================================================

class UserRole(str, Enum):
    """
    Роли доступа согласно контракту безопасности:
    - VIEWER: чтение справочников и результатов.
    - ENGINEER: запуск расчётов, экстракции и работа с моделями.
    - ADMIN: конфигурация EDA-интеграций, управление ключами и обучение AI.
    """
    VIEWER = "viewer"
    ENGINEER = "engineer"
    ADMIN = "admin"


ROLE_HIERARCHY: dict[UserRole, int] = {
    UserRole.VIEWER: 10,
    UserRole.ENGINEER: 20,
    UserRole.ADMIN: 30,
}


def resolve_user_role(user: Any) -> Optional[UserRole]:
    """
    Извлекает роль из объекта пользователя или сервисного принципала.
    """
    if not user or not getattr(user, "is_authenticated", False):
        return None

    if getattr(user, "is_superuser", False) or getattr(user, "is_staff", False):
        return UserRole.ADMIN

    raw_role = getattr(user, "role", None)
    if isinstance(raw_role, UserRole):
        return raw_role
    if isinstance(raw_role, str):
        try:
            return UserRole(raw_role.lower())
        except ValueError:
            logger.warning("Неизвестная роль пользователя: %s", raw_role)
    return UserRole.VIEWER


def has_minimum_role(user: Any, required_role: UserRole) -> bool:
    """
    Проверяет, что уровень прав пользователя не ниже требуемого по иерархии.
    """
    current_role = resolve_user_role(user)
    if current_role is None:
        return False
    return ROLE_HIERARCHY[current_role] >= ROLE_HIERARCHY[required_role]


# =============================================================================
# 2. Сервисный принципал для аутентификации по X-Api-Key
# =============================================================================

@dataclass
class ServicePrincipal:
    """
    Представление внешнего сервиса (Jenkins, CI/CD, ночные скрипты),
    аутентифицированного по заголовку `X-Api-Key`.
    По спецификации обладает правами `engineer` в пределах разрешённых проектов.
    """
    key_id: str
    name: str
    role: UserRole = UserRole.ENGINEER
    allowed_project_ids: Optional[list[int]] = None
    is_active: bool = True

    @property
    def is_authenticated(self) -> bool:
        return self.is_active

    @property
    def is_anonymous(self) -> bool:
        return False

    @property
    def is_staff(self) -> bool:
        return self.role == UserRole.ADMIN

    @property
    def is_superuser(self) -> bool:
        return False

    @property
    def id(self) -> str:
        return f"apikey:{self.key_id}"

    @property
    def pk(self) -> str:
        return self.id

    def can_access_project(self, project_id: Optional[int]) -> bool:
        if project_id is None or self.allowed_project_ids is None:
            return True
        return int(project_id) in self.allowed_project_ids


# =============================================================================
# 3. Классы аутентификации DRF (`bearerAuth` и `apiKeyAuth`)
# =============================================================================

class BearerJwtAuthentication(BaseAuthentication):
    """
    Схема `bearerAuth` (HTTP Bearer JWT).
    Делегирует проверку токена настроенному JWT-бэкенду (например, SimpleJWT или PyJWT)
    и обогащает пользователя ролью из claims (`role`).
    """
    keyword = "Bearer"

    def authenticate(self, request: Request) -> Optional[tuple[Any, dict[str, Any]]]:
        auth_header = request.headers.get("Authorization", "")
        if not auth_header:
            return None

        parts = auth_header.split()
        if len(parts) != 2 or parts[0].lower() != self.keyword.lower():
            return None

        raw_token = parts[1]
        user, payload = self._decode_and_resolve_user(raw_token)
        return user, payload

    def authenticate_header(self, request: Request) -> str:
        return f'{self.keyword} realm="api", charset="UTF-8"'

    def _decode_and_resolve_user(self, raw_token: str) -> tuple[Any, dict[str, Any]]:
        """
        Декодирует JWT и возвращает кортеж (user, token_payload).
        Поддерживает интеграцию с `rest_framework_simplejwt` или `PyJWT`.
        """
        try:
            from rest_framework_simplejwt.authentication import JWTAuthentication
            jwt_auth = JWTAuthentication()
            validated_token = jwt_auth.get_validated_token(raw_token)
            user = jwt_auth.get_user(validated_token)
            payload = dict(validated_token.payload)

            if "role" in payload and not getattr(user, "role", None):
                setattr(user, "role", payload["role"])
            return user, payload
        except ImportError:
            pass
        except Exception as exc:
            raise AuthenticationFailed("Недействительный или просроченный JWT-токен.") from exc

        try:
            import jwt
            secret = getattr(settings, "JWT_SIGNING_KEY", settings.SECRET_KEY)
            algorithms = getattr(settings, "JWT_ALGORITHMS", ["HS256"])
            payload = jwt.decode(raw_token, secret, algorithms=algorithms)
        except ImportError as exc:
            raise AuthenticationFailed("JWT-провайдер не настроен на сервере.") from exc
        except jwt.ExpiredSignatureError as exc:
            raise AuthenticationFailed("Срок действия токена истёк.") from exc
        except jwt.InvalidTokenError as exc:
            raise AuthenticationFailed("Некорректный JWT-токен.") from exc

        from django.contrib.auth import get_user_model
        UserModel = get_user_model()
        user_id = payload.get("sub") or payload.get("user_id")
        if not user_id:
            raise AuthenticationFailed("В токене отсутствует идентификатор субъекта (sub).")

        try:
            user = UserModel.objects.get(pk=user_id, is_active=True)
        except UserModel.DoesNotExist as exc:
            raise AuthenticationFailed("Пользователь не найден или деактивирован.") from exc

        if "role" in payload and not getattr(user, "role", None):
            setattr(user, "role", payload["role"])

        return user, payload


class ApiKeyAuthentication(BaseAuthentication):
    """
    Схема `apiKeyAuth` (заголовок `X-Api-Key`).
    Используется для автоматизации (Jenkins, ночные прогоны, внешние скрипты).
    """
    header_name = "X-Api-Key"

    def authenticate(self, request: Request) -> Optional[tuple[ServicePrincipal, dict[str, Any]]]:
        raw_key = request.headers.get(self.header_name)
        if not raw_key:
            return None

        principal = self._verify_api_key(raw_key.strip())
        if principal is None or not principal.is_active:
            raise AuthenticationFailed("Недействительный или отозванный X-Api-Key.")

        return principal, {"auth_scheme": "apiKeyAuth", "key_id": principal.key_id}

    def authenticate_header(self, request: Request) -> str:
        return 'ApiKey header="X-Api-Key"'

    def _verify_api_key(self, raw_key: str) -> Optional[ServicePrincipal]:
        """
        Проверяет ключ по БД (если модель ServiceApiKey определена) или по словарю
        `settings.SERVICE_API_KEYS`.
        """
        key_hash = hashlib.sha256(raw_key.encode("utf-8")).hexdigest()

        # 1. Поиск в БД, если в проекте зарегистрирована модель ServiceApiKey
        principal = self._lookup_in_database(key_hash)
        if principal is not None:
            return principal

        # 2. Резервный поиск в settings.SERVICE_API_KEYS:
        # Формат: {"<sha256_hash>": {"key_id": "jenkins-ci", "name": "Jenkins", "allowed_project_ids": [1, 2]}}
        configured_keys: dict[str, dict[str, Any]] = getattr(settings, "SERVICE_API_KEYS", {})
        for stored_hash, meta in configured_keys.items():
            if hmac.compare_digest(key_hash, stored_hash):
                return ServicePrincipal(
                    key_id=meta.get("key_id", stored_hash[:12]),
                    name=meta.get("name", "Automation Service"),
                    role=UserRole.ENGINEER,
                    allowed_project_ids=meta.get("allowed_project_ids"),
                    is_active=meta.get("is_active", True),
                )

        return None

    def _lookup_in_database(self, key_hash: str) -> Optional[ServicePrincipal]:
        try:
            from django.apps import apps
            ApiKeyModel = apps.get_model("core", "ServiceApiKey")
        except LookupError:
            return None

        record = ApiKeyModel.objects.filter(key_hash=key_hash, is_active=True).first()
        if record is None:
            return None

        expires_at = getattr(record, "expires_at", None)
        if expires_at and expires_at <= timezone.now():
            return None

        return ServicePrincipal(
            key_id=str(record.pk),
            name=getattr(record, "name", f"key-{record.pk}"),
            role=UserRole.ENGINEER,
            allowed_project_ids=getattr(record, "allowed_project_ids", None),
            is_active=True,
        )


# =============================================================================
# 4. Классы прав доступа DRF (Permissions)
# =============================================================================

class HasRolePermission(BasePermission):
    """
    Базовый класс проверки минимальной роли и границ проекта (для X-Api-Key).
    """
    required_role: UserRole = UserRole.VIEWER
    message = "Недостаточно прав для выполнения данной операции."

    def has_permission(self, request: Request, view: Any) -> bool:
        user = request.user
        if not user or not getattr(user, "is_authenticated", False):
            return False

        if not has_minimum_role(user, self.required_role):
            return False

        # Для сервисных ключей проверяем ограничение по allowed_project_ids, если project_id передан
        if isinstance(user, ServicePrincipal):
            project_id = (
                request.query_params.get("project_id")
                or (request.data.get("project_id") if isinstance(request.data, dict) else None)
            )
            if project_id is not None:
                try:
                    if not user.can_access_project(int(project_id)):
                        self.message = "Сервисный ключ не имеет доступа к указанному проекту."
                        return False
                except (TypeError, ValueError):
                    return False

        return True


class IsViewerOrAbove(HasRolePermission):
    """Доступ для ролей: viewer, engineer, admin."""
    required_role = UserRole.VIEWER


class IsEngineerOrAbove(HasRolePermission):
    """Доступ для ролей: engineer, admin (и сервисных ключей X-Api-Key)."""
    required_role = UserRole.ENGINEER
    message = "Операция доступна только пользователям с ролью engineer или admin."


class IsAdminRole(HasRolePermission):
    """Доступ исключительно для роли admin (конфигурация интеграций, обучение AI)."""
    required_role = UserRole.ADMIN
    message = "Операция доступна исключительно администратору (роль admin)."


class ReadOnlyOrEngineerPermission(HasRolePermission):
    """
    GET/HEAD/OPTIONS разрешены с ролью viewer и выше;
    изменяющие запросы (POST/PUT/PATCH/DELETE) требуют роль engineer и выше.
    """
    def has_permission(self, request: Request, view: Any) -> bool:
        if request.method in SAFE_METHODS:
            self.required_role = UserRole.VIEWER
        else:
            self.required_role = UserRole.ENGINEER
        return super().has_permission(request, view)


# =============================================================================
# 5. Настройка для OpenAPI / drf-spectacular
# =============================================================================

SPECTACULAR_SECURITY_SCHEMES = {
    "bearerAuth": {
        "type": "http",
        "scheme": "bearer",
        "bearerFormat": "JWT",
    },
    "apiKeyAuth": {
        "type": "apiKey",
        "in": "header",
        "name": "X-Api-Key",
    },
}
