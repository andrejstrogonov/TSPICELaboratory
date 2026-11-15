import hashlib
import uuid
from typing import Any, Optional

from django.core.cache import cache
from django.db.models import Q, QuerySet
from django.shortcuts import get_object_or_404
from django.urls import reverse
from rest_framework import serializers, status
from rest_framework.parsers import FormParser, MultiPartParser
from rest_framework.permissions import AllowAny, IsAuthenticated
from rest_framework.request import Request
from rest_framework.response import Response
from rest_framework.views import APIView

# Предполагаемые модели, сериализаторы и сервисы вашего проекта
from .models import DeviceProfile, DeviceStructure
from .serializers import (
    CadImportRequestSerializer,
    DeviceCapabilitiesSerializer,
    DeviceFamilyLabelSerializer,
    DeviceStructureCreateSerializer,
    DeviceStructureListItemSerializer,
    DeviceStructureSerializer,
    DeviceStructureUpdateSerializer,
    GeometryFieldDefinitionSerializer,
    GeometryRangeSerializer,
    MaterialSerializer,
    ParameterSetExportSerializer,
    ProfileSerializer,
    StlGeometrySerializer,
    ValidationReportSerializer,
)
from .services import (
    CadGeometryValidationError,
    CadImportService,
    DeviceCatalogService,
    StructureConflictError,
    StructureService,
    StructureValidationService,
    UnsupportedCadFormatError,
)

MAX_CAD_UPLOAD_BYTES = 100 * 1024 * 1024  # 100 МБ для STL-моделей


class IdempotencyAndETagMixin:
    """
    Миксин для работы с заголовками Idempotency-Key, X-Request-Id, ETag и If-Match.
    """

    IDEMPOTENCY_TTL = 60 * 60 * 24

    def _get_request_id(self, request: Request) -> str:
        return request.headers.get("X-Request-Id", str(uuid.uuid4()))

    def _compute_etag(self, obj: Any) -> str:
        """
        Формирует ETag на основе идентификатора и версии/времени обновления.
        """
        version_token = getattr(obj, "version", None) or getattr(obj, "updated_at", obj.pk)
        raw = f"{obj.__class__.__name__}:{obj.pk}:{version_token}".encode("utf-8")
        return f'"{hashlib.sha256(raw).hexdigest()[:16]}"'

    def _check_if_match(self, request: Request, obj: Any) -> Optional[Response]:
        """
        Проверяет заголовок If-Match при изменении или удалении ресурса.
        """
        if_match = request.headers.get("If-Match")
        if not if_match:
            return Response(
                {"detail": "Заголовок If-Match обязателен для этой операции."},
                status=status.HTTP_409_CONFLICT,
            )
        current_etag = self._compute_etag(obj)
        if if_match.strip('"') != current_etag.strip('"'):
            return Response(
                {"detail": "Конфликт версий (ETag не совпадает с актуальным состоянием ресурса)."},
                status=status.HTTP_409_CONFLICT,
            )
        return None

    def _get_cached_response(self, request: Request, endpoint_prefix: str) -> Optional[dict[str, Any]]:
        idem_key = request.headers.get("Idempotency-Key")
        if not idem_key:
            return None
        cache_key = f"idem:{endpoint_prefix}:{request.user.id}:{idem_key}"
        return cache.get(cache_key)

    def _save_cached_response(
        self,
        request: Request,
        endpoint_prefix: str,
        payload: Any,
        status_code: int,
        headers: Optional[dict[str, str]] = None,
    ) -> None:
        idem_key = request.headers.get("Idempotency-Key")
        if not idem_key:
            return
        cache_key = f"idem:{endpoint_prefix}:{request.user.id}:{idem_key}"
        cache.set(
            cache_key,
            {"data": payload, "status": status_code, "headers": headers or {}},
            self.IDEMPOTENCY_TTL,
        )


class BaseDeviceView(IdempotencyAndETagMixin, APIView):
    """
    Базовый контроллер для домена Devices & Catalog с общей валидацией,
    пагинацией и сборкой стандартных заголовков.
    """

    permission_classes = [IsAuthenticated]

    def _validate_request(
        self,
        serializer_cls: type[serializers.Serializer],
        data: Any,
        request_id: Optional[str] = None,
        instance: Any = None,
        partial: bool = False,
    ) -> tuple[Optional[serializers.Serializer], Optional[Response]]:
        serializer = serializer_cls(instance=instance, data=data, partial=partial)
        if not serializer.is_valid():
            headers = {"X-Request-Id": request_id} if request_id else None
            response = Response(
                {
                    "detail": "Ошибка валидации входных данных.",
                    "errors": serializer.errors,
                },
                status=status.HTTP_422_UNPROCESSABLE_ENTITY,
                headers=headers,
            )
            return None, response
        return serializer, None

    def _paginate_queryset(self, request: Request, queryset: QuerySet) -> tuple[list[Any], dict[str, int]]:
        try:
            page = max(1, int(request.query_params.get("page", 1)))
            page_size = min(100, max(1, int(request.query_params.get("page_size", 20))))
        except ValueError:
            page, page_size = 1, 20

        offset = (page - 1) * page_size
        total = queryset.count()
        items = list(queryset[offset : offset + page_size])

        meta = {
            "total": total,
            "page": page,
            "page_size": page_size,
        }
        return items, meta

    def _build_created_response(
        self,
        request: Request,
        structure: DeviceStructure,
        request_id: str,
    ) -> tuple[Response, dict[str, str]]:
        res_serializer = DeviceStructureSerializer(structure)
        location_url = request.build_absolute_uri(
            reverse("structure-detail", kwargs={"structure_id": structure.id})
        )
        etag = self._compute_etag(structure)

        cache_headers = {"Location": location_url, "ETag": etag}
        response_headers = {**cache_headers, "X-Request-Id": request_id}

        response = Response(
            res_serializer.data,
            status=status.HTTP_201_CREATED,
            headers=response_headers,
        )
        return response, cache_headers


# =============================================================================
# 1. Справочники (Catalog)
# =============================================================================


class DeviceFamilyListView(APIView):
    """
    GET /catalog/device-families — Справочник семейств приборов (публичный, security: [])
    """

    permission_classes = [AllowAny]

    def get(self, request: Request) -> Response:
        families = DeviceCatalogService.list_device_families()
        serializer = DeviceFamilyLabelSerializer(families, many=True)
        return Response(serializer.data, status=status.HTTP_200_OK)


class GeometryFieldListView(BaseDeviceView):
    """
    GET /catalog/geometry-fields — Поля геометрии семейства с диапазонами для GUI
    """

    def get(self, request: Request) -> Response:
        device_family = request.query_params.get("device_family")
        if not device_family:
            return Response(
                {"detail": "Обязательный query-параметр 'device_family' не передан."},
                status=status.HTTP_400_BAD_REQUEST,
            )
        if not DeviceCatalogService.is_valid_family(device_family):
            return Response(
                {"detail": f"Неизвестное семейство прибора '{device_family}'."},
                status=status.HTTP_400_BAD_REQUEST,
            )

        fields = DeviceCatalogService.list_geometry_fields(device_family)
        serializer = GeometryFieldDefinitionSerializer(fields, many=True)
        return Response(serializer.data, status=status.HTTP_200_OK)


class GeometryRangeListView(BaseDeviceView):
    """
    GET /catalog/geometry-ranges — Диапазоны значений полей геометрии
    """

    def get(self, request: Request) -> Response:
        device_family = request.query_params.get("device_family")
        if not device_family or not DeviceCatalogService.is_valid_family(device_family):
            return Response(
                {"detail": "Некорректный или отсутствующий параметр 'device_family'."},
                status=status.HTTP_400_BAD_REQUEST,
            )

        ranges = DeviceCatalogService.list_geometry_ranges(device_family)
        serializer = GeometryRangeSerializer(ranges, many=True)
        return Response(serializer.data, status=status.HTTP_200_OK)


class MaterialListView(APIView):
    """
    GET /catalog/materials — Справочник материалов слоёв (публичный, security: [])
    """

    permission_classes = [AllowAny]

    def get(self, request: Request) -> Response:
        materials = DeviceCatalogService.list_materials()
        serializer = MaterialSerializer(materials, many=True)
        return Response(serializer.data, status=status.HTTP_200_OK)


class DeviceCapabilitiesView(BaseDeviceView):
    """
    GET /catalog/capabilities — Возможности семейства (методы, экспорт, симуляторы)
    """

    def get(self, request: Request) -> Response:
        device_family = request.query_params.get("device_family")
        if not device_family or not DeviceCatalogService.is_valid_family(device_family):
            return Response(
                {"detail": "Параметр 'device_family' обязателен и должен содержать корректное семейство."},
                status=status.HTTP_400_BAD_REQUEST,
            )

        capabilities = DeviceCatalogService.get_capabilities(device_family)
        serializer = DeviceCapabilitiesSerializer(data=capabilities)
        serializer.is_valid(raise_exception=True)
        return Response(serializer.data, status=status.HTTP_200_OK)


# =============================================================================
# 2. Профили конфигурации (Profiles)
# =============================================================================


class ProfileListView(BaseDeviceView):
    """
    GET /profiles — Профили конфигурации семейств
    """

    def get(self, request: Request) -> Response:
        queryset = DeviceProfile.objects.all().order_by("id")

        device_family = request.query_params.get("device_family")
        if device_family:
            queryset = queryset.filter(device_family=device_family)

        items, meta = self._paginate_queryset(request, queryset)
        serializer = ProfileSerializer(items, many=True)
        return Response({"items": serializer.data, **meta}, status=status.HTTP_200_OK)


class ProfileDetailView(BaseDeviceView):
    """
    GET /profiles/{profile_id} — Профиль по идентификатору с заголовком ETag
    """

    def get(self, request: Request, profile_id: int) -> Response:
        profile = get_object_or_404(DeviceProfile, id=profile_id)
        serializer = ProfileSerializer(profile)
        return Response(
            serializer.data,
            status=status.HTTP_200_OK,
            headers={"ETag": self._compute_etag(profile)},
        )


# =============================================================================
# 3. Структуры приборов (Structures)
# =============================================================================


class StructureListCreateView(BaseDeviceView):
    """
    GET /structures — Список структур (краткие записи)
    POST /structures — Создать параметрическую структуру
    """

    ALLOWED_SORT_FIELDS = {
        "created_at",
        "-created_at",
        "updated_at",
        "-updated_at",
        "name",
        "-name",
    }

    def get(self, request: Request) -> Response:
        queryset = DeviceStructure.objects.all()

        device_family = request.query_params.get("device_family")
        if device_family:
            queryset = queryset.filter(device_family=device_family)

        search = request.query_params.get("search")
        if search:
            queryset = queryset.filter(
                Q(name__icontains=search) | Q(description__icontains=search)
            )

        sort_by = request.query_params.get("sort", "-created_at")
        if sort_by not in self.ALLOWED_SORT_FIELDS:
            sort_by = "-created_at"
        queryset = queryset.order_by(sort_by)

        items, meta = self._paginate_queryset(request, queryset)
        serializer = DeviceStructureListItemSerializer(items, many=True)
        return Response({"items": serializer.data, **meta}, status=status.HTTP_200_OK)

    def post(self, request: Request) -> Response:
        request_id = self._get_request_id(request)

        cached = self._get_cached_response(request, "structures_create")
        if cached:
            headers = {**cached["headers"], "X-Request-Id": request_id}
            return Response(cached["data"], status=cached["status"], headers=headers)

        req_serializer, error_response = self._validate_request(
            DeviceStructureCreateSerializer, request.data, request_id
        )
        if error_response:
            return error_response

        validation_report = StructureValidationService.validate(req_serializer.validated_data)
        if not validation_report["valid"]:
            return Response(
                {
                    "detail": "Нарушены инварианты геометрии или обязательные поля семейства.",
                    "report": validation_report,
                },
                status=status.HTTP_422_UNPROCESSABLE_ENTITY,
                headers={"X-Request-Id": request_id},
            )

        structure = StructureService.create_structure(
            validated_data=req_serializer.validated_data,
            user=request.user,
        )
        response, cache_headers = self._build_created_response(request, structure, request_id)

        self._save_cached_response(
            request,
            "structures_create",
            response.data,
            status.HTTP_201_CREATED,
            headers=cache_headers,
        )
        return response


class StructureDetailView(BaseDeviceView):
    """
    GET /structures/{structure_id} — Структура с полным набором параметров
    PATCH /structures/{structure_id} — Частично обновить структуру (требует If-Match)
    DELETE /structures/{structure_id} — Удалить структуру (требует If-Match)
    """

    def get(self, request: Request, structure_id: int) -> Response:
        structure = get_object_or_404(DeviceStructure, id=structure_id)
        serializer = DeviceStructureSerializer(structure)
        return Response(
            serializer.data,
            status=status.HTTP_200_OK,
            headers={"ETag": self._compute_etag(structure)},
        )

    def patch(self, request: Request, structure_id: int) -> Response:
        request_id = self._get_request_id(request)
        structure = get_object_or_404(DeviceStructure, id=structure_id)

        conflict_resp = self._check_if_match(request, structure)
        if conflict_resp:
            conflict_resp["X-Request-Id"] = request_id
            return conflict_resp

        if "device_id" in request.data and request.data["device_id"] != structure.device_id:
            return Response(
                {"detail": "Изменение device_id запрещено; для другой геометрии создайте новую структуру."},
                status=status.HTTP_409_CONFLICT,
                headers={"X-Request-Id": request_id},
            )

        req_serializer, error_response = self._validate_request(
            DeviceStructureUpdateSerializer,
            request.data,
            request_id,
            instance=structure,
            partial=True,
        )
        if error_response:
            return error_response

        try:
            updated_structure = StructureService.update_structure(
                structure=structure,
                validated_data=req_serializer.validated_data,
                user=request.user,
            )
        except StructureConflictError as exc:
            return Response(
                {"detail": str(exc)},
                status=status.HTTP_409_CONFLICT,
                headers={"X-Request-Id": request_id},
            )

        res_serializer = DeviceStructureSerializer(updated_structure)
        return Response(
            res_serializer.data,
            status=status.HTTP_200_OK,
            headers={
                "ETag": self._compute_etag(updated_structure),
                "X-Request-Id": request_id,
            },
        )

    def delete(self, request: Request, structure_id: int) -> Response:
        structure = get_object_or_404(DeviceStructure, id=structure_id)

        conflict_resp = self._check_if_match(request, structure)
        if conflict_resp:
            return conflict_resp

        try:
            StructureService.delete_structure(structure)
        except StructureConflictError as exc:
            return Response({"detail": str(exc)}, status=status.HTTP_409_CONFLICT)

        return Response(status=status.HTTP_204_NO_CONTENT)


class StructureValidateView(BaseDeviceView):
    """
    POST /structures/validate — Проверка структуры без сохранения
    """

    def post(self, request: Request) -> Response:
        request_id = self._get_request_id(request)

        req_serializer, error_response = self._validate_request(
            DeviceStructureCreateSerializer, request.data, request_id
        )
        if error_response:
            return error_response

        report = StructureValidationService.validate(req_serializer.validated_data)
        res_serializer = ValidationReportSerializer(data=report)
        res_serializer.is_valid(raise_exception=True)

        return Response(
            res_serializer.data,
            status=status.HTTP_200_OK,
            headers={"X-Request-Id": request_id},
        )


class StructureImportCadView(BaseDeviceView):
    """
    POST /structures/import-cad — Импорт CAD-геометрии (STL) (FR-IMP-02)
    """

    parser_classes = [MultiPartParser, FormParser]

    def post(self, request: Request) -> Response:
        request_id = self._get_request_id(request)
        uploaded_file = request.FILES.get("file")

        if not uploaded_file:
            return Response(
                {"detail": "STL-файл ('file') обязателен."},
                status=status.HTTP_400_BAD_REQUEST,
                headers={"X-Request-Id": request_id},
            )

        if uploaded_file.size > MAX_CAD_UPLOAD_BYTES:
            return Response(
                {"detail": f"Размер STL-файла превышает лимит ({MAX_CAD_UPLOAD_BYTES} байт)."},
                status=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                headers={"X-Request-Id": request_id},
            )

        req_serializer, error_response = self._validate_request(
            CadImportRequestSerializer, request.data, request_id
        )
        if error_response:
            return error_response

        try:
            structure = CadImportService.import_stl(
                file_obj=uploaded_file,
                metadata=req_serializer.validated_data,
                user=request.user,
            )
        except UnsupportedCadFormatError as exc:
            return Response(
                {
                    "type": "about:blank",
                    "title": "Unsupported Media Type",
                    "status": 415,
                    "detail": str(exc),
                },
                status=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                content_type="application/problem+json",
                headers={"X-Request-Id": request_id},
            )
        except CadGeometryValidationError as exc:
            return Response(
                {"detail": str(exc)},
                status=status.HTTP_422_UNPROCESSABLE_ENTITY,
                headers={"X-Request-Id": request_id},
            )

        response, _ = self._build_created_response(request, structure, request_id)
        return response


class StructureCadGeometryView(BaseDeviceView):
    """
    GET /structures/{structure_id}/cad — Метаданные CAD-геометрии (STL)
    """

    def get(self, request: Request, structure_id: int) -> Response:
        structure = get_object_or_404(DeviceStructure, id=structure_id)
        cad_geometry = getattr(structure, "cad_geometry", None)

        if not cad_geometry:
            return Response(
                {"detail": "У данной структуры отсутствует привязанная CAD-геометрия."},
                status=status.HTTP_404_NOT_FOUND,
            )

        serializer = StlGeometrySerializer(cad_geometry)
        return Response(serializer.data, status=status.HTTP_200_OK)


class StructureExportView(BaseDeviceView):
    """
    GET /structures/{structure_id}/export — Экспорт полного набора параметров (FR-IMP-03)
    """

    def get(self, request: Request, structure_id: int) -> Response:
        structure = get_object_or_404(DeviceStructure, id=structure_id)
        export_payload = StructureService.export_full_parameter_set(structure)

        serializer = ParameterSetExportSerializer(data=export_payload)
        serializer.is_valid(raise_exception=True)
        return Response(serializer.data, status=status.HTTP_200_OK)
