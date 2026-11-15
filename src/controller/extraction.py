import uuid
from django.core.cache import cache
from django.shortcuts import get_object_or_404
from django.urls import reverse
from rest_framework import serializers, status
from rest_framework.parsers import JSONParser, MultiPartParser, FormParser
from rest_framework.permissions import AllowAny, IsAuthenticated
from rest_framework.response import Response
from rest_framework.views import APIView

# Предполагаемые модели, сериализаторы и сервисы вашего проекта
from .models import MeasurementSet, ExtractionSession, SpiceModel
from .serializers import (
    MeasurementImportRequestSerializer,
    MeasurementSetSerializer,
    MeasurementPointSerializer,
    ExtractionSessionCreateSerializer,
    ExtractionSessionAcceptedSerializer,
    ExtractionSessionSerializer,
    ExtractionProgressSerializer,
    CurveMetricsResultSerializer,
    MetricDefinitionSerializer,
    SpiceModelSerializer,
    SpiceExportRequestSerializer,
    SpiceExportResponseSerializer,
    AnalysisReportSerializer,
)
from .services import (
    MeasurementImportService,
    MeasurementParsingError,
    ExtractionFittingService,
    CurveMetricsService,
    SpiceExportService,
    AnalysisReportService,
)


MAX_MEASUREMENT_UPLOAD_BYTES = 50 * 1024 * 1024  # 50 МБ (настраивается под NFR)


class IdempotencyMixin:
    """
    Миксин для обработки заголовков Idempotency-Key и X-Request-Id.
    """
    IDEMPOTENCY_TTL = 60 * 60 * 24

    def _get_request_id(self, request):
        return request.headers.get('X-Request-Id', str(uuid.uuid4()))

    def _get_cached_response(self, request, endpoint_prefix):
        idem_key = request.headers.get('Idempotency-Key')
        if not idem_key:
            return None
        cache_key = f"idem:{endpoint_prefix}:{request.user.id}:{idem_key}"
        return cache.get(cache_key)

    def _save_cached_response(self, request, endpoint_prefix, payload, status_code, headers=None):
        idem_key = request.headers.get('Idempotency-Key')
        if not idem_key:
            return
        cache_key = f"idem:{endpoint_prefix}:{request.user.id}:{idem_key}"
        cache.set(
            cache_key,
            {"data": payload, "status": status_code, "headers": headers or {}},
            self.IDEMPOTENCY_TTL,
        )


def validate_422(serializer):
    """
    Преобразует ошибки сериализатора DRF (400) в 422 Unprocessable Entity согласно контракту.
    """
    if not serializer.is_valid():
        return Response(
            {"detail": "Ошибка валидации входных данных", "errors": serializer.errors},
            status=status.HTTP_422_UNPROCESSABLE_ENTITY,
        )
    return None


def parse_positive_int_query(value, param_name):
    """
    Валидация query-параметров типа int64 с minimum: 1 (например, structure_id).
    """
    if value is None:
        return None, None
    try:
        parsed = int(value)
        if parsed < 1:
            raise ValueError()
        return parsed, None
    except ValueError:
        err = Response(
            {"detail": f"Параметр '{param_name}' должен быть целым числом >= 1."},
            status=status.HTTP_422_UNPROCESSABLE_ENTITY,
        )
        return None, err


# =============================================================================
# 1. Измерения (Measurements)
# =============================================================================

class MeasurementImportView(IdempotencyMixin, APIView):
    """
    POST /measurements/import — Импорт измерений из CSV/лога (FR-IMP-01)
    """
    permission_classes = [IsAuthenticated]
    parser_classes = [MultiPartParser, FormParser]

    def post(self, request):
        request_id = self._get_request_id(request)
        uploaded_file = request.FILES.get('file')

        if not uploaded_file:
            return Response(
                {"detail": "Файл измерений ('file') обязателен."},
                status=status.HTTP_422_UNPROCESSABLE_ENTITY,
                headers={'X-Request-Id': request_id},
            )

        if uploaded_file.size > MAX_MEASUREMENT_UPLOAD_BYTES:
            return Response(
                {"detail": f"Размер файла превышает допустимый лимит ({MAX_MEASUREMENT_UPLOAD_BYTES} байт)."},
                status=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                headers={'X-Request-Id': request_id},
            )

        req_serializer = MeasurementImportRequestSerializer(data=request.data)
        error_response = validate_422(req_serializer)
        if error_response:
            error_response['X-Request-Id'] = request_id
            return error_response

        try:
            measurement_set = MeasurementImportService.import_csv(
                file_obj=uploaded_file,
                metadata=req_serializer.validated_data,
                user=request.user,
            )
        except MeasurementParsingError as exc:
            # exc.code: 'INVALID_MEASUREMENT_COLUMNS' или 'EMPTY_MEASUREMENT'
            return Response(
                {"code": exc.code, "detail": str(exc)},
                status=status.HTTP_422_UNPROCESSABLE_ENTITY,
                headers={'X-Request-Id': request_id},
            )

        res_serializer = MeasurementSetSerializer(measurement_set)
        location_url = request.build_absolute_uri(
            reverse('measurement-set-detail', kwargs={'set_id': measurement_set.id})
        )
        return Response(
            res_serializer.data,
            status=status.HTTP_201_CREATED,
            headers={'Location': location_url, 'X-Request-Id': request_id},
        )


class MeasurementSetListView(APIView):
    """
    GET /measurements — Список наборов измерений
    """
    permission_classes = [IsAuthenticated]

    def get(self, request):
        queryset = MeasurementSet.objects.all().order_by('-created_at')

        structure_id, err = parse_positive_int_query(request.query_params.get('structure_id'), 'structure_id')
        if err:
            return err
        if structure_id is not None:
            queryset = queryset.filter(structure_id=structure_id)

        source = request.query_params.get('source')
        if source:
            queryset = queryset.filter(source=source)

        page = max(1, int(request.query_params.get('page', 1)))
        page_size = max(1, int(request.query_params.get('page_size', 20)))
        offset = (page - 1) * page_size

        total = queryset.count()
        items = queryset[offset:offset + page_size]

        serializer = MeasurementSetSerializer(items, many=True)
        return Response({
            "items": serializer.data,
            "total": total,
            "page": page,
            "page_size": page_size,
        }, status=status.HTTP_200_OK)


class MeasurementSetDetailView(APIView):
    """
    GET /measurements/{set_id} — Набор измерений с описанием свипов
    DELETE /measurements/{set_id} — Удалить набор измерений
    """
    permission_classes = [IsAuthenticated]

    def get(self, request, set_id):
        measurement_set = get_object_or_404(MeasurementSet, id=set_id)
        serializer = MeasurementSetSerializer(measurement_set)
        return Response(serializer.data, status=status.HTTP_200_OK)

    def delete(self, request, set_id):
        measurement_set = get_object_or_404(MeasurementSet, id=set_id)
        measurement_set.delete()
        return Response(status=status.HTTP_204_NO_CONTENT)


class MeasurementPointsListView(APIView):
    """
    GET /measurements/{set_id}/points — Точки измерений (постранично)
    """
    permission_classes = [IsAuthenticated]

    def get(self, request, set_id):
        measurement_set = get_object_or_404(MeasurementSet, id=set_id)
        points_qs = measurement_set.points.all().order_by('id')

        page = max(1, int(request.query_params.get('page', 1)))
        page_size = max(1, int(request.query_params.get('page_size', 500)))
        offset = (page - 1) * page_size

        total = points_qs.count()
        items = points_qs[offset:offset + page_size]

        serializer = MeasurementPointSerializer(items, many=True)
        return Response({
            "items": serializer.data,
            "total": total,
            "page": page,
            "page_size": page_size,
        }, status=status.HTTP_200_OK)


# =============================================================================
# 2. Сессии экстракции (Extraction Sessions)
# =============================================================================

class ExtractionSessionListCreateView(IdempotencyMixin, APIView):
    """
    GET /extraction/sessions — Список сессий экстракции
    POST /extraction/sessions — Запустить многопараметрический фиттинг (FR-EXT-02)
    """
    permission_classes = [IsAuthenticated]

    def get(self, request):
        queryset = ExtractionSession.objects.all().order_by('-created_at')

        structure_id, err = parse_positive_int_query(request.query_params.get('structure_id'), 'structure_id')
        if err:
            return err
        if structure_id is not None:
            queryset = queryset.filter(structure_id=structure_id)

        session_status = request.query_params.get('status')
        if session_status:
            queryset = queryset.filter(status=session_status)

        page = max(1, int(request.query_params.get('page', 1)))
        page_size = max(1, int(request.query_params.get('page_size', 20)))
        offset = (page - 1) * page_size

        total = queryset.count()
        items = queryset[offset:offset + page_size]

        serializer = ExtractionSessionSerializer(items, many=True)
        return Response({
            "items": serializer.data,
            "total": total,
            "page": page,
            "page_size": page_size,
        }, status=status.HTTP_200_OK)

    def post(self, request):
        request_id = self._get_request_id(request)

        cached = self._get_cached_response(request, "extraction_sessions_create")
        if cached:
