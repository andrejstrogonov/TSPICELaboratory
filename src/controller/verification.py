import uuid
from django.core.cache import cache
from django.shortcuts import get_object_or_404
from rest_framework import status
from rest_framework.exceptions import ValidationError
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from rest_framework.views import APIView

# Предполагаемые модели, сериализаторы и сервисы вашего проекта
from .models import VerificationRun
from .serializers import (
    VerificationRunSerializer,
    VerificationRunRequestSerializer,
    VerificationAcceptedSerializer,
    VerilogACompileRequestSerializer,
    VerilogACompileResultSerializer,
    LoopbackRequestSerializer,
    LoopbackResponseSerializer,
    HysteresisValidationSerializer,
)
from .services import (
    EdaVerificationService,
    OpenVafCompilerService,
    LoopbackVerificationService,
    HysteresisValidationService,
    ExternalToolUnavailableError,
)


class IdempotencyMixin:
    """
    Вспомогательный миксин для обработки заголовка Idempotency-Key
    и трассировки X-Request-Id.
    """
    IDEMPOTENCY_TTL = 60 * 60 * 24  # 24 часа

    def _get_request_id(self, request):
        return request.headers.get('X-Request-Id', str(uuid.uuid4()))

    def _get_idempotency_key(self, request):
        return request.headers.get('Idempotency-Key')

    def _get_cached_response(self, request, endpoint_prefix):
        idem_key = self._get_idempotency_key(request)
        if not idem_key:
            return None
        cache_key = f"idem:{endpoint_prefix}:{request.user.id}:{idem_key}"
        return cache.get(cache_key)

    def _save_cached_response(self, request, endpoint_prefix, payload, status_code):
        idem_key = self._get_idempotency_key(request)
        if not idem_key:
            return
        cache_key = f"idem:{endpoint_prefix}:{request.user.id}:{idem_key}"
        cache.set(cache_key, {"data": payload, "status": status_code}, self.IDEMPOTENCY_TTL)


def validate_422(serializer):
    """
    В DRF ошибки сериализатора по умолчанию возвращают 400.
    Этот хелпер преобразует их в 422 Unprocessable Entity согласно OpenAPI-контракту.
    """
    if not serializer.is_valid():
        return Response(
            {"detail": "Ошибка валидации входных данных", "errors": serializer.errors},
            status=status.HTTP_422_UNPROCESSABLE_ENTITY
        )
    return None


class VerificationRunListCreateView(IdempotencyMixin, APIView):
    """
    GET /verification/runs — Прогоны верификации (список с фильтрацией)
    POST /verification/runs — Запустить верификацию в EDA (FR-EXT-06)
    """
    permission_classes = [IsAuthenticated]

    def get(self, request):
        queryset = VerificationRun.objects.all().order_by('-created_at')

        # Фильтрация по structure_id (integer >= 1)
        structure_id = request.query_params.get('structure_id')
        if structure_id is not None:
            try:
                struct_id_int = int(structure_id)
                if struct_id_int < 1:
                    raise ValueError()
                queryset = queryset.filter(structure_id=struct_id_int)
            except ValueError:
                return Response(
                    {"detail": "Параметр structure_id должен быть целым числом >= 1."},
                    status=status.HTTP_422_UNPROCESSABLE_ENTITY
                )

        # Фильтрация по режиму верификации
        mode = request.query_params.get('mode')
        if mode:
            queryset = queryset.filter(mode=mode)

        # Пагинация
        page = max(1, int(request.query_params.get('page', 1)))
        page_size = max(1, int(request.query_params.get('page_size', 20)))
        offset = (page - 1) * page_size

        total = queryset.count()
        items = queryset[offset:offset + page_size]

        serializer = VerificationRunSerializer(items, many=True)
        return Response({
            "items": serializer.data,
            "total": total,
            "page": page,
            "page_size": page_size
        }, status=status.HTTP_200_OK)

    def post(self, request):
        request_id = self._get_request_id(request)

        # Проверка идемпотентности
        cached = self._get_cached_response(request, "verification_runs")
        if cached:
            return Response(cached["data"], status=cached["status"], headers={'X-Request-Id': request_id})

        req_serializer = VerificationRunRequestSerializer(data=request.data)
        error_response = validate_422(req_serializer)
        if error_response:
            return error_response

        try:
            # Постановка задачи прогона ngspice / Micro-Cap в очередь
            accepted_data = EdaVerificationService.enqueue_run(
                validated_data=req_serializer.validated_data,
                user=request.user,
                request_id=request_id
            )
        except ExternalToolUnavailableError as exc:
            return Response(
                {"detail": f"EDA-симулятор временно недоступен: {str(exc)}"},
                status=status.HTTP_503_SERVICE_UNAVAILABLE,
                headers={'X-Request-Id': request_id}
            )

        res_serializer = VerificationAcceptedSerializer(data=accepted_data)
        res_serializer.is_valid(raise_exception=True)

        self._save_cached_response(request, "verification_runs", res_serializer.data, status.HTTP_202_ACCEPTED)
        return Response(
            res_serializer.data,
            status=status.HTTP_202_ACCEPTED,
            headers={'X-Request-Id': request_id}
        )


class VerificationRunDetailView(APIView):
    """
    GET /verification/runs/{run_id} — Прогон верификации с результатом, кривыми и вердиктом
    """
    permission_classes = [IsAuthenticated]

    def get(self, request, run_id):
        run = get_object_or_404(VerificationRun, id=run_id)  # Вернёт 404 NotFound автоматически
        serializer = VerificationRunSerializer(run)
        return Response(serializer.data, status=status.HTTP_200_OK)


class VerilogACompileView(APIView):
    """
    POST /verification/verilog-a/compile — Компиляция Verilog-A через OpenVAF в OSDI (FR-EXT-05)
    """
    permission_classes = [IsAuthenticated]

    def post(self, request):
        req_serializer = VerilogACompileRequestSerializer(data=request.data)
        error_response = validate_422(req_serializer)
        if error_response:
            return error_response

        try:
            compile_result = OpenVafCompilerService.compile(req_serializer.validated_data)
        except ExternalToolUnavailableError as exc:
            return Response(
                {"detail": f"Компилятор OpenVAF недоступен: {str(exc)}"},
                status=status.HTTP_503_SERVICE_UNAVAILABLE
            )

        res_serializer = VerilogACompileResultSerializer(data=compile_result)
        res_serializer.is_valid(raise_exception=True)
        return Response(res_serializer.data, status=status.HTTP_200_OK)


class LoopbackVerificationView(IdempotencyMixin, APIView):
    """
    POST /verification/loopback — Loopback-самотестирование (FR-EXT-07)
    Эталонная карта -> генерация ВАХ -> экстракция -> сравнение параметров.
    """
    permission_classes = [IsAuthenticated]

    def post(self, request):
        cached = self._get_cached_response(request, "verification_loopback")
        if cached:
            return Response(cached["data"], status=cached["status"])

        req_serializer = LoopbackRequestSerializer(data=request.data)
        error_response = validate_422(req_serializer)
        if error_response:
            return error_response

        loopback_data = LoopbackVerificationService.enqueue_loopback(
            validated_data=req_serializer.validated_data,
            user=request.user
        )

        res_serializer = LoopbackResponseSerializer(data=loopback_data)
        res_serializer.is_valid(raise_exception=True)

        self._save_cached_response(request, "verification_loopback", res_serializer.data, status.HTTP_202_ACCEPTED)
        return Response(res_serializer.data, status=status.HTTP_202_ACCEPTED)


class HysteresisValidationView(APIView):
    """
    POST /verification/hysteresis — Валидация гистерезиса мемристора (FR-EXT-05)
    Проверка формы петли и площади выше порога.
    """
    permission_classes = [IsAuthenticated]

    def post(self, request):
        req_serializer = VerificationRunRequestSerializer(data=request.data)
        error_response = validate_422(req_serializer)
        if error_response:
            return error_response

        validation_result = HysteresisValidationService.validate_loop(
            validated_data=req_serializer.validated_data
        )

        res_serializer = HysteresisValidationSerializer(data=validation_result)
        res_serializer.is_valid(raise_exception=True)
        return Response(res_serializer.data, status=status.HTTP_200_OK)
