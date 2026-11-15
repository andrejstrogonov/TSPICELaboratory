import uuid
from typing import Any, Optional

from django.core.cache import cache
from django.db.models import QuerySet
from django.shortcuts import get_object_or_404
from rest_framework import serializers, status
from rest_framework.permissions import IsAuthenticated
from rest_framework.request import Request
from rest_framework.response import Response
from rest_framework.views import APIView

# Предполагаемые модели, сериализаторы и сервисы вашего проекта
from .models import SurrogateModel, TrainingRun
from .serializers import (
    InferenceRequestSerializer,
    InferenceResultSerializer,
    ModelGateReportSerializer,
    OnnxExportResultSerializer,
    SurrogateModelCreateSerializer,
    SurrogateModelSerializer,
    TrainingAcceptedSerializer,
    TrainingRequestSerializer,
    TrainingRunSerializer,
)
from .services import (
    OnnxExportValidationError,
    SurrogateGateService,
    SurrogateInferenceService,
    SurrogateOnnxService,
    SurrogateRegistryService,
    SurrogateTrainingService,
)


SUPPORTED_SURROGATE_STATUSES = {"draft", "validating", "active", "archived", "failed"}


class IdempotencyMixin:
    """
    Миксин для обработки заголовков Idempotency-Key и X-Request-Id.
    """

    IDEMPOTENCY_TTL = 60 * 60 * 24  # 24 часа

    def _get_request_id(self, request: Request) -> str:
        return request.headers.get("X-Request-Id", str(uuid.uuid4()))

    def _get_cached_response(self, request: Request, endpoint_prefix: str) -> Optional[dict]:
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


class BaseAiView(IdempotencyMixin, APIView):
    """
    Базовый контроллер для эндпоинтов домена AI с общей валидацией 422 и пагинацией.
    """

    permission_classes = [IsAuthenticated]

    def _validate_request(
        self,
        serializer_cls: type[serializers.Serializer],
        data: Any,
        request_id: str,
    ) -> tuple[Optional[serializers.Serializer], Optional[Response]]:
        serializer = serializer_cls(data=data)
        if not serializer.is_valid():
            response = Response(
                {
                    "detail": "Ошибка валидации входных данных.",
                    "errors": serializer.errors,
                },
                status=status.HTTP_422_UNPROCESSABLE_ENTITY,
                headers={"X-Request-Id": request_id},
            )
            return None, response
        return serializer, None

    def _paginate_queryset(self, request: Request, queryset: QuerySet) -> tuple[list, dict]:
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


# =============================================================================
# 1. Реестр суррогатных моделей (/ai/surrogates)
# =============================================================================

class SurrogateListCreateView(BaseAiView):
    """
    GET /ai/surrogates — Реестр суррогатных моделей
    POST /ai/surrogates — Зарегистрировать суррогатную модель в состоянии draft
    """

    def get(self, request: Request) -> Response:
        queryset = SurrogateModel.objects.all().order_by("-created_at")

        family = request.query_params.get("family")
        if family:
            queryset = queryset.filter(family=family)

        model_status = request.query_params.get("status")
        if model_status:
            if model_status not in SUPPORTED_SURROGATE_STATUSES:
                return Response(
                    {
                        "detail": f"Недопустимый статус модели '{model_status}'.",
                        "allowed": sorted(SUPPORTED_SURROGATE_STATUSES),
                    },
                    status=status.HTTP_422_UNPROCESSABLE_ENTITY,
                )
            queryset = queryset.filter(status=model_status)

        items, meta = self._paginate_queryset(request, queryset)
        serializer = SurrogateModelSerializer(items, many=True)

        return Response(
            {"items": serializer.data, **meta},
            status=status.HTTP_200_OK,
        )

    def post(self, request: Request) -> Response:
        request_id = self._get_request_id(request)

        req_serializer, error_response = self._validate_request(
            SurrogateModelCreateSerializer, request.data, request_id
        )
        if error_response:
            return error_response

        # Создаётся в состоянии 'draft'; переход в 'active' возможен только после гейтов
        surrogate = SurrogateRegistryService.register_draft(
            validated_data=req_serializer.validated_data,
            user=request.user,
        )

        res_serializer = SurrogateModelSerializer(surrogate)
        return Response(
            res_serializer.data,
            status=status.HTTP_201_CREATED,
            headers={"X-Request-Id": request_id},
        )


class SurrogateDetailView(BaseAiView):
    """
    GET /ai/surrogates/{surrogate_id} — Запись суррогатной модели в реестре
    """

    def get(self, request: Request, surrogate_id: int) -> Response:
        surrogate = get_object_or_404(SurrogateModel, id=surrogate_id)
        serializer = SurrogateModelSerializer(surrogate)
        return Response(serializer.data, status=status.HTTP_200_OK)


# =============================================================================
# 2. Обучение суррогатных моделей (/ai/surrogates/{surrogate_id}/train и /ai/trainings)
# =============================================================================

class SurrogateTrainView(BaseAiView):
    """
    POST /ai/surrogates/{surrogate_id}/train — Запустить асинхронное обучение на прогонах FEM
    """

    def post(self, request: Request, surrogate_id: int) -> Response:
        request_id = self._get_request_id(request)
        surrogate = get_object_or_404(SurrogateModel, id=surrogate_id)

        cached = self._get_cached_response(request, f"ai_train_{surrogate_id}")
        if cached:
            headers = {**cached["headers"], "X-Request-Id": request_id}
            return Response(cached["data"], status=cached["status"], headers=headers)

        req_serializer, error_response = self._validate_request(
            TrainingRequestSerializer, request.data, request_id
        )
        if error_response:
            return error_response

        # Ставим задачу обучения в очередь; прогресс транслируется через /jobs/{job_id}/events
        accepted_data = SurrogateTrainingService.enqueue_training(
            surrogate=surrogate,
            validated_data=req_serializer.validated_data,
            user=request.user,
            request_id=request_id,
        )

        res_serializer = TrainingAcceptedSerializer(data=accepted_data)
        res_serializer.is_valid(raise_exception=True)

        self._save_cached_response(
            request,
            f"ai_train_{surrogate_id}",
            res_serializer.data,
            status.HTTP_202_ACCEPTED,
        )
        return Response(
            res_serializer.data,
            status=status.HTTP_202_ACCEPTED,
            headers={"X-Request-Id": request_id},
        )


class TrainingRunListView(BaseAiView):
    """
    GET /ai/trainings — Журнал прогонов обучения
    """

    def get(self, request: Request) -> Response:
        queryset = TrainingRun.objects.all().order_by("-created_at")

        surrogate_id = request.query_params.get("surrogate_id")
        if surrogate_id is not None:
            try:
                surrogate_id_int = int(surrogate_id)
                if surrogate_id_int < 1:
                    raise ValueError()
                queryset = queryset.filter(surrogate_id=surrogate_id_int)
            except ValueError:
                return Response(
                    {"detail": "Параметр 'surrogate_id' должен быть целым числом >= 1."},
                    status=status.HTTP_422_UNPROCESSABLE_ENTITY,
                )

        run_status = request.query_params.get("status")
        if run_status:
            queryset = queryset.filter(status=run_status)

        items, meta = self._paginate_queryset(request, queryset)
        serializer = TrainingRunSerializer(items, many=True)

        return Response(
            {"items": serializer.data, **meta},
            status=status.HTTP_200_OK,
        )


class TrainingRunDetailView(BaseAiView):
    """
    GET /ai/trainings/{training_id} — Прогон обучения с метриками по эпохам
    """

    def get(self, request: Request, training_id: int) -> Response:
        training_run = get_object_or_404(TrainingRun, id=training_id)
        serializer = TrainingRunSerializer(training_run)
        return Response(serializer.data, status=status.HTTP_200_OK)


# =============================================================================
# 3. Инференс, гейты качества и ONNX-экспорт
# =============================================================================

class SurrogateInferenceView(BaseAiView):
    """
    POST /ai/inference — Пробный инференс warm start (без записи в задачу расчёта)
    """

    def post(self, request: Request) -> Response:
        request_id = self._get_request_id(request)

        req_serializer, error_response = self._validate_request(
            InferenceRequestSerializer, request.data, request_id
        )
        if error_response:
            return error_response

        # Если передан compare_with_fem=True, сервис также вычисляет невязку относительно FEM
        inference_result = SurrogateInferenceService.run_warm_start(
            validated_data=req_serializer.validated_data
        )

        res_serializer = InferenceResultSerializer(data=inference_result)
        res_serializer.is_valid(raise_exception=True)

        return Response(
            res_serializer.data,
            status=status.HTTP_200_OK,
            headers={"X-Request-Id": request_id},
        )


class SurrogateGatesView(BaseAiView):
    """
    GET /ai/surrogates/{surrogate_id}/gates — Отчёт о проверках-гейтах модели
    (физическая невязка, монотонность ВАХ, C2-непрерывность, экономия итераций).
    """

    def get(self, request: Request, surrogate_id: int) -> Response:
        surrogate = get_object_or_404(SurrogateModel, id=surrogate_id)
        gate_report = SurrogateGateService.get_gate_report(surrogate)

        serializer = ModelGateReportSerializer(data=gate_report)
        serializer.is_valid(raise_exception=True)
        return Response(serializer.data, status=status.HTTP_200_OK)


class SurrogateOnnxExportView(BaseAiView):
    """
    POST /ai/surrogates/{surrogate_id}/onnx — Экспорт модели в ONNX для Rust-ядра
    и проверка паритета «ONNX ≈ PyTorch».
    """

    def post(self, request: Request, surrogate_id: int) -> Response:
        request_id = self._get_request_id(request)
        surrogate = get_object_or_404(SurrogateModel, id=surrogate_id)

        try:
            export_result = SurrogateOnnxService.export_and_verify_parity(surrogate)
        except OnnxExportValidationError as exc:
            return Response(
                {"detail": str(exc)},
                status=status.HTTP_422_UNPROCESSABLE_ENTITY,
                headers={"X-Request-Id": request_id},
            )

        res_serializer = OnnxExportResultSerializer(data=export_result)
        res_serializer.is_valid(raise_exception=True)

        return Response(
            res_serializer.data,
            status=status.HTTP_200_OK,
            headers={"X-Request-Id": request_id},
        )


# =============================================================================
# Маршрутизация (для подключения в urls.py)
# =============================================================================
# from django.urls import path
#
# urlpatterns = [
#     path("ai/surrogates", SurrogateListCreateView.as_view(), name="ai-surrogates"),
#     path("ai/surrogates/<int:surrogate_id>", SurrogateDetailView.as_view(), name="ai-surrogate-detail"),
#     path("ai/surrogates/<int:surrogate_id>/train", SurrogateTrainView.as_view(), name="ai-surrogate-train"),
#     path("ai/surrogates/<int:surrogate_id>/gates", SurrogateGatesView.as_view(), name="ai-surrogate-gates"),
#     path("ai/surrogates/<int:surrogate_id>/onnx", SurrogateOnnxExportView.as_view(), name="ai-surrogate-onnx"),
#     path("ai/trainings", TrainingRunListView.as_view(), name="ai-trainings"),
#     path("ai/trainings/<int:training_id>", TrainingRunDetailView.as_view(), name="ai-training-detail"),
#     path("ai/inference", SurrogateInferenceView.as_view(), name="ai-inference"),
# ]
