import uuid
from typing import Any, Optional

from django.core.cache import cache
from django.db.models import QuerySet
from django.shortcuts import get_object_or_404
from rest_framework import serializers, status
from rest_framework.permissions import IsAuthenticated
from rest_framework.request import Request
from rest_framework.response import Response
from rest_framework.throttling import UserRateThrottle
from rest_framework.views import APIView

# Предполагаемые модели, сериализаторы и сервисы вашего проекта
from .models import CharacteristicSet, SolverTask
from .serializers import (
    CharacteristicsBatchRequestSerializer,
    CharacteristicsBatchResponseSerializer,
    CharacteristicSetSerializer,
    FemCalculationAcceptedSerializer,
    FemCalculationRequestSerializer,
    FemCalculationResponseSerializer,
    FieldInterpolationQuerySerializer,
    FieldInterpolationResultSerializer,
    FieldMapPageSerializer,
    FieldMapSerializer,
    MemoryCharacteristicsRequestSerializer,
    MemoryCharacteristicsResponseSerializer,
    MemoryEstimateSerializer,
    MeshEstimateRequestSerializer,
    PdeCharacteristicsRequestSerializer,
    PdeCharacteristicsResponseSerializer,
    QuantumPoissonRequestSerializer,
    QuantumPoissonResponseSerializer,
    SolverTaskSerializer,
    TaskCancelRequestSerializer,
)
from .services import (
    CharacteristicSolverService,
    ComputeServiceUnavailableError,
    FemSolverService,
    FieldMapService,
    MeshBudgetExceededError,
    QuantumPoissonSolverService,
    SolverExecutionError,
    SolverTaskService,
)


class SolverRateThrottle(UserRateThrottle):
    """
    Ограничение частоты тяжёлых расчётов (429 Too Many Requests).
    Настраивается в settings.py -> REST_FRAMEWORK['DEFAULT_THROTTLE_RATES']['solver_compute'].
    """

    scope = "solver_compute"

    def get_remaining(self) -> int:
        if hasattr(self, "history") and hasattr(self, "num_requests"):
            return max(0, self.num_requests - len(self.history))
        return 0


class BaseComputeView(APIView):
    """
    Базовый класс для эндпоинтов расчётов (Compute) с поддержкой:
    - Idempotency-Key и X-Request-Id
    - RFC 7240 Prefer: respond-async=false (синхронный экспресс-режим)
    - Валидации входных данных с кодом 422 Unprocessable Entity
    - Безопасной пагинации
    """

    permission_classes = [IsAuthenticated]
    IDEMPOTENCY_TTL = 60 * 60 * 24

    def _get_request_id(self, request: Request) -> str:
        return request.headers.get("X-Request-Id", str(uuid.uuid4()))

    def _is_sync_preferred(self, request: Request) -> bool:
        prefer_header = request.headers.get("Prefer", "").lower()
        return "respond-async=false" in prefer_header

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
                    "detail": "Ошибка валидации параметров расчёта.",
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
        return items, {"total": total, "page": page, "page_size": page_size}


# =============================================================================
# 1. FEM и Пуассон–Шрёдингер (синхронный экспресс-режим + асинхронная очередь)
# =============================================================================

class FemCalculationView(BaseComputeView):
    """
    POST /solver/fem — 3D FEM-расчёт (тепло, электрика, индукция) (FR-CORE-02/03/04).
    По умолчанию возвращает 202 Accepted; заголовок Prefer: respond-async=false
    запускает синхронный экспресс-расчёт (200 OK) с заголовком X-RateLimit-Remaining.
    """

    throttle_classes = [SolverRateThrottle]

    def _get_rate_limit_remaining(self) -> str:
        for throttle in self.get_throttles():
            if isinstance(throttle, SolverRateThrottle):
                return str(throttle.get_remaining())
        return "0"

    def post(self, request: Request) -> Response:
        request_id = self._get_request_id(request)
        sync_mode = self._is_sync_preferred(request)
        cache_scope = f"solver_fem:{'sync' if sync_mode else 'async'}"

        cached = self._get_cached_response(request, cache_scope)
        if cached:
            headers = {**cached["headers"], "X-Request-Id": request_id}
            return Response(cached["data"], status=cached["status"], headers=headers)

        req_serializer, error_response = self._validate_request(
            FemCalculationRequestSerializer, request.data, request_id
        )
        if error_response:
            return error_response

        try:
            if sync_mode:
                result_data = FemSolverService.run_sync(
                    validated_data=req_serializer.validated_data,
                    user=request.user,
                    request_id=request_id,
                )
                res_serializer = FemCalculationResponseSerializer(data=result_data)
                res_serializer.is_valid(raise_exception=True)

                extra_headers = {"X-RateLimit-Remaining": self._get_rate_limit_remaining()}
                self._save_cached_response(
                    request,
                    cache_scope,
                    res_serializer.data,
                    status.HTTP_200_OK,
                    headers=extra_headers,
                )
                return Response(
                    res_serializer.data,
                    status=status.HTTP_200_OK,
                    headers={**extra_headers, "X-Request-Id": request_id},
                )

            accepted_data = FemSolverService.enqueue_async(
                validated_data=req_serializer.validated_data,
                user=request.user,
                request_id=request_id,
            )
            res_serializer = FemCalculationAcceptedSerializer(data=accepted_data)
            res_serializer.is_valid(raise_exception=True)

            self._save_cached_response(
                request, cache_scope, res_serializer.data, status.HTTP_202_ACCEPTED
            )
            return Response(
                res_serializer.data,
                status=status.HTTP_202_ACCEPTED,
                headers={"X-Request-Id": request_id},
            )

        except MeshBudgetExceededError as exc:
            return Response(
                {"detail": f"Размер сетки или объём VRAM превышает допустимый лимит: {exc}"},
                status=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                headers={"X-Request-Id": request_id},
            )
        except ComputeServiceUnavailableError as exc:
            return Response(
                {"detail": f"Вычислительный узел или CUDA-воркер недоступен: {exc}"},
                status=status.HTTP_503_SERVICE_UNAVAILABLE,
                headers={"X-Request-Id": request_id},
            )
        except SolverExecutionError as exc:
            return Response(
                {"detail": f"Внутренняя ошибка решателя FEM: {exc}"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
                headers={"X-Request-Id": request_id},
            )


class QuantumPoissonCalculationView(BaseComputeView):
    """
    POST /solver/quantum-poisson — Самосогласованный расчёт Пуассона–Шрёдингера.
    Поддерживает асинхронную очередь (202) и экспресс-режим через Prefer: respond-async=false (200).
    """

    def post(self, request: Request) -> Response:
        request_id = self._get_request_id(request)
        sync_mode = self._is_sync_preferred(request)
        cache_scope = f"solver_qp:{'sync' if sync_mode else 'async'}"

        cached = self._get_cached_response(request, cache_scope)
        if cached:
            headers = {**cached["headers"], "X-Request-Id": request_id}
            return Response(cached["data"], status=cached["status"], headers=headers)

        req_serializer, error_response = self._validate_request(
            QuantumPoissonRequestSerializer, request.data, request_id
        )
        if error_response:
            return error_response

        if sync_mode:
            result_data = QuantumPoissonSolverService.run_sync(
                validated_data=req_serializer.validated_data,
                user=request.user,
            )
            res_serializer = QuantumPoissonResponseSerializer(data=result_data)
            res_serializer.is_valid(raise_exception=True)

            self._save_cached_response(
                request, cache_scope, res_serializer.data, status.HTTP_200_OK
            )
            return Response(
                res_serializer.data,
                status=status.HTTP_200_OK,
                headers={"X-Request-Id": request_id},
            )

        accepted_data = QuantumPoissonSolverService.enqueue_async(
            validated_data=req_serializer.validated_data,
            user=request.user,
            request_id=request_id,
        )
        res_serializer = FemCalculationAcceptedSerializer(data=accepted_data)
        res_serializer.is_valid(raise_exception=True)

        self._save_cached_response(
            request, cache_scope, res_serializer.data, status.HTTP_202_ACCEPTED
        )
        return Response(
            res_serializer.data,
            status=status.HTTP_202_ACCEPTED,
            headers={"X-Request-Id": request_id},
        )


# =============================================================================
# 2. Reduced-order PDE, state-модели памяти, пакетный расчёт и оценка сетки
# =============================================================================

class PdeCharacteristicsView(BaseComputeView):
    """
    POST /solver/pde-characteristics — Семейства ВАХ/ВФХ по reduced-order PDE
    """

    def post(self, request: Request) -> Response:
        request_id = self._get_request_id(request)
        req_serializer, error_response = self._validate_request(
            PdeCharacteristicsRequestSerializer, request.data, request_id
        )
        if error_response:
            return error_response

        curves_data = CharacteristicSolverService.compute_pde(req_serializer.validated_data)
        res_serializer = PdeCharacteristicsResponseSerializer(data=curves_data)
        res_serializer.is_valid(raise_exception=True)
        return Response(res_serializer.data, status=status.HTTP_200_OK)


class MemoryCharacteristicsView(BaseComputeView):
    """
    POST /solver/memory-characteristics — Кривые memristor / memtransistor (state-модель x(t))
    """

    def post(self, request: Request) -> Response:
        request_id = self._get_request_id(request)
        req_serializer, error_response = self._validate_request(
            MemoryCharacteristicsRequestSerializer, request.data, request_id
        )
        if error_response:
            return error_response

        curves_data = CharacteristicSolverService.compute_memory(req_serializer.validated_data)
        res_serializer = MemoryCharacteristicsResponseSerializer(data=curves_data)
        res_serializer.is_valid(raise_exception=True)
        return Response(res_serializer.data, status=status.HTTP_200_OK)


class CharacteristicsBatchView(BaseComputeView):
    """
    POST /solver/characteristics/batch — Пакетный расчёт всех характеристик семейства
    """

    def post(self, request: Request) -> Response:
        request_id = self._get_request_id(request)
        req_serializer, error_response = self._validate_request(
            CharacteristicsBatchRequestSerializer, request.data, request_id
        )
        if error_response:
            return error_response

        batch_data = CharacteristicSolverService.compute_batch(req_serializer.validated_data)
        res_serializer = CharacteristicsBatchResponseSerializer(data=batch_data)
        res_serializer.is_valid(raise_exception=True)
        return Response(res_serializer.data, status=status.HTTP_200_OK)


class MeshEstimateView(BaseComputeView):
    """
    POST /solver/mesh-estimate — Оценка бюджета сетки (числа узлов) и памяти VRAM
    """

    def post(self, request: Request) -> Response:
        request_id = self._get_request_id(request)
        req_serializer, error_response = self._validate_request(
            MeshEstimateRequestSerializer, request.data, request_id
        )
        if error_response:
            return error_response

        estimate_data = FemSolverService.estimate_mesh_budget(req_serializer.validated_data)
        res_serializer = MemoryEstimateSerializer(data=estimate_data)
        res_serializer.is_valid(raise_exception=True)
        return Response(res_serializer.data, status=status.HTTP_200_OK)


# =============================================================================
# 3. Журнал задач расчёта (Tasks) и отмена
# =============================================================================

class SolverTaskListView(BaseComputeView):
    """
    GET /solver/tasks — Журнал задач расчёта с фильтрацией по structure_id и status
    """

    def get(self, request: Request) -> Response:
        queryset = SolverTask.objects.all().order_by("-created_at")

        structure_id = request.query_params.get("structure_id")
        if structure_id is not None:
            try:
                struct_id_int = int(structure_id)
                if struct_id_int < 1:
                    raise ValueError()
                queryset = queryset.filter(structure_id=struct_id_int)
            except ValueError:
                return Response(
                    {"detail": "Параметр 'structure_id' должен быть целым числом >= 1."},
                    status=status.HTTP_422_UNPROCESSABLE_ENTITY,
                )

        task_status = request.query_params.get("status")
        if task_status:
            queryset = queryset.filter(status=task_status)

        items, meta = self._paginate_queryset(request, queryset)
        serializer = SolverTaskSerializer(items, many=True)
        return Response({"items": serializer.data, **meta}, status=status.HTTP_200_OK)


class SolverTaskDetailView(BaseComputeView):
    """
    GET /solver/tasks/{task_id} — Задача расчёта с журналом сходимости
    """

    def get(self, request: Request, task_id: uuid.UUID) -> Response:
        task = get_object_or_404(SolverTask, id=task_id)
        serializer = SolverTaskSerializer(task)
        return Response(serializer.data, status=status.HTTP_200_OK)


class SolverTaskCancelView(BaseComputeView):
    """
    POST /solver/tasks/{task_id}/cancel — Отменить задачу, освободить VRAM и удалить промежуточные данные
    """

    def post(self, request: Request, task_id: uuid.UUID) -> Response:
        task = get_object_or_404(SolverTask, id=task_id)

        if task.status in ("completed", "failed", "cancelled"):
            return Response(
                {"detail": f"Невозможно отменить задачу в терминальном статусе '{task.status}'."},
                status=status.HTTP_409_CONFLICT,
            )

        cancel_options = {}
        if request.data:
            req_serializer = TaskCancelRequestSerializer(data=request.data)
            if req_serializer.is_valid():
                cancel_options = req_serializer.validated_data

        updated_task = SolverTaskService.cancel_task(
            task=task,
            options=cancel_options,
            cancelled_by=request.user,
        )
        res_serializer = SolverTaskSerializer(updated_task)
        return Response(res_serializer.data, status=status.HTTP_202_ACCEPTED)


# =============================================================================
# 4. Карты физических полей, постраничная выгрузка, интерполяция и наборы кривых
# =============================================================================

class TaskFieldMapView(BaseComputeView):
    """
    GET /solver/tasks/{task_id}/fields/{quantity} — Метаданные карты и значения (для небольших сеток)
    """

    def get(self, request: Request, task_id: uuid.UUID, quantity: str) -> Response:
        task = get_object_or_404(SolverTask, id=task_id)
        field_map = FieldMapService.get_field_map(task=task, quantity=quantity)

        if field_map is None:
            return Response(
                {"detail": f"Карта поля '{quantity}' для задачи {task_id} не найдена."},
                status=status.HTTP_404_NOT_FOUND,
            )

        serializer = FieldMapSerializer(data=field_map)
        serializer.is_valid(raise_exception=True)
        return Response(serializer.data, status=status.HTTP_200_OK)


class TaskFieldMapPageView(BaseComputeView):
    """
    GET /solver/tasks/{task_id}/fields/{quantity}/pages — Постраничная выгрузка значений большой карты
    """

    MAX_PAGE_COUNT = 1_000_000
    DEFAULT_PAGE_COUNT = 65_536

    def get(self, request: Request, task_id: uuid.UUID, quantity: str) -> Response:
        task = get_object_or_404(SolverTask, id=task_id)

        try:
            offset = int(request.query_params.get("offset", 0))
            count = int(request.query_params.get("count", self.DEFAULT_PAGE_COUNT))
            if offset < 0 or count < 1 or count > self.MAX_PAGE_COUNT:
                raise ValueError()
        except ValueError:
            return Response(
                {
                    "detail": (
                        f"Некорректные параметры пагинации: offset >= 0, "
                        f"1 <= count <= {self.MAX_PAGE_COUNT}."
                    )
                },
                status=status.HTTP_422_UNPROCESSABLE_ENTITY,
            )

        page_data = FieldMapService.get_field_map_page(
            task=task,
            quantity=quantity,
            offset=offset,
            count=count,
        )
        if page_data is None:
            return Response(
                {"detail": f"Данные поля '{quantity}' для задачи {task_id} не найдены."},
                status=status.HTTP_404_NOT_FOUND,
            )

        serializer = FieldMapPageSerializer(data=page_data)
        serializer.is_valid(raise_exception=True)
        return Response(serializer.data, status=status.HTTP_200_OK)


class TaskFieldInterpolateView(BaseComputeView):
    """
    POST /solver/tasks/{task_id}/fields/interpolate — Интерполяция значения поля в произвольной точке (FR-GUI-05)
    """

    def post(self, request: Request, task_id: uuid.UUID) -> Response:
        request_id = self._get_request_id(request)
        task = get_object_or_404(SolverTask, id=task_id)

        req_serializer, error_response = self._validate_request(
            FieldInterpolationQuerySerializer, request.data, request_id
        )
        if error_response:
            return error_response

        interp_result = FieldMapService.interpolate_at_point(
            task=task,
            query_params=req_serializer.validated_data,
        )
        if interp_result is None:
            return Response(
                {"detail": "Поля задачи недоступны для интерполяции."},
                status=status.HTTP_404_NOT_FOUND,
            )

        res_serializer = FieldInterpolationResultSerializer(data=interp_result)
        res_serializer.is_valid(raise_exception=True)
        return Response(res_serializer.data, status=status.HTTP_200_OK)


class CharacteristicSetDetailView(BaseComputeView):
    """
    GET /solver/characteristic-sets/{set_id} — Сохранённый набор кривых для графиков GUI
    """

    def get(self, request: Request, set_id: int) -> Response:
        char_set = get_object_or_404(CharacteristicSet, id=set_id)
        serializer = CharacteristicSetSerializer(char_set)
        return Response(serializer.data, status=status.HTTP_200_OK)


# =============================================================================
# Маршрутизация (urls.py) для подключения в приложении compute
# =============================================================================
# from django.urls import path
#
# urlpatterns = [
#     path("solver/fem", FemCalculationView.as_view(), name="solver-fem"),
#     path("solver/quantum-poisson", QuantumPoissonCalculationView.as_view(), name="solver-quantum-poisson"),
#     path("solver/pde-characteristics", PdeCharacteristicsView.as_view(), name="solver-pde-characteristics"),
#     path("solver/memory-characteristics", MemoryCharacteristicsView.as_view(), name="solver-memory-characteristics"),
#     path("solver/characteristics/batch", CharacteristicsBatchView.as_view(), name="solver-characteristics-batch"),
#     path("solver/mesh-estimate", MeshEstimateView.as_view(), name="solver-mesh-estimate"),
#     path("solver/tasks", SolverTaskListView.as_view(), name="solver-tasks-list"),
#     path("solver/tasks/<uuid:task_id>", SolverTaskDetailView.as_view(), name="solver-task-detail"),
#     path("solver/tasks/<uuid:task_id>/cancel", SolverTaskCancelView.as_view(), name="solver-task-cancel"),
#     path("solver/tasks/<uuid:task_id>/fields/interpolate", TaskFieldInterpolateView.as_view(), name="solver-task-field-interpolate"),
#     path("solver/tasks/<uuid:task_id>/fields/<str:quantity>", TaskFieldMapView.as_view(), name="solver-task-field-map"),
#     path("solver/tasks/<uuid:task_id>/fields/<str:quantity>/pages", TaskFieldMapPageView.as_view(), name="solver-task-field-map-page"),
#     path("solver/characteristic-sets/<int:set_id>", CharacteristicSetDetailView.as_view(), name="solver-characteristic-set-detail"),
# ]
