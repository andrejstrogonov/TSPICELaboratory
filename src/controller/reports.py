import os
import uuid
from django.core.cache import cache
from django.http import FileResponse
from django.shortcuts import get_object_or_404
from rest_framework import status
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from rest_framework.views import APIView

# Предполагаемые модели, сериализаторы и сервисы вашего проекта
from .models import Report
from .serializers import (
    ReportSerializer,
    ReportDetailSerializer,
    ReportRequestSerializer,
)
from .services import (
    ReportGenerationService,
    ReportStorageService,
    LatexServiceUnavailableError,
)


class IdempotencyMixin:
    """
    Миксин для обработки заголовков Idempotency-Key и X-Request-Id.
    """
    IDEMPOTENCY_TTL = 60 * 60 * 24  # 24 часа

    def _get_request_id(self, request):
        return request.headers.get('X-Request-Id', str(uuid.uuid4()))

    def _get_cached_response(self, request, endpoint_prefix):
        idem_key = request.headers.get('Idempotency-Key')
        if not idem_key:
            return None
        cache_key = f"idem:{endpoint_prefix}:{request.user.id}:{idem_key}"
        return cache.get(cache_key)

    def _save_cached_response(self, request, endpoint_prefix, payload, status_code):
        idem_key = request.headers.get('Idempotency-Key')
        if not idem_key:
            return
        cache_key = f"idem:{endpoint_prefix}:{request.user.id}:{idem_key}"
        cache.set(cache_key, {"data": payload, "status": status_code}, self.IDEMPOTENCY_TTL)


def validate_422(serializer):
    """
    Преобразует ошибки валидации DRF (400) в 422 Unprocessable Entity согласно контракту.
    """
    if not serializer.is_valid():
        return Response(
            {"detail": "Ошибка валидации параметров отчёта", "errors": serializer.errors},
            status=status.HTTP_422_UNPROCESSABLE_ENTITY
        )
    return None


class ReportListCreateView(IdempotencyMixin, APIView):
    """
    GET /reports — Список отчётов с фильтрацией и пагинацией
    POST /reports — Сгенерировать LaTeX-отчёт (асинхронная сборка PDF)
    """
    permission_classes = [IsAuthenticated]

    def get(self, request):
        queryset = Report.objects.all().order_by('-created_at')

        # Фильтр по structure_id (integer >= 1)
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

        # Фильтр по статусу (StatusQuery)
        report_status = request.query_params.get('status')
        if report_status:
            queryset = queryset.filter(status=report_status)

        # Пагинация
        page = max(1, int(request.query_params.get('page', 1)))
        page_size = max(1, int(request.query_params.get('page_size', 20)))
        offset = (page - 1) * page_size

        total = queryset.count()
        items = queryset[offset:offset + page_size]

        serializer = ReportSerializer(items, many=True)
        return Response({
            "items": serializer.data,
            "total": total,
            "page": page,
            "page_size": page_size
        }, status=status.HTTP_200_OK)

    def post(self, request):
        request_id = self._get_request_id(request)

        # Проверка идемпотентности
        cached = self._get_cached_response(request, "reports_create")
        if cached:
            return Response(cached["data"], status=cached["status"], headers={'X-Request-Id': request_id})

        req_serializer = ReportRequestSerializer(data=request.data)
        error_response = validate_422(req_serializer)
        if error_response:
            return error_response

        try:
            # Ставим задачу сборки LaTeX и компиляции PDF в очередь (юнит report)
            report = ReportGenerationService.enqueue_report(
                validated_data=req_serializer.validated_data,
                user=request.user,
                request_id=request_id
            )
        except LatexServiceUnavailableError as exc:
            return Response(
                {"detail": f"Сервис генерации отчётов LaTeX недоступен: {str(exc)}"},
                status=status.HTTP_503_SERVICE_UNAVAILABLE,
                headers={'X-Request-Id': request_id}
            )

        res_serializer = ReportSerializer(report)
        self._save_cached_response(request, "reports_create", res_serializer.data, status.HTTP_202_ACCEPTED)

        return Response(
            res_serializer.data,
            status=status.HTTP_202_ACCEPTED,
            headers={'X-Request-Id': request_id}
        )


class ReportDetailView(APIView):
    """
    GET /reports/{report_id} — Метаданные отчёта и исходный код LaTeX
    """
    permission_classes = [IsAuthenticated]

    def get(self, request, report_id):
        report = get_object_or_404(Report, id=report_id)  # Вернёт 404 NotFound автоматически
        serializer = ReportDetailSerializer(report)
        return Response(serializer.data, status=status.HTTP_200_OK)


class ReportDownloadView(APIView):
    """
    GET /reports/{report_id}/download — Скачать скомпилированный PDF отчёта
    """
    permission_classes = [IsAuthenticated]

    def get(self, request, report_id):
        report = get_object_or_404(Report, id=report_id)

        # Если PDF ещё компилируется или сборка завершилась ошибкой, отдаём 409 Conflict
        if report.status != 'completed' or not report.pdf_path:
            return Response(
                {"detail": f"PDF отчёта недоступен для скачивания (текущий статус: '{report.status}')."},
                status=status.HTTP_409_CONFLICT
            )

        # Получаем файловый дескриптор из локального хранилища или S3/MinIO
        pdf_file = ReportStorageService.open_pdf(report)
        if pdf_file is None:
            return Response(
                {"detail": "Файл PDF не найден в хранилище."},
                status=status.HTTP_404_NOT_FOUND
            )

        filename = f"report_{report.structure_id}_{report.id}.pdf"
        return FileResponse(
            pdf_file,
            as_attachment=True,
            filename=filename,
            content_type='application/pdf'
        )
