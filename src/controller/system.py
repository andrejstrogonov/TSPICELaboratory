import json
import time
import uuid
from django.http import StreamingHttpResponse
from django.shortcuts import get_object_or_404
from rest_framework import status
from rest_framework.permissions import AllowAny, IsAuthenticated
from rest_framework.response import Response
from rest_framework.views import APIView

# Предполагаемые модели и сервисы вашего проекта
from .models import Job
from .serializers import JobSerializer, HealthStatusSerializer
from .services import SystemHealthService, JobCancellationService, JobEventStreamService


class HealthCheckView(APIView):
    """
    GET /health
    Состояние сервиса и его зависимостей (БД, хранилище, CUDA, EDA, LaTeX).
    """
    permission_classes = [AllowAny]  # security: []

    def get(self, request):
        health_data, is_healthy = SystemHealthService.check_all()
        serializer = HealthStatusSerializer(data=health_data)
        serializer.is_valid(raise_exception=True)

        response_status = status.HTTP_200_OK if is_healthy else status.HTTP_503_SERVICE_UNAVAILABLE
        return Response(serializer.data, status=response_status)


class ApiVersionView(APIView):
    """
    GET /version
    Версия контракта API для проверки совместимости GUI.
    """
    permission_classes = [AllowAny]  # security: []

    def get(self, request):
        data = {
            "api_version": "v1",
            "contract_version": "1.0.0",
            "git_commit": "a1b2c3d"  # Можно подтягивать из env или настроек Django
        }
        return Response(data, status=status.HTTP_200_OK)


class JobListView(APIView):
    """
    GET /jobs
    Журнал фоновых задач с пагинацией, сортировкой и фильтрацией.
    """
    permission_classes = [IsAuthenticated]  # 401 Unauthorized при отсутствии токена

    def get(self, request):
        queryset = Job.objects.all()

        # Фильтрация по query-параметрам
        kind = request.query_params.get('kind')
        job_status = request.query_params.get('status')
        since_stream = request.query_params.get('since_stream')
        sort_by = request.query_params.get('sort', '-created_at')

        if kind:
            queryset = queryset.filter(kind=kind)
        if job_status:
            queryset = queryset.filter(status=job_status)
        if since_stream:
            queryset = queryset.filter(created_at__gte=since_stream)

        queryset = queryset.order_by(sort_by)

        # Базовая пагинация (можно заменить на стандартный PageNumberPagination DRF)
        page = int(request.query_params.get('page', 1))
        page_size = int(request.query_params.get('page_size', 20))
        offset = (page - 1) * page_size

        total = queryset.count()
        items = queryset[offset:offset + page_size]

        serializer = JobSerializer(items, many=True)
        return Response({
            "items": serializer.data,
            "total": total,
            "page": page,
            "page_size": page_size
        }, status=status.HTTP_200_OK)


class JobDetailView(APIView):
    """
    GET /jobs/{job_id} — Состояние задачи
    DELETE /jobs/{job_id} — Отмена задачи
    """
    permission_classes = [IsAuthenticated]

    def _get_request_id(self, request):
        return request.headers.get('X-Request-Id', str(uuid.uuid4()))

    def get(self, request, job_id):
        job = get_object_or_404(Job, id=job_id)  # Вернет 404 NotFound автоматически
        serializer = JobSerializer(job)

        response = Response(serializer.data, status=status.HTTP_200_OK)
        response['X-Request-Id'] = self._get_request_id(request)
        return response

    def delete(self, request, job_id):
        job = get_object_or_404(Job, id=job_id)
        request_id = self._get_request_id(request)

        # Если задача уже в терминальном состоянии, отмена невозможна (409 Conflict)
        if job.status in ['completed', 'failed', 'cancelled']:
            return Response(
                {"detail": f"Невозможно отменить задачу в статусе '{job.status}'."},
                status=status.HTTP_409_CONFLICT,
                headers={'X-Request-Id': request_id}
            )

        # Публикуем cancel_requested в брокер/очередь
        updated_job = JobCancellationService.request_cancel(job)
        serializer = JobSerializer(updated_job)

        return Response(
            serializer.data,
            status=status.HTTP_202_ACCEPTED,
            headers={'X-Request-Id': request_id}
        )


class JobEventsStreamView(APIView):
    """
    GET /jobs/{job_id}/events
    Поток прогресса задачи через Server-Sent Events (text/event-stream).
    """
    permission_classes = [IsAuthenticated]

    def get(self, request, job_id):
        job = get_object_or_404(Job, id=job_id)
        last_event_id = request.headers.get('Last-Event-ID')
        request_id = request.headers.get('X-Request-Id', str(uuid.uuid4()))

        def event_stream():
            """
            Генератор SSE-событий.
            В реальном проекте лучше читать из Redis Pub/Sub или БД.
            """
            for event in JobEventStreamService.listen(job_id=job.id, last_event_id=last_event_id):
                event_id = event['id']
                event_type = event['type']  # 'progress' или 'done'
                payload = json.dumps(event['data'], ensure_ascii=False)

                yield f"id: {event_id}\n"
                yield f"event: {event_type}\n"
                yield f"data: {payload}\n\n"

                if event_type == 'done':
                    break

        response = StreamingHttpResponse(
            event_stream(),
            content_type='text/event-stream'
        )
        response['Cache-Control'] = 'no-cache'
        response['X-Accel-Buffering'] = 'no'  # Отключает буферизацию в Nginx
        response['X-Request-Id'] = request_id
        return response
