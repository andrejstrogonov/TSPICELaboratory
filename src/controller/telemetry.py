import json
import time
from django.http import StreamingHttpResponse
from rest_framework import status
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from rest_framework.throttling import UserRateThrottle
from rest_framework.views import APIView

# Предполагаемые сервисы и сериализаторы вашего проекта
from .serializers import (
    TelemetrySnapshotSerializer,
    NodeTelemetrySerializer,
)
from .services import TelemetryCollectorService, TelemetryStreamService


class TelemetryStreamThrottle(UserRateThrottle):
    """
    Ограничение частоты подключений к SSE-потоку телеметрии (429 Too Many Requests).
    Настраивается в settings.py -> REST_FRAMEWORK['DEFAULT_THROTTLE_RATES']['telemetry_stream'].
    """
    scope = 'telemetry_stream'


class GpuTelemetryView(APIView):
    """
    GET /telemetry/gpu
    Снимок телеметрии GPU (загрузка, VRAM, температура, питание).
    Если данных нет, возвращает source='unavailable' со статусом 200 OK.
    """
    permission_classes = [IsAuthenticated]

    def get(self, request):
        snapshot = TelemetryCollectorService.get_gpu_snapshot()

        # Если GPU отсутствует или драйвер недоступен, сервис возвращает заглушку
        if not snapshot:
            snapshot = {
                "source": "unavailable",
                "devices": []
            }

        serializer = TelemetrySnapshotSerializer(data=snapshot)
        serializer.is_valid(raise_exception=True)
        return Response(serializer.data, status=status.HTTP_200_OK)


class NodeTelemetryView(APIView):
    """
    GET /telemetry/node
    Снимок телеметрии узла (CPU, память, диск расчётной станции).
    """
    permission_classes = [IsAuthenticated]

    def get(self, request):
        node_data = TelemetryCollectorService.get_node_snapshot()
        serializer = NodeTelemetrySerializer(data=node_data)
        serializer.is_valid(raise_exception=True)
        return Response(serializer.data, status=status.HTTP_200_OK)


class TelemetryStreamView(APIView):
    """
    GET /telemetry/stream
    Поток кадров телеметрии (SSE) с монотонным sequence для панели мониторинга GUI.
    """
    permission_classes = [IsAuthenticated]
    throttle_classes = [TelemetryStreamThrottle]  # Вернёт 429 при превышении лимита

    def get(self, request):
        # Поддерживаем как query-параметр SinceStream, так и стандартный заголовок SSE
        since_param = request.query_params.get('since') or request.headers.get('Last-Event-ID')
        try:
            start_sequence = int(since_param) if since_param is not None else 0
        except ValueError:
            start_sequence = 0

        def event_stream():
            """
            Генератор кадров телеметрии.
            Внутри сервиса желательно использовать asyncio или подписку на Redis/цикл опроса.
            """
            sequence = start_sequence
            try:
                for frame in TelemetryStreamService.stream_frames(since_sequence=start_sequence):
                    sequence = frame.get("sequence", sequence + 1)
                    frame["sequence"] = sequence

                    payload = json.dumps(frame, ensure_ascii=False)
                    yield f"id: {sequence}\n"
                    yield "event: telemetry\n"
                    yield f"data: {payload}\n\n"
            except GeneratorExit:
                # Клиент закрыл соединение (закрыта вкладка GUI)
                TelemetryStreamService.on_client_disconnect(request.user)

        response = StreamingHttpResponse(
            event_stream(),
            content_type='text/event-stream'
        )
        response['Cache-Control'] = 'no-cache'
        response['X-Accel-Buffering'] = 'no'  # Отключает буферизацию в Nginx
        return response
