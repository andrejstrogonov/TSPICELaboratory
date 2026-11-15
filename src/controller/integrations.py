from rest_framework import serializers, status
from rest_framework.permissions import BasePermission, IsAuthenticated
from rest_framework.response import Response
from rest_framework.views import APIView

# Предполагаемые сериализаторы и сервисы вашего проекта
from .serializers import EdaIntegrationSerializer
from .services import EdaIntegrationService, EdaToolUnavailableError


SUPPORTED_SIMULATORS = {"ngspice", "microcap", "openvaf", "tanner_tspice"}


class IsAdminRole(BasePermission):
    """
    Разрешает доступ только пользователям с ролью 'admin' (403 Forbidden для остальных).
    """
    message = "Изменение настроек EDA-интеграций доступно только роли admin."

    def has_permission(self, request, view):
        user = request.user
        if not user or not user.is_authenticated:
            return False
        # Проверяем либо стандартный флаг is_staff, либо кастомное поле role
        return getattr(user, "role", None) == "admin" or user.is_staff


class EdaIntegrationUpdateRequestSerializer(serializers.Serializer):
    """
    Встроенная валидация тела запроса PUT /integrations/eda/{simulator} строго по контракту.
    """
    executable_path = serializers.CharField(max_length=512, required=True, allow_blank=False)
    timeout_s = serializers.IntegerField(min_value=1, max_value=3600, default=300, required=False)


def validate_simulator_param(simulator: str):
    """
    Проверяет, что переданный в URL симулятор входит в допустимый enum EdaSimulator.
    """
    if simulator not in SUPPORTED_SIMULATORS:
        return Response(
            {
                "detail": f"Неизвестный симулятор '{simulator}'.",
                "allowed": sorted(SUPPORTED_SIMULATORS),
            },
            status=status.HTTP_422_UNPROCESSABLE_ENTITY,
        )
    return None


class EdaIntegrationListView(APIView):
    """
    GET /integrations/eda
    Возвращает список настроек подключения ко всем EDA-симуляторам (без секретов).
    """
    permission_classes = [IsAuthenticated]

    def get(self, request):
        integrations = EdaIntegrationService.list_all()
        serializer = EdaIntegrationSerializer(integrations, many=True)
        return Response(serializer.data, status=status.HTTP_200_OK)


class EdaIntegrationDetailView(APIView):
    """
    PUT /integrations/eda/{simulator} — Обновить настройки и выполнить проверку запуска (admin)
    DELETE /integrations/eda/{simulator} — Сбросить настройки симулятора (admin)
    """
    permission_classes = [IsAuthenticated, IsAdminRole]

    def put(self, request, simulator: str):
        sim_error = validate_simulator_param(simulator)
        if sim_error:
            return sim_error

        req_serializer = EdaIntegrationUpdateRequestSerializer(data=request.data)
        if not req_serializer.is_valid():
            return Response(
                {"detail": "Ошибка валидации настроек симулятора.", "errors": req_serializer.errors},
                status=status.HTTP_422_UNPROCESSABLE_ENTITY,
            )

        # Сохраняем настройки и сразу выполняем тестовый запуск
        integration_data = EdaIntegrationService.update_and_check(
            simulator=simulator,
            executable_path=req_serializer.validated_data["executable_path"],
            timeout_s=req_serializer.validated_data["timeout_s"],
            updated_by=request.user,
        )

        res_serializer = EdaIntegrationSerializer(data=integration_data)
        res_serializer.is_valid(raise_exception=True)
        return Response(res_serializer.data, status=status.HTTP_200_OK)

    def delete(self, request, simulator: str):
        sim_error = validate_simulator_param(simulator)
        if sim_error:
            return sim_error

        EdaIntegrationService.reset(simulator=simulator, reset_by=request.user)
        return Response(status=status.HTTP_204_NO_CONTENT)


class EdaIntegrationCheckView(APIView):
    """
    POST /integrations/eda/{simulator}/check
    Запускает симулятор с тестовой командой и сообщает версию или причину недоступности.
    """
    permission_classes = [IsAuthenticated]

    def post(self, request, simulator: str):
        sim_error = validate_simulator_param(simulator)
        if sim_error:
            return sim_error

        try:
            check_result = EdaIntegrationService.check_availability(simulator=simulator)
        except EdaToolUnavailableError as exc:
            return Response(
                {
                    "code": "EDA_TOOL_UNAVAILABLE",
                    "detail": str(exc),
                    "simulator": simulator,
                },
                status=status.HTTP_503_SERVICE_UNAVAILABLE,
            )

        res_serializer = EdaIntegrationSerializer(data=check_result)
        res_serializer.is_valid(raise_exception=True)
        return Response(res_serializer.data, status=status.HTTP_200_OK)
