fworkspace "TSpice Laboratory E2E" "Архитектурная модель системы моделирования полупроводников" {

    model {
        engineer = person "Инженер-исследователь" "Пользователь системы, задает параметры структур и запускает расчеты"
        qa = person "QA Инженер" "Тестирует систему на соответствие стандартам ISTQB"

        softwareSystem = softwareSystem "TSpice Laboratory System" "Главный программный комплекс" {
            ui = container "PyQt6 GUI" "Графический интерфейс пользователя" "Python, PyQt6"
            apiGateway = container "REST API Gateway" "Точка входа для интеграции и UI" "FastAPI, /api/v1"
            backendApp = container "FastAPI Server" "Бизнес-логика, оркестрация расчетов, изоляция VRAM OOM" "Python, FastAPI" {
                meshEngine = component "TCAD Mesh & FEM Engine" "Генерация сетки и решение уравнения Лапласа" "Python, SciPy"
                optEngine = component "Optimization Engine" "Многопараметрический фиттинг моделей" "Python, scipy.optimize"
                pinnModel = component "PINN Inference Model" "Нейросетевое ускорение поиска стартовых точек" "PyTorch FP64"
                latexGen = component "LaTeX Report Generator" "Сборка отчетных PDF документов" "Python, Jinja2/LaTeX"
            }
            database = container "SQLite Database" "Хранение данных сессий и параметров структур" "SQLite file"
            
            # Внешние симуляторы
            ngspice = container "Ngspice Component" "Проверка Verilog-A и Loopback-тестирование" "Ngspice / OpenVAF"
            microcap = container "Micro-Cap 12 Wrapper" "Финальная верификация сквозного прогона нетлиста" "Micro-Cap CLI"
        }

        # Инфраструктура развертывания
        jenkins = deploymentNode "Jenkins Server" {
            deploymentInstance dockerBuild = infrastructureNode "Docker Engine (Build)" "Сборка образов и деплой через SSH"
        }

        winServer = deploymentNode "Windows Server Compute Node" {
            gpu = infrastructureNode "NVIDIA GeForce RTX 4090" "Аппаратное ускорение вычислений" "Hardware GPU"
            cuda = infrastructureNode "NVIDIA CUDA Runtime" "Среда исполнения FP64 матриц" "CUDA 12.x"
            
            dockerWin = deploymentNode "Docker Engine Windows" {
                containerInstance appContainer = softwareSystem.apiGateway
                # Контейнер включает в себя и бэкенд с БД
            }
        }

        # Связи между пользователями и компонентами
        engineer -> ui "Использует для ввода параметров"
        ui -> apiGateway "Вызывает эндпоинты по HTTP/JSON"
        apiGateway -> backendApp "Перенаправляет запросы"
        
        backendApp.optEngine -> backendApp.pinnModel "Запрашивает стартовые приближения"
        backendApp.optEngine -> microcap "Проводит финальный прогон нетлиста"
        backendApp.optEngine -> ngspice "Выполняет проверочные тесты"
        
        backendApp -> database "Записывает результаты сессий"
        
        # Инфраструктурные связи деплоя
        dockerBuild -> dockerWin "Деплоит контейнеры по SSH"
        dockerWin -> cuda "Пробрасывает доступ к ресурсам"
        cuda -> gpu "Использует для расчетов"
    }

    views {
        container softwareSystem "Containers_View" "Диаграмма контейнеров архитектуры системы" {
            include *
            autolayout lr
        }
        
        component backendApp "Components_View" "Внутреннее устройство FastAPI сервера" {
            include *
            autolayout topbottom
        }

        theme default
    }
}


