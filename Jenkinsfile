pipeline {
    agent any

    environment {
        // Конфигурация Docker реестра в Nexus
        NEXUS_REGISTRY   = 'localhost:8082'
        APP_NAME         = 'tspice-laboratory-e2e'
        VERSION          = "1.0.0-${BUILD_NUMBER}"

        // Конфигурация RAW-репозитория для логов и статусов
        NEXUS_URL        = 'http://localhost:8081'
        RAW_REPOSITORY   = 'pipeline-statuses'

        // Учетные данные
        NEXUS_CREDENTIALS_ID = 'nexus-artifactory-credentials'
    }

    stages {
        stage('Pipeline Execution') {
            steps {
                script {
                    // Инициализируем статусы всех стейджей как NOT_STARTED
                    def stagesList = [
                        'Checkout',
                        'Build-Docker-Image',
                        'Push-to-Nexus',
                        'Deploy-to-Windows',
                        'Verify-Deployment',
                        'Health-Check'
                    ]

                    for (st in stagesList) {
                        sendStageStatus(st, 'NOT_STARTED', 'Stage has not been executed yet.')
                    }

                    // --- STAGE: Checkout ---
                    runStage('Checkout') {
                        git branch: 'main', url: 'https://github.com'
                    }

                    // --- STAGE: Build Docker Image ---
                    runStage('Build-Docker-Image') {
                        echo "Building image: ${NEXUS_REGISTRY}/${APP_NAME}:${VERSION}"
                        docker.build("${NEXUS_REGISTRY}/${APP_NAME}:${VERSION}")
                    }

                    // --- STAGE: Push to Nexus Registry ---
                    runStage('Push-to-Nexus') {
                        docker.withRegistry("http://${NEXUS_REGISTRY}", "${NEXUS_CREDENTIALS_ID}") {
                            docker.image("${NEXUS_REGISTRY}/${APP_NAME}:${VERSION}").push()
                        }
                    }

                    // --- STAGE: Deploy to Windows GPU Server ---
                    runStage('Deploy-to-Windows') {
                        sshagent(['windows-gpu-server-credentials']) {
                            sh """
                                ssh user@windows-server "
                                    docker stop ${APP_NAME} || true
                                    docker rm ${APP_NAME} || true
                                    docker pull ${NEXUS_REGISTRY}/${APP_NAME}:${VERSION}
                                    docker run -d \\
                                        --gpus all \\
                                        -p 8000:80 \\
                                        -v \\\\host.docker.internal\\data:/app/data \\
                                        --name ${APP_NAME} \\
                                        --restart unless-stopped \\
                                        ${NEXUS_REGISTRY}/${APP_NAME}:${VERSION}
                                "
                            """
                        }
                    }

                    // --- STAGE: Verify Deployment ---
                    runStage('Verify-Deployment') {
                        sleep 10
                        sh """
                            ssh user@windows-server "docker ps | grep ${APP_NAME}"
                        """
                    }

                    // --- STAGE: Health Check ---
                    runStage('Health-Check') {
                        def maxRetries = 5
                        def retryCount = 0
                        def isHealthy = false

                        while (retryCount < maxRetries && !isHealthy) {
                            def response = sh(script: "curl -f -s -o /dev/null -w '%{http_code}' http://windows-server:8000/health || echo 'failed'", returnStdout: true).trim()
                            if (response == '200') {
                                isHealthy = true
                            } else {
                                retryCount++
                                sleep 10
                            }
                        }

                        if (!isHealthy) {
                            error "Health check timed out. Application is not responding on port 8000."
                        }
                    }
                }
            }
        }
    }

    post {
        success {
            echo "Deployment successful!"
            slackSend channel: '#deployments', message: "✅ Deployment of ${APP_NAME}:${VERSION} successful!"
        }
        failure {
            echo "Deployment failed!"
            slackSend channel: '#deployments', message: "❌ Deployment of ${APP_NAME}:${VERSION} failed. Check Nexus for stage logs."
        }
    }
}

/**
 * Обертка для безопасного выполнения стейджа.
 * Принимает имя стейджа и блок кода. Если код падает, статус FAILURE мгновенно летит в Nexus.
 */
def runStage(String stageName, Closure body) {
    echo "=== Starting Stage: ${stageName} ==="
    // Обновляем статус на RUNNING перед стартом
    sendStageStatus(stageName, 'RUNNING', 'Stage is currently executing...')

    try {
        body()
        sendStageStatus(stageName, 'SUCCESS', 'Stage completed successfully.')
        echo "=== Completed Stage: ${stageName} ==="
    } catch (Exception e) {
        // Фиксируем точную причину падения
        def errorMessage = e.getMessage() ?: e.toString()
        sendStageStatus(stageName, 'FAILURE', "Error details:\n${errorMessage}")
        echo "=== FAILED Stage: ${stageName} ==="
        // Пробрасываем ошибку дальше, чтобы остановить пайплайн
        throw e
    }
}

/**
 * Функция отправки статусов и логов в Nexus Raw-репозиторий
 */
def sendStageStatus(String stageName, String status, String message) {
    // Безопасное извлечение креденшелов из Jenkins Store
    withCredentials([usernamePassword(credentialsId: "${NEXUS_CREDENTIALS_ID}", usernameVariable: 'NEXUS_USER', passwordVariable: 'NEXUS_PASSWORD')]) {
        def timestamp = sh(script: "date '+%Y-%m-%d %H:%M:%S'", returnStdout: true).trim()

        // Формируем структурированное содержимое лога
        def logContent = """=========================================
PROJECT:    ${APP_NAME}
BUILD:      #${BUILD_NUMBER}
STAGE:      ${stageName}
STATUS:     ${status}
TIMESTAMP:  ${timestamp}
=========================================
${message}
"""

        def fileName = "status-${stageName}.txt"
        writeFile file: fileName, text: logContent

        // Отправка в Nexus. Флаг -s -S убирает лишний прогресс-бар из консоли Jenkins
        sh """
            curl -s -S -u "${NEXUS_USER}:${NEXUS_PASSWORD}" \
            --upload-file ./${fileName} \
            "${NEXUS_URL}/repository/${RAW_REPOSITORY}/${APP_NAME}/${BUILD_NUMBER}/${fileName}"
        """
    }
}

