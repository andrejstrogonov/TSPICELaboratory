pipeline {
    agent any

    environment {
        // Конфигурация Docker реестра в Nexus
        NEXUS_REGISTRY       = 'localhost:8082'
        APP_NAME             = 'tspice-laboratory-e2e'
        VERSION              = "1.0.0-${BUILD_NUMBER}"

        // Конфигурация RAW-репозитория для логов и статусов
        NEXUS_URL            = 'http://localhost:8081'
        RAW_REPOSITORY       = 'pipeline-statuses'

        // Учетные данные в Jenkins Credentials Store
        NEXUS_CREDENTIALS_ID = 'nexus-artifactory-credentials'
        NEO4J_CREDENTIALS_ID = 'neo4j-prod-credentials'
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
                        withCredentials([usernamePassword(credentialsId: "${NEO4J_CREDENTIALS_ID}", usernameVariable: 'NEO4J_USER', passwordVariable: 'NEO4J_PASS')]) {
                            sshagent(['windows-gpu-server-credentials']) {
                                sh """
                                    ssh user@windows-server "
                                        docker stop ${APP_NAME} || true
                                        docker rm ${APP_NAME} || true
                                        docker pull ${NEXUS_REGISTRY}/${APP_NAME}:${VERSION}
                                        docker run -d \\
                                            --gpus all \\
                                            -p 8000:8000 \\
                                            -p 7474:7474 \\
                                            -p 7687:7687 \\
                                            -e NEO4J_AUTH=${NEO4J_USER}/${NEO4J_PASS} \\
                                            -v \\\\host.docker.internal\\data:/app/data \\
                                            -v \\\\host.docker.internal\\neo4j-data:/var/lib/neo4j/data \\
                                            --name ${APP_NAME} \\
                                            --restart unless-stopped \\
                                            ${NEXUS_REGISTRY}/${APP_NAME}:${VERSION}
                                    "
                                """
                            }
                        }
                    }

                    // --- STAGE: Verify Deployment ---
                    runStage('Verify-Deployment') {
                        sleep 10
                        sh """
                            ssh user@windows-server "docker ps | grep ${APP_NAME}"
                        """
                    }

                    // --- STAGE: Health Check (Django + Neo4j) ---
                    runStage('Health-Check') {
                        def maxRetries = 8
                        def retryCount = 0
                        def isHealthy = false

                        while (retryCount < maxRetries && !isHealthy) {
                            def appStatus = sh(script: "curl -f -s -o /dev/null -w '%{http_code}' http://windows-server:8000/health || echo 'failed'", returnStdout: true).trim()
                            def neo4jStatus = sh(script: "curl -f -s -o /dev/null -w '%{http_code}' http://windows-server:7474 || echo 'failed'", returnStdout: true).trim()

                            if (appStatus == '200' && neo4jStatus == '200') {
                                isHealthy = true
                            } else {
                                echo "Waiting for services (Attempt ${retryCount + 1}/${maxRetries}): Django=${appStatus}, Neo4j=${neo4jStatus}"
                                retryCount++
                                sleep 10
                            }
                        }

                        if (!isHealthy) {
                            error "Health check timed out. Django (8000) or Neo4j (7474) is not responding."
                        }
                    }
                }
            }
        }
    }

    post {
        success {
            echo "Deployment successful!"
            slackSend channel: '#deployments', message: "✅ Deployment of ${APP_NAME}:${VERSION} (with Neo4j) successful!"
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
    sendStageStatus(stageName, 'RUNNING', 'Stage is currently executing...')

    try {
        body()
        sendStageStatus(stageName, 'SUCCESS', 'Stage completed successfully.')
        echo "=== Completed Stage: ${stageName} ==="
    } catch (Exception e) {
        def errorMessage = e.getMessage() ?: e.toString()
        sendStageStatus(stageName, 'FAILURE', "Error details:\n${errorMessage}")
        echo "=== FAILED Stage: ${stageName} ==="
        throw e
    }
}

/**
 * Функция отправки статусов и логов в Nexus Raw-репозиторий
 */
def sendStageStatus(String stageName, String status, String message) {
    withCredentials([usernamePassword(credentialsId: "${NEXUS_CREDENTIALS_ID}", usernameVariable: 'NEXUS_USER', passwordVariable: 'NEXUS_PASSWORD')]) {
        def timestamp = sh(script: "date '+%Y-%m-%d %H:%M:%S'", returnStdout: true).trim()

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

        sh """
            curl -s -S -u "${NEXUS_USER}:${NEXUS_PASSWORD}" \
            --upload-file ./${fileName} \
            "${NEXUS_URL}/repository/${RAW_REPOSITORY}/${APP_NAME}/${BUILD_NUMBER}/${fileName}"
        """
    }
}

