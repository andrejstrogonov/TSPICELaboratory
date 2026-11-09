pipeline {
    agent any

    environment {
        DOCKER_REGISTRY = 'my-registry.com'
        APP_NAME = 'tspice-laboratory-e2e'
        VERSION = '1.0.0'
    }

    stages {
        stage('Checkout') {
            steps {
                git branch: 'main', url: 'https://github.com/your-repo/tspice-laboratory.git'
            }
        }

        stage('Build Docker Image') {
            steps {
                script {
                    docker.build("${DOCKER_REGISTRY}/${APP_NAME}:${VERSION}")
                }
            }
        }

        stage('Push to Registry') {
            steps {
                script {
                    docker.withRegistry("https://${DOCKER_REGISTRY}", 'docker-hub-credentials') {
                        docker.image("${DOCKER_REGISTRY}/${APP_NAME}:${VERSION}").push()
                    }
                }
            }
        }

        stage('Deploy to Windows GPU Server') {
            steps {
                script {
                    // Подключение к Windows GPU серверу
                    sshagent(['windows-gpu-server-credentials']) {
                        sh '''
                            # Подключение к Windows серверу и деплой
                            echo "Deploying to Windows GPU Server..."

                            # Копирование образа на сервер (если нужно)
                            # docker save ${DOCKER_REGISTRY}/${APP_NAME}:${VERSION} | ssh user@windows-server "docker load"

                            # Запуск контейнера на Windows сервере с GPU поддержкой
                            ssh user@windows-server "
                                docker run -d \
                                --gpus all \
                                -p 8000:8000 \
                                -v \\\\host.docker.internal\\data:/app/data \
                                --name ${APP_NAME} \
                                ${DOCKER_REGISTRY}/${APP_NAME}:${VERSION}
                            "
                        '''
                    }
                }
            }
        }

        stage('Verify Deployment') {
            steps {
                script {
                    sleep 10
                    sh '''
                        ssh user@windows-server "
                            docker ps | grep ${APP_NAME}
                        "
                    '''
                }
            }
        }

        stage('Health Check') {
            steps {
                script {
                    def maxRetries = 5
                    def retryCount = 0

                    while (retryCount < maxRetries) {
                        try {
                            def response = sh(script: "curl -f http://windows-server:8000/health || exit 1", returnStatus: true)
                            if (response == 0) {
                                echo "Health check passed"
                                break
                            } else {
                                echo "Health check failed, retrying..."
                                retryCount++
                                sleep 10
                            }
                        } catch (Exception e) {
                            echo "Health check failed: ${e}"
                            retryCount++
                            sleep 10
                        }
                    }
                }
            }
        }
    }

    post {
        success {
            echo "Deployment successful!"
            // Отправка уведомления
            slackSend channel: '#deployments', message: "✅ Deployment of ${APP_NAME} successful!"
        }
        failure {
            echo "Deployment failed!"
            // Отправка уведомления об ошибке
            slackSend channel: '#deployments', message: "❌ Deployment of ${APP_NAME} failed!"
        }
    }
}
