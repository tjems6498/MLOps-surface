def component = [
		Base: true,
		Preprocess: true,
		Hyper: true,
		Train: true,
		Test: true,
		Serve: true
]

pipeline {
	agent any
	stages {
		stage("Checkout") {
			steps {
				checkout scm
			}
		}
		stage("Build and Push") {
			steps {
                script {
					component.each{ entry ->
						stage ("${entry.key}"){
							if (entry.value){
								var = entry.key
								withCredentials([[$class: 'UsernamePasswordMultiBinding',
								credentialsId: 'docker_credentials',
								usernameVariable: 'DOCKER_USER_ID',
								passwordVariable: 'DOCKER_USER_PASSWORD'
								]]){
							    sh "docker-compose build ${var.toLowerCase()}"
								sh "docker tag surface_pipeline_${var.toLowerCase()}:latest ${DOCKER_USER_ID}/surface_pipeline_${var.toLowerCase()}:${BUILD_NUMBER}"
								sh "docker login -u ${DOCKER_USER_ID} -p ${DOCKER_USER_PASSWORD}"
								sh "docker push ${DOCKER_USER_ID}/surface_pipeline_${var.toLowerCase()}:${BUILD_NUMBER}"
								}
							}
						}
					}
				}
			}	


		}
	}
}
