pipeline {
  agent {
    kubernetes {
      defaultContainer 'tools'
      yaml """
apiVersion: v1
kind: Pod
spec:
  restartPolicy: Never
  containers:
    - name: tools
      image: python:3.11-slim
      command:
        - cat
      tty: true
    - name: kaniko
      image: gcr.io/kaniko-project/executor:debug
      command:
        - /busybox/cat
      tty: true
      volumeMounts:
        - name: kaniko-docker-config
          mountPath: /kaniko/.docker
  volumes:
    - name: kaniko-docker-config
      emptyDir: {}
"""
    }
  }

  options {
    timestamps()
    disableConcurrentBuilds()
    skipDefaultCheckout(true)
  }

  parameters {
    string(name: 'IMAGE_REPOSITORY', defaultValue: 'amdp-registry.skala-ai.com/skala26a-cloud/onramp-api', description: 'Harbor image repository for onramp-api')
    string(name: 'GITOPS_REPOSITORY', defaultValue: 'https://github.com/OnRamp-2026/gitops.git', description: 'GitOps repository URL')
  }

  environment {
    GITOPS_VALUES_FILE = 'apps/onramp-api/values-dev.yaml'
  }

  stages {
    stage('Prepare Tools') {
      steps {
        sh '''
          set -eu
          apt-get update
          apt-get install -y --no-install-recommends git ca-certificates
          rm -rf /var/lib/apt/lists/*
        '''
      }
    }

    stage('Checkout') {
      steps {
        checkout scm
        script {
          env.IMAGE_TAG = sh(script: 'git rev-parse --short=12 HEAD', returnStdout: true).trim()
        }
      }
    }

    stage('Lint and Test') {
      steps {
        sh '''
          python -m venv .venv
          . .venv/bin/activate
          pip install --upgrade pip
          pip install ".[dev]"
          ruff format --check app tests
          ruff check app tests
          pytest tests/unit -v
        '''
      }
    }

    stage('Build and Push Image') {
      steps {
        container('kaniko') {
          withCredentials([usernamePassword(
            credentialsId: 'harbor-robot-credential',
            usernameVariable: 'HARBOR_USERNAME',
            passwordVariable: 'HARBOR_PASSWORD'
          )]) {
            sh '''
              set -eu
              REGISTRY_HOST="${IMAGE_REPOSITORY%%/*}"
              AUTH="$(printf '%s:%s' "${HARBOR_USERNAME}" "${HARBOR_PASSWORD}" | base64 | tr -d '\\n')"
              cat > /kaniko/.docker/config.json <<EOF
{"auths":{"${REGISTRY_HOST}":{"auth":"${AUTH}"}}}
EOF
              /kaniko/executor \
                --context "${WORKSPACE}" \
                --dockerfile "${WORKSPACE}/Dockerfile" \
                --destination "${IMAGE_REPOSITORY}:${IMAGE_TAG}" \
                --digest-file "${WORKSPACE}/image-digest.txt"
            '''
          }
        }
        script {
          env.IMAGE_DIGEST = readFile('image-digest.txt').trim()
          env.FULL_IMAGE = "${env.IMAGE_REPOSITORY}@${env.IMAGE_DIGEST}"
          echo "Built image: ${env.FULL_IMAGE}"
        }
      }
    }

    stage('Update GitOps Image Digest') {
      when {
        branch 'main'
      }
      steps {
        withCredentials([usernamePassword(
          credentialsId: 'github-gitops-write-token',
          usernameVariable: 'GITOPS_USERNAME',
          passwordVariable: 'GITOPS_TOKEN'
        )]) {
          sh '''
            set -eu
            rm -rf gitops
            ENCODED_GITOPS_USERNAME="$(python -c 'import os, urllib.parse; print(urllib.parse.quote(os.environ["GITOPS_USERNAME"], safe=""))')"
            ENCODED_GITOPS_TOKEN="$(python -c 'import os, urllib.parse; print(urllib.parse.quote(os.environ["GITOPS_TOKEN"], safe=""))')"
            AUTHED_REPO="$(printf '%s' "${GITOPS_REPOSITORY}" | sed "s#https://#https://${ENCODED_GITOPS_USERNAME}:${ENCODED_GITOPS_TOKEN}@#")"
            git clone "${AUTHED_REPO}" gitops
            cd gitops
            git config user.name "onramp-jenkins"
            git config user.email "onramp-jenkins@users.noreply.github.com"
            python - <<'PY'
from pathlib import Path
import os
import re

path = Path(os.environ["GITOPS_VALUES_FILE"])
text = path.read_text()
text = re.sub(r"(repository:\s*).+", lambda m: m.group(1) + os.environ["IMAGE_REPOSITORY"], text, count=1)
text = re.sub(r"(tag:\s*).+", lambda m: m.group(1) + os.environ["IMAGE_TAG"], text, count=1)
text = re.sub(r"(digest:\s*).+", lambda m: m.group(1) + os.environ["IMAGE_DIGEST"], text, count=1)
path.write_text(text)
PY
            git diff -- "${GITOPS_VALUES_FILE}"
            if git diff --quiet -- "${GITOPS_VALUES_FILE}"; then
              echo "No GitOps image digest change."
              exit 0
            fi
            git add "${GITOPS_VALUES_FILE}"
            git commit -m "chore: update onramp-api image ${IMAGE_TAG} [skip ci]"
            git push origin main
          '''
        }
      }
    }
  }

  post {
    always {
      sh 'rm -rf .venv gitops image-digest.txt || true'
    }
  }
}
