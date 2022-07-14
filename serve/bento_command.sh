#!/bin/bash

TOKEN=$1

bentoml yatai login --api-token $TOKEN --endpoint=http://116.47.188.227:30080
bentoml build
bentoml push surface_clf:latest
