#!/usr/bin/env bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
mkdir -p $DIR/_build
cp $DIR/../environment.yml $DIR/_build/
docker build -t dementrock/deeprlbootcamp $DIR
