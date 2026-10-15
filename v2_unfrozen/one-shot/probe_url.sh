#!/bin/bash
URL='https://huggingface.co/datasets/openslr/librispeech_asr/resolve/refs%2Fconvert%2Fparquet/clean/train.100/0000.parquet'
curl -sIL -o /dev/null -w '%{http_code} %{size_download} %{url_effective}\n' "$URL"
