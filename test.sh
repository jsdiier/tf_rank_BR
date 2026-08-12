#!/bin/bash

source /etc/profile
source ./common.conf

nowt=`date +"%Y%m%d%H%M"`

set -x
set -e

CONF_FILE=./common.conf

# Optional arguments keep the old no-argument behavior intact:
#   bash test.sh [test_start_day] [test_end_day] [checkpoint_path] [log_prefix]
run_test_start_day=${1:-$test_start_day}
run_test_end_day=${2:-$test_end_day}
checkpoint_path=${3:-}
log_prefix=${4:-test_log}

exec 1>"./log/${log_prefix}_${run_test_end_day}_$nowt" 2>&1

HADOOP_HDFS_HOME=/usr/local/hadoop-current
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HADOOP_HDFS_HOME/lib/native:${JAVA_HOME}/jre/lib/amd64/server
test_args=(-data "$train_hdfs_dir" -start_day "$run_test_start_day" -end_day "$run_test_end_day")
if [[ -n "$checkpoint_path" ]]; then
    test_args+=( -checkpoint_path "$checkpoint_path" )
fi

CLASSPATH=$(${HADOOP_HDFS_HOME}/bin/hadoop classpath --glob) \
$python -u test.py "${test_args[@]}"
