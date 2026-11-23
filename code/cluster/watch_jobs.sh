#!/bin/bash
# Edge watcher with cause (from thomouvic/alliance-canada README).
# Usage: watch_jobs.sh <cluster> <logdir> <jobid> [jobid...]
set -u
cluster="$1"; logdir="$2"; shift 2
declare -A done_state crash_seen
for j in "$@"; do done_state[$j]=""; crash_seen[$j]=""; done
echo "WATCHER_STARTED: $*"
while :; do
  all_done=1
  for j in "$@"; do
    [ -n "${done_state[$j]}" ] && continue
    state=$(ssh -o BatchMode=yes "$cluster" "sacct -j $j --format=State -n -X | head -1 | xargs" 2>/dev/null)
    if [ -z "${crash_seen[$j]}" ]; then
      hit=$(ssh -o BatchMode=yes "$cluster" "grep -lE 'Traceback|CUDA error|OutOfMemory|Killed' $logdir/*$j*.err 2>/dev/null" | head -1)
      [ -n "$hit" ] && { echo "JOB_CRASH_SIGNAL: $j (error text in .err while state=$state)"; crash_seen[$j]=1; }
    fi
    case "$state" in
      COMPLETED|FAILED|CANCELLED*|TIMEOUT|OUT_OF_MEMORY|NODE_FAIL|PREEMPTED|BOOT_FAIL|DEADLINE)
        echo "JOB_TERMINAL: $j $state"
        ssh -o BatchMode=yes "$cluster" "sacct -j $j --format=JobID,State,Elapsed,ExitCode -n; echo '--- .err tail ---'; tail -n 15 $logdir/*$j*.err 2>/dev/null; echo '--- .out tail ---'; tail -n 30 $logdir/*$j*.out 2>/dev/null"
        done_state[$j]="$state" ;;
      *) all_done=0 ;;
    esac
  done
  [ "$all_done" = 1 ] && exit 0
  sleep 20
done
