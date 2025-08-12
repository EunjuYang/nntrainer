# Android Performance Optimization Guide for CausalLM with MoE

## 문제 분석

Android에서 Qwen MoE 레이어의 성능이 저하되는 주요 원인:

1. **I/O 병목 현상**: mmap을 통한 expert weight 로딩 시 페이지 폴트로 인한 블로킹
2. **캐시 관리 비효율**: 고정된 16개 expert 캐시 크기
3. **메모리 접근 패턴 최적화 부재**: Android에서 madvise가 비활성화됨
4. **동기식 I/O**: 비동기 프리페칭 미지원

## 적용된 최적화

### 1. 코드 레벨 최적화

#### A. 동적 캐시 크기 조정
```cpp
// qwen_moe_layer_cached.cpp
// 환경 변수를 통해 캐시 크기 조정 가능
export NNTRAINER_MOE_CACHE_SIZE=32  # 기본값: 16
```

#### B. mmap 최적화
```cpp
// tensor.cpp - Android 전용 mmap 플래그 추가
#ifdef __ANDROID__
  // MAP_POPULATE: 페이지 프리폴팅 (선택적)
  // MADV_SEQUENTIAL: 순차 접근 패턴 힌트
  // MADV_WILLNEED: 프리페칭 힌트
#endif
```

### 2. 환경 변수 설정

최적화를 위한 주요 환경 변수:

```bash
# MoE 캐시 설정
export NNTRAINER_MOE_CACHE_SIZE=24      # Expert 캐시 크기 (메모리에 따라 조정)
export NNTRAINER_MOE_PREFETCH=1         # Expert 프리페칭 활성화

# mmap 최적화
export NNTRAINER_USE_MADVISE=1          # madvise 활성화 (순차 접근 패턴)
export NNTRAINER_MADVISE_WILLNEED=1     # MADV_WILLNEED 활성화 (선택적, 메모리 사용량 증가)
export NNTRAINER_MMAP_POPULATE=1        # MAP_POPULATE 활성화 (선택적, 초기 로딩 시간 증가)

# OpenMP 최적화
export OMP_NUM_THREADS=$(nproc)         # CPU 코어 수에 맞춰 설정
export OMP_PROC_BIND=true               # 스레드 바인딩
export OMP_PLACES=cores                 # 코어별 배치
```

### 3. 시스템 레벨 최적화 (Root 필요)

```bash
# android_optimize.sh 실행
chmod +x android_optimize.sh
su -c ./android_optimize.sh

# 또는 ADB를 통해
adb shell su -c "cd /path/to/app && ./android_optimize.sh"
```

## 성능 튜닝 가이드

### 메모리별 권장 설정

#### 고성능 디바이스 (RAM > 8GB)
```bash
export NNTRAINER_MOE_CACHE_SIZE=32
export NNTRAINER_USE_MADVISE=1
export NNTRAINER_MADVISE_WILLNEED=1
export NNTRAINER_MMAP_POPULATE=1
```

#### 중간 성능 디바이스 (RAM 4-8GB)
```bash
export NNTRAINER_MOE_CACHE_SIZE=24
export NNTRAINER_USE_MADVISE=1
export NNTRAINER_MADVISE_WILLNEED=1
# MMAP_POPULATE는 비활성화 (메모리 절약)
```

#### 저성능 디바이스 (RAM < 4GB)
```bash
export NNTRAINER_MOE_CACHE_SIZE=16
export NNTRAINER_USE_MADVISE=1
# WILLNEED와 POPULATE 모두 비활성화
```

### I/O 성능 개선

#### 1. 블록 디바이스 read-ahead 증가
```bash
echo 2048 > /sys/block/mmcblk0/queue/read_ahead_kb
```

#### 2. I/O 스케줄러 최적화
```bash
# SSD/eMMC의 경우 noop 또는 deadline 사용
echo noop > /sys/block/mmcblk0/queue/scheduler
```

#### 3. 파일시스템 최적화
```bash
# noatime 마운트 옵션 사용
mount -o remount,noatime /data
```

## 실행 예시

### 기본 실행
```bash
./nntrainer_causallm /path/to/model
```

### 최적화된 실행
```bash
# 1. 최적화 스크립트 실행
source ./android_optimize.sh

# 2. 애플리케이션 실행
./nntrainer_causallm /path/to/model
```

### 커스텀 설정으로 실행
```bash
# 대용량 캐시와 적극적인 프리페칭
NNTRAINER_MOE_CACHE_SIZE=48 \
NNTRAINER_USE_MADVISE=1 \
NNTRAINER_MADVISE_WILLNEED=1 \
NNTRAINER_MMAP_POPULATE=1 \
OMP_NUM_THREADS=8 \
./nntrainer_causallm /path/to/model
```

## 성능 모니터링

### 1. I/O 통계 확인
```bash
# iostat 사용 (busybox 필요)
iostat -x 1

# /proc/diskstats 직접 확인
cat /proc/diskstats
```

### 2. 메모리 사용량 모니터링
```bash
# 프로세스별 메모리 사용량
cat /proc/$(pidof nntrainer_causallm)/status | grep -E "VmRSS|VmSwap"

# 전체 메모리 상태
cat /proc/meminfo | grep -E "MemFree|Cached|SwapFree"
```

### 3. CPU 사용률 확인
```bash
top -p $(pidof nntrainer_causallm)
```

## 트러블슈팅

### 문제: 메모리 부족 (OOM)
**해결책**:
- `NNTRAINER_MOE_CACHE_SIZE` 감소
- `NNTRAINER_MMAP_POPULATE` 비활성화
- swap 파일 생성 (성능 저하 감수)

### 문제: 초기 로딩 시간이 너무 김
**해결책**:
- `NNTRAINER_MMAP_POPULATE` 비활성화
- `NNTRAINER_MADVISE_WILLNEED` 비활성화

### 문제: 런타임 중 지연 발생
**해결책**:
- `NNTRAINER_MOE_CACHE_SIZE` 증가
- `NNTRAINER_USE_MADVISE=1` 설정
- CPU governor를 performance로 설정

## 벤치마킹

성능 측정을 위한 스크립트:

```bash
#!/bin/bash
# benchmark.sh

echo "Starting benchmark..."

# Baseline (no optimization)
time ./nntrainer_causallm /path/to/model > baseline.log 2>&1

# With optimizations
source ./android_optimize.sh
time ./nntrainer_causallm /path/to/model > optimized.log 2>&1

# Compare results
echo "Baseline time:"
grep "real" baseline.log
echo "Optimized time:"
grep "real" optimized.log
```

## 추가 최적화 아이디어

1. **비동기 I/O**: io_uring 지원 추가 (커널 5.1+)
2. **메모리 풀링**: expert weight를 위한 전용 메모리 풀
3. **압축**: zstd/lz4를 사용한 weight 압축
4. **예측 기반 캐싱**: 사용 패턴 학습을 통한 프리페칭

## 참고 자료

- [Android NDK 성능 가이드](https://developer.android.com/ndk/guides/cpu-features)
- [Linux 메모리 관리](https://www.kernel.org/doc/html/latest/admin-guide/mm/index.html)
- [mmap 최적화 기법](https://www.kernel.org/doc/html/latest/admin-guide/mm/transhuge.html)