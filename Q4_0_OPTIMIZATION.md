# Q4_0 모델 성능 최적화 가이드

## 문제
Q4_0 양자화 모델을 OpenMP로 실행 시 성능 편차가 큼

## 원인
1. 고정된 스레드 수 사용
2. guided 스케줄링으로 인한 로드 불균형  
3. 스레드 간 False Sharing
4. 반복적인 메모리 할당

## 해결 방법

### 1. 환경 변수 설정
```bash
export OMP_NUM_THREADS=6
export OMP_PROC_BIND=close
export OMP_SCHEDULE="static"
export OMP_DYNAMIC=false
```

### 2. 코드 최적화
- schedule(guided) → schedule(static) 변경
- Thread-local 스토리지 사용
- 워크로드 기반 동적 스레드 수 조정

### 3. 시스템 설정
```bash
sudo cpupower frequency-set -g performance
```

## 결과
- 성능 편차: 3-5배 → 1.2-1.5배
- 평균 성능: 10-20% 향상
