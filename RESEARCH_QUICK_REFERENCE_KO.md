> 이 문서는 빠른 한국어 확인용 요약본이다.
> 최종 연구 및 구현 판단에서는 `docs/FINAL_RESEARCH_SPEC.md`와
> `docs/DECISIONS.md`의 active decisions를 우선한다.

# 최종 연구 방향 빠른 확인본

## 최종 하드웨어

- Isaac Sim
- Universal Robots UR10e
- OnRobot RG6 gripper
- 손목 장착 Zivid 2 3D/RGB-D camera

이 구성을 최종 기준으로 사용한다. 다른 로봇은 최종 시스템으로 대체하지 않는다.

## 연구를 아주 쉽게 설명하면

사용자가 물체를 찾아 달라고 했는데, 로봇이 그 물체가 어느 통 안에 있는지, 뒤에 있는지, 다른 물체에 가려졌는지 확실하지 않은 상황을 다룬다.

로봇은 바로 집지 않는다. 카메라 위치를 바꾸거나, 덮개를 치우거나, 가림 물체를 옮겼을 때 어떤 정보가 생길지 먼저 예측한다. 그다음 잘못 집을 위험과 전체 작업 비용이 가장 작아지는 행동을 선택한다.

## 다른 논문들과의 차별점

- 박신규 교수님 기존 논문은 Scene Graph를 현재 상태를 기억하고 장기 작업 순서를 세우는 데 사용한다. 새 연구는 Scene Graph 관계에 확률을 저장하고, 그 불확실성이 다음 로봇 행동을 직접 결정하게 한다.
- RoboEXP는 환경을 탐색하며 Scene Graph를 만든다. 새 연구는 특정 사용자의 물체 회수 요청에 필요한 정보만 모으고 잘못된 선택 위험을 줄인다.
- RoboRetriever는 이미 재관찰, 물체 이동, Scene Graph, 회수를 통합한다. 따라서 새 연구는 단순 통합이 아니라 관계 확률의 보정, 행동 후 미래 확률 예측, 잘못된 행동 비용 최소화가 핵심이다.
- VLMPC는 행동 후 미래 영상을 예측한다. 새 연구는 행동 후 `물체가 어느 위치나 관계에 있을 확률`이 어떻게 바뀌는지를 예측한다.
- ReKep은 관계를 제약식으로 만들어 동작을 최적화한다. 새 연구는 여러 가능한 장면 가설의 확률을 유지하고, 어느 행동이 잘못된 grasp 위험을 가장 줄이는지 선택한다.
- SCOUT는 불확실한 물체를 더 잘 보기 위한 viewpoint를 고르지만, 논문에서 relation-edge uncertainty는 향후 연구로 남긴다. 새 연구는 관계 불확실성을 manipulation decision과 연결한다.

## 핵심 차별점 네 가지

1. 물체뿐 아니라 `inside`, `behind`, `occluded_by`, `covered_by` 관계에도 확률을 저장한다.
2. 카메라 이동이나 물체 조작 뒤에 그 확률이 어떻게 바뀔지 미리 예측한다.
3. 단순히 정보량을 늘리는 것이 아니라 잘못된 grasp와 전체 작업 비용을 줄이는 행동을 선택한다.
4. 가장 가능성이 높은 곳을 확인했는데 물체가 없으면, 그 실패를 새로운 증거로 반영하여 다른 가설로 재계획한다.

## 주요 평가 지표

- Task Success Rate
- Wrong Commitment Rate
- Total Cost to Successful Retrieval
- 보조: calibration error, relation accuracy, planning time, collision rate

## 중요한 경계

`VLM + Scene Graph + MPC를 연결했다`는 것만으로는 차별점이 아니다. 최종 논문은 반드시 행동에 따른 미래 belief를 예측하고, relation uncertainty와 negative evidence가 실제 행동 선택과 성공률을 바꾼다는 것을 보여야 한다.
