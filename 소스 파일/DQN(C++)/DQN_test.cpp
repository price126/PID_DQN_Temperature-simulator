#include <torch/torch.h>
#include <iostream>
#include <random>
#include <deque>
#include <tuple>
#include <algorithm>

// torch::nn::Module을 상속받은 dqn이라는 이름의 신경망 구조를 정의
struct dqn : torch::nn::Module {
public:
    // 생성자에서 입력 차원과 출력 차원을 받아, 두 개의 완전 연결 계층(fc1, fc2)을 만듭니다.
    dqn(int64_t input_dim, int64_t output_dim)
        : fc1(torch::nn::Linear(input_dim, 128)),     // 입력 차원을 128로 매핑하는 계층
        fc2(torch::nn::Linear(128, output_dim)) {     // 128을 출력 차원으로 매핑하는 계층
        register_module("fc1", fc1);
        register_module("fc2", fc2); 
    }

    // 순방향 패스를 정의합니다. 입력 x를 받아, 두 계층을 거치며 활성화 함수 relu를 사용합니다.
    torch::Tensor forward(torch::Tensor x) {
        x = torch::relu(fc1->forward(x));  // 첫 번째 계층을 거치고 relu 활성화 함수 적용
        x = fc2->forward(x);              // 두 번째 계층을 거칩니다.
        return x;                         // 결과를 반환
    }

private:
    torch::nn::Linear fc1{ nullptr }, fc2{ nullptr };
};

// 온도 조절을 학습하는 환경을 정의
class temperatureenvironment {
public:
    // 생성자에서 목표 온도와 가능한 액션을 설정합니다.
    temperatureenvironment(double target_temperature = 27.0, std::vector<double> actions = { -0.001, -0.005, -0.01, -0.05, -0.1, -0.5, -1.0, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0 })
        : target_temperature(target_temperature), actions(actions) {
        reset();
    }

    // 환경을 초기화하는 메소드입니다. 현재 온도를 무작위로 설정합니다.
    double reset() {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<> dis(0.0, 30.0); // 0~30도 사이의 온도를 랜덤하게 설정
        current_temperature = dis(gen);
        return current_temperature;
    }

    // 한 단계를 진행하는 메소드입니다. 선택한 액션에 따라 현재 온도를 변경하고, 보상을 계산하고, 목표 온도에 도달했는지를 반환합니다.
    std::tuple<double, double, bool> step(int action_idx) {
        current_temperature += actions[action_idx];  // 선택한 액션에 따라 온도 변경
        double reward = -std::pow((current_temperature - target_temperature), 2);  // 보상은 목표 온도와의 차이의 제곱에 마이너스를 취한 값
        bool done = std::abs(current_temperature - target_temperature) < 0.1;  // 목표 온도에 가까워졌는지 판단
        return std::make_tuple(current_temperature, reward, done);  // 현재 온도, 보상, 완료 여부를 반환
    }

    // 상태 차원(여기서는 온도이므로 1)을 반환하는 메소드입니다.
    int get_state_dim() {
        return 1;
    }

    // 가능한 액션의 수를 반환하는 메소드입니다.
    size_t get_action_dim() {
        return actions.size();
    }

private:
    double current_temperature;  // 현재 온도
    double target_temperature;   // 목표 온도
    std::vector<double> actions; // 가능한 액션
};

// DQN 에이전트를 정의
class dqnagent {
public:
    // 생성자에서는 입력 차원, 출력 차원, 감마(큐러닝학습시 사용되는 할인계수) 값을 받습니다.
    dqnagent(int64_t input_dim, int64_t output_dim, double gamma = 0.99) {
        model = std::make_shared<dqn>(input_dim, output_dim);  // DQN 모델 생성
        torch::load(model, "model.pt");  // 학습된 모델을 로드합니다.
    }

    // 주어진 상태에서 가장 높은 Q 값을 가지는 액션을 반환하는 메소드
    int get_action(torch::Tensor state) {
        return torch::argmax(model->forward(state)).item<int>();  // 모델의 출력에서 가장 높은 값을 가지는 인덱스를 반환합니다.
    }

    // 모델을 로드하는 메소드
    void load_model(std::string path) {
        torch::load(model, path);
    }

private:
    std::shared_ptr<dqn> model;  // DQN 모델
};

// 에이전트를 테스트하는 함수
int test_agent() {
    auto env = std::make_shared<temperatureenvironment>();  // 환경 생성
    auto agent = std::make_shared<dqnagent>(env->get_state_dim(), env->get_action_dim());  // 에이전트 생성

    int num_test_episodes = 5;  // 테스트 에피소드 수
    for (int i_episode = 1; i_episode <= num_test_episodes; i_episode++) {
        double raw_state = env->reset();  // 환경 초기화
        auto state = torch::tensor({ raw_state });  // 상태 텐서 생성

        // 에피소드가 끝날 때까지 반복합니다.
        for (int t = 0;; t++) {
            auto action = agent->get_action(state);  // 에이전트가 액션을 선택합니다.

            // 환경을 한 단계 진행하고, 다음 상태, 보상, 종료 여부를 얻습니다.
            double raw_next_state, reward;
            bool done;
            std::tie(raw_next_state, reward, done) = env->step(action);
            auto next_state = torch::tensor({ raw_next_state });

            // 상태, 액션, 보상 등을 출력합니다.
            std::cout << "episode: " << i_episode
                << ", time step: " << t
                << ", action: " << action
                << ", reward: " << reward
                << ", current temperature: " << state.item<float>()
                << ", next temperature: " << next_state.item<float>()
                << std::endl;

            state = next_state;  // 상태를 업데이트합니다.

            // 에피소드가 종료되었다면, 반복을 종료합니다.
            if (done) {
                std::cout << "episode " << i_episode << " finished after " << t + 1 << " timesteps\n";
                break;
            }
        }
    }

    return 0;
}

// 에이전트 테스트
int main() {
    test_agent();
    return 0;
}
