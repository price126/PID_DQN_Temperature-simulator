#include <torch/torch.h>
#include <aten/tensor.h>
#include <torch/script.h>
#include <torch/nn/module.h>
#include <torch/nn/modules/linear.h>
#include <torch/optim.h>
#include <iostream>
#include <vector>
#include <deque>
#include <random>
#include <algorithm>
#include <iterator>
#include <cmath>

// 아래 dqn 구조체는 신경망 모델을 나타냅니다.
// 이 모델은 두 개의 선형 레이어(fc1, fc2)를 가지고 있습니다.
struct dqn : torch::nn::Module {
public:
    dqn(int64_t input_dim, int64_t output_dim)
        : fc1(torch::nn::Linear(input_dim, 128)),
        fc2(torch::nn::Linear(128, output_dim)) {
        register_module("fc1", fc1);
        register_module("fc2", fc2);
    }

    // 신경망의 순전파를 정의합니다.
    torch::Tensor forward(torch::Tensor x) {
        x = torch::relu(fc1->forward(x));
        x = fc2->forward(x);
        return x;
    }

    // 주어진 소스의 가중치를 복사합니다.
    void copy_weights(const dqn& source) {
        fc1->weight.data().copy_(source.fc1->weight.data());
        fc1->bias.data().copy_(source.fc1->bias.data());
        fc2->weight.data().copy_(source.fc2->weight.data());
        fc2->bias.data().copy_(source.fc2->bias.data());
    }

private:
    torch::nn::Linear fc1{ nullptr }, fc2{ nullptr };
};

// 온도환경 클래스로, 에이전트가 상호작용하는 환경을 나타냅니다.
class temperatureenvironment {
public:
    temperatureenvironment(double target_temperature = 27.0, std::vector<double> actions = { -0.001, -0.005, -0.01, -0.05, -0.1, -0.5, -1.0, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0 })
        : target_temperature(target_temperature), actions(actions) {
        reset();
    }

    // 환경을 초기화합니다.
    double reset() {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<> dis(0.0, 30.0); // 0~30도 사이의 온도를 랜덤하게 설정
        current_temperature = dis(gen);
        return current_temperature;
    }

    // 에이전트가 취한 행동의 결과를 반환합니다.
    std::tuple<double, double, bool> step(int action_idx) {
        current_temperature += actions[action_idx];
        double reward = -std::pow((current_temperature - target_temperature), 2);
        bool done = std::abs(current_temperature - target_temperature) < 0.1;
        return std::make_tuple(current_temperature, reward, done);
    }

    int get_state_dim() {
        return 1;
    }

    size_t get_action_dim() {
        return actions.size();
    }

private:
    double current_temperature;
    double target_temperature;
    std::vector<double> actions;
};


class dqnagent {
public:

    // 학습률 조정 파라미터
    double lr = 0.001; // 초기 학습률

    // 생성자에서 에이전트의 초기 설정을 합니다.
    dqnagent(int64_t input_dim, int64_t output_dim, double gamma = 0.99, double lr = 0.001, double eps_start = 0.9, double eps_end = 0.05, int64_t eps_decay = 350) {
        model = std::make_shared<dqn>(input_dim, output_dim);
        target_model = std::make_shared<dqn>(input_dim, output_dim);

        torch::save(model, "temp.pt");
        torch::load(target_model, "temp.pt");

        optimizer = std::make_shared<torch::optim::Adam>(model->parameters(), torch::optim::AdamOptions(lr));

        this->gamma = gamma;
        this->eps_start = eps_start;
        this->eps_end = eps_end;
        this->eps_decay = eps_decay;
        steps_done = 0;
    }

    // 현재 상태를 기반으로 에이전트가 취할 행동을 결정합니다.
    int get_action(torch::Tensor state) {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<> dis(0, 1);
        double eps_threshold = eps_end + (eps_start - eps_end) * std::exp(-1. * static_cast<double>(steps_done) / eps_decay);
        steps_done++;
        if (dis(gen) < eps_threshold) {
            std::uniform_int_distribution<> disint(0, 13);
            return disint(gen);
        }
        else {
            return torch::argmax(model->forward(state)).item<int>();
        }
    }

    // 모델을 업데이트하는 함수입니다.
    void update_model() {
        if (memory.size() < batch_size) return;

        std::random_device rd;
        std::mt19937 g(rd());

        std::vector<size_t> indices(memory.size());
        std::iota(indices.begin(), indices.end(), 0);
        std::shuffle(indices.begin(), indices.end(), g);

        std::vector<size_t> indices_batch(indices.begin(), indices.begin() + batch_size);
        std::vector<std::tuple<torch::Tensor, int64_t, double, torch::Tensor, bool>> batch;
        for (auto i : indices_batch) {
            batch.push_back(memory[i]);
        }

        std::vector<torch::Tensor> states, next_states;
        std::vector<int64_t> actions;
        std::vector<double> rewards;
        std::vector<bool> dones;

        for (auto& transition : batch) {
            states.push_back(std::get<0>(transition));
            actions.push_back(std::get<1>(transition));
            rewards.push_back(std::get<2>(transition));
            next_states.push_back(std::get<3>(transition));
            dones.push_back(std::get<4>(transition));
        }

        auto q_values = model->forward(torch::stack(states));
        auto next_q_values = target_model->forward(torch::stack(next_states));

        torch::Tensor y = q_values.clone().detach();
        for (size_t i = 0; i < batch_size; i++) {
            if (dones[i]) {
                y[i][actions[i]] = rewards[i];
            }
            else {
                y[i][actions[i]] = rewards[i] + gamma * next_q_values[i].max().item<double>();
            }
        }

        optimizer->zero_grad();
        auto loss = torch::nn::functional::mse_loss(q_values, y.detach());
        loss.backward();
        optimizer->step();
    }

    // 타겟 모델을 업데이트합니다.
    void update_target_model() {
        target_model->copy_weights(*model);
    }

    // 에이전트의 경험을 메모리에 저장합니다.
    void memory_push(std::tuple<torch::Tensor, int64_t, double, torch::Tensor, bool> transition) {
        if (memory.size() == memory_maxlen) {
            memory.pop_front();
        }
        memory.push_back(transition);
    }

    // 학습된 모델을 저장.
    void save_model(std::string path) {
        torch::save(model, path);
    }

    // 저장된 모델을 불러오기.
    void load_model(std::string path) {
        torch::load(model, path);
    }

private:
    std::shared_ptr<dqn> model, target_model;
    std::shared_ptr<torch::optim::Adam> optimizer;
    double gamma;
    double eps_start, eps_end, eps_decay;
    int64_t steps_done;
    std::deque<std::tuple<torch::Tensor, int64_t, double, torch::Tensor, bool>> memory;
    const int64_t batch_size = 64;
    const int64_t memory_maxlen = 10000;
};


int main() {

    // 환경과 에이전트를 초기화합니다.
    auto env = std::make_shared<temperatureenvironment>();
    auto agent = std::make_shared<dqnagent>(env->get_state_dim(), env->get_action_dim());

    // 학습을 위해 1000회의 에피소드를 실행합니다.
    int num_episodes = 1000;
    std::vector<double> rewards; // 에피소드별 총 보상을 저장할 리스트

    for (int i_episode = 1; i_episode <= num_episodes; i_episode++) {

        // 각 에피소드 시작시 환경은 초기화되며, 초기 상태가 주어집니다.
        double raw_state = env->reset();
        auto state = torch::tensor({ raw_state });

        double total_reward = 0.0; // 에피소드별 총 보상을 저장할 변수
        for (int t = 0;; t++) {
            // 에이전트는 현재 상태를 기반으로 행동을 결정하고, 해당 행동을 실행
            auto action = agent->get_action(state);

            // 환경은 에이전트의 행동에 따라 다음 상태와 보상, 그리고 에피소드 종료 여부를 반환
            double raw_next_state, reward;
            bool done;
            std::tie(raw_next_state, reward, done) = env->step(action);
            total_reward += reward; // 보상을 누적합니다.
            auto next_state = torch::tensor({ raw_next_state });

            // 에피소드의 진행 상황을 출력
            std::cout << "episode: " << i_episode
                << ", time step: " << t
                << ", action: " << action
                << ", reward: " << reward
                << ", current temperature: " << state.item<float>()
                << ", next temperature: " << next_state.item<float>()
                << std::endl;

            // 에이전트는 방금 경험한 상태, 행동, 보상, 다음 상태, 종료 여부를 메모리에 저장합니다.
            agent->memory_push(std::make_tuple(state, action, reward, next_state, done));
            state = next_state;

            // 그후 메모리에 저장된 경험을 바탕으로 모델을 업데이트합니다.
            agent->update_model();
            if (t % 1000 == 0) {
                agent->update_target_model(); // 매 1000번의 스텝마다 타깃 모델을 업데이트하도록 했습니다.
            }

            // 에피소드가 종료되면, 해당 정보를 출력하고 에피소드를 종료
            if (done) {
                rewards.push_back(total_reward); // 에피소드별 총 보상을 리스트에 추가합니다.
                std::cout << "episode " << i_episode << " finished after " << t + 1 << " timesteps\n";
                break;
            }
        }

        // 각 에피소드가 종료될 때마다, 현재까지 학습된 모델을 파일로 저장
        agent->save_model("model.pt");
    }
    double average_reward = std::accumulate(rewards.begin(), rewards.end(), 0.0) / rewards.size(); // 평균 보상을 계산합니다.
    std::cout << "Average reward: " << average_reward << std::endl; // 평균 보상을 출력합니다.

    return 0;
}




