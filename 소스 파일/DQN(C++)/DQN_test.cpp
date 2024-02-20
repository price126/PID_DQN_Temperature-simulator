#include <torch/torch.h>
#include <iostream>
#include <random>
#include <deque>
#include <tuple>
#include <algorithm>

// torch::nn::Module�� ��ӹ��� dqn�̶�� �̸��� �Ű�� ������ ����
struct dqn : torch::nn::Module {
public:
    // �����ڿ��� �Է� ������ ��� ������ �޾�, �� ���� ���� ���� ����(fc1, fc2)�� ����ϴ�.
    dqn(int64_t input_dim, int64_t output_dim)
        : fc1(torch::nn::Linear(input_dim, 128)),     // �Է� ������ 128�� �����ϴ� ����
        fc2(torch::nn::Linear(128, output_dim)) {     // 128�� ��� �������� �����ϴ� ����
        register_module("fc1", fc1);
        register_module("fc2", fc2); 
    }

    // ������ �н��� �����մϴ�. �Է� x�� �޾�, �� ������ ��ġ�� Ȱ��ȭ �Լ� relu�� ����մϴ�.
    torch::Tensor forward(torch::Tensor x) {
        x = torch::relu(fc1->forward(x));  // ù ��° ������ ��ġ�� relu Ȱ��ȭ �Լ� ����
        x = fc2->forward(x);              // �� ��° ������ ��Ĩ�ϴ�.
        return x;                         // ����� ��ȯ
    }

private:
    torch::nn::Linear fc1{ nullptr }, fc2{ nullptr };
};

// �µ� ������ �н��ϴ� ȯ���� ����
class temperatureenvironment {
public:
    // �����ڿ��� ��ǥ �µ��� ������ �׼��� �����մϴ�.
    temperatureenvironment(double target_temperature = 27.0, std::vector<double> actions = { -0.001, -0.005, -0.01, -0.05, -0.1, -0.5, -1.0, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0 })
        : target_temperature(target_temperature), actions(actions) {
        reset();
    }

    // ȯ���� �ʱ�ȭ�ϴ� �޼ҵ��Դϴ�. ���� �µ��� �������� �����մϴ�.
    double reset() {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<> dis(0.0, 30.0); // 0~30�� ������ �µ��� �����ϰ� ����
        current_temperature = dis(gen);
        return current_temperature;
    }

    // �� �ܰ踦 �����ϴ� �޼ҵ��Դϴ�. ������ �׼ǿ� ���� ���� �µ��� �����ϰ�, ������ ����ϰ�, ��ǥ �µ��� �����ߴ����� ��ȯ�մϴ�.
    std::tuple<double, double, bool> step(int action_idx) {
        current_temperature += actions[action_idx];  // ������ �׼ǿ� ���� �µ� ����
        double reward = -std::pow((current_temperature - target_temperature), 2);  // ������ ��ǥ �µ����� ������ ������ ���̳ʽ��� ���� ��
        bool done = std::abs(current_temperature - target_temperature) < 0.1;  // ��ǥ �µ��� ����������� �Ǵ�
        return std::make_tuple(current_temperature, reward, done);  // ���� �µ�, ����, �Ϸ� ���θ� ��ȯ
    }

    // ���� ����(���⼭�� �µ��̹Ƿ� 1)�� ��ȯ�ϴ� �޼ҵ��Դϴ�.
    int get_state_dim() {
        return 1;
    }

    // ������ �׼��� ���� ��ȯ�ϴ� �޼ҵ��Դϴ�.
    size_t get_action_dim() {
        return actions.size();
    }

private:
    double current_temperature;  // ���� �µ�
    double target_temperature;   // ��ǥ �µ�
    std::vector<double> actions; // ������ �׼�
};

// DQN ������Ʈ�� ����
class dqnagent {
public:
    // �����ڿ����� �Է� ����, ��� ����, ����(ť�����н��� ���Ǵ� ���ΰ��) ���� �޽��ϴ�.
    dqnagent(int64_t input_dim, int64_t output_dim, double gamma = 0.99) {
        model = std::make_shared<dqn>(input_dim, output_dim);  // DQN �� ����
        torch::load(model, "model.pt");  // �н��� ���� �ε��մϴ�.
    }

    // �־��� ���¿��� ���� ���� Q ���� ������ �׼��� ��ȯ�ϴ� �޼ҵ�
    int get_action(torch::Tensor state) {
        return torch::argmax(model->forward(state)).item<int>();  // ���� ��¿��� ���� ���� ���� ������ �ε����� ��ȯ�մϴ�.
    }

    // ���� �ε��ϴ� �޼ҵ�
    void load_model(std::string path) {
        torch::load(model, path);
    }

private:
    std::shared_ptr<dqn> model;  // DQN ��
};

// ������Ʈ�� �׽�Ʈ�ϴ� �Լ�
int test_agent() {
    auto env = std::make_shared<temperatureenvironment>();  // ȯ�� ����
    auto agent = std::make_shared<dqnagent>(env->get_state_dim(), env->get_action_dim());  // ������Ʈ ����

    int num_test_episodes = 5;  // �׽�Ʈ ���Ǽҵ� ��
    for (int i_episode = 1; i_episode <= num_test_episodes; i_episode++) {
        double raw_state = env->reset();  // ȯ�� �ʱ�ȭ
        auto state = torch::tensor({ raw_state });  // ���� �ټ� ����

        // ���Ǽҵ尡 ���� ������ �ݺ��մϴ�.
        for (int t = 0;; t++) {
            auto action = agent->get_action(state);  // ������Ʈ�� �׼��� �����մϴ�.

            // ȯ���� �� �ܰ� �����ϰ�, ���� ����, ����, ���� ���θ� ����ϴ�.
            double raw_next_state, reward;
            bool done;
            std::tie(raw_next_state, reward, done) = env->step(action);
            auto next_state = torch::tensor({ raw_next_state });

            // ����, �׼�, ���� ���� ����մϴ�.
            std::cout << "episode: " << i_episode
                << ", time step: " << t
                << ", action: " << action
                << ", reward: " << reward
                << ", current temperature: " << state.item<float>()
                << ", next temperature: " << next_state.item<float>()
                << std::endl;

            state = next_state;  // ���¸� ������Ʈ�մϴ�.

            // ���Ǽҵ尡 ����Ǿ��ٸ�, �ݺ��� �����մϴ�.
            if (done) {
                std::cout << "episode " << i_episode << " finished after " << t + 1 << " timesteps\n";
                break;
            }
        }
    }

    return 0;
}

// ������Ʈ �׽�Ʈ
int main() {
    test_agent();
    return 0;
}
