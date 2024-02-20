#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>
#include <queue>
#include <random>

double alpha, beta; // 알파와 베타는 온도 조절에 사용되는 계수
// 알파는 주변 환경과의 열전달 계수, 베타는 히터의 열전달 계수

double T_ambient, T_desired = 27, T_start; // 환경 온도(T_ambient), 목표 온도(T_desired), 시작 온도(T_start)

double next_temp(double u, double T, double dt) { // 다음 온도를 계산하는 함수
    return T + alpha * (T_ambient - T) * dt + beta * u * dt;
}

class PIDController { // PID 컨트롤러 클래스
    double Kp, Ki, Kd; // 비례(P), 적분(I), 미분(D) 게인.  (컨트롤러의 반응 속도와 안정성에 영향을 줍니다)
    double set_point;  // 목표 온도
    double int_term, derivative_term, last_error; // 적분항, 미분항, 마지막 오차
    std::queue<double> delay_queue; // 딜레이 큐. 이는 컨트롤 신호가 실제로 적용되기까지의 딜레이를 모사하기 위해 사용
    int delay_steps; // 딜레이 스텝 수. 이는 컨트롤 신호가 실제로 적용되기까지의 시간을 결정

public:
    PIDController(double Kp, double Ki, double Kd, double set_point, int delay_steps)
        : Kp(Kp), Ki(Ki), Kd(Kd), set_point(set_point), int_term(0), derivative_term(0), last_error(nan("")), delay_steps(delay_steps) {}

    double get_control(double measurement, double dt) { // 컨트롤 신호를 계산하는 함수
        double error = set_point - measurement; // 오차를 계산
        int_term += error * Ki * dt; // 적분항을 업데이트
        if (!std::isnan(last_error)) {
            derivative_term = (error - last_error) / dt * Kd; // 미분항을 업데이트
        }
        last_error = error; // 마지막 오차를 업데이트
        double control = Kp * error + int_term + derivative_term; // 컨트롤 신호 계산
        delay_queue.push(control); // 딜레이 큐에 컨트롤 신호를 추가
        if (delay_queue.size() > delay_steps) {
            control = delay_queue.front(); // 딜레이 큐가 가득 차면, 가장 먼저 추가되었던 컨트롤 신호를 사용 (이를 통해 일정 딜레이 후 신호가 전달되도록 합니다)
            delay_queue.pop();
        }
        return control; // 컨트롤 신호를 반환
    }
};

void simulate_temp(PIDController& controller, int num_steps = 20, double noise_stddev = 0.5) { // 온도 시뮬레이션 함수
    std::default_random_engine generator;
    std::normal_distribution<double> distribution(0.0, noise_stddev); // 노이즈 생성을 위한 정규 분포

    double dt = 0.1; // 시간 간격. 이는 각 시뮬레이션 스텝 사이의 시간을 나타냅니다.
    double T = T_start; // 초기 온도
    std::vector<double> T_list{ T }; // 온도 기록을 위한 벡터

    std::cout << "초기 온도: " << T << std::endl;

    for (int k = 1; k <= num_steps; ++k) { // 시뮬레이션 루프
        double u = controller.get_control(T, dt); // 컨트롤 신호를 계산
        u = std::clamp(u, 0.0, 1.0); // 컨트롤 신호를 0과 1 사이로 제한합니다. 이는 히터의 출력을 나타냅니다.
        T = next_temp(u, T, dt); // 온도를 업데이트합니다.
        T += distribution(generator); // 노이즈를 추가 (실제 환경에서의 노이즈를 대략적으로 모사)
        T_list.push_back(T); // 온도를 기록
        std::cout << "step " << k << ": 온도 = " << T << std::endl; // 결과를 출력
    }
}

double get_input(const std::string& prompt, double min_value, double max_value) { // 사용자 입력을 받는 함수
    double value;
    while (true) {
        std::cout << prompt;
        std::cin >> value;
        if (std::cin.fail() || value < min_value || value > max_value) {
            std::cin.clear();
            std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
            std::cout << "잘못된 입력입니다. " << min_value << "와 " << max_value << " 사이의 값을 입력해주세요.\n";
        }
        else {
            break;
        }
    }
    return value;
}

int main() {
    alpha = get_input("주변 환경과의 열전달 계수 값을 입력해주세요 (예: 1): ", 0.00001, 100);
    beta = get_input("히터의 열전달 계수 값을 입력해주세요 (예: 40): ", 0.00001, 100);
    T_ambient = get_input("주변 온도를 입력해주세요 (예: 20): ", -273.15, 100);  // 절대영도 이하로 내려갈 수 없으므로
    T_start = get_input("시작 온도를 입력해주세요 (예: 21): ", -273.15, 100);

    double Kp = get_input("PID 컨트롤러의 Kp 값을 입력해주세요 (예: 0.6): ", 0.00001, 100);
    double Ki = get_input("PID 컨트롤러의 Ki 값을 입력해주세요 (예: 0.2): ", 0.00001, 100);
    double Kd = get_input("PID 컨트롤러의 Kd 값을 입력해주세요 (예: 0.02): ", 0.00001, 100);

    int num_steps = get_input("시뮬레이션을 진행할 step 수를 입력해주세요 (예: 30): ", 1, 1000);
    int delay_steps = get_input("딜레이를 적용할 step 수를 입력해주세요 (예: 5): ", 0, num_steps);

    PIDController pid_controller(Kp, Ki, Kd, T_desired, delay_steps); // PID 컨트롤러 생성
    simulate_temp(pid_controller, num_steps); // 온도 시뮬레이션 실행
    return 0;
}
