#include <iostream>
#include <vector>
#include <algorithm>

double alpha, beta; // 알파와 베타는 온도 조절에 사용되는 계수
//알파는 주변 환경과의 열전달 계수, 베타는 히터의 열전달 계수

double t_ambient, t_desired = 27, t_start; // 환경 온도(t_ambient), 목표 온도(t_desired), 시작 온도(t_start)

double next_temp(double u, double t, double dt) { // 다음 온도를 계산하는 함수
    return t + alpha * (t_ambient - t) * dt + beta * u * dt;
}

class pidcontroller { // pid 컨트롤러 클래스
    double kp, ki, kd; // 비례(p), 적분(i), 미분(d) 게인.  (컨트롤러의 반응 속도와 안정성에 영향을 줍니다)
    double set_point;  // 목표 온도
    double int_term, derivative_term, last_error; // 적분항, 미분항, 마지막 오차

public:
    pidcontroller(double kp, double ki, double kd, double set_point)
        : kp(kp), ki(ki), kd(kd), set_point(set_point), int_term(0), derivative_term(0), last_error(0) {}

    double get_control(double measurement, double dt) { // 컨트롤 신호를 계산하는 함수
        double error = set_point - measurement; // 오차를 계산
        int_term += error * ki * dt; // 적분항을 업데이트
        derivative_term = (error - last_error) / dt * kd; // 미분항을 업데이트
        last_error = error; // 마지막 오차를 업데이트
        return kp * error + int_term + derivative_term; // 컨트롤 신호 계산
    }
};

void simulate_temp(pidcontroller& controller, int num_steps = 20) { // 온도 시뮬레이션 함수
    double dt = 0.1; // 시간 간격. 이는 각 시뮬레이션 스텝 사이의 시간을 나타냅니다.
    double t = t_start; // 초기 온도
    std::vector<double> t_list{ t }; // 온도 기록을 위한 벡터

    std::cout << "초기 온도: " << t << std::endl;

    for (int k = 1; k <= num_steps; ++k) { // 시뮬레이션 루프
        double u = controller.get_control(t, dt); // 컨트롤 신호를 계산
        u = std::clamp(u, 0.0, 1.0); // 컨트롤 신호를 0과 1 사이로 제한합니다. 이는 히터의 출력을 나타냅니다.
        t = next_temp(u, t, dt); // 온도를 업데이트합니다.
        t_list.push_back(t); // 온도를 기록
        std::cout << "step " << k << ": 온도 = " << t << std::endl; // 결과를 출력
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
    t_ambient = get_input("주변 온도를 입력해주세요 (예: 20): ", -273.15, 100);  // 절대영도 이하로 내려갈 수 없으므로
    t_start = get_input("시작 온도를 입력해주세요 (예: 21): ", -273.15, 100);

    double kp = get_input("pid 컨트롤러의 kp 값을 입력해주세요 (예: 0.6): ", 0.00001, 100);
    double ki = get_input("pid 컨트롤러의 ki 값을 입력해주세요 (예: 0.2): ", 0.00001, 100);
    double kd = get_input("pid 컨트롤러의 kd 값을 입력해주세요 (예: 0.02): ", 0.00001, 100);

    int num_steps = get_input("시뮬레이션을 진행할 step 수를 입력해주세요 (예: 30): ", 1, 1000);

    pidcontroller pid_controller(kp, ki, kd, t_desired); // pid 컨트롤러 생성
    simulate_temp(pid_controller, num_steps); // 온도 시뮬레이션 실행
    return 0;
}
