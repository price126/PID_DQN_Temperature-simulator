#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>
#include <queue>
#include <random>

double alpha, beta; // ���Ŀ� ��Ÿ�� �µ� ������ ���Ǵ� ���
// ���Ĵ� �ֺ� ȯ����� ������ ���, ��Ÿ�� ������ ������ ���

double T_ambient, T_desired = 27, T_start; // ȯ�� �µ�(T_ambient), ��ǥ �µ�(T_desired), ���� �µ�(T_start)

double next_temp(double u, double T, double dt) { // ���� �µ��� ����ϴ� �Լ�
    return T + alpha * (T_ambient - T) * dt + beta * u * dt;
}

class PIDController { // PID ��Ʈ�ѷ� Ŭ����
    double Kp, Ki, Kd; // ���(P), ����(I), �̺�(D) ����.  (��Ʈ�ѷ��� ���� �ӵ��� �������� ������ �ݴϴ�)
    double set_point;  // ��ǥ �µ�
    double int_term, derivative_term, last_error; // ������, �̺���, ������ ����
    std::queue<double> delay_queue; // ������ ť. �̴� ��Ʈ�� ��ȣ�� ������ ����Ǳ������ �����̸� ����ϱ� ���� ���
    int delay_steps; // ������ ���� ��. �̴� ��Ʈ�� ��ȣ�� ������ ����Ǳ������ �ð��� ����

public:
    PIDController(double Kp, double Ki, double Kd, double set_point, int delay_steps)
        : Kp(Kp), Ki(Ki), Kd(Kd), set_point(set_point), int_term(0), derivative_term(0), last_error(nan("")), delay_steps(delay_steps) {}

    double get_control(double measurement, double dt) { // ��Ʈ�� ��ȣ�� ����ϴ� �Լ�
        double error = set_point - measurement; // ������ ���
        int_term += error * Ki * dt; // �������� ������Ʈ
        if (!std::isnan(last_error)) {
            derivative_term = (error - last_error) / dt * Kd; // �̺����� ������Ʈ
        }
        last_error = error; // ������ ������ ������Ʈ
        double control = Kp * error + int_term + derivative_term; // ��Ʈ�� ��ȣ ���
        delay_queue.push(control); // ������ ť�� ��Ʈ�� ��ȣ�� �߰�
        if (delay_queue.size() > delay_steps) {
            control = delay_queue.front(); // ������ ť�� ���� ����, ���� ���� �߰��Ǿ��� ��Ʈ�� ��ȣ�� ��� (�̸� ���� ���� ������ �� ��ȣ�� ���޵ǵ��� �մϴ�)
            delay_queue.pop();
        }
        return control; // ��Ʈ�� ��ȣ�� ��ȯ
    }
};

void simulate_temp(PIDController& controller, int num_steps = 20, double noise_stddev = 0.5) { // �µ� �ùķ��̼� �Լ�
    std::default_random_engine generator;
    std::normal_distribution<double> distribution(0.0, noise_stddev); // ������ ������ ���� ���� ����

    double dt = 0.1; // �ð� ����. �̴� �� �ùķ��̼� ���� ������ �ð��� ��Ÿ���ϴ�.
    double T = T_start; // �ʱ� �µ�
    std::vector<double> T_list{ T }; // �µ� ����� ���� ����

    std::cout << "�ʱ� �µ�: " << T << std::endl;

    for (int k = 1; k <= num_steps; ++k) { // �ùķ��̼� ����
        double u = controller.get_control(T, dt); // ��Ʈ�� ��ȣ�� ���
        u = std::clamp(u, 0.0, 1.0); // ��Ʈ�� ��ȣ�� 0�� 1 ���̷� �����մϴ�. �̴� ������ ����� ��Ÿ���ϴ�.
        T = next_temp(u, T, dt); // �µ��� ������Ʈ�մϴ�.
        T += distribution(generator); // ����� �߰� (���� ȯ�濡���� ����� �뷫������ ���)
        T_list.push_back(T); // �µ��� ���
        std::cout << "step " << k << ": �µ� = " << T << std::endl; // ����� ���
    }
}

double get_input(const std::string& prompt, double min_value, double max_value) { // ����� �Է��� �޴� �Լ�
    double value;
    while (true) {
        std::cout << prompt;
        std::cin >> value;
        if (std::cin.fail() || value < min_value || value > max_value) {
            std::cin.clear();
            std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
            std::cout << "�߸��� �Է��Դϴ�. " << min_value << "�� " << max_value << " ������ ���� �Է����ּ���.\n";
        }
        else {
            break;
        }
    }
    return value;
}

int main() {
    alpha = get_input("�ֺ� ȯ����� ������ ��� ���� �Է����ּ��� (��: 1): ", 0.00001, 100);
    beta = get_input("������ ������ ��� ���� �Է����ּ��� (��: 40): ", 0.00001, 100);
    T_ambient = get_input("�ֺ� �µ��� �Է����ּ��� (��: 20): ", -273.15, 100);  // ���뿵�� ���Ϸ� ������ �� �����Ƿ�
    T_start = get_input("���� �µ��� �Է����ּ��� (��: 21): ", -273.15, 100);

    double Kp = get_input("PID ��Ʈ�ѷ��� Kp ���� �Է����ּ��� (��: 0.6): ", 0.00001, 100);
    double Ki = get_input("PID ��Ʈ�ѷ��� Ki ���� �Է����ּ��� (��: 0.2): ", 0.00001, 100);
    double Kd = get_input("PID ��Ʈ�ѷ��� Kd ���� �Է����ּ��� (��: 0.02): ", 0.00001, 100);

    int num_steps = get_input("�ùķ��̼��� ������ step ���� �Է����ּ��� (��: 30): ", 1, 1000);
    int delay_steps = get_input("�����̸� ������ step ���� �Է����ּ��� (��: 5): ", 0, num_steps);

    PIDController pid_controller(Kp, Ki, Kd, T_desired, delay_steps); // PID ��Ʈ�ѷ� ����
    simulate_temp(pid_controller, num_steps); // �µ� �ùķ��̼� ����
    return 0;
}
