#include <iostream>
#include <vector>
#include <algorithm>

double alpha, beta; // ���Ŀ� ��Ÿ�� �µ� ������ ���Ǵ� ���
//���Ĵ� �ֺ� ȯ����� ������ ���, ��Ÿ�� ������ ������ ���

double t_ambient, t_desired = 27, t_start; // ȯ�� �µ�(t_ambient), ��ǥ �µ�(t_desired), ���� �µ�(t_start)

double next_temp(double u, double t, double dt) { // ���� �µ��� ����ϴ� �Լ�
    return t + alpha * (t_ambient - t) * dt + beta * u * dt;
}

class pidcontroller { // pid ��Ʈ�ѷ� Ŭ����
    double kp, ki, kd; // ���(p), ����(i), �̺�(d) ����.  (��Ʈ�ѷ��� ���� �ӵ��� �������� ������ �ݴϴ�)
    double set_point;  // ��ǥ �µ�
    double int_term, derivative_term, last_error; // ������, �̺���, ������ ����

public:
    pidcontroller(double kp, double ki, double kd, double set_point)
        : kp(kp), ki(ki), kd(kd), set_point(set_point), int_term(0), derivative_term(0), last_error(0) {}

    double get_control(double measurement, double dt) { // ��Ʈ�� ��ȣ�� ����ϴ� �Լ�
        double error = set_point - measurement; // ������ ���
        int_term += error * ki * dt; // �������� ������Ʈ
        derivative_term = (error - last_error) / dt * kd; // �̺����� ������Ʈ
        last_error = error; // ������ ������ ������Ʈ
        return kp * error + int_term + derivative_term; // ��Ʈ�� ��ȣ ���
    }
};

void simulate_temp(pidcontroller& controller, int num_steps = 20) { // �µ� �ùķ��̼� �Լ�
    double dt = 0.1; // �ð� ����. �̴� �� �ùķ��̼� ���� ������ �ð��� ��Ÿ���ϴ�.
    double t = t_start; // �ʱ� �µ�
    std::vector<double> t_list{ t }; // �µ� ����� ���� ����

    std::cout << "�ʱ� �µ�: " << t << std::endl;

    for (int k = 1; k <= num_steps; ++k) { // �ùķ��̼� ����
        double u = controller.get_control(t, dt); // ��Ʈ�� ��ȣ�� ���
        u = std::clamp(u, 0.0, 1.0); // ��Ʈ�� ��ȣ�� 0�� 1 ���̷� �����մϴ�. �̴� ������ ����� ��Ÿ���ϴ�.
        t = next_temp(u, t, dt); // �µ��� ������Ʈ�մϴ�.
        t_list.push_back(t); // �µ��� ���
        std::cout << "step " << k << ": �µ� = " << t << std::endl; // ����� ���
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
    t_ambient = get_input("�ֺ� �µ��� �Է����ּ��� (��: 20): ", -273.15, 100);  // ���뿵�� ���Ϸ� ������ �� �����Ƿ�
    t_start = get_input("���� �µ��� �Է����ּ��� (��: 21): ", -273.15, 100);

    double kp = get_input("pid ��Ʈ�ѷ��� kp ���� �Է����ּ��� (��: 0.6): ", 0.00001, 100);
    double ki = get_input("pid ��Ʈ�ѷ��� ki ���� �Է����ּ��� (��: 0.2): ", 0.00001, 100);
    double kd = get_input("pid ��Ʈ�ѷ��� kd ���� �Է����ּ��� (��: 0.02): ", 0.00001, 100);

    int num_steps = get_input("�ùķ��̼��� ������ step ���� �Է����ּ��� (��: 30): ", 1, 1000);

    pidcontroller pid_controller(kp, ki, kd, t_desired); // pid ��Ʈ�ѷ� ����
    simulate_temp(pid_controller, num_steps); // �µ� �ùķ��̼� ����
    return 0;
}
