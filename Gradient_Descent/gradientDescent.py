import numpy as np

def calculate_total_error(b, m, data_points):
    error_sum = 0
    for point in data_points:
        x_val = point[0]
        y_val = point[1]
        error_sum += (y_val - (m * x_val + b)) ** 2
    return error_sum / len(data_points)

def compute_gradients(b_curr, m_curr, data_points, lr):
    b_grad = 0
    m_grad = 0
    num_points = len(data_points)
    for point in data_points:
        x_val = point[0]
        y_val = point[1]
        b_grad += -(2/num_points) * (y_val - ((m_curr * x_val) + b_curr))
        m_grad += -(2/num_points) * x_val * (y_val - ((m_curr * x_val) + b_curr))
    updated_b = b_curr - (lr * b_grad)
    updated_m = m_curr - (lr * m_grad)
    return updated_b, updated_m

def gradient_descent(points, start_b, start_m, lr, iterations):
    b = start_b
    m = start_m
    for _ in range(iterations):
        b, m = compute_gradients(b, m, np.array(points), lr)
    return b, m

def main():
    data_points = np.genfromtxt("data.csv", delimiter=",")
    learning_rate = 0.0001
    initial_b = 0  # Initial guess for y-intercept
    initial_m = 0  # Initial guess for slope
    total_iterations = 10000

    initial_error = calculate_total_error(initial_b, initial_m, data_points)
    print(f"Starting gradient descent with b = {initial_b}, m = {initial_m}, initial error = {initial_error}")
    print("Executing gradient descent...")
    
    final_b, final_m = gradient_descent(data_points, initial_b, initial_m, learning_rate, total_iterations)
    final_error = calculate_total_error(final_b, final_m, data_points)

    print(f"After {total_iterations} iterations: b = {final_b}, m = {final_m}, final error = {final_error}")

if __name__ == '__main__':
    main()
