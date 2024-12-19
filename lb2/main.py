import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

def load_signal(file_path):
    data = pd.read_excel(file_path, header=None)
    time = data.iloc[:, 0].values  # время
    signal = data.iloc[:, 2].values  # данные после фильтрации
    return time, signal

def cluster_signal(signal, k):
    signal = signal.reshape(-1, 1)  # Преобразование для k-means
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10).fit(signal)
    clusters = kmeans.labels_
    return clusters

# Расчёт гистограммы кластеров
def calculate_histogram(clusters, k):
    hist, _ = np.histogram(clusters, bins=np.arange(k + 1) - 0.5)
    probabilities = hist / np.sum(hist)
    return probabilities

# Расчёт разделённой энтропии
def calculate_partitioned_entropy(time, clusters, k, h, tau_end, nt):
    tauspan = np.arange(h, tau_end + h, h)  # Диапазон tau
    hp = []  # Массив для хранения значений h_p(τ)
    nl = len(time) - int(tau_end / h)
    tl = np.zeros(nl)  # Время покидания кластеров

    for tau in tauspan:
        ns = int(tau / h)
        idx_old = clusters[:len(clusters) - ns]
        idx_new = clusters[ns:]

        # Найдём индексы точек, покинувших кластеры
        changes = idx_old != idx_new
        leave_times = (tl == 0) & changes[:nl]
        tl[leave_times] = tau

        s = 0
        for i in range(k):
            ni = (idx_old == i) & changes
            hist, _ = np.histogram(idx_new[ni], bins=np.arange(k + 1) - 0.5)
            hist = hist[hist > 0]  # Уберём нули
            probabilities = hist / np.sum(hist) if np.sum(hist) > 0 else []
            if len(probabilities) > 0:
                hi = -np.sum(probabilities * np.log(probabilities))
                s += hi

        hp.append(s)

    tau0 = np.max(tl)
    n0 = int(tau0 / h)

    # Проверяем, чтобы n1 > n0, иначе увеличиваем tau_end
    n1 = min(n0 + nt, len(tauspan))
    if n1 <= n0:
        # print("Warning: Insufficient range for linear approximation. Adjusting tau_end...")
        tau_end += h * nt
        return calculate_partitioned_entropy(time, clusters, k, h, tau_end, nt)

    ts = tauspan[n0:n1]
    hps = np.array(hp)[n0:n1]

    print(f"tau0: {tau0}, n0: {n0}, n1: {n1}")
    print(f"ts: {ts}")
    print(f"hps: {hps}")

    if len(ts) == 0 or len(hps) == 0:
        raise ValueError("Empty ts or hps array. Check input parameters.")

    coefficients = np.polyfit(ts, hps, 1)
    h_ks = coefficients[0]  # Наклон аппроксимационной линии

    return h_ks, ts, hps

def plot_entropy(ts, hps, h_ks, output_path):
    plt.figure()
    plt.plot(ts, hps, label="h_p(τ)")
    plt.plot(ts, np.polyval([h_ks, 0], ts), linestyle="--", label=f"h_KS = {h_ks:.3f}")
    plt.xlabel("τ")
    plt.ylabel("h_p(τ)")
    plt.title(f"Kolmogorov-Sinai Entropy: h_KS = {h_ks:.3f}")
    plt.legend()
    plt.grid()
    plt.savefig(output_path)
    plt.show()

def main():
    file_path = "./Data/3.xlsx"
    time, signal = load_signal(file_path)

    k = 128  # Количество кластеров
    h = 0.01  # Шаг дискретизации
    tau_end = 1.0  # Максимальное tau
    nt = 20  # Увеличиваем nt для большего диапазона

    clusters = cluster_signal(signal, k)
    h_ks, ts, hps = calculate_partitioned_entropy(time, clusters, k, h, tau_end, nt)

    output_path = "F3.png"
    plot_entropy(ts, hps, h_ks, output_path)

if __name__ == "__main__":
    main()
