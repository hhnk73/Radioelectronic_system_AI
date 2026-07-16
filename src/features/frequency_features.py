import numpy as np


def calculate_fft(values, sampling_rate):
    values = np.asarray(values)
    values = values - np.mean(values)

    fft_values = np.fft.rfft(values)
    frequencies = np.fft.rfftfreq(len(values), d=1 / sampling_rate)
    amplitudes = np.abs(fft_values)

    return frequencies, amplitudes


def calculate_band_energy(frequencies, amplitudes, low, high):
    mask = (frequencies >= low) & (frequencies < high)

    if not np.any(mask):
        return 0.0

    return float(np.sum(amplitudes[mask] ** 2))


def calculate_dominant_frequency(frequencies, amplitudes):
    if len(amplitudes) <= 1:
        return 0.0

    amplitudes_without_dc = amplitudes.copy()
    amplitudes_without_dc[0] = 0

    dominant_index = np.argmax(amplitudes_without_dc)

    return float(frequencies[dominant_index])


def calculate_spectral_centroid(frequencies, amplitudes):
    amplitude_sum = np.sum(amplitudes)

    if amplitude_sum == 0:
        return 0.0

    return float(np.sum(frequencies * amplitudes) / amplitude_sum)


def calculate_spectral_spread(frequencies, amplitudes):
    centroid = calculate_spectral_centroid(frequencies, amplitudes)
    amplitude_sum = np.sum(amplitudes)

    if amplitude_sum == 0:
        return 0.0

    spread = np.sqrt(np.sum(((frequencies - centroid) ** 2) * amplitudes) / amplitude_sum)

    return float(spread)


def safe_divide(a, b):
    if abs(b) < 1e-12:
        return 0.0

    return float(a / b)


def extract_frequency_features_for_signal(values, sampling_rate, prefix):
    frequencies, amplitudes = calculate_fft(values, sampling_rate)

    energy_0_5 = calculate_band_energy(frequencies, amplitudes, 0, 5)
    energy_5_15 = calculate_band_energy(frequencies, amplitudes, 5, 15)
    energy_15_30 = calculate_band_energy(frequencies, amplitudes, 15, 30)
    energy_30_50 = calculate_band_energy(frequencies, amplitudes, 30, 50)
    energy_0_10 = calculate_band_energy(frequencies, amplitudes, 0, 10)
    energy_10_25 = calculate_band_energy(frequencies, amplitudes, 10, 25)
    energy_25_50 = calculate_band_energy(frequencies, amplitudes, 25, 50)
    total_energy = float(np.sum(amplitudes ** 2))

    features = {
        f"{prefix}_dominant_frequency": calculate_dominant_frequency(frequencies, amplitudes),
        f"{prefix}_spectral_centroid": calculate_spectral_centroid(frequencies, amplitudes),
        f"{prefix}_spectral_spread": calculate_spectral_spread(frequencies, amplitudes),
        f"{prefix}_energy_0_5_hz": energy_0_5,
        f"{prefix}_energy_5_15_hz": energy_5_15,
        f"{prefix}_energy_15_30_hz": energy_15_30,
        f"{prefix}_energy_30_50_hz": energy_30_50,
        f"{prefix}_energy_0_10_hz": energy_0_10,
        f"{prefix}_energy_10_25_hz": energy_10_25,
        f"{prefix}_energy_25_50_hz": energy_25_50,
        f"{prefix}_total_spectral_energy": total_energy,
        f"{prefix}_low_to_mid_energy_ratio": safe_divide(energy_0_10, energy_10_25),
        f"{prefix}_high_to_mid_energy_ratio": safe_divide(energy_25_50, energy_10_25),
        f"{prefix}_high_frequency_energy_ratio": safe_divide(energy_25_50, total_energy),
        f"{prefix}_low_frequency_energy_ratio": safe_divide(energy_0_10, total_energy)
    }

    return features


def extract_frequency_features(window, sampling_rate):
    features = {}

    vibration_columns = [
        "vibration_x",
        "vibration_y",
        "vibration_z"
    ]

    for column in vibration_columns:
        features.update(
            extract_frequency_features_for_signal(
                values=window[column].values,
                sampling_rate=sampling_rate,
                prefix=column
            )
        )

    vibration_magnitude = np.sqrt(
        window["vibration_x"].values ** 2
        + window["vibration_y"].values ** 2
        + window["vibration_z"].values ** 2
    )

    features.update(
        extract_frequency_features_for_signal(
            values=vibration_magnitude,
            sampling_rate=sampling_rate,
            prefix="vibration_magnitude"
        )
    )

    features["axis_energy_ratio_xy"] = safe_divide(
        features["vibration_x_total_spectral_energy"],
        features["vibration_y_total_spectral_energy"]
    )

    features["axis_energy_ratio_xz"] = safe_divide(
        features["vibration_x_total_spectral_energy"],
        features["vibration_z_total_spectral_energy"]
    )

    features["axis_energy_ratio_yz"] = safe_divide(
        features["vibration_y_total_spectral_energy"],
        features["vibration_z_total_spectral_energy"]
    )

    return features
