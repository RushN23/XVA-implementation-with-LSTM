# XVA-implementation-with-LSTM


## Project Overview

This project demonstrates an advanced approach to calculating Credit Valuation Adjustment (CVA) by integrating Monte Carlo simulations with Long Short-Term Memory (LSTM) neural networks. The primary goal is to leverage the predictive power of LSTMs to forecast Expected Positive Exposure (EPE), thereby accelerating the CVA calculation process and enabling efficient stress testing, which are typically computationally intensive when relying solely on Monte Carlo methods.

## Features

*   **Monte Carlo Simulation:** Simulates key market risk factors:
    *   **FX Rates:** Modeled using Geometric Brownian Motion (GBM).
    *   **Interest Rates:** Modeled using the Hull-White model.
    *   **Hazard Rates:** Modeled using the Cox-Ingersoll-Ross (CIR) model.
*   **Portfolio Mark-to-Market (MtM):** Calculates the MtM for a portfolio of financial instruments (e.g., FX Forwards, IR Swaps) under simulated market conditions.
*   **Exposure Calculation:** Determines counterparty exposure, incorporating netting and collateral agreements.
*   **Wrong-Way Risk (WWR) Adjustment:** Applies a simplified adjustment to hazard rates to account for the correlation between exposure and counterparty credit quality.
*   **Expected Positive Exposure (EPE):** Calculates the average positive exposure across all simulation paths.
*   **Credit Valuation Adjustment (CVA):** Computes CVA based on EPE, hazard rates, Loss Given Default (LGD), and discount factors.
*   **LSTM-based EPE Forecasting:** Trains a stacked LSTM neural network on Monte Carlo-generated EPE data to learn its temporal patterns and predict future EPE values.
*   **Accelerated CVA Calculation:** Uses LSTM-predicted EPE to calculate CVA, offering a faster alternative to full Monte Carlo re-simulations.
*   **Backtesting:** Evaluates the accuracy of the LSTM model's EPE predictions using Mean Squared Error (MSE).
*   **Stress Testing:** Demonstrates how to apply market shocks to EPE to assess the impact on CVA under adverse scenarios.

## Project Structure

The core logic is contained within `xva_lstm_advanced_final.py`.


## Getting Started

### Prerequisites

*   Python 3.x
*   `numpy`
*   `pandas` (though not directly used in the provided `xva_lstm_advanced_final.py` for data loading, it's common for financial data handling)
*   `tensorflow` (for Keras and LSTM)
*   `scikit-learn` (for `MinMaxScaler` and `train_test_split`)
*   `matplotlib` (for plotting)

You can install the required libraries using pip:

```bash
pip install numpy pandas tensorflow scikit-learn matplotlib
