# project
기계학습 기말

# Markov Chain Routine Recommender (Python)

A simple simulation-based recommender that models daily states with a Markov Chain and compares different policies to maximize expected reward.

## 1) Overview
This project models a user's daily condition as a **state** (e.g., energy level, mood, day type) and uses a **transition matrix** to simulate how the state changes over time.  
Given candidate activities (study, exercise, rest, movie, etc.), it evaluates policies via Monte Carlo simulation and recommends an activity with high **expected reward**.

## 2) Key Features
- Define states and actions (activities)
- Simulate state transitions with a Markov Chain
- Compare policies:
  - Greedy (choose best immediate reward)
  - Heuristic (rule-based)
  - Random baseline
- Visualize:
  - State distribution over time
  - Average reward comparison across policies

## 3) Tech Stack
- Python 3.10+
- numpy, pandas
- matplotlib
- (optional) jupyter

## 4) Project Structure

## 5) How to Run

```bash
pip install -r requirements.txt
python src/simulate.py

