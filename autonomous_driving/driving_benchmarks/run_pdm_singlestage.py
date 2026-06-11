"""Single-stage (classic NAVSIM v1) PDMS evaluation — runs on mini/navtest without
the two-stage synthetic-scene machinery. CPU-only, non-reactive (log-replay) traffic.

Usage:
  python run_pdm_singlestage.py train_test_split=navmini agent=constant_velocity_agent \
         experiment_name=cv_agent metric_cache_path=$NAVSIM_EXP_ROOT/metric_cache
"""
import logging
import os
import traceback
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
from hydra.utils import instantiate
from omegaconf import DictConfig

from navsim.agents.abstract_agent import AbstractAgent
from navsim.common.dataclasses import SensorConfig
from navsim.common.dataloader import MetricCacheLoader, SceneFilter, SceneLoader
from navsim.evaluate.pdm_score import pdm_score
from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_scorer import PDMScorer
from navsim.planning.simulation.planner.pdm_planner.simulation.pdm_simulator import PDMSimulator

logger = logging.getLogger(__name__)

CONFIG_PATH = "config/pdm_scoring"
CONFIG_NAME = "default_run_pdm_score"


@hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_NAME, version_base=None)
def main(cfg: DictConfig) -> None:
    # an empty dir to satisfy SceneLoader's synthetic-path args (unused for single stage)
    empty_dir = Path(os.environ["NAVSIM_EXP_ROOT"]) / "_empty"
    empty_dir.mkdir(parents=True, exist_ok=True)

    simulator: PDMSimulator = instantiate(cfg.simulator)
    scorer: PDMScorer = instantiate(cfg.scorer)
    agent: AbstractAgent = instantiate(cfg.agent)
    agent.initialize()

    metric_cache_loader = MetricCacheLoader(Path(cfg.metric_cache_path))
    scene_filter: SceneFilter = instantiate(cfg.train_test_split.scene_filter)
    scene_loader = SceneLoader(
        synthetic_sensor_path=empty_dir,
        original_sensor_path=empty_dir,
        data_path=Path(cfg.navsim_log_path),
        synthetic_scenes_path=empty_dir,
        scene_filter=scene_filter,
        sensor_config=agent.get_sensor_config(),
    )

    # classic NAVSIM v1: non-reactive (log-replay) background traffic
    traffic_agents_policy = instantiate(cfg.traffic_agents_policy.non_reactive, simulator.proposal_sampling)

    tokens = list(set(scene_loader.tokens_stage_one) & set(metric_cache_loader.tokens))
    logger.info(f"Scoring {len(tokens)} navmini scenarios (single-stage PDMS, CPU)...")

    rows = []
    for i, token in enumerate(tokens):
        try:
            metric_cache = metric_cache_loader.get_from_token(token)
            agent_input = scene_loader.get_agent_input_from_token(token)
            if agent.requires_scene:
                scene = scene_loader.get_scene_from_token(token)
                trajectory = agent.compute_trajectory(agent_input, scene)
            else:
                trajectory = agent.compute_trajectory(agent_input)
            row, _ = pdm_score(
                metric_cache=metric_cache,
                model_trajectory=trajectory,
                future_sampling=simulator.proposal_sampling,
                simulator=simulator,
                scorer=scorer,
                traffic_agents_policy=traffic_agents_policy,
            )
            row = row.copy()
            row["token"] = token
            rows.append(row)
        except Exception:
            logger.warning(f"Agent failed for token {token}")
            traceback.print_exc()
        if (i + 1) % 25 == 0:
            logger.info(f"  {i + 1}/{len(tokens)} done")

    df = pd.concat(rows, ignore_index=True)

    # report: mean over scenarios of the final PDMS and the interpretable sub-metrics
    sub_cols = [
        "no_at_fault_collisions", "drivable_area_compliance", "driving_direction_compliance",
        "traffic_light_compliance", "ego_progress", "time_to_collision_within_bound",
        "lane_keeping", "history_comfort", "comfort", "pdm_score",
    ]
    print("\n" + "=" * 48)
    print(f" NAVSIM single-stage PDMS  —  agent: {cfg.agent.get('_target_', 'agent').split('.')[-1]}")
    print(f" split: navmini   scenarios: {len(df)}")
    print("=" * 48)
    for c in sub_cols:
        if c in df.columns:
            label = "PDMS (final)" if c == "pdm_score" else c
            print(f"  {label:34s} {df[c].astype(float).mean():.4f}")
    print("=" * 48)

    out = Path(cfg.output_dir) / "single_stage_scores.csv"
    df.to_csv(out, index=False)
    print(f"per-scenario CSV: {out}")


if __name__ == "__main__":
    main()
