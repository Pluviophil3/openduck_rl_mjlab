
python scripts/train.py \
  OpenDuck-Tracking-No-State-Estimation \
  --motion-file=src/assets/motions/open_duck/forward_headshake_40deg_04hz_50hz_realxml_backlash.npz \
  --env.scene.num-envs=4096 \
  --agent.run-name=forward_bachlash_unlimited