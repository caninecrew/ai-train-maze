from games.registry import list_games


def test_registered_adapters_contract():
    for game in list_games():
        env = game.make_env(render_mode=None)
        obs, info = env.reset()
        assert env.observation_space.contains(obs)
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        assert env.observation_space.contains(obs)
        assert isinstance(reward, float)
        assert terminated in (True, False)
        assert truncated in (True, False)
        env.close()
