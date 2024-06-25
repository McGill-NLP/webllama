import gymnasium as gym
import browsergym.core  # register the openended task as a gym environment
from examples.browsergym.agent import WebLinxAgent

agent = WebLinxAgent()

env = gym.make(
    "browsergym/openended",
    headless=False,
    wait_for_user_message=False,
    action_mapping=agent.get_action_mapping(),
    task_kwargs={"start_url": "chrome://newtab"},
    # task_kwargs={"start_url": "https://en.wikipedia.org"},
)

agent.reset()
obs, info = env.reset()

done = False
while not done:
    action = agent.get_action(obs)
    obs, reward, terminated, truncated, info = env.step(action)
