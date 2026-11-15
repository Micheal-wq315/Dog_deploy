"""Differentiable rewards, full-return actor objective and TD(lambda) critic targets."""

import math
from contextlib import contextmanager
import torch
from .world_model import GRAVITY, LIN_VEL, ANG_VEL, LEG_POS, LEG_VEL, WHEEL_VEL
from .policy import frame, push_history, critic_input


@contextmanager
def frozen(*modules):
  """Freeze weights without disabling gradients through inputs/actions."""
  parameters = [(p, p.requires_grad) for m in modules for p in m.parameters()]
  try:
    for p, _ in parameters:
      p.requires_grad_(False)
    yield
  finally:
    for p, enabled in parameters:
      p.requires_grad_(enabled)


def survival(state):
  gravity = torch.nn.functional.normalize(state[:, GRAVITY], dim=-1, eps=1e-6)
  # Smooth proxy for the environment's 70-degree fall cutoff, width ~5 deg.
  return torch.sigmoid((-gravity[:, 2] - math.cos(math.radians(70))) / 0.08)


from .task import reward_terms


def lambda_returns(rewards, continuation, next_values, gamma=0.99, lam=0.95):
  carry = next_values[-1]
  result = []
  for t in reversed(range(len(rewards))):
    carry = rewards[t] + gamma * continuation[t] * ((1 - lam) * next_values[t] + lam * carry)
    result.append(carry)
  return torch.stack(result[::-1])


def discounted_objective(rewards, continuation, terminal_value, gamma=.99):
  alive = torch.ones_like(terminal_value)
  total = torch.zeros_like(terminal_value)
  discount = 1.
  for reward, cont in zip(rewards, continuation):
    total = total + discount * alive * reward
    alive = alive * cont
    discount *= gamma
  return total + discount * alive * terminal_value


def make_imagined_step(model, actor, constants, weights, env_action_order,
                       dt=.02, noise=True, field=None):
  """Pure single-step graph, shared by eager and compiled execution."""
  def step(state, previous, history, pose, scan, command, context, bias):
    action = actor(history)
    model_context = context if field is None else torch.cat((context, scan), -1)
    collision_prob = None
    contacts = None
    if constants.get('nonwheel_collision', False):
      nxt, delta, contact_prediction, collision_logit = model.forward_all(state, action, previous, model_context)
      collision_prob = collision_logit.squeeze(-1).sigmoid()
      contacts = model.decode_contacts(contact_prediction)[0]
    else:
      nxt, delta = model(state, action, previous, model_context)
    if field is not None:
      from .terrain import advance_pose
      pose = advance_pose(pose, delta)
    next_scan = None if field is None else field.scan(nxt, pose)
    cont = survival(nxt)
    if collision_prob is not None:
      cont = cont * (1. - collision_prob)
    terms = reward_terms(nxt, action, previous, command, constants, cont,
                         scan=next_scan, pose=pose, field=field, contacts=contacts)
    reward = dt * sum(weights[name] * value for name, value in terms.items())
    latest = frame(nxt, action, command, bias, noise=noise)
    history = push_history(history, latest, actor.history)
    return nxt, action, history, pose, next_scan, reward, cont
  return step


def rollout(model, actor, target_critic, seed, constants, weights, env_action_order,
            horizon=16, dt=0.02, gamma=0.99, lam=0.95, noise=True, field=None,
            step_fn=None):
  """Actor: full discounted return + terminal value. Critic: detached TD(lambda).

  Caller freezes dynamics and target critic weights, preserving state gradients.
  Reuse one terrain scan at each state, including the rollout's initial state.
  """
  state, previous, history = seed["state"], seed["previous"], seed["history"]
  command, context, bias = seed["command"], seed["context"], seed["bias"]
  pose = seed.get("world_pose")
  def value_input(s, prev, scan):
    base = critic_input(s, command, context, prev)
    return base if field is None else torch.cat((base, scan), -1)
  if step_fn is None:
    step_fn = make_imagined_step(model, actor, constants, weights, env_action_order,
                                 dt, noise, field)
  rewards, continuations, inputs, next_inputs = [], [], [], []
  scan = None if field is None else field.scan(state, pose)
  for _ in range(horizon):
    inputs.append(value_input(state, previous, scan).detach())
    state, previous, history, pose, scan, reward, cont = step_fn(
      state, previous, history, pose, scan, command, context, bias)
    rewards.append(reward)
    continuations.append(cont)
    next_inputs.append(value_input(state, previous, scan).detach())
  final_value = target_critic(value_input(state, previous, scan)).squeeze(-1)
  r, c = torch.stack(rewards), torch.stack(continuations)
  objective = discounted_objective(r, c, final_value, gamma)
  with torch.no_grad():
    next_values = target_critic(torch.stack(next_inputs)).squeeze(-1)
    targets = lambda_returns(r.detach(), c.detach(), next_values, gamma, lam)
  return -objective.mean(), torch.stack(inputs), targets
