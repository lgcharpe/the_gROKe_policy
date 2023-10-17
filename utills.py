import numpy as np
import yaml
from pymgrid.microgrid.utils.step import MicrogridStep
from pymgrid import Microgrid
from pymgrid.envs import DiscreteMicrogridEnv
from pymgrid.algos import RuleBasedControl
from pymgrid.microgrid import DEFAULT_HORIZON
from pymgrid.modules.base import BaseTimeSeriesMicrogridModule


class RenewableModuleCustom(BaseTimeSeriesMicrogridModule):
    """
    A renewable energy module.

    The classic examples of renewables are photovoltaics (PV) and wind turbines.

    Parameters
    ----------
    time_series : array-like, shape (n_steps, )
        Time series of renewable production.

    forecaster : callable, float, "oracle", or None, default None.
        Function that gives a forecast n-steps ahead.

        * If ``callable``, must take as arguments ``(val_c: float, val_{c+n}: float, n: int)``, where

          * ``val_c`` is the current value in the time series: ``self.time_series[self.current_step]``

          * ``val_{c+n}`` is the value in the time series n steps in the future

          * n is the number of steps in the future at which we are forecasting.

          The output ``forecast = forecaster(val_c, val_{c+n}, n)`` must have the same sign
          as the inputs ``val_c`` and ``val_{c+n}``.

        * If ``float``, serves as a standard deviation for a mean-zero gaussian noise function
          that is added to the true value.

        * If ``"oracle"``, gives a perfect forecast.

        * If ``None``, no forecast.

    forecast_horizon : int.
        Number of steps in the future to forecast. If forecaster is None, ignored and 0 is returned.

    forecaster_increase_uncertainty : bool, default False
        Whether to increase uncertainty for farther-out dates if using a GaussianNoiseForecaster. Ignored otherwise.

    provided_energy_name: str, default "renewable_used"
        Name of the energy provided by this module, to be used in logging.

    raise_errors : bool, default False
        Whether to raise errors if bounds are exceeded in an action.
        If False, actions are clipped to the limit possible.

    """
    module_type = ('renewable', 'flex')
    yaml_tag = u"!RenewableModule"
    yaml_loader = yaml.SafeLoader
    yaml_dumper = yaml.SafeDumper

    state_components = np.array(["renewable"], dtype=object)

    def __init__(self,
                 time_series,
                 raise_errors=False,
                 forecaster=None,
                 forecast_horizon=DEFAULT_HORIZON,
                 forecaster_increase_uncertainty=False,
                 forecaster_relative_noise=False,
                 initial_step=0,
                 final_step=-1,
                 normalized_action_bounds=(0, 1),
                 provided_energy_name='renewable_used',
                 operating_cost=0.0):
        super().__init__(
            time_series,
            raise_errors,
            forecaster=forecaster,
            forecast_horizon=forecast_horizon,
            forecaster_increase_uncertainty=forecaster_increase_uncertainty,
            #forecaster_relative_noise=forecaster_relative_noise,
            initial_step=initial_step,
            final_step=final_step,
            normalized_action_bounds=normalized_action_bounds,
            provided_energy_name=provided_energy_name,
            absorbed_energy_name=None
        )
        self.operating_cost = operating_cost

    def step(self, action):
        """
        Take one step in the module, attempting to draw or send ``action`` amount of energy.

        Parameters
        ----------
        action : float or np.ndarray, shape (1,)
            The amount of energy to draw or send.

            If ``normalized``, the action is assumed to be normalized and is un-normalized into the range
            [:attr:`.BaseModule.min_act`, :attr:`.BaseModule.max_act`].

            If the **unnormalized** action is positive, the module acts as a source and provides energy to the
            microgrid. Otherwise, the module acts as a sink and absorbs energy.

            If the unnormalized action implies acting as a sink and ``is_sink`` is False -- or the converse -- an
            ``AssertionError`` is raised.

        normalized : bool, default True
            Whether ``action`` is normalized. If True, action is assumed to be normalized and is un-normalized into the
            range [:attr:`.BaseModule.min_act`, :attr:`.BaseModule.max_act`].

        Raises
        ------
        AssertionError
            If action implies acting as a source and module is not a source. Likewise if action implies acting as a
            sink and module is not a sink.

        Returns
        -------
        observation : np.ndarray
            State of the module after taking action ``action``.
        reward : float
            Reward/cost after taking the action.
        done : bool
            Whether the module terminates.
        info : dict
            Additional information from this step.
            Will include either``provided_energy`` or ``absorbed_energy`` as a key, denoting the amount of energy
            this module provided to or absorbed from the microgrid.

        """

        if action == 1:
            reward, done, info = self.update(self.max_production, as_source=True)
        else:
            reward, done, info = self.update(0, as_source=True)

        state_dict = self.state_dict()
        self._log(state_dict, reward=reward, **info)
        self._update_step()

        obs = self.to_normalized(self.state, obs=True)

        return obs, reward, done, info

    def update(self, external_energy_change, as_source=False, as_sink=False):
        assert as_source, f'Class {self.__class__.__name__} can only be used as a source.'
        assert external_energy_change <= self.current_renewable, f'Cannot provide more than {self.current_renewable}'

        info = {'provided_energy': external_energy_change,
                'curtailment': self.current_renewable-external_energy_change}

        reward = -1.0 * self.operating_cost * external_energy_change

        return reward, self._done(), info

    @property
    def max_production(self):
        return self.current_renewable

    @property
    def current_renewable(self):
        """
        Current renewable production.

        Returns
        -------
        renewable : float
            Renewable production.

        """
        return self._time_series[self._current_step].item()

    @property
    def is_source(self):
        return True

    @property
    def production_marginal_cost(self):
        return self.operating_cost

from pymgrid.modules.grid_module import GridModule

class GridModuleCustom(GridModule):

    module_type = ('grid', 'controllable')

    yaml_tag = u"!GridModule"
    yaml_loader = yaml.SafeLoader
    yaml_dumper = yaml.SafeDumper

    state_components = np.array(['import_price', 'export_price', 'co2_per_kwh', 'grid_status'], dtype=object)

    def step(self, unbalanced_load, sold):

        if unbalanced_load > 0:
            reward, done, info = self.update(unbalanced_load, sold, as_sink=True)
        else:
            reward, done, info = self.update(-unbalanced_load, sold, as_source=True) 

        state_dict = self.state_dict()
        self._log(state_dict, reward=reward, **info)
        self._update_step()

        obs = self.to_normalized(self.state, obs=True)

        return obs, reward, done, info

    def update(self, external_energy_change, sold, as_source=False, as_sink=False):
        assert as_source + as_sink == 1, 'Must act as either source or sink but not both or neither.'
        reward_external = self.get_cost(external_energy_change, as_source, as_sink)
        reward_sold = self.get_cost(sold, False, True)
        if as_source:
            info = {"provided_energy": external_energy_change, "absorbed_energy": sold,
                'co2_production': self.get_co2_production(external_energy_change, as_source, as_sink)}
        else:
            info = {"provided_energy": 0.0, "absorbed_energy": sold + external_energy_change,
                'co2_production': self.get_co2_production(external_energy_change, as_source, as_sink)}

        return reward_external + reward_sold, self._done(), info

from pymgrid.modules.genset_module import GensetModule

class GensetModuleDiscrete(GensetModule):

    module_type = 'genset', 'flex'
    yaml_tag = f"!Genset"
    yaml_dumper = yaml.SafeDumper
    yaml_loader = yaml.SafeLoader

    _energy_pos = 1

    def __init__(self,
                 running_min_production,
                 running_max_production,
                 genset_cost,
                 co2_per_unit=0.0,
                 cost_per_unit_co2=0.0,
                 start_up_time=0,
                 wind_down_time=0,
                 allow_abortion=True,
                 init_start_up=True,
                 initial_step=0,
                 normalized_action_bounds=(0, 1),
                 raise_errors=False,
                 provided_energy_name='genset_production',
                 num_buckets=1):

        super().__init__(running_min_production,
                         running_max_production,
                         genset_cost,
                         co2_per_unit,
                         cost_per_unit_co2,
                         start_up_time,
                         wind_down_time,
                         allow_abortion,
                         init_start_up,
                         initial_step,
                         normalized_action_bounds,
                         raise_errors,
                         provided_energy_name)

        self.increments = (self.running_max_production - self.running_min_production) / num_buckets
        self.num_buckets = num_buckets

    def step(self, action):
        """
        Take one step in the module, attempting to draw or send ``action`` amount of energy.

        Parameters
        ----------
        action : float or np.ndarray, shape (1,)
            The amount of energy to draw or send.

            If ``normalized``, the action is assumed to be normalized and is un-normalized into the range
            [:attr:`.BaseModule.min_act`, :attr:`.BaseModule.max_act`].

            If the **unnormalized** action is positive, the module acts as a source and provides energy to the
            microgrid. Otherwise, the module acts as a sink and absorbs energy.

            If the unnormalized action implies acting as a sink and ``is_sink`` is False -- or the converse -- an
            ``AssertionError`` is raised.

        normalized : bool, default True
            Whether ``action`` is normalized. If True, action is assumed to be normalized and is un-normalized into the
            range [:attr:`.BaseModule.min_act`, :attr:`.BaseModule.max_act`].

        Raises
        ------
        AssertionError
            If action implies acting as a source and module is not a source. Likewise if action implies acting as a
            sink and module is not a sink.

        Returns
        -------
        observation : np.ndarray
            State of the module after taking action ``action``.
        reward : float
            Reward/cost after taking the action.
        done : bool
            Whether the module terminates.
        info : dict
            Additional information from this step.
            Will include either``provided_energy`` or ``absorbed_energy`` as a key, denoting the amount of energy
            this module provided to or absorbed from the microgrid.

        """

        reward, done, info = self.update(action * self.increments, as_source=True)

        state_dict = self.state_dict()
        self._log(state_dict, reward=reward, **info)
        self._update_step()

        obs = self.to_normalized(self.state, obs=True)

        return obs, reward, done, info

class MicrogridStepCustom(MicrogridStep):

    def append(self, module_name, obs, reward, done, info):
        try:
            self._obs[module_name] = obs
        except KeyError:
            self._obs[module_name] = obs
        self._reward += reward
        if done:
            self._done = True

        try:
            self._info[module_name].append(info)
        except KeyError:
            self._info[module_name] = [info]

        for key, value in info.items():
            try:
                self._info[key].append(value)
            except KeyError:
                pass

class Microgrid2(Microgrid):

    yaml_tag = u"!Microgrid"
    """Tag used for yaml serialization."""
    yaml_dumper = yaml.SafeDumper
    yaml_loader = yaml.SafeLoader

    def run(self, control, normalized=True):
        """

        Run the microgrid for a single step.

        Parameters
        ----------
        control : dict[str, list[float]]
            Actions to pass to each fixed module.
        normalized : bool, default True
            Whether ``control`` is a normalized value or not. If not, each module de-normalizes its respective action.

        Returns
        -------
        observation : dict[str, list[float]]
            Observations of each module after using the passed ``control``.
        reward : float
            Reward/cost of running the microgrid. A positive value implies revenue while a negative
            value is a cost.
        done : bool
            Whether the microgrid terminates.
        info : dict
            Additional information from this step.

        """
        control_copy = control.copy()
        microgrid_step = MicrogridStepCustom(reward_shaping_func=self.reward_shaping_func, cost_info=self.get_cost_info())

        for name, modules in self.fixed.iterdict():
            for module in modules:
                o, r, d, i = module.step(0.0, normalized=False)
                microgrid_step.append(name, o, r, d, i)

        fixed_provided, fixed_consumed, _, _ = microgrid_step.balance()
        log_dict = self._get_log_dict(fixed_provided, fixed_consumed, prefix='fixed')

        sold = 0.0
        for name, modules in self.flex.iterdict():
            try:
                module_controls = control_copy.pop(name)
            except KeyError:
                raise ValueError(f'Control for module "{name}" not found. Available controls:\n\t{control.keys()}')
            else:
                try:
                    _zip = zip(modules, module_controls)
                except TypeError:
                    _zip = zip(modules, [module_controls])

            for module, _control in _zip:
                module_step = module.step(_control)  # obs, reward, done, info.
                microgrid_step.append(name, module_step[0], module_step[1], module_step[2], module_step[3])
                if control_copy["status"][name] == 1:
                    sold += module_step[3]["provided_energy"]
                

        flex_fixed_provided, flex_fixed_consumed, _, _ = microgrid_step.balance()
        difference = flex_fixed_provided - flex_fixed_consumed - sold

        log_dict = self._get_log_dict(
            flex_fixed_provided-fixed_provided,
            flex_fixed_consumed-fixed_consumed,
            log_dict=log_dict,
            prefix='flex'
        )

        

        # if difference > 0, have an excess. Try to use flex sinks to dissapate
        # otherwise, insufficient. Use flex sources to make up

        for name, modules in self.controllable.iterdict():
            for module in modules:
                if name == "grid":
                    module_step = module.step(difference, sold)
                else:
                    module_step = module.step(difference)
                microgrid_step.append(name, *module_step)

        provided, consumed, reward, shaped_reward = microgrid_step.balance()

        log_dict = self._get_log_dict(
            provided-flex_fixed_provided,
            consumed-flex_fixed_consumed,
            log_dict=log_dict,
            prefix='controllable'
        )

        log_dict = self._get_log_dict(provided, consumed, log_dict=log_dict, prefix='overall')

        self._balance_logger.log(reward=reward, shaped_reward=shaped_reward, **log_dict)

        if not np.isclose(provided, consumed):
            raise RuntimeError('Microgrid modules unable to balance energy production with consumption.\n'
                               '')


        return microgrid_step.output()

    def reset(self):
        """
        Reset the microgrid and flush the log.

        Returns
        -------
        dict[str, list[float]]
            Observations from resetting the modules as well as the flushed balance log.
        """
        self._set_trajectory()
        return {
            **{module.name[0]: module.reset() for module in self.module_list},
        }
