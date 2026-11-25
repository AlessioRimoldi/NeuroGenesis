package neurogenesis.mc.sim;

import neurogenesis.mc.proto.v1.Reset;
import neurogenesis.mc.proto.v1.Spec;
import neurogenesis.mc.proto.v1.Step;
import neurogenesis.mc.proto.v1.StepResult;

/** What the lockstep loop drives. All methods are called on the server thread. */
public interface Simulation {
	Spec spec();

	/** Synchronously sets up a new episode (arena, agents). Throws IllegalArgumentException for bad configs. */
	void reset(Reset reset);

	/** True once a reset has happened, so steps are allowed. */
	boolean ready();

	/** Installs the actions to apply during the coming ticks of a step. */
	void applyActions(Step step);

	/** Called after every game tick while a session is attached. */
	void afterTick();

	/** Builds the observation for all agents after the last tick of a step. */
	StepResult.Builder observe();

	/** The Python session ended: remove agents, release resources. */
	void detach();
}
