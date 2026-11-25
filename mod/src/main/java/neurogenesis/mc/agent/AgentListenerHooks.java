package neurogenesis.mc.agent;

/** Implemented on ServerGamePacketListenerImpl by mixin. */
public interface AgentListenerHooks {
	/** As if the client had reported that it finished loading the world (ends spawn invulnerability). */
	void ng$markClientLoaded();
}
