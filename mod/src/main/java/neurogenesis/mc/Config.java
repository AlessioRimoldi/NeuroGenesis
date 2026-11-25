package neurogenesis.mc;

/**
 * Runtime configuration, read from system properties (-Dneurogenesis.*) first, then environment variables (NG_*).
 */
public record Config(String bindAddress, int port, boolean controlEnabled) {
	public static final int DEFAULT_PORT = 25570;

	public static Config load() {
		String bind = get("neurogenesis.bind", "NG_CONTROL_BIND", "0.0.0.0");
		int port = Integer.parseInt(get("neurogenesis.port", "NG_CONTROL_PORT", Integer.toString(DEFAULT_PORT)));
		// Fabric's GameTest runner sets this property; tests drive agents directly, without the socket.
		boolean gameTest = System.getProperty("fabric-api.gametest") != null;
		boolean enabled = !gameTest && Boolean.parseBoolean(get("neurogenesis.control", "NG_CONTROL", "true"));
		return new Config(bind, port, enabled);
	}

	private static String get(String property, String env, String fallback) {
		String value = System.getProperty(property);
		if (value == null || value.isBlank()) {
			value = System.getenv(env);
		}
		return value == null || value.isBlank() ? fallback : value.trim();
	}
}
