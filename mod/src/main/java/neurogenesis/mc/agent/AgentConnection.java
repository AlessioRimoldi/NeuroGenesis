package neurogenesis.mc.agent;

import io.netty.channel.ChannelFutureListener;
import io.netty.channel.embedded.EmbeddedChannel;
import net.minecraft.network.Connection;
import net.minecraft.network.PacketListener;
import net.minecraft.network.ProtocolInfo;
import net.minecraft.network.protocol.Packet;
import net.minecraft.network.protocol.PacketFlow;
import org.jspecify.annotations.Nullable;

import neurogenesis.mc.mixin.ConnectionAccessor;

/**
 * A connection with no client behind it: outgoing packets are dropped and it never disconnects.
 * Adapted from Carpet's FakeClientConnection (MIT, see THIRD_PARTY_NOTICES).
 */
public final class AgentConnection extends Connection {
	public AgentConnection() {
		super(PacketFlow.SERVERBOUND);
		// An open channel makes isConnected() true, which some vanilla code checks (e.g. ender pearl teleports).
		((ConnectionAccessor) (Object) this).ng$setChannel(new EmbeddedChannel());
	}

	@Override
	public void send(Packet<?> packet, @Nullable ChannelFutureListener listener, boolean flush) {
	}

	@Override
	public void setReadOnly() {
	}

	@Override
	public void handleDisconnection() {
	}

	@Override
	public void setListenerForServerboundHandshake(PacketListener listener) {
	}

	@Override
	public <T extends PacketListener> void setupInboundProtocol(ProtocolInfo<T> protocol, T listener) {
	}
}
