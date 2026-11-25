package neurogenesis.mc.input;

import java.util.ArrayList;
import java.util.List;

import neurogenesis.mc.proto.v1.AgentAction;
import neurogenesis.mc.proto.v1.ClickSlot;
import neurogenesis.mc.proto.v1.InventoryAction;

/** The keys an agent holds for the current step, plus one-shot actions consumed on the step's first tick. */
public final class AgentInput {
	public boolean forward, back, left, right, jump, sneak, sprint, attack, use;

	public float cameraDyaw, cameraDpitch;
	public int hotbar;
	public boolean drop, dropStack, swapHands;
	public InventoryAction inventory = InventoryAction.INVENTORY_NONE;
	public final List<ClickSlot> clicks = new ArrayList<>();
	private boolean oneShotsPending;

	public void set(AgentAction a) {
		forward = a.getForward();
		back = a.getBack();
		left = a.getLeft();
		right = a.getRight();
		jump = a.getJump();
		sneak = a.getSneak();
		sprint = a.getSprint();
		attack = a.getAttack();
		use = a.getUse();
		cameraDyaw = a.getCameraDyaw();
		cameraDpitch = a.getCameraDpitch();
		hotbar = a.getHotbar();
		drop = a.getDrop();
		dropStack = a.getDropStack();
		swapHands = a.getSwapHands();
		inventory = a.getInventory();
		clicks.clear();
		clicks.addAll(a.getClicksList());
		oneShotsPending = true;
	}

	public void releaseAll() {
		forward = back = left = right = jump = sneak = sprint = attack = use = false;
		clearOneShots();
	}

	public boolean hasOneShots() {
		return oneShotsPending;
	}

	public void clearOneShots() {
		cameraDyaw = cameraDpitch = 0;
		hotbar = 0;
		drop = dropStack = swapHands = false;
		inventory = InventoryAction.INVENTORY_NONE;
		clicks.clear();
		oneShotsPending = false;
	}
}
