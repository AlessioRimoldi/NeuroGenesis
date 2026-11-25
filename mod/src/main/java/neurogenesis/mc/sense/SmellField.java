package neurogenesis.mc.sense;

import java.util.List;

import net.minecraft.world.phys.Vec3;

/**
 * A 2-D diffusing scalar field over the arena floor ("smell"). Each game tick: sources emit, the field diffuses
 * (explicit finite differences, sub-stepped for stability) and decays. The arena walls are no-flux boundaries.
 *
 * <p>Pure Java with no Minecraft state, so it can be unit-tested. Coordinates are world blocks; the grid covers
 * [originX, originX + sizeX) x [originZ, originZ + sizeZ) with square cells of {@code cellSize} blocks.
 */
public final class SmellField {
	private final double originX, originZ, cellSize;
	private final int nx, nz;
	private final double diffusion, decay, emission;
	private final int substeps;
	private final double alpha;   // D * dt / dx^2 per substep
	private double[] c, next;

	public SmellField(double originX, double originZ, int sizeX, int sizeZ, double cellSize,
					  double diffusion, double decay, double emission) {
		if (cellSize <= 0 || diffusion < 0 || decay < 0 || decay >= 1) {
			throw new IllegalArgumentException("invalid smell parameters");
		}
		this.originX = originX;
		this.originZ = originZ;
		this.cellSize = cellSize;
		this.nx = Math.max(1, (int) Math.ceil(sizeX / cellSize));
		this.nz = Math.max(1, (int) Math.ceil(sizeZ / cellSize));
		this.diffusion = diffusion;
		this.decay = decay;
		this.emission = emission;
		double perTick = diffusion / (cellSize * cellSize);
		this.substeps = Math.max(1, (int) Math.ceil(perTick / 0.24));
		this.alpha = perTick / substeps;
		this.c = new double[nx * nz];
		this.next = new double[nx * nz];
	}

	public int nx() {
		return nx;
	}

	public int nz() {
		return nz;
	}

	public int substeps() {
		return substeps;
	}

	/** Advances one game tick with the given source positions (world coordinates; y ignored). */
	public void step(List<Vec3> sources) {
		for (Vec3 s : sources) {
			int i = cellX(s.x);
			int k = cellZ(s.z);
			if (i >= 0 && k >= 0) {
				c[k * nx + i] += emission;
			}
		}
		for (int s = 0; s < substeps; s++) {
			for (int k = 0; k < nz; k++) {
				for (int i = 0; i < nx; i++) {
					int idx = k * nx + i;
					double center = c[idx];
					// No-flux boundary: a missing neighbour contributes the centre value (zero gradient).
					double w = i > 0 ? c[idx - 1] : center;
					double e = i < nx - 1 ? c[idx + 1] : center;
					double n = k > 0 ? c[idx - nx] : center;
					double so = k < nz - 1 ? c[idx + nx] : center;
					next[idx] = center + alpha * (w + e + n + so - 4 * center);
				}
			}
			double[] t = c;
			c = next;
			next = t;
		}
		if (decay > 0) {
			double keep = 1.0 - decay;
			for (int idx = 0; idx < c.length; idx++) {
				c[idx] *= keep;
			}
		}
	}

	/** Bilinear sample at a world position (cell values live at cell centres; clamped at the edges). */
	public double sample(double x, double z) {
		double gx = (x - originX) / cellSize - 0.5;
		double gz = (z - originZ) / cellSize - 0.5;
		gx = Math.max(0, Math.min(nx - 1, gx));
		gz = Math.max(0, Math.min(nz - 1, gz));
		int i0 = (int) Math.floor(gx);
		int k0 = (int) Math.floor(gz);
		int i1 = Math.min(i0 + 1, nx - 1);
		int k1 = Math.min(k0 + 1, nz - 1);
		double fx = gx - i0;
		double fz = gz - k0;
		double a = c[k0 * nx + i0] * (1 - fx) + c[k0 * nx + i1] * fx;
		double b = c[k1 * nx + i0] * (1 - fx) + c[k1 * nx + i1] * fx;
		return a * (1 - fz) + b * fz;
	}

	public double total() {
		double sum = 0;
		for (double v : c) {
			sum += v;
		}
		return sum;
	}

	public double get(int i, int k) {
		return c[k * nx + i];
	}

	private int cellX(double x) {
		int i = (int) Math.floor((x - originX) / cellSize);
		return i >= 0 && i < nx ? i : -1;
	}

	private int cellZ(double z) {
		int k = (int) Math.floor((z - originZ) / cellSize);
		return k >= 0 && k < nz ? k : -1;
	}
}
