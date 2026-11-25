package neurogenesis.mc.sense;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.List;

import net.minecraft.world.phys.Vec3;
import org.junit.jupiter.api.Test;

class SmellFieldTest {
	private static final List<Vec3> NONE = List.of();

	@Test
	void massIsConservedWithoutDecay() {
		SmellField f = new SmellField(0, 0, 20, 20, 1, 0.2, 0, 1);
		List<Vec3> src = List.of(new Vec3(3.5, 64, 4.5));
		for (int t = 0; t < 300; t++) {
			f.step(src);
		}
		assertEquals(300.0, f.total(), 1e-6); // walls are no-flux: everything emitted stays in the arena
		for (int t = 0; t < 300; t++) {
			f.step(NONE);
		}
		assertEquals(300.0, f.total(), 1e-6);
	}

	@Test
	void fieldIsSymmetricAroundACentredSource() {
		SmellField f = new SmellField(0, 0, 21, 21, 1, 0.3, 0.01, 1);
		List<Vec3> src = List.of(new Vec3(10.5, 64, 10.5));
		for (int t = 0; t < 200; t++) {
			f.step(src);
		}
		for (int d = 1; d <= 8; d++) {
			double e = f.get(10 + d, 10), w = f.get(10 - d, 10), n = f.get(10, 10 - d), s = f.get(10, 10 + d);
			assertEquals(e, w, 1e-9);
			assertEquals(n, s, 1e-9);
			assertEquals(e, n, 1e-9);
		}
	}

	@Test
	void concentrationDecreasesWithDistanceAtSteadyState() {
		SmellField f = new SmellField(0, 0, 40, 5, 1, 0.2, 0.01, 1);
		List<Vec3> src = List.of(new Vec3(0.5, 64, 2.5));
		for (int t = 0; t < 3000; t++) {
			f.step(src);
		}
		for (int i = 1; i < 40; i++) {
			assertTrue(f.get(i, 2) < f.get(i - 1, 2), "not monotone at " + i);
		}
		// 1-D steady state with decay: c(x) ~ exp(-x / sqrt(D / k)); check the decay length within 15 %.
		double lambda = -1.0 / Math.log(f.get(11, 2) / f.get(10, 2));
		assertEquals(Math.sqrt(0.2 / 0.01), lambda, Math.sqrt(0.2 / 0.01) * 0.15);
	}

	@Test
	void largeDiffusionIsSubSteppedAndStaysStable() {
		SmellField f = new SmellField(0, 0, 10, 10, 0.5, 2.0, 0.0, 1);
		assertTrue(f.substeps() >= 33); // D dt / dx^2 = 8 per tick needs >= 8 / 0.24 substeps
		List<Vec3> src = List.of(new Vec3(5, 64, 5));
		for (int t = 0; t < 100; t++) {
			f.step(src);
		}
		for (int k = 0; k < f.nz(); k++) {
			for (int i = 0; i < f.nx(); i++) {
				assertTrue(f.get(i, k) >= 0 && f.get(i, k) < 100);
			}
		}
	}

	@Test
	void sampleInterpolatesBetweenCellCentres() {
		SmellField f = new SmellField(0, 0, 3, 1, 1, 0.0, 0.0, 1);
		f.step(List.of(new Vec3(1.5, 64, 0.5)));            // c = [0, 1, 0]
		assertEquals(1.0, f.sample(1.5, 0.5), 1e-12);
		assertEquals(0.5, f.sample(1.0, 0.5), 1e-12);
		assertEquals(0.0, f.sample(0.2, 0.5), 1e-12);        // clamped at the edge cell centre
	}
}
