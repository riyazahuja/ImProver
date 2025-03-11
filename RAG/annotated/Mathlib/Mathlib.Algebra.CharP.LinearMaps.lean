/-- For a commutative semiring `R` and a `R`-module `M`, if `M` contains an
  element `x` such that `r • x = 0` implies `r = 0` (finding such element usually
  depends on specific `•`), then the characteristic of `R` is equal to the
  characteristic of the `R`-linear endomorphisms of `M`.-/
theorem charP_end {p : ℕ} [hchar : CharP R p]
    (hreduction : ∃ x : M, ∀ r : R, r • x = 0 → r = 0) : CharP (M →ₗ[R] M) p where
  cast_eq_zero_iff' n := by
    have exact : (n : M →ₗ[R] M) = (n : R) • 1 := by
      simp only [Nat.cast_smul_eq_nsmul, nsmul_eq_mul, mul_one]
    /-
      R : Type u_1
      M : Type u_2
      inst✝² : CommSemiring R
      inst✝¹ : AddCommMonoid M
      inst✝ : Module R M
      p : Nat
      hchar : CharP R p
      hreduction : Exists fun x => ∀ (r : R), Eq (HSMul.hSMul r x) 0 → Eq r 0
      n : Nat
      exact : Eq (↑n) (HSMul.hSMul (↑n) 1)
      ⊢ Iff (Eq (↑n) 0) (Dvd.dvd p n)
    -/
    rw [exact, LinearMap.ext_iff, ← hchar.1]
    exact ⟨fun h ↦ Exists.casesOn hreduction fun x hx ↦ hx n (h x),
      fun h ↦ (congrArg (fun t ↦ ∀ x, t • x = 0) h).mpr fun x ↦ zero_smul R x⟩


/-- For a division ring `D` with center `k`, the ring of `k`-linear endomorphisms
  of `D` has the same characteristic as `D`-/
instance {D : Type*} [DivisionRing D] {p : ℕ} [CharP D p] :
    CharP (D →ₗ[(Subring.center D)] D) p :=
  charP_of_injective_ringHom (Algebra.lmul (Subring.center D) D).toRingHom.injective p


instance {D : Type*} [DivisionRing D] {p : ℕ} [ExpChar D p] :
    ExpChar (D →ₗ[Subring.center D] D) p :=
  expChar_of_injective_ringHom (Algebra.lmul (Subring.center D) D).toRingHom.injective p

