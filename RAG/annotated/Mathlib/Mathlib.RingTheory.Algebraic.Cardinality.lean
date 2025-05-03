theorem lift_cardinalMk_le_sigma_polynomial :
    lift.{u} #L ≤ #(Σ p : R[X], { x : L // x ∈ p.aroots L }) := by
  have := @lift_mk_le_lift_mk_of_injective L (Σ p : R[X], {x : L | x ∈ p.aroots L})
    (fun x : L =>
      let p := Classical.indefiniteDescription _ (Algebra.IsAlgebraic.isAlgebraic x)
      ⟨p.1, x, by
        dsimp
        have := (Polynomial.map_ne_zero_iff (NoZeroSMulDivisors.algebraMap_injective R L)).2 p.2.1
        rw [Polynomial.mem_roots this, Polynomial.IsRoot, Polynomial.eval_map,
          ← Polynomial.aeval_def, p.2.2]⟩)
    fun x y => by
      intro h
      simp only [Set.coe_setOf, ne_eq, Set.mem_setOf_eq, Sigma.mk.inj_iff] at h
      refine (Subtype.heq_iff_coe_eq ?_).1 h.2
      simp only [h.1, forall_true_iff]
  /-
    R : Type u
    inst✝⁵ : CommRing R
    L : Type v
    inst✝⁴ : CommRing L
    inst✝³ : IsDomain L
    inst✝² : Algebra R L
    inst✝¹ : NoZeroSMulDivisors R L
    inst✝ : Algebra.IsAlgebraic R L
    this : LE.le (Cardinal.lift.{max v u, v} (Cardinal.mk L)) (Cardinal.lift.{v, m …
    ⊢ LE.le (Cardinal.lift.{u, v} (Cardinal.mk L)) (Cardinal.mk (Sigma fun p => Su …
  -/
  rwa [lift_umax, lift_id'.{v}] at this
  /-
    🎉 no goals
  -/


theorem lift_cardinalMk_le_max : lift.{u} #L ≤ lift.{v} #R ⊔ ℵ₀ :=
  calc
    lift.{u} #L ≤ #(Σ p : R[X], { x : L // x ∈ p.aroots L }) :=
      lift_cardinalMk_le_sigma_polynomial R L
    _ = Cardinal.sum fun p : R[X] => #{x : L | x ∈ p.aroots L} := by
      /-
        R : Type u
        inst✝⁵ : CommRing R
        L : Type v
        inst✝⁴ : CommRing L
        inst✝³ : IsDomain L
        inst✝² : Algebra R L
        inst✝¹ : NoZeroSMulDivisors R L
        inst✝ : Algebra.IsAlgebraic R L
        ⊢ Eq (Cardinal.mk (Sigma fun p => Subtype fun x => Membership.mem (p.aroots L) …
      -/
      rw [← mk_sigma]; rfl
                       /-
                         🎉 no goals
                       -/
    _ ≤ Cardinal.sum.{u, v} fun _ : R[X] => ℵ₀ :=
      (sum_le_sum _ _ fun _ => (Multiset.finite_toSet _).lt_aleph0.le)
                                    /-
                                      R : Type u
                                      inst✝⁵ : CommRing R
                                      L : Type v
                                      inst✝⁴ : CommRing L
                                      inst✝³ : IsDomain L
                                      inst✝² : Algebra R L
                                      inst✝¹ : NoZeroSMulDivisors R L
                                      inst✝ : Algebra.IsAlgebraic R L
                                      ⊢ Eq (Cardinal.sum fun x => Cardinal.aleph0) (HMul.hMul (Cardinal.lift.{v, u}  …
                                    -/
    _ = lift.{v} #(R[X]) * ℵ₀ := by rw [sum_const, lift_aleph0]
                                    /-
                                      🎉 no goals
                                    -/
    _ ≤ lift.{v} (#R ⊔ ℵ₀) ⊔ ℵ₀ ⊔ ℵ₀ := (mul_le_max _ _).trans <| by
      /-
        R : Type u
        inst✝⁵ : CommRing R
        L : Type v
        inst✝⁴ : CommRing L
        inst✝³ : IsDomain L
        inst✝² : Algebra R L
        inst✝¹ : NoZeroSMulDivisors R L
        inst✝ : Algebra.IsAlgebraic R L
        ⊢ LE.le (Max.max (Max.max (Cardinal.lift.{v, u} (Cardinal.mk (Polynomial R)))  …
      -/
      gcongr; simp only [lift_le, Polynomial.cardinalMk_le_max]
              /-
                🎉 no goals
              -/
                /-
                  R : Type u
                  inst✝⁵ : CommRing R
                  L : Type v
                  inst✝⁴ : CommRing L
                  inst✝³ : IsDomain L
                  inst✝² : Algebra R L
                  inst✝¹ : NoZeroSMulDivisors R L
                  inst✝ : Algebra.IsAlgebraic R L
                  ⊢ Eq (Max.max (Max.max (Cardinal.lift.{v, u} (Max.max (Cardinal.mk R) Cardinal …
                -/
    _ = _ := by simp
                /-
                  🎉 no goals
                -/


theorem cardinalMk_le_sigma_polynomial :
    #L ≤ #(Σ p : R[X], { x : L // x ∈ p.aroots L }) := by
  /-
    R : Type u
    inst✝⁵ : CommRing R
    L : Type u
    inst✝⁴ : CommRing L
    inst✝³ : IsDomain L
    inst✝² : Algebra R L
    inst✝¹ : NoZeroSMulDivisors R L
    inst✝ : Algebra.IsAlgebraic R L
    ⊢ LE.le (Cardinal.mk L) (Cardinal.mk (Sigma fun p => Subtype fun x => Membersh …
  -/
  simpa only [lift_id] using lift_cardinalMk_le_sigma_polynomial R L
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-11-10")]
alias cardinal_mk_le_sigma_polynomial := cardinalMk_le_sigma_polynomial


/-- The cardinality of an algebraic extension is at most the maximum of the cardinality
of the base ring or `ℵ₀`. -/
@[stacks 09GK]
theorem cardinalMk_le_max : #L ≤ max #R ℵ₀ := by
  /-
    R : Type u
    inst✝⁵ : CommRing R
    L : Type u
    inst✝⁴ : CommRing L
    inst✝³ : IsDomain L
    inst✝² : Algebra R L
    inst✝¹ : NoZeroSMulDivisors R L
    inst✝ : Algebra.IsAlgebraic R L
    ⊢ LE.le (Cardinal.mk L) (Max.max (Cardinal.mk R) Cardinal.aleph0)
  -/
  simpa only [lift_id] using lift_cardinalMk_le_max R L
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-11-10")] alias cardinal_mk_le_max := cardinalMk_le_max


