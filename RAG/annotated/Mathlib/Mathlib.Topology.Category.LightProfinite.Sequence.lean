/-- The continuous map from `ℕ∪{∞}` to `ℝ` sending `n` to `1/(n+1)` and `∞` to `0`. -/
noncomputable def natUnionInftyEmbedding : C(OnePoint ℕ, ℝ) where
  toFun
    | ∞ => 0
    | OnePoint.some n => 1 / (n+1 : ℝ)
  continuous_toFun := OnePoint.continuous_iff_from_nat _ |>.mpr
    tendsto_one_div_add_atTop_nhds_zero_nat


/--
The continuous map from `ℕ∪{∞}` to `ℝ` sending `n` to `1/(n+1)` and `∞` to `0` is a closed
embedding.
-/
lemma isClosedEmbedding_natUnionInftyEmbedding : IsClosedEmbedding natUnionInftyEmbedding := by
  refine .of_continuous_injective_isClosedMap
    natUnionInftyEmbedding.continuous ?_ ?_
    /-
      case refine_1
      ⊢ Function.Injective ⇑LightProfinite.natUnionInftyEmbedding
    -/
  · rintro (_|n) (_|m) h
      /-
        case refine_1.none.none
        h : Eq (LightProfinite.natUnionInftyEmbedding Option.none) (LightProfinite.nat …
        ⊢ Eq Option.none Option.none
      -/
    · rfl
      /-
        🎉 no goals
      -/
      /-
        case refine_1.none.some
        m : Nat
        h : Eq (LightProfinite.natUnionInftyEmbedding Option.none) (LightProfinite.nat …
        ⊢ Eq Option.none (Option.some m)
      -/
    · simp only [natUnionInftyEmbedding, one_div, ContinuousMap.coe_mk, zero_eq_inv] at h
      /-
        case refine_1.none.some
        m : Nat
        h : Eq 0 (HAdd.hAdd (↑m) 1)
        ⊢ Eq Option.none (Option.some m)
      -/
      rw [← Nat.cast_one, ← Nat.cast_add, eq_comm, Nat.cast_eq_zero] at h
      /-
        case refine_1.none.some
        m : Nat
        h : Eq (HAdd.hAdd m 1) 0
        ⊢ Eq Option.none (Option.some m)
      -/
      simp at h
      /-
        🎉 no goals
      -/
      /-
        case refine_1.some.none
        n : Nat
        h : Eq (LightProfinite.natUnionInftyEmbedding (Option.some n)) (LightProfinite …
        ⊢ Eq (Option.some n) Option.none
      -/
    · simp only [natUnionInftyEmbedding, one_div, ContinuousMap.coe_mk, inv_eq_zero] at h
      /-
        case refine_1.some.none
        n : Nat
        h : Eq (HAdd.hAdd (↑n) 1) 0
        ⊢ Eq (Option.some n) Option.none
      -/
      rw [← Nat.cast_one, ← Nat.cast_add, Nat.cast_eq_zero] at h
      /-
        case refine_1.some.none
        n : Nat
        h : Eq (HAdd.hAdd n 1) 0
        ⊢ Eq (Option.some n) Option.none
      -/
      simp at h
      /-
        🎉 no goals
      -/
    · simp only [natUnionInftyEmbedding, one_div, ContinuousMap.coe_mk, inv_inj, add_left_inj,
        Nat.cast_inj] at h
      /-
        case refine_1.some.some
        n m : Nat
        h : Eq n m
        ⊢ Eq (Option.some n) (Option.some m)
      -/
      rw [h]
      /-
        🎉 no goals
      -/
    /-
      case refine_2
      ⊢ IsClosedMap ⇑LightProfinite.natUnionInftyEmbedding
    -/
  · exact fun _ hC => (hC.isCompact.image natUnionInftyEmbedding.continuous).isClosed
    /-
      🎉 no goals
    -/


@[deprecated (since := "2024-10-20")]
alias closedEmbedding_natUnionInftyEmbedding := isClosedEmbedding_natUnionInftyEmbedding


instance : MetrizableSpace (OnePoint ℕ) := isClosedEmbedding_natUnionInftyEmbedding.metrizableSpace


/-- The one point compactification of the natural numbers as a light profinite set. -/
abbrev NatUnionInfty : LightProfinite := of (OnePoint ℕ)


@[inherit_doc]
scoped notation "ℕ∪{∞}" => NatUnionInfty


instance : Coe ℕ ℕ∪{∞} := optionCoe


lemma continuous_iff_convergent {Y : Type*} [TopologicalSpace Y] (f : ℕ∪{∞} → Y) :
    Continuous f ↔ Tendsto (fun x : ℕ ↦ f x) atTop (𝓝 (f ∞)) :=
  continuous_iff_from_nat f


