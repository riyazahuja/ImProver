/-- This is a special case of `NormedSpace.discreteTopology_zmultiples`. It exists only to simplify
dependencies. -/
instance {a : ℝ} : DiscreteTopology (AddSubgroup.zmultiples a) := by
  /-
    α : Type u
    β : Type v
    γ : Type w
    a : Real
    ⊢ DiscreteTopology (Subtype fun x => Membership.mem (AddSubgroup.zmultiples a) …
  -/
  rcases eq_or_ne a 0 with (rfl | ha)
    /-
      case inl
      α : Type u
      β : Type v
      γ : Type w
      ⊢ DiscreteTopology (Subtype fun x => Membership.mem (AddSubgroup.zmultiples 0) …
    -/
  · rw [AddSubgroup.zmultiples_zero_eq_bot]
    /-
      case inl
      α : Type u
      β : Type v
      γ : Type w
      ⊢ DiscreteTopology (Subtype fun x => Membership.mem Bot.bot x)
    -/
    exact Subsingleton.discreteTopology (α := (⊥ : Submodule ℤ ℝ))
    /-
      🎉 no goals
    -/
  /-
    case inr
    α : Type u
    β : Type v
    γ : Type w
    a : Real
    ha : Ne a 0
    ⊢ DiscreteTopology (Subtype fun x => Membership.mem (AddSubgroup.zmultiples a) …
  -/
  rw [discreteTopology_iff_isOpen_singleton_zero, isOpen_induced_iff]
  /-
    case inr
    α : Type u
    β : Type v
    γ : Type w
    a : Real
    ha : Ne a 0
    ⊢ Exists fun t => And (IsOpen t) (Eq (Set.preimage Subtype.val t) (Singleton.s …
  -/
  refine ⟨ball 0 |a|, isOpen_ball, ?_⟩
  /-
    case inr
    α : Type u
    β : Type v
    γ : Type w
    a : Real
    ha : Ne a 0
    ⊢ Eq (Set.preimage Subtype.val (Metric.ball 0 (abs a))) (Singleton.singleton 0)
  -/
  ext ⟨x, hx⟩
  /-
    case inr.h.mk
    α : Type u
    β : Type v
    γ : Type w
    a : Real
    ha : Ne a 0
    x : Real
    hx : Membership.mem (AddSubgroup.zmultiples a) x
    ⊢ Iff (Membership.mem (Set.preimage Subtype.val (Metric.ball 0 (abs a))) ⟨x, h …
  -/
  obtain ⟨k, rfl⟩ := AddSubgroup.mem_zmultiples_iff.mp hx
  /-
    case inr.h.mk.intro
    α : Type u
    β : Type v
    γ : Type w
    a : Real
    ha : Ne a 0
    k : Int
    hx : Membership.mem (AddSubgroup.zmultiples a) (HSMul.hSMul k a)
    ⊢ Iff (Membership.mem (Set.preimage Subtype.val (Metric.ball 0 (abs a))) ⟨HSMu …
  -/
  simp [ha, Real.dist_eq, abs_mul, (by norm_cast : |(k : ℝ)| < 1 ↔ |k| < 1)]
  /-
    🎉 no goals
  -/


/-- Under the coercion from `ℤ` to `ℝ`, inverse images of compact sets are finite. -/
theorem tendsto_coe_cofinite : Tendsto ((↑) : ℤ → ℝ) cofinite (cocompact ℝ) := by
  /-
    ⊢ Filter.Tendsto Int.cast Filter.cofinite (Filter.cocompact Real)
  -/
  apply (castAddHom ℝ).tendsto_coe_cofinite_of_discrete cast_injective
  /-
    ⊢ DiscreteTopology (Subtype fun x => Membership.mem (Int.castAddHom Real).rang …
  -/
  rw [range_castAddHom]
  /-
    ⊢ DiscreteTopology (Subtype fun x => Membership.mem (AddSubgroup.zmultiples 1) …
  -/
  infer_instance
  /-
    🎉 no goals
  -/


/-- For nonzero `a`, the "multiples of `a`" map `zmultiplesHom` from `ℤ` to `ℝ` is discrete, i.e.
inverse images of compact sets are finite. -/
theorem tendsto_zmultiplesHom_cofinite {a : ℝ} (ha : a ≠ 0) :
    Tendsto (zmultiplesHom ℝ a) cofinite (cocompact ℝ) := by
  /-
    a : Real
    ha : Ne a 0
    ⊢ Filter.Tendsto (⇑((zmultiplesHom Real) a)) Filter.cofinite (Filter.cocompact …
  -/
  apply (zmultiplesHom ℝ a).tendsto_coe_cofinite_of_discrete <| smul_left_injective ℤ ha
  /-
    a : Real
    ha : Ne a 0
    ⊢ DiscreteTopology (Subtype fun x => Membership.mem ((zmultiplesHom Real) a).r …
  -/
  rw [AddSubgroup.range_zmultiplesHom]
  /-
    a : Real
    ha : Ne a 0
    ⊢ DiscreteTopology (Subtype fun x => Membership.mem (AddSubgroup.zmultiples a) …
  -/
  infer_instance
  /-
    🎉 no goals
  -/


/-- The subgroup "multiples of `a`" (`zmultiples a`) is a discrete subgroup of `ℝ`, i.e. its
intersection with compact sets is finite. -/
theorem tendsto_zmultiples_subtype_cofinite (a : ℝ) :
    Tendsto (zmultiples a).subtype cofinite (cocompact ℝ) :=
  (zmultiples a).tendsto_coe_cofinite_of_discrete


