/-- A bounded variation function into `ℝ` is differentiable almost everywhere. Superseded by
`ae_differentiableWithinAt_of_mem`. -/
theorem ae_differentiableWithinAt_of_mem_real {f : ℝ → ℝ} {s : Set ℝ}
    (h : LocallyBoundedVariationOn f s) : ∀ᵐ x, x ∈ s → DifferentiableWithinAt ℝ f s x := by
  obtain ⟨p, q, hp, hq, rfl⟩ : ∃ p q, MonotoneOn p s ∧ MonotoneOn q s ∧ f = p - q :=
    h.exists_monotoneOn_sub_monotoneOn
  filter_upwards [hp.ae_differentiableWithinAt_of_mem, hq.ae_differentiableWithinAt_of_mem] with
    x hxp hxq xs
  /-
    case h
    s : Set Real
    p q : Real → Real
    hp : MonotoneOn p s
    hq : MonotoneOn q s
    h : LocallyBoundedVariationOn (HSub.hSub p q) s
    x : Real
    hxp : Membership.mem s x → DifferentiableWithinAt Real p s x
    hxq : Membership.mem s x → DifferentiableWithinAt Real q s x
    xs : Membership.mem s x
    ⊢ DifferentiableWithinAt Real (HSub.hSub p q) s x
  -/
  exact (hxp xs).sub (hxq xs)
  /-
    🎉 no goals
  -/


/-- A bounded variation function into a finite dimensional product vector space is differentiable
almost everywhere. Superseded by `ae_differentiableWithinAt_of_mem`. -/
theorem ae_differentiableWithinAt_of_mem_pi {ι : Type*} [Fintype ι] {f : ℝ → ι → ℝ} {s : Set ℝ}
    (h : LocallyBoundedVariationOn f s) : ∀ᵐ x, x ∈ s → DifferentiableWithinAt ℝ f s x := by
  /-
    ι : Type u_4
    inst✝ : Fintype ι
    f : Real → ι → Real
    s : Set Real
    h : LocallyBoundedVariationOn f s
    ⊢ Filter.Eventually (fun x => Membership.mem s x → DifferentiableWithinAt Real …
  -/
  have A : ∀ i : ι, LipschitzWith 1 fun x : ι → ℝ => x i := fun i => LipschitzWith.eval i
  have : ∀ i : ι, ∀ᵐ x, x ∈ s → DifferentiableWithinAt ℝ (fun x : ℝ => f x i) s x := fun i ↦ by
    apply ae_differentiableWithinAt_of_mem_real
    exact LipschitzWith.comp_locallyBoundedVariationOn (A i) h
  /-
    ι : Type u_4
    inst✝ : Fintype ι
    f : Real → ι → Real
    s : Set Real
    h : LocallyBoundedVariationOn f s
    A : ∀ (i : ι), LipschitzWith 1 fun x => x i
    this : ∀ (i : ι), Filter.Eventually (fun x => Membership.mem s x → Differentia …
    ⊢ Filter.Eventually (fun x => Membership.mem s x → DifferentiableWithinAt Real …
  -/
  filter_upwards [ae_all_iff.2 this] with x hx xs
  /-
    case h
    ι : Type u_4
    inst✝ : Fintype ι
    f : Real → ι → Real
    s : Set Real
    h : LocallyBoundedVariationOn f s
    A : ∀ (i : ι), LipschitzWith 1 fun x => x i
    this : ∀ (i : ι), Filter.Eventually (fun x => Membership.mem s x → Differentia …
    x : Real
    hx : ∀ (i : ι), Membership.mem s x → DifferentiableWithinAt Real (fun x => f x …
    xs : Membership.mem s x
    ⊢ DifferentiableWithinAt Real f s x
  -/
  exact differentiableWithinAt_pi.2 fun i => hx i xs
  /-
    🎉 no goals
  -/


/-- A real function into a finite dimensional real vector space with bounded variation on a set
is differentiable almost everywhere in this set. -/
theorem ae_differentiableWithinAt_of_mem {f : ℝ → V} {s : Set ℝ}
    (h : LocallyBoundedVariationOn f s) : ∀ᵐ x, x ∈ s → DifferentiableWithinAt ℝ f s x := by
  /-
    V : Type u_3
    inst✝² : NormedAddCommGroup V
    inst✝¹ : NormedSpace Real V
    inst✝ : FiniteDimensional Real V
    f : Real → V
    s : Set Real
    h : LocallyBoundedVariationOn f s
    ⊢ Filter.Eventually (fun x => Membership.mem s x → DifferentiableWithinAt Real …
  -/
  let A := (Basis.ofVectorSpace ℝ V).equivFun.toContinuousLinearEquiv
  suffices H : ∀ᵐ x, x ∈ s → DifferentiableWithinAt ℝ (A ∘ f) s x by
    filter_upwards [H] with x hx xs
    have : f = (A.symm ∘ A) ∘ f := by
      simp only [ContinuousLinearEquiv.symm_comp_self, Function.id_comp]
    rw [this]
    exact A.symm.differentiableAt.comp_differentiableWithinAt x (hx xs)
  /-
    V : Type u_3
    inst✝² : NormedAddCommGroup V
    inst✝¹ : NormedSpace Real V
    inst✝ : FiniteDimensional Real V
    f : Real → V
    s : Set Real
    h : LocallyBoundedVariationOn f s
    A : ContinuousLinearEquiv (RingHom.id Real) V (↑(Basis.ofVectorSpaceIndex Real …
    ⊢ Filter.Eventually (fun x => Membership.mem s x → DifferentiableWithinAt Real …
  -/
  apply ae_differentiableWithinAt_of_mem_pi
  /-
    case h
    V : Type u_3
    inst✝² : NormedAddCommGroup V
    inst✝¹ : NormedSpace Real V
    inst✝ : FiniteDimensional Real V
    f : Real → V
    s : Set Real
    h : LocallyBoundedVariationOn f s
    A : ContinuousLinearEquiv (RingHom.id Real) V (↑(Basis.ofVectorSpaceIndex Real …
    ⊢ LocallyBoundedVariationOn (Function.comp (⇑A) f) s
  -/
  exact A.lipschitz.comp_locallyBoundedVariationOn h
  /-
    🎉 no goals
  -/


/-- A real function into a finite dimensional real vector space with bounded variation on a set
is differentiable almost everywhere in this set. -/
theorem ae_differentiableWithinAt {f : ℝ → V} {s : Set ℝ} (h : LocallyBoundedVariationOn f s)
    (hs : MeasurableSet s) : ∀ᵐ x ∂volume.restrict s, DifferentiableWithinAt ℝ f s x := by
  /-
    V : Type u_3
    inst✝² : NormedAddCommGroup V
    inst✝¹ : NormedSpace Real V
    inst✝ : FiniteDimensional Real V
    f : Real → V
    s : Set Real
    h : LocallyBoundedVariationOn f s
    hs : MeasurableSet s
    ⊢ Filter.Eventually (fun x => DifferentiableWithinAt Real f s x) (MeasureTheor …
  -/
  rw [ae_restrict_iff' hs]
  /-
    V : Type u_3
    inst✝² : NormedAddCommGroup V
    inst✝¹ : NormedSpace Real V
    inst✝ : FiniteDimensional Real V
    f : Real → V
    s : Set Real
    h : LocallyBoundedVariationOn f s
    hs : MeasurableSet s
    ⊢ Filter.Eventually (fun x => Membership.mem s x → DifferentiableWithinAt Real …
  -/
  exact h.ae_differentiableWithinAt_of_mem
  /-
    🎉 no goals
  -/


/-- A real function into a finite dimensional real vector space with bounded variation
is differentiable almost everywhere. -/
theorem ae_differentiableAt {f : ℝ → V} (h : LocallyBoundedVariationOn f univ) :
    ∀ᵐ x, DifferentiableAt ℝ f x := by
  /-
    V : Type u_3
    inst✝² : NormedAddCommGroup V
    inst✝¹ : NormedSpace Real V
    inst✝ : FiniteDimensional Real V
    f : Real → V
    h : LocallyBoundedVariationOn f Set.univ
    ⊢ Filter.Eventually (fun x => DifferentiableAt Real f x) (MeasureTheory.ae Mea …
  -/
  filter_upwards [h.ae_differentiableWithinAt_of_mem] with x hx
  /-
    case h
    V : Type u_3
    inst✝² : NormedAddCommGroup V
    inst✝¹ : NormedSpace Real V
    inst✝ : FiniteDimensional Real V
    f : Real → V
    h : LocallyBoundedVariationOn f Set.univ
    x : Real
    hx : Membership.mem Set.univ x → DifferentiableWithinAt Real f Set.univ x
    ⊢ DifferentiableAt Real f x
  -/
  rw [differentiableWithinAt_univ] at hx
  /-
    case h
    V : Type u_3
    inst✝² : NormedAddCommGroup V
    inst✝¹ : NormedSpace Real V
    inst✝ : FiniteDimensional Real V
    f : Real → V
    h : LocallyBoundedVariationOn f Set.univ
    x : Real
    hx : Membership.mem Set.univ x → DifferentiableAt Real f x
    ⊢ DifferentiableAt Real f x
  -/
  exact hx (mem_univ _)
  /-
    🎉 no goals
  -/


/-- A real function into a finite dimensional real vector space which is Lipschitz on a set
is differentiable almost everywhere in this set. For the general Rademacher theorem assuming
that the source space is finite dimensional, see `LipschitzOnWith.ae_differentiableWithinAt_of_mem`.
-/
theorem LipschitzOnWith.ae_differentiableWithinAt_of_mem_real {C : ℝ≥0} {f : ℝ → V} {s : Set ℝ}
    (h : LipschitzOnWith C f s) : ∀ᵐ x, x ∈ s → DifferentiableWithinAt ℝ f s x :=
  h.locallyBoundedVariationOn.ae_differentiableWithinAt_of_mem


/-- A real function into a finite dimensional real vector space which is Lipschitz on a set
is differentiable almost everywhere in this set. For the general Rademacher theorem assuming
that the source space is finite dimensional, see `LipschitzOnWith.ae_differentiableWithinAt`. -/
theorem LipschitzOnWith.ae_differentiableWithinAt_real {C : ℝ≥0} {f : ℝ → V} {s : Set ℝ}
    (h : LipschitzOnWith C f s) (hs : MeasurableSet s) :
    ∀ᵐ x ∂volume.restrict s, DifferentiableWithinAt ℝ f s x :=
  h.locallyBoundedVariationOn.ae_differentiableWithinAt hs


/-- A real Lipschitz function into a finite dimensional real vector space is differentiable
almost everywhere. For the general Rademacher theorem assuming
that the source space is finite dimensional, see `LipschitzWith.ae_differentiableAt`. -/
theorem LipschitzWith.ae_differentiableAt_real {C : ℝ≥0} {f : ℝ → V} (h : LipschitzWith C f) :
    ∀ᵐ x, DifferentiableAt ℝ f x :=
  (h.locallyBoundedVariationOn univ).ae_differentiableAt

