theorem nonempty_algHom_of_exist_roots (h : ∀ x : E, ∃ y : K, aeval y (minpoly F x) = 0) :
    Nonempty (E →ₐ[F] K) := by
  /-
    F : Type u_1
    E : Type u_2
    K : Type u_3
    inst✝⁴ : Field F
    inst✝³ : Field E
    inst✝² : Field K
    inst✝¹ : Algebra F E
    inst✝ : Algebra F K
    alg : Algebra.IsAlgebraic F E
    h : ∀ (x : E), Exists fun y => Eq ((Polynomial.aeval y) (minpoly F x)) 0
    ⊢ Nonempty (AlgHom F E K)
  -/
  refine Lifts.nonempty_algHom_of_exist_lifts_finset fun S ↦ ⟨⟨adjoin F S, ?_⟩, subset_adjoin _ _⟩
  /-
    F : Type u_1
    E : Type u_2
    K : Type u_3
    inst✝⁴ : Field F
    inst✝³ : Field E
    inst✝² : Field K
    inst✝¹ : Algebra F E
    inst✝ : Algebra F K
    alg : Algebra.IsAlgebraic F E
    h : ∀ (x : E), Exists fun y => Eq ((Polynomial.aeval y) (minpoly F x)) 0
    S : Finset E
    ⊢ AlgHom F (Subtype fun x => Membership.mem (IntermediateField.adjoin F ↑S) x) K
  -/
  let p := (S.prod <| minpoly F).map (algebraMap F K)
  /-
    F : Type u_1
    E : Type u_2
    K : Type u_3
    inst✝⁴ : Field F
    inst✝³ : Field E
    inst✝² : Field K
    inst✝¹ : Algebra F E
    inst✝ : Algebra F K
    alg : Algebra.IsAlgebraic F E
    h : ∀ (x : E), Exists fun y => Eq ((Polynomial.aeval y) (minpoly F x)) 0
    S : Finset E
    p : Polynomial K := Polynomial.map (algebraMap F K) (S.prod (minpoly F))
    ⊢ AlgHom F (Subtype fun x => Membership.mem (IntermediateField.adjoin F ↑S) x) K
  -/
  let K' := SplittingField p
  have splits s (hs : s ∈ S) : (minpoly F s).Splits (algebraMap F K') := by
    apply splits_of_splits_of_dvd _
      (Finset.prod_ne_zero_iff.mpr fun _ _ ↦ minpoly.ne_zero <| (alg.isIntegral).1 _)
      ((splits_map_iff _ _).mp <| SplittingField.splits p) (Finset.dvd_prod_of_mem _ hs)
  /-
    F : Type u_1
    E : Type u_2
    K : Type u_3
    inst✝⁴ : Field F
    inst✝³ : Field E
    inst✝² : Field K
    inst✝¹ : Algebra F E
    inst✝ : Algebra F K
    alg : Algebra.IsAlgebraic F E
    h : ∀ (x : E), Exists fun y => Eq ((Polynomial.aeval y) (minpoly F x)) 0
    S : Finset E
    p : Polynomial K := Polynomial.map (algebraMap F K) (S.prod (minpoly F))
    K' : Type u_3 := p.SplittingField
    splits : ∀ (s : E), Membership.mem S s → Polynomial.Splits (algebraMap F K') ( …
    ⊢ AlgHom F (Subtype fun x => Membership.mem (IntermediateField.adjoin F ↑S) x) K
  -/
  let K₀ := (⊥ : IntermediateField K K').restrictScalars F
  /-
    F : Type u_1
    E : Type u_2
    K : Type u_3
    inst✝⁴ : Field F
    inst✝³ : Field E
    inst✝² : Field K
    inst✝¹ : Algebra F E
    inst✝ : Algebra F K
    alg : Algebra.IsAlgebraic F E
    h : ∀ (x : E), Exists fun y => Eq ((Polynomial.aeval y) (minpoly F x)) 0
    S : Finset E
    p : Polynomial K := Polynomial.map (algebraMap F K) (S.prod (minpoly F))
    K' : Type u_3 := p.SplittingField
    splits : ∀ (s : E), Membership.mem S s → Polynomial.Splits (algebraMap F K') ( …
    K₀ : IntermediateField F K' := IntermediateField.restrictScalars F Bot.bot
    ⊢ AlgHom F (Subtype fun x => Membership.mem (IntermediateField.adjoin F ↑S) x) K
  -/
  let FS := adjoin F (S : Set E)
  /-
    F : Type u_1
    E : Type u_2
    K : Type u_3
    inst✝⁴ : Field F
    inst✝³ : Field E
    inst✝² : Field K
    inst✝¹ : Algebra F E
    inst✝ : Algebra F K
    alg : Algebra.IsAlgebraic F E
    h : ∀ (x : E), Exists fun y => Eq ((Polynomial.aeval y) (minpoly F x)) 0
    S : Finset E
    p : Polynomial K := Polynomial.map (algebraMap F K) (S.prod (minpoly F))
    K' : Type u_3 := p.SplittingField
    splits : ∀ (s : E), Membership.mem S s → Polynomial.Splits (algebraMap F K') ( …
    K₀ : IntermediateField F K' := IntermediateField.restrictScalars F Bot.bot
    FS : IntermediateField F E := IntermediateField.adjoin F ↑S
    ⊢ AlgHom F (Subtype fun x => Membership.mem (IntermediateField.adjoin F ↑S) x) K
  -/
  let Ω := FS →ₐ[F] K'
  /-
    F : Type u_1
    E : Type u_2
    K : Type u_3
    inst✝⁴ : Field F
    inst✝³ : Field E
    inst✝² : Field K
    inst✝¹ : Algebra F E
    inst✝ : Algebra F K
    alg : Algebra.IsAlgebraic F E
    h : ∀ (x : E), Exists fun y => Eq ((Polynomial.aeval y) (minpoly F x)) 0
    S : Finset E
    p : Polynomial K := Polynomial.map (algebraMap F K) (S.prod (minpoly F))
    K' : Type u_3 := p.SplittingField
    splits : ∀ (s : E), Membership.mem S s → Polynomial.Splits (algebraMap F K') ( …
    K₀ : IntermediateField F K' := IntermediateField.restrictScalars F Bot.bot
    FS : IntermediateField F E := IntermediateField.adjoin F ↑S
    Ω : Type (max u_2 u_3) := AlgHom F (Subtype fun x => Membership.mem FS x) K'
    ⊢ AlgHom F (Subtype fun x => Membership.mem (IntermediateField.adjoin F ↑S) x) K
  -/
  have := finiteDimensional_adjoin (S := (S : Set E)) fun _ _ ↦ (alg.isIntegral).1 _
  /-
    F : Type u_1
    E : Type u_2
    K : Type u_3
    inst✝⁴ : Field F
    inst✝³ : Field E
    inst✝² : Field K
    inst✝¹ : Algebra F E
    inst✝ : Algebra F K
    alg : Algebra.IsAlgebraic F E
    h : ∀ (x : E), Exists fun y => Eq ((Polynomial.aeval y) (minpoly F x)) 0
    S : Finset E
    p : Polynomial K := Polynomial.map (algebraMap F K) (S.prod (minpoly F))
    K' : Type u_3 := p.SplittingField
    splits : ∀ (s : E), Membership.mem S s → Polynomial.Splits (algebraMap F K') ( …
    K₀ : IntermediateField F K' := IntermediateField.restrictScalars F Bot.bot
    FS : IntermediateField F E := IntermediateField.adjoin F ↑S
    Ω : Type (max u_2 u_3) := AlgHom F (Subtype fun x => Membership.mem FS x) K'
    this : FiniteDimensional F (Subtype fun x => Membership.mem (IntermediateField …
    ⊢ AlgHom F (Subtype fun x => Membership.mem (IntermediateField.adjoin F ↑S) x) K
  -/
  let M (ω : Ω) := Subalgebra.toSubmodule (K₀.comap ω).toSubalgebra
  have : ⋃ ω : Ω, (M ω : Set FS) = Set.univ :=
    Set.eq_univ_of_forall fun ⟨α, hα⟩ ↦ Set.mem_iUnion.mpr <| by
      have ⟨β, hβ⟩ := h α
      let ϕ : F⟮α⟯ →ₐ[F] K' := (IsScalarTower.toAlgHom _ _ _).comp ((AdjoinRoot.liftHom _ _ hβ).comp
        (adjoinRootEquivAdjoin F <| (alg.isIntegral).1 _).symm.toAlgHom)
      have ⟨ω, hω⟩ := exists_algHom_adjoin_of_splits
        (fun s hs ↦ ⟨(alg.isIntegral).1 _, splits s hs⟩) ϕ (adjoin_simple_le_iff.mpr hα)
      refine ⟨ω, β, ((DFunLike.congr_fun hω <| AdjoinSimple.gen F α).trans ?_).symm⟩
      rw [AlgHom.comp_apply, AlgHom.comp_apply, AlgEquiv.coe_algHom,
        adjoinRootEquivAdjoin_symm_apply_gen, AdjoinRoot.liftHom_root]
      rfl
  have ω : ∃ ω : Ω, ⊤ ≤ M ω := by
    cases finite_or_infinite F
    · have ⟨α, hα⟩ := exists_primitive_element_of_finite_bot F FS
      have ⟨ω, hω⟩ := Set.mem_iUnion.mp (this ▸ Set.mem_univ α)
      exact ⟨ω, show ⊤ ≤ K₀.comap ω by rwa [← hα, adjoin_simple_le_iff]⟩
    · simp_rw [top_le_iff, Subspace.exists_eq_top_of_iUnion_eq_univ this]
  exact ((botEquiv K K').toAlgHom.restrictScalars F).comp
    (ω.choose.codRestrict K₀.toSubalgebra fun x ↦ ω.choose_spec trivial)


theorem nonempty_algHom_of_minpoly_eq
    (h : ∀ x : E, ∃ y : K, minpoly F x = minpoly F y) :
    Nonempty (E →ₐ[F] K) :=
                                                                     /-
                                                                       F : Type u_1
                                                                       E : Type u_2
                                                                       K : Type u_3
                                                                       inst✝⁴ : Field F
                                                                       inst✝³ : Field E
                                                                       inst✝² : Field K
                                                                       inst✝¹ : Algebra F E
                                                                       inst✝ : Algebra F K
                                                                       alg : Algebra.IsAlgebraic F E
                                                                       h : ∀ (x : E), Exists fun y => Eq (minpoly F x) (minpoly F y)
                                                                       x : E
                                                                       y : K
                                                                       hy : Eq (minpoly F x) (minpoly F y)
                                                                       ⊢ Eq ((Polynomial.aeval y) (minpoly F x)) 0
                                                                     -/
  nonempty_algHom_of_exist_roots fun x ↦ have ⟨y, hy⟩ := h x; ⟨y, by rw [hy, minpoly.aeval]⟩
                                                                     /-
                                                                       🎉 no goals
                                                                     -/


theorem nonempty_algHom_of_range_minpoly_subset
    (h : Set.range (@minpoly F E _ _ _) ⊆ Set.range (@minpoly F K _ _ _)) :
    Nonempty (E →ₐ[F] K) :=
  nonempty_algHom_of_minpoly_eq fun x ↦ have ⟨y, hy⟩ := h ⟨x, rfl⟩; ⟨y, hy.symm⟩


theorem nonempty_algEquiv_of_range_minpoly_eq
    (h : Set.range (@minpoly F E _ _ _) = Set.range (@minpoly F K _ _ _)) :
    Nonempty (E ≃ₐ[F] K) :=
  have ⟨σ⟩ := nonempty_algHom_of_range_minpoly_subset h.le
  have : Algebra.IsAlgebraic F K := ⟨fun y ↦ IsIntegral.isAlgebraic <| by
    /-
      F : Type u_1
      E : Type u_2
      K : Type u_3
      inst✝⁴ : Field F
      inst✝³ : Field E
      inst✝² : Field K
      inst✝¹ : Algebra F E
      inst✝ : Algebra F K
      alg : Algebra.IsAlgebraic F E
      h : Eq (Set.range (minpoly F)) (Set.range (minpoly F))
      σ : AlgHom F E K
      y : K
      ⊢ IsIntegral F y
    -/
    by_contra hy
    /-
      F : Type u_1
      E : Type u_2
      K : Type u_3
      inst✝⁴ : Field F
      inst✝³ : Field E
      inst✝² : Field K
      inst✝¹ : Algebra F E
      inst✝ : Algebra F K
      alg : Algebra.IsAlgebraic F E
      h : Eq (Set.range (minpoly F)) (Set.range (minpoly F))
      σ : AlgHom F E K
      y : K
      hy : Not (IsIntegral F y)
      ⊢ False
    -/
    have ⟨x, hx⟩ := h.ge ⟨y, rfl⟩
    /-
      F : Type u_1
      E : Type u_2
      K : Type u_3
      inst✝⁴ : Field F
      inst✝³ : Field E
      inst✝² : Field K
      inst✝¹ : Algebra F E
      inst✝ : Algebra F K
      alg : Algebra.IsAlgebraic F E
      h : Eq (Set.range (minpoly F)) (Set.range (minpoly F))
      σ : AlgHom F E K
      y : K
      hy : Not (IsIntegral F y)
      x : E
      hx : Eq (minpoly F x) (minpoly F y)
      ⊢ False
    -/
    rw [minpoly.eq_zero hy] at hx
    /-
      F : Type u_1
      E : Type u_2
      K : Type u_3
      inst✝⁴ : Field F
      inst✝³ : Field E
      inst✝² : Field K
      inst✝¹ : Algebra F E
      inst✝ : Algebra F K
      alg : Algebra.IsAlgebraic F E
      h : Eq (Set.range (minpoly F)) (Set.range (minpoly F))
      σ : AlgHom F E K
      y : K
      hy : Not (IsIntegral F y)
      x : E
      hx : Eq (minpoly F x) 0
      ⊢ False
    -/
    exact minpoly.ne_zero ((alg.isIntegral).1 x) hx⟩
    /-
      🎉 no goals
    -/
  have ⟨τ⟩ := nonempty_algHom_of_range_minpoly_subset h.ge
  ⟨.ofBijective _ (Algebra.IsAlgebraic.algHom_bijective₂ σ τ).1⟩


theorem nonempty_algHom_of_aeval_eq_zero_subset
    (h : {p : F[X] | ∃ x : E, aeval x p = 0} ⊆ {p | ∃ y : K, aeval y p = 0}) :
    Nonempty (E →ₐ[F] K) :=
  nonempty_algHom_of_minpoly_eq fun x ↦
    have ⟨y, hy⟩ := h ⟨_, minpoly.aeval F x⟩
    ⟨y, (minpoly.eq_iff_aeval_minpoly_eq_zero <| (alg.isIntegral).1 x).mpr hy⟩


theorem nonempty_algEquiv_of_aeval_eq_zero_eq [Algebra.IsAlgebraic F K]
    (h : {p : F[X] | ∃ x : E, aeval x p = 0} = {p | ∃ y : K, aeval y p = 0}) :
    Nonempty (E ≃ₐ[F] K) :=
  have ⟨σ⟩ := nonempty_algHom_of_aeval_eq_zero_subset h.le
  have ⟨τ⟩ := nonempty_algHom_of_aeval_eq_zero_subset h.ge
  ⟨.ofBijective _ (Algebra.IsAlgebraic.algHom_bijective₂ σ τ).1⟩


theorem _root_.IsAlgClosure.of_exist_roots
    (h : ∀ p : F[X], p.Monic → Irreducible p → ∃ x : E, aeval x p = 0) :
    IsAlgClosure F E :=
  .of_splits fun p _ _ ↦
    have ⟨σ⟩ := nonempty_algHom_of_exist_roots fun x : p.SplittingField ↦
      have := Algebra.IsAlgebraic.isIntegral (K := F).1 x
      h _ (minpoly.monic this) (minpoly.irreducible this)
    splits_of_algHom (SplittingField.splits _) σ


