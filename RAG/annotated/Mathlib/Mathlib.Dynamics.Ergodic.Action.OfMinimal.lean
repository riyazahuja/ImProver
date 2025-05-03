/-- Let `M` act continuously on an R₁ topological space `X`.
Let `μ` be a finite inner regular measure on `X` which is ergodic with respect to this action.
If a null measurable set `s` is a.e. equal
to its preimages under the action of a dense set of elements of `M`,
then it is either null or conull. -/
@[to_additive "Let `M` act continuously on an R₁ topological space `X`.
Let `μ` be a finite inner regular measure on `X` which is ergodic with respect to this action.
If a null measurable set `s` is a.e. equal
to its preimages under the action of a dense set of elements of `M`,
then it is either null or conull."]
theorem aeconst_of_dense_setOf_preimage_smul_ae (hsm : NullMeasurableSet s μ)
    (hd : Dense {g : M | (g • ·) ⁻¹' s =ᵐ[μ] s}) : EventuallyConst s (ae μ) := by
  /-
    M : Type u_1
    inst✝⁹ : TopologicalSpace M
    X : Type u_2
    inst✝⁸ : TopologicalSpace X
    inst✝⁷ : R1Space X
    inst✝⁶ : MeasurableSpace X
    inst✝⁵ : BorelSpace X
    inst✝⁴ : SMul M X
    inst✝³ : ContinuousSMul M X
    μ : MeasureTheory.Measure X
    inst✝² : MeasureTheory.IsFiniteMeasure μ
    inst✝¹ : μ.InnerRegular
    inst✝ : ErgodicSMul M X μ
    s : Set X
    hsm : MeasureTheory.NullMeasurableSet s μ
    hd : Dense (setOf fun g => (MeasureTheory.ae μ).EventuallyEq (Set.preimage (fu …
    ⊢ Filter.EventuallyConst s (MeasureTheory.ae μ)
  -/
  borelize M
  /-
    M : Type u_1
    inst✝⁹ : TopologicalSpace M
    X : Type u_2
    inst✝⁸ : TopologicalSpace X
    inst✝⁷ : R1Space X
    inst✝⁶ : MeasurableSpace X
    inst✝⁵ : BorelSpace X
    inst✝⁴ : SMul M X
    inst✝³ : ContinuousSMul M X
    μ : MeasureTheory.Measure X
    inst✝² : MeasureTheory.IsFiniteMeasure μ
    inst✝¹ : μ.InnerRegular
    inst✝ : ErgodicSMul M X μ
    s : Set X
    hsm : MeasureTheory.NullMeasurableSet s μ
    hd : Dense (setOf fun g => (MeasureTheory.ae μ).EventuallyEq (Set.preimage (fu …
    this✝¹ : MeasurableSpace M := borel M
    this✝ : BorelSpace M
    ⊢ Filter.EventuallyConst s (MeasureTheory.ae μ)
  -/
  refine aeconst_of_forall_preimage_smul_ae_eq M hsm ?_
  /-
    M : Type u_1
    inst✝⁹ : TopologicalSpace M
    X : Type u_2
    inst✝⁸ : TopologicalSpace X
    inst✝⁷ : R1Space X
    inst✝⁶ : MeasurableSpace X
    inst✝⁵ : BorelSpace X
    inst✝⁴ : SMul M X
    inst✝³ : ContinuousSMul M X
    μ : MeasureTheory.Measure X
    inst✝² : MeasureTheory.IsFiniteMeasure μ
    inst✝¹ : μ.InnerRegular
    inst✝ : ErgodicSMul M X μ
    s : Set X
    hsm : MeasureTheory.NullMeasurableSet s μ
    hd : Dense (setOf fun g => (MeasureTheory.ae μ).EventuallyEq (Set.preimage (fu …
    this✝¹ : MeasurableSpace M := borel M
    this✝ : BorelSpace M
    ⊢ ∀ (g : M), (MeasureTheory.ae μ).EventuallyEq (Set.preimage (fun x => HSMul.h …
  -/
  rwa [dense_iff_closure_eq, IsClosed.closure_eq, eq_univ_iff_forall] at hd
  /-
    M : Type u_1
    inst✝⁹ : TopologicalSpace M
    X : Type u_2
    inst✝⁸ : TopologicalSpace X
    inst✝⁷ : R1Space X
    inst✝⁶ : MeasurableSpace X
    inst✝⁵ : BorelSpace X
    inst✝⁴ : SMul M X
    inst✝³ : ContinuousSMul M X
    μ : MeasureTheory.Measure X
    inst✝² : MeasureTheory.IsFiniteMeasure μ
    inst✝¹ : μ.InnerRegular
    inst✝ : ErgodicSMul M X μ
    s : Set X
    hsm : MeasureTheory.NullMeasurableSet s μ
    hd : Eq (closure (setOf fun g => (MeasureTheory.ae μ).EventuallyEq (Set.preima …
    this✝¹ : MeasurableSpace M := borel M
    this✝ : BorelSpace M
    ⊢ IsClosed (setOf fun g => (MeasureTheory.ae μ).EventuallyEq (Set.preimage (fu …
  -/
  let f : C(M × X, X) := ⟨(· • ·).uncurry, continuous_smul⟩
  exact isClosed_setOf_preimage_ae_eq f.curry.continuous (measurePreserving_smul · μ) _ hsm
    (measure_ne_top _ _)


@[to_additive]
theorem aeconst_of_dense_setOf_preimage_smul_eq (hsm : NullMeasurableSet s μ)
    (hd : Dense {g : M | (g • ·) ⁻¹' s = s}) : EventuallyConst s (ae μ) :=
  aeconst_of_dense_setOf_preimage_smul_ae hsm <| hd.mono fun _ h ↦ mem_setOf.2 <| .of_eq h


/-- If a monoid `M` continuously acts on an R₁ topological space `X`,
`g` is an element of `M such that its natural powers are dense in `M`,
and `μ` is a finite inner regular measure on `X` which is ergodic with respect to the action of `M`,
then the scalar multiplication by `g` is an ergodic map. -/
@[to_additive "If an additive monoid `M` continuously acts on an R₁ topological space `X`,
`g` is an element of `M such that its natural multiples are dense in `M`,
and `μ` is a finite inner regular measure on `X` which is ergodic with respect to the action of `M`,
then the vector addition of `g` is an ergodic map."]
theorem ergodic_smul_of_denseRange_pow {M : Type*} [Monoid M] [TopologicalSpace M]
    [MulAction M X] [ContinuousSMul M X] {g : M} (hg : DenseRange (g ^ · : ℕ → M))
    (μ : Measure X) [IsFiniteMeasure μ] [μ.InnerRegular] [ErgodicSMul M X μ] :
    Ergodic (g • ·) μ := by
  /-
    X : Type u_2
    inst✝¹⁰ : TopologicalSpace X
    inst✝⁹ : R1Space X
    inst✝⁸ : MeasurableSpace X
    inst✝⁷ : BorelSpace X
    M : Type u_3
    inst✝⁶ : Monoid M
    inst✝⁵ : TopologicalSpace M
    inst✝⁴ : MulAction M X
    inst✝³ : ContinuousSMul M X
    g : M
    hg : DenseRange fun x => HPow.hPow g x
    μ : MeasureTheory.Measure X
    inst✝² : MeasureTheory.IsFiniteMeasure μ
    inst✝¹ : μ.InnerRegular
    inst✝ : ErgodicSMul M X μ
    ⊢ Ergodic (fun x => HSMul.hSMul g x) μ
  -/
  borelize M
  /-
    X : Type u_2
    inst✝¹⁰ : TopologicalSpace X
    inst✝⁹ : R1Space X
    inst✝⁸ : MeasurableSpace X
    inst✝⁷ : BorelSpace X
    M : Type u_3
    inst✝⁶ : Monoid M
    inst✝⁵ : TopologicalSpace M
    inst✝⁴ : MulAction M X
    inst✝³ : ContinuousSMul M X
    g : M
    hg : DenseRange fun x => HPow.hPow g x
    μ : MeasureTheory.Measure X
    inst✝² : MeasureTheory.IsFiniteMeasure μ
    inst✝¹ : μ.InnerRegular
    inst✝ : ErgodicSMul M X μ
    this✝¹ : MeasurableSpace M := borel M
    this✝ : BorelSpace M
    ⊢ Ergodic (fun x => HSMul.hSMul g x) μ
  -/
  refine ⟨measurePreserving_smul _ _, ⟨fun s hsm hs ↦ ?_⟩⟩
  /-
    X : Type u_2
    inst✝¹⁰ : TopologicalSpace X
    inst✝⁹ : R1Space X
    inst✝⁸ : MeasurableSpace X
    inst✝⁷ : BorelSpace X
    M : Type u_3
    inst✝⁶ : Monoid M
    inst✝⁵ : TopologicalSpace M
    inst✝⁴ : MulAction M X
    inst✝³ : ContinuousSMul M X
    g : M
    hg : DenseRange fun x => HPow.hPow g x
    μ : MeasureTheory.Measure X
    inst✝² : MeasureTheory.IsFiniteMeasure μ
    inst✝¹ : μ.InnerRegular
    inst✝ : ErgodicSMul M X μ
    this✝¹ : MeasurableSpace M := borel M
    this✝ : BorelSpace M
    s : Set X
    hsm : MeasurableSet s
    hs : Eq (Set.preimage (fun x => HSMul.hSMul g x) s) s
    ⊢ Filter.EventuallyConst s (MeasureTheory.ae μ)
  -/
  refine aeconst_of_dense_setOf_preimage_smul_eq hsm.nullMeasurableSet (hg.mono ?_)
  /-
    X : Type u_2
    inst✝¹⁰ : TopologicalSpace X
    inst✝⁹ : R1Space X
    inst✝⁸ : MeasurableSpace X
    inst✝⁷ : BorelSpace X
    M : Type u_3
    inst✝⁶ : Monoid M
    inst✝⁵ : TopologicalSpace M
    inst✝⁴ : MulAction M X
    inst✝³ : ContinuousSMul M X
    g : M
    hg : DenseRange fun x => HPow.hPow g x
    μ : MeasureTheory.Measure X
    inst✝² : MeasureTheory.IsFiniteMeasure μ
    inst✝¹ : μ.InnerRegular
    inst✝ : ErgodicSMul M X μ
    this✝¹ : MeasurableSpace M := borel M
    this✝ : BorelSpace M
    s : Set X
    hsm : MeasurableSet s
    hs : Eq (Set.preimage (fun x => HSMul.hSMul g x) s) s
    ⊢ HasSubset.Subset (Set.range fun x => HPow.hPow g x) (setOf fun g => Eq (Set. …
  -/
  refine range_subset_iff.2 fun n ↦ ?_
  /-
    X : Type u_2
    inst✝¹⁰ : TopologicalSpace X
    inst✝⁹ : R1Space X
    inst✝⁸ : MeasurableSpace X
    inst✝⁷ : BorelSpace X
    M : Type u_3
    inst✝⁶ : Monoid M
    inst✝⁵ : TopologicalSpace M
    inst✝⁴ : MulAction M X
    inst✝³ : ContinuousSMul M X
    g : M
    hg : DenseRange fun x => HPow.hPow g x
    μ : MeasureTheory.Measure X
    inst✝² : MeasureTheory.IsFiniteMeasure μ
    inst✝¹ : μ.InnerRegular
    inst✝ : ErgodicSMul M X μ
    this✝¹ : MeasurableSpace M := borel M
    this✝ : BorelSpace M
    s : Set X
    hsm : MeasurableSet s
    hs : Eq (Set.preimage (fun x => HSMul.hSMul g x) s) s
    n : Nat
    ⊢ Membership.mem (setOf fun g => Eq (Set.preimage (fun x => HSMul.hSMul g x) s …
  -/
  rw [mem_setOf, ← smul_iterate, preimage_iterate_eq, iterate_fixed hs]
  /-
    🎉 no goals
  -/


/-- If `N` acts continuously and ergodically on `X` and `M` acts minimally on `N`,
then the corresponding action of `M` on `X` is ergodic. -/
@[to_additive
  "If `N` acts additively continuously and ergodically on `X` and `M` acts minimally on `N`,
then the corresponding action of `M` on `X` is ergodic."]
theorem ErgodicSMul.trans_isMinimal (N : Type*) [MulAction M N]
    [Monoid N] [TopologicalSpace N] [MulAction.IsMinimal M N]
    [MulAction N X] [IsScalarTower M N X] [ContinuousSMul N X] [ErgodicSMul N X μ] :
    ErgodicSMul M X μ where
  measure_preimage_smul c s hsm := by
    /-
      M : Type u_1
      X : Type u_2
      inst✝¹⁵ : Monoid M
      inst✝¹⁴ : SMul M X
      inst✝¹³ : TopologicalSpace X
      inst✝¹² : R1Space X
      inst✝¹¹ : MeasurableSpace X
      inst✝¹⁰ : BorelSpace X
      μ : MeasureTheory.Measure X
      inst✝⁹ : MeasureTheory.IsFiniteMeasure μ
      inst✝⁸ : μ.InnerRegular
      N : Type u_3
      inst✝⁷ : MulAction M N
      inst✝⁶ : Monoid N
      inst✝⁵ : TopologicalSpace N
      inst✝⁴ : MulAction.IsMinimal M N
      inst✝³ : MulAction N X
      inst✝² : IsScalarTower M N X
      inst✝¹ : ContinuousSMul N X
      inst✝ : ErgodicSMul N X μ
      c : M
      s : Set X
      hsm : MeasurableSet s
      ⊢ Eq (μ (Set.preimage (fun x => HSMul.hSMul c x) s)) (μ s)
    -/
    simpa only [smul_one_smul] using SMulInvariantMeasure.measure_preimage_smul (c • 1 : N) hsm
    /-
      🎉 no goals
    -/
  aeconst_of_forall_preimage_smul_ae_eq {s} hsm hs := by
    /-
      M : Type u_1
      X : Type u_2
      inst✝¹⁵ : Monoid M
      inst✝¹⁴ : SMul M X
      inst✝¹³ : TopologicalSpace X
      inst✝¹² : R1Space X
      inst✝¹¹ : MeasurableSpace X
      inst✝¹⁰ : BorelSpace X
      μ : MeasureTheory.Measure X
      inst✝⁹ : MeasureTheory.IsFiniteMeasure μ
      inst✝⁸ : μ.InnerRegular
      N : Type u_3
      inst✝⁷ : MulAction M N
      inst✝⁶ : Monoid N
      inst✝⁵ : TopologicalSpace N
      inst✝⁴ : MulAction.IsMinimal M N
      inst✝³ : MulAction N X
      inst✝² : IsScalarTower M N X
      inst✝¹ : ContinuousSMul N X
      inst✝ : ErgodicSMul N X μ
      s : Set X
      hsm : MeasurableSet s
      hs : ∀ (g : M), (MeasureTheory.ae μ).EventuallyEq (Set.preimage (fun x => HSMu …
      ⊢ Filter.EventuallyConst s (MeasureTheory.ae μ)
    -/
    refine aeconst_of_dense_setOf_preimage_smul_ae (M := N) hsm.nullMeasurableSet ?_
    /-
      M : Type u_1
      X : Type u_2
      inst✝¹⁵ : Monoid M
      inst✝¹⁴ : SMul M X
      inst✝¹³ : TopologicalSpace X
      inst✝¹² : R1Space X
      inst✝¹¹ : MeasurableSpace X
      inst✝¹⁰ : BorelSpace X
      μ : MeasureTheory.Measure X
      inst✝⁹ : MeasureTheory.IsFiniteMeasure μ
      inst✝⁸ : μ.InnerRegular
      N : Type u_3
      inst✝⁷ : MulAction M N
      inst✝⁶ : Monoid N
      inst✝⁵ : TopologicalSpace N
      inst✝⁴ : MulAction.IsMinimal M N
      inst✝³ : MulAction N X
      inst✝² : IsScalarTower M N X
      inst✝¹ : ContinuousSMul N X
      inst✝ : ErgodicSMul N X μ
      s : Set X
      hsm : MeasurableSet s
      hs : ∀ (g : M), (MeasureTheory.ae μ).EventuallyEq (Set.preimage (fun x => HSMu …
      ⊢ Dense (setOf fun g => (MeasureTheory.ae μ).EventuallyEq (Set.preimage (fun x …
    -/
    refine (MulAction.dense_orbit M 1).mono ?_
    /-
      M : Type u_1
      X : Type u_2
      inst✝¹⁵ : Monoid M
      inst✝¹⁴ : SMul M X
      inst✝¹³ : TopologicalSpace X
      inst✝¹² : R1Space X
      inst✝¹¹ : MeasurableSpace X
      inst✝¹⁰ : BorelSpace X
      μ : MeasureTheory.Measure X
      inst✝⁹ : MeasureTheory.IsFiniteMeasure μ
      inst✝⁸ : μ.InnerRegular
      N : Type u_3
      inst✝⁷ : MulAction M N
      inst✝⁶ : Monoid N
      inst✝⁵ : TopologicalSpace N
      inst✝⁴ : MulAction.IsMinimal M N
      inst✝³ : MulAction N X
      inst✝² : IsScalarTower M N X
      inst✝¹ : ContinuousSMul N X
      inst✝ : ErgodicSMul N X μ
      s : Set X
      hsm : MeasurableSet s
      hs : ∀ (g : M), (MeasureTheory.ae μ).EventuallyEq (Set.preimage (fun x => HSMu …
      ⊢ HasSubset.Subset (MulAction.orbit M 1) (setOf fun g => (MeasureTheory.ae μ). …
    -/
    rintro _ ⟨g, rfl⟩
    /-
      case intro
      M : Type u_1
      X : Type u_2
      inst✝¹⁵ : Monoid M
      inst✝¹⁴ : SMul M X
      inst✝¹³ : TopologicalSpace X
      inst✝¹² : R1Space X
      inst✝¹¹ : MeasurableSpace X
      inst✝¹⁰ : BorelSpace X
      μ : MeasureTheory.Measure X
      inst✝⁹ : MeasureTheory.IsFiniteMeasure μ
      inst✝⁸ : μ.InnerRegular
      N : Type u_3
      inst✝⁷ : MulAction M N
      inst✝⁶ : Monoid N
      inst✝⁵ : TopologicalSpace N
      inst✝⁴ : MulAction.IsMinimal M N
      inst✝³ : MulAction N X
      inst✝² : IsScalarTower M N X
      inst✝¹ : ContinuousSMul N X
      inst✝ : ErgodicSMul N X μ
      s : Set X
      hsm : MeasurableSet s
      hs : ∀ (g : M), (MeasureTheory.ae μ).EventuallyEq (Set.preimage (fun x => HSMu …
      g : M
      ⊢ Membership.mem (setOf fun g => (MeasureTheory.ae μ).EventuallyEq (Set.preima …
    -/
    simpa using hs g
    /-
      🎉 no goals
    -/


@[to_additive]
theorem aeconst_of_dense_aestabilizer_smul (hsm : NullMeasurableSet s μ)
    (hd : Dense (MulAction.aestabilizer G μ s : Set G)) : EventuallyConst s (ae μ) :=
  aeconst_of_dense_setOf_preimage_smul_ae hsm <| (hd.preimage (isOpenMap_inv _)).mono fun g hg ↦ by
    /-
      G : Type u_1
      inst✝¹¹ : Group G
      inst✝¹⁰ : TopologicalSpace G
      inst✝⁹ : ContinuousInv G
      X : Type u_2
      inst✝⁸ : TopologicalSpace X
      inst✝⁷ : R1Space X
      inst✝⁶ : MeasurableSpace X
      inst✝⁵ : BorelSpace X
      inst✝⁴ : MulAction G X
      inst✝³ : ContinuousSMul G X
      μ : MeasureTheory.Measure X
      inst✝² : MeasureTheory.IsFiniteMeasure μ
      inst✝¹ : μ.InnerRegular
      inst✝ : ErgodicSMul G X μ
      s : Set X
      hsm : MeasureTheory.NullMeasurableSet s μ
      hd : Dense ↑(MulAction.aestabilizer G μ s)
      g : G
      hg : Membership.mem (Set.preimage Inv.inv ↑(MulAction.aestabilizer G μ s)) g
      ⊢ Membership.mem (setOf fun g => (MeasureTheory.ae μ).EventuallyEq (Set.preima …
    -/
    simpa only [preimage_smul] using hg
    /-
      🎉 no goals
    -/


/-- If a monoid `M` continuously acts on an R₁ topological space `X`,
`g` is an element of `M such that its integer powers are dense in `M`,
and `μ` is a finite inner regular measure on `X` which is ergodic with respect to the action of `M`,
then the scalar multiplication by `g` is an ergodic map. -/
@[to_additive "If an additive monoid `M` continuously acts on an R₁ topological space `X`,
`g` is an element of `M such that its integer multiples are dense in `M`,
and `μ` is a finite inner regular measure on `X` which is ergodic with respect to the action of `M`,
then the vector addition of `g` is an ergodic map."]
theorem ergodic_smul_of_denseRange_zpow {g : G} (hg : DenseRange (g ^ · : ℤ → G))
    (μ : Measure X) [IsFiniteMeasure μ] [μ.InnerRegular] [ErgodicSMul G X μ] :
    Ergodic (g • ·) μ := by
  /-
    G : Type u_1
    inst✝¹¹ : Group G
    inst✝¹⁰ : TopologicalSpace G
    inst✝⁹ : ContinuousInv G
    X : Type u_2
    inst✝⁸ : TopologicalSpace X
    inst✝⁷ : R1Space X
    inst✝⁶ : MeasurableSpace X
    inst✝⁵ : BorelSpace X
    inst✝⁴ : MulAction G X
    inst✝³ : ContinuousSMul G X
    g : G
    hg : DenseRange fun x => HPow.hPow g x
    μ : MeasureTheory.Measure X
    inst✝² : MeasureTheory.IsFiniteMeasure μ
    inst✝¹ : μ.InnerRegular
    inst✝ : ErgodicSMul G X μ
    ⊢ Ergodic (fun x => HSMul.hSMul g x) μ
  -/
  borelize G
  /-
    G : Type u_1
    inst✝¹¹ : Group G
    inst✝¹⁰ : TopologicalSpace G
    inst✝⁹ : ContinuousInv G
    X : Type u_2
    inst✝⁸ : TopologicalSpace X
    inst✝⁷ : R1Space X
    inst✝⁶ : MeasurableSpace X
    inst✝⁵ : BorelSpace X
    inst✝⁴ : MulAction G X
    inst✝³ : ContinuousSMul G X
    g : G
    hg : DenseRange fun x => HPow.hPow g x
    μ : MeasureTheory.Measure X
    inst✝² : MeasureTheory.IsFiniteMeasure μ
    inst✝¹ : μ.InnerRegular
    inst✝ : ErgodicSMul G X μ
    this✝¹ : MeasurableSpace G := borel G
    this✝ : BorelSpace G
    ⊢ Ergodic (fun x => HSMul.hSMul g x) μ
  -/
  refine ⟨measurePreserving_smul _ _, ⟨fun s hsm hs ↦ ?_⟩⟩
  /-
    G : Type u_1
    inst✝¹¹ : Group G
    inst✝¹⁰ : TopologicalSpace G
    inst✝⁹ : ContinuousInv G
    X : Type u_2
    inst✝⁸ : TopologicalSpace X
    inst✝⁷ : R1Space X
    inst✝⁶ : MeasurableSpace X
    inst✝⁵ : BorelSpace X
    inst✝⁴ : MulAction G X
    inst✝³ : ContinuousSMul G X
    g : G
    hg : DenseRange fun x => HPow.hPow g x
    μ : MeasureTheory.Measure X
    inst✝² : MeasureTheory.IsFiniteMeasure μ
    inst✝¹ : μ.InnerRegular
    inst✝ : ErgodicSMul G X μ
    this✝¹ : MeasurableSpace G := borel G
    this✝ : BorelSpace G
    s : Set X
    hsm : MeasurableSet s
    hs : Eq (Set.preimage (fun x => HSMul.hSMul g x) s) s
    ⊢ Filter.EventuallyConst s (MeasureTheory.ae μ)
  -/
  refine aeconst_of_dense_aestabilizer_smul hsm.nullMeasurableSet (hg.mono ?_)
  rw [← Subgroup.coe_zpowers, SetLike.coe_subset_coe, ← Subgroup.zpowers_inv, Subgroup.zpowers_le,
    MulAction.mem_aestabilizer, ← preimage_smul, hs]

 
/-- If the left multiplication by `g` is ergodic
with respect to a measure which is positive on nonempty open sets,
then the integer powers of `g` are dense in `G`. -/
@[to_additive "If the left addition of `g` is ergodic
with respect to a measure which is positive on nonempty open sets,
then the integer multiples of `g` are dense in `G`."]
theorem DenseRange.zpow_of_ergodic_mul_left [OpensMeasurableSpace G]
    {μ : Measure G} [μ.IsOpenPosMeasure] {g : G} (hg : Ergodic (g * ·) μ) :
    DenseRange (g ^ · : ℤ → G) := by
  /-
    G : Type u_1
    inst✝⁵ : Group G
    inst✝⁴ : TopologicalSpace G
    inst✝³ : TopologicalGroup G
    inst✝² : MeasurableSpace G
    inst✝¹ : OpensMeasurableSpace G
    μ : MeasureTheory.Measure G
    inst✝ : μ.IsOpenPosMeasure
    g : G
    hg : Ergodic (fun x => HMul.hMul g x) μ
    ⊢ DenseRange fun x => HPow.hPow g x
  -/
  intro a
  /-
    G : Type u_1
    inst✝⁵ : Group G
    inst✝⁴ : TopologicalSpace G
    inst✝³ : TopologicalGroup G
    inst✝² : MeasurableSpace G
    inst✝¹ : OpensMeasurableSpace G
    μ : MeasureTheory.Measure G
    inst✝ : μ.IsOpenPosMeasure
    g : G
    hg : Ergodic (fun x => HMul.hMul g x) μ
    a : G
    ⊢ Membership.mem (closure (Set.range fun x => HPow.hPow g x)) a
  -/
  by_contra h
  obtain ⟨V, hV₁, hVo, hV⟩ :
      ∃ V : Set G, 1 ∈ V ∧ IsOpen V ∧ ∀ x ∈ V, ∀ y ∈ V, ∀ m : ℤ, g ^ m ≠ a * x / y := by
    rw [← mem_compl_iff, ← interior_compl, mem_interior_iff_mem_nhds] at h
    have : Tendsto (fun (x, y) ↦ a * x / y) (𝓝 1) (𝓝 a) :=
      Continuous.tendsto' (by fun_prop) _ _ (by simp)
    rw [nhds_prod_eq] at this
    simpa [(nhds_basis_opens (1 : G)).prod_self.mem_iff, prod_subset_iff, and_assoc] using this h
  /-
    case intro.intro.intro
    G : Type u_1
    inst✝⁵ : Group G
    inst✝⁴ : TopologicalSpace G
    inst✝³ : TopologicalGroup G
    inst✝² : MeasurableSpace G
    inst✝¹ : OpensMeasurableSpace G
    μ : MeasureTheory.Measure G
    inst✝ : μ.IsOpenPosMeasure
    g : G
    hg : Ergodic (fun x => HMul.hMul g x) μ
    a : G
    h : Not (Membership.mem (closure (Set.range fun x => HPow.hPow g x)) a)
    V : Set G
    hV₁ : Membership.mem V 1
    hVo : IsOpen V
    hV : ∀ (x : G), Membership.mem V x → ∀ (y : G), Membership.mem V y → ∀ (m : In …
    ⊢ False
  -/
  set s := ⋃ m : ℤ, g ^ m • V
  /-
    case intro.intro.intro
    G : Type u_1
    inst✝⁵ : Group G
    inst✝⁴ : TopologicalSpace G
    inst✝³ : TopologicalGroup G
    inst✝² : MeasurableSpace G
    inst✝¹ : OpensMeasurableSpace G
    μ : MeasureTheory.Measure G
    inst✝ : μ.IsOpenPosMeasure
    g : G
    hg : Ergodic (fun x => HMul.hMul g x) μ
    a : G
    h : Not (Membership.mem (closure (Set.range fun x => HPow.hPow g x)) a)
    V : Set G
    hV₁ : Membership.mem V 1
    hVo : IsOpen V
    hV : ∀ (x : G), Membership.mem V x → ∀ (y : G), Membership.mem V y → ∀ (m : In …
    s : Set G := Set.iUnion fun m => HSMul.hSMul (HPow.hPow g m) V
    ⊢ False
  -/
  have hso : IsOpen s := isOpen_iUnion fun m ↦ hVo.smul _
  /-
    case intro.intro.intro
    G : Type u_1
    inst✝⁵ : Group G
    inst✝⁴ : TopologicalSpace G
    inst✝³ : TopologicalGroup G
    inst✝² : MeasurableSpace G
    inst✝¹ : OpensMeasurableSpace G
    μ : MeasureTheory.Measure G
    inst✝ : μ.IsOpenPosMeasure
    g : G
    hg : Ergodic (fun x => HMul.hMul g x) μ
    a : G
    h : Not (Membership.mem (closure (Set.range fun x => HPow.hPow g x)) a)
    V : Set G
    hV₁ : Membership.mem V 1
    hVo : IsOpen V
    hV : ∀ (x : G), Membership.mem V x → ∀ (y : G), Membership.mem V y → ∀ (m : In …
    s : Set G := Set.iUnion fun m => HSMul.hSMul (HPow.hPow g m) V
    hso : IsOpen s
    ⊢ False
  -/
  have hsne : s.Nonempty := ⟨1, mem_iUnion.2 ⟨0, by simpa⟩⟩
  have hd : Disjoint s (a • V) := by
    simp_rw [s, disjoint_iUnion_left, disjoint_left]
    rintro m _ ⟨x, hx, rfl⟩ ⟨y, hy, hxy⟩
    apply hV y hy x hx m
    simp_all
  have hgs : (g * ·) ⁻¹' s = s := by
    simp only [s, preimage_iUnion, ← smul_eq_mul, preimage_smul]
    refine iUnion_congr_of_surjective _ (add_left_surjective (-1)) fun m ↦ ?_
    simp [zpow_add, mul_smul]
  cases hg.measure_self_or_compl_eq_zero hso.measurableSet hgs with
  | inl h => exact hso.measure_ne_zero _ hsne h
  | inr h =>
    refine (hVo.smul a).measure_ne_zero μ (.image _ ⟨1, hV₁⟩) (measure_mono_null ?_ h)
    rwa [disjoint_right] at hd


@[to_additive]
theorem ergodic_mul_left_of_denseRange_pow (hg : DenseRange (g ^ · : ℕ → G))
    (μ : Measure G) [IsFiniteMeasure μ] [μ.InnerRegular] [μ.IsMulLeftInvariant] :
    Ergodic (g * ·) μ :=
  ergodic_smul_of_denseRange_pow hg μ


@[to_additive]
theorem ergodic_mul_left_of_denseRange_zpow (hg : DenseRange (g ^ · : ℤ → G))
    (μ : Measure G) [IsFiniteMeasure μ] [μ.InnerRegular] [μ.IsMulLeftInvariant] :
    Ergodic (g * ·) μ :=
  ergodic_smul_of_denseRange_zpow hg μ


@[to_additive]
theorem ergodic_mul_left_iff_denseRange_zpow (μ : Measure G) [IsFiniteMeasure μ]
    [μ.InnerRegular] [μ.IsMulLeftInvariant] [NeZero μ] :
    Ergodic (g * ·) μ ↔ DenseRange (g ^ · : ℤ → G) :=
  ⟨.zpow_of_ergodic_mul_left, (ergodic_mul_left_of_denseRange_zpow · μ)⟩


/-- Let `f : G →* G` be a group endomorphism of a topological group with second countable topology.
If the preimages of `1` under the iterations of `f` are dense,
then it is preergodic with respect to any finite inner regular left invariant measure. -/
@[to_additive "Let `f : G →+ G` be an additive group endomorphism
of a topological additive group with second countable topology.
If the preimages of `0` under the iterations of `f` are dense,
then it is preergodic with respect to any finite inner regular left invariant measure."]
theorem preErgodic_of_dense_iUnion_preimage_one
    {μ : Measure G} [IsFiniteMeasure μ] [μ.InnerRegular] [μ.IsMulLeftInvariant]
    (f : G →* G) (hf : Dense (⋃ n, f^[n] ⁻¹' 1)) : PreErgodic f μ := by
  /-
    G : Type u_1
    inst✝⁸ : Group G
    inst✝⁷ : TopologicalSpace G
    inst✝⁶ : TopologicalGroup G
    inst✝⁵ : SecondCountableTopology G
    inst✝⁴ : MeasurableSpace G
    inst✝³ : BorelSpace G
    μ : MeasureTheory.Measure G
    inst✝² : MeasureTheory.IsFiniteMeasure μ
    inst✝¹ : μ.InnerRegular
    inst✝ : μ.IsMulLeftInvariant
    f : MonoidHom G G
    hf : Dense (Set.iUnion fun n => Set.preimage (Nat.iterate (⇑f) n) 1)
    ⊢ PreErgodic (⇑f) μ
  -/
  refine ⟨fun s hsm hs ↦ aeconst_of_dense_setOf_preimage_smul_eq (M := G) hsm.nullMeasurableSet ?_⟩
  /-
    G : Type u_1
    inst✝⁸ : Group G
    inst✝⁷ : TopologicalSpace G
    inst✝⁶ : TopologicalGroup G
    inst✝⁵ : SecondCountableTopology G
    inst✝⁴ : MeasurableSpace G
    inst✝³ : BorelSpace G
    μ : MeasureTheory.Measure G
    inst✝² : MeasureTheory.IsFiniteMeasure μ
    inst✝¹ : μ.InnerRegular
    inst✝ : μ.IsMulLeftInvariant
    f : MonoidHom G G
    hf : Dense (Set.iUnion fun n => Set.preimage (Nat.iterate (⇑f) n) 1)
    s : Set G
    hsm : MeasurableSet s
    hs : Eq (Set.preimage (⇑f) s) s
    ⊢ Dense (setOf fun g => Eq (Set.preimage (fun x => HSMul.hSMul g x) s) s)
  -/
  refine hf.mono <| iUnion_subset fun n x hx ↦ ?_
  have hsn : f^[n] ⁻¹' s = s := by
    rw [preimage_iterate_eq, iterate_fixed hs]
  /-
    G : Type u_1
    inst✝⁸ : Group G
    inst✝⁷ : TopologicalSpace G
    inst✝⁶ : TopologicalGroup G
    inst✝⁵ : SecondCountableTopology G
    inst✝⁴ : MeasurableSpace G
    inst✝³ : BorelSpace G
    μ : MeasureTheory.Measure G
    inst✝² : MeasureTheory.IsFiniteMeasure μ
    inst✝¹ : μ.InnerRegular
    inst✝ : μ.IsMulLeftInvariant
    f : MonoidHom G G
    hf : Dense (Set.iUnion fun n => Set.preimage (Nat.iterate (⇑f) n) 1)
    s : Set G
    hsm : MeasurableSet s
    hs : Eq (Set.preimage (⇑f) s) s
    n : Nat
    x : G
    hx : Membership.mem (Set.preimage (Nat.iterate (⇑f) n) 1) x
    hsn : Eq (Set.preimage (Nat.iterate (⇑f) n) s) s
    ⊢ Membership.mem (setOf fun g => Eq (Set.preimage (fun x => HSMul.hSMul g x) s …
  -/
  rw [mem_preimage, Set.mem_one] at hx
  /-
    G : Type u_1
    inst✝⁸ : Group G
    inst✝⁷ : TopologicalSpace G
    inst✝⁶ : TopologicalGroup G
    inst✝⁵ : SecondCountableTopology G
    inst✝⁴ : MeasurableSpace G
    inst✝³ : BorelSpace G
    μ : MeasureTheory.Measure G
    inst✝² : MeasureTheory.IsFiniteMeasure μ
    inst✝¹ : μ.InnerRegular
    inst✝ : μ.IsMulLeftInvariant
    f : MonoidHom G G
    hf : Dense (Set.iUnion fun n => Set.preimage (Nat.iterate (⇑f) n) 1)
    s : Set G
    hsm : MeasurableSet s
    hs : Eq (Set.preimage (⇑f) s) s
    n : Nat
    x : G
    hx : Eq (Nat.iterate (⇑f) n x) 1
    hsn : Eq (Set.preimage (Nat.iterate (⇑f) n) s) s
    ⊢ Membership.mem (setOf fun g => Eq (Set.preimage (fun x => HSMul.hSMul g x) s …
  -/
  rw [mem_setOf, ← hsn]
  /-
    G : Type u_1
    inst✝⁸ : Group G
    inst✝⁷ : TopologicalSpace G
    inst✝⁶ : TopologicalGroup G
    inst✝⁵ : SecondCountableTopology G
    inst✝⁴ : MeasurableSpace G
    inst✝³ : BorelSpace G
    μ : MeasureTheory.Measure G
    inst✝² : MeasureTheory.IsFiniteMeasure μ
    inst✝¹ : μ.InnerRegular
    inst✝ : μ.IsMulLeftInvariant
    f : MonoidHom G G
    hf : Dense (Set.iUnion fun n => Set.preimage (Nat.iterate (⇑f) n) 1)
    s : Set G
    hsm : MeasurableSet s
    hs : Eq (Set.preimage (⇑f) s) s
    n : Nat
    x : G
    hx : Eq (Nat.iterate (⇑f) n x) 1
    hsn : Eq (Set.preimage (Nat.iterate (⇑f) n) s) s
    ⊢ Eq (Set.preimage (fun x_1 => HSMul.hSMul x x_1) (Set.preimage (Nat.iterate ( …
  -/
  ext y
  /-
    case h
    G : Type u_1
    inst✝⁸ : Group G
    inst✝⁷ : TopologicalSpace G
    inst✝⁶ : TopologicalGroup G
    inst✝⁵ : SecondCountableTopology G
    inst✝⁴ : MeasurableSpace G
    inst✝³ : BorelSpace G
    μ : MeasureTheory.Measure G
    inst✝² : MeasureTheory.IsFiniteMeasure μ
    inst✝¹ : μ.InnerRegular
    inst✝ : μ.IsMulLeftInvariant
    f : MonoidHom G G
    hf : Dense (Set.iUnion fun n => Set.preimage (Nat.iterate (⇑f) n) 1)
    s : Set G
    hsm : MeasurableSet s
    hs : Eq (Set.preimage (⇑f) s) s
    n : Nat
    x : G
    hx : Eq (Nat.iterate (⇑f) n x) 1
    hsn : Eq (Set.preimage (Nat.iterate (⇑f) n) s) s
    y : G
    ⊢ Iff (Membership.mem (Set.preimage (fun x_1 => HSMul.hSMul x x_1) (Set.preima …
  -/
  simp [hx]
  /-
    🎉 no goals
  -/


/-- Let `f : G →* G` be a continuous surjective group endomorphism
of a compact topological group with second countable topology.
If the preimages of `1` under the iterations of `f` are dense,
then `f` is ergodic with respect to any finite inner regular left invariant measure. -/
@[to_additive "Let `f : G →+ G` be a continuous surjective additive group endomorphism
of a compact topological additive group with second countable topology.
If the preimages of `0` under the iterations of `f` are dense,
then `f` is ergodic with respect to any finite inner regular left invariant measure."]
theorem ergodic_of_dense_iUnion_preimage_one [CompactSpace G] {μ : Measure G} [μ.IsHaarMeasure]
    (f : G →* G) (hf : Dense (⋃ n, f^[n] ⁻¹' 1)) (hcont : Continuous f) (hsurj : Surjective f) :
    Ergodic f μ :=
  ⟨f.measurePreserving hcont hsurj rfl, f.preErgodic_of_dense_iUnion_preimage_one hf⟩


