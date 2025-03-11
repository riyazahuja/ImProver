/-- When `α` is compact, the bounded continuous maps `α →ᵇ β` are
equivalent to `C(α, β)`.
-/
@[simps (config := .asFn)]
def equivBoundedOfCompact : C(α, β) ≃ (α →ᵇ β) :=
  ⟨mkOfCompact, BoundedContinuousFunction.toContinuousMap, fun f => by
    /-
      α : Type u_1
      β : Type u_2
      E : Type u_3
      inst✝³ : TopologicalSpace α
      inst✝² : CompactSpace α
      inst✝¹ : PseudoMetricSpace β
      inst✝ : SeminormedAddCommGroup E
      f : ContinuousMap α β
      ⊢ Eq (BoundedContinuousFunction.mkOfCompact f).toContinuousMap f
    -/
    ext
    /-
      case h
      α : Type u_1
      β : Type u_2
      E : Type u_3
      inst✝³ : TopologicalSpace α
      inst✝² : CompactSpace α
      inst✝¹ : PseudoMetricSpace β
      inst✝ : SeminormedAddCommGroup E
      f : ContinuousMap α β
      a✝ : α
      ⊢ Eq ((BoundedContinuousFunction.mkOfCompact f).toContinuousMap a✝) (f a✝)
    -/
    rfl, fun f => by
    /-
      🎉 no goals
    -/
    /-
      α : Type u_1
      β : Type u_2
      E : Type u_3
      inst✝³ : TopologicalSpace α
      inst✝² : CompactSpace α
      inst✝¹ : PseudoMetricSpace β
      inst✝ : SeminormedAddCommGroup E
      f : BoundedContinuousFunction α β
      ⊢ Eq (BoundedContinuousFunction.mkOfCompact f.toContinuousMap) f
    -/
    ext
    /-
      case h
      α : Type u_1
      β : Type u_2
      E : Type u_3
      inst✝³ : TopologicalSpace α
      inst✝² : CompactSpace α
      inst✝¹ : PseudoMetricSpace β
      inst✝ : SeminormedAddCommGroup E
      f : BoundedContinuousFunction α β
      x✝ : α
      ⊢ Eq ((BoundedContinuousFunction.mkOfCompact f.toContinuousMap) x✝) (f x✝)
    -/
    rfl⟩
    /-
      🎉 no goals
    -/


theorem isUniformInducing_equivBoundedOfCompact : IsUniformInducing (equivBoundedOfCompact α β) :=
  IsUniformInducing.mk'
    (by
      /-
        α : Type u_1
        β : Type u_2
        inst✝² : TopologicalSpace α
        inst✝¹ : CompactSpace α
        inst✝ : PseudoMetricSpace β
        ⊢ ∀ (s : Set (Prod (ContinuousMap α β) (ContinuousMap α β))), Iff (Membership. …
      -/
      simp only [hasBasis_compactConvergenceUniformity.mem_iff, uniformity_basis_dist_le.mem_iff]
      exact fun s =>
        ⟨fun ⟨⟨a, b⟩, ⟨_, ⟨ε, hε, hb⟩⟩, hs⟩ =>
          ⟨{ p | ∀ x, (p.1 x, p.2 x) ∈ b }, ⟨ε, hε, fun _ h x => hb ((dist_le hε.le).mp h x)⟩,
            fun f g h => hs fun x _ => h x⟩,
          fun ⟨_, ⟨ε, hε, ht⟩, hs⟩ =>
          ⟨⟨Set.univ, { p | dist p.1 p.2 ≤ ε }⟩, ⟨isCompact_univ, ⟨ε, hε, fun _ h => h⟩⟩,
            fun ⟨f, g⟩ h => hs _ _ (ht ((dist_le hε.le).mpr fun x => h x (mem_univ x)))⟩⟩)


@[deprecated (since := "2024-10-05")]
alias uniformInducing_equivBoundedOfCompact := isUniformInducing_equivBoundedOfCompact


theorem isUniformEmbedding_equivBoundedOfCompact : IsUniformEmbedding (equivBoundedOfCompact α β) :=
  { isUniformInducing_equivBoundedOfCompact α β with
    injective := (equivBoundedOfCompact α β).injective }


@[deprecated (since := "2024-10-01")]
alias uniformEmbedding_equivBoundedOfCompact := isUniformEmbedding_equivBoundedOfCompact


/-- When `α` is compact, the bounded continuous maps `α →ᵇ 𝕜` are
additively equivalent to `C(α, 𝕜)`.
-/
-- Porting note: the following `simps` received a "maximum recursion depth" error
-- @[simps! (config := .asFn) apply symm_apply]
def addEquivBoundedOfCompact [AddMonoid β] [LipschitzAdd β] : C(α, β) ≃+ (α →ᵇ β) :=
  ({ toContinuousMapAddHom α β, (equivBoundedOfCompact α β).symm with } : (α →ᵇ β) ≃+ C(α, β)).symm

-- Porting note: added this `simp` lemma manually because of the `simps` error above

@[simp]
theorem addEquivBoundedOfCompact_symm_apply [AddMonoid β] [LipschitzAdd β] :
    ⇑((addEquivBoundedOfCompact α β).symm) = toContinuousMapAddHom α β :=
  rfl

-- Porting note: added this `simp` lemma manually because of the `simps` error above

@[simp]
theorem addEquivBoundedOfCompact_apply [AddMonoid β] [LipschitzAdd β] :
    ⇑(addEquivBoundedOfCompact α β) = mkOfCompact :=
  rfl


instance instPseudoMetricSpace : PseudoMetricSpace C(α, β) :=
  (isUniformEmbedding_equivBoundedOfCompact α β).comapPseudoMetricSpace _


instance instMetricSpace {β : Type*} [MetricSpace β] :
    MetricSpace C(α, β) :=
  (isUniformEmbedding_equivBoundedOfCompact α β).comapMetricSpace _



/-- When `α` is compact, and `β` is a metric space, the bounded continuous maps `α →ᵇ β` are
isometric to `C(α, β)`.
-/
@[simps! (config := .asFn) toEquiv apply symm_apply]
def isometryEquivBoundedOfCompact : C(α, β) ≃ᵢ (α →ᵇ β) where
  isometry_toFun _ _ := rfl
  toEquiv := equivBoundedOfCompact α β


@[simp]
theorem _root_.BoundedContinuousFunction.dist_mkOfCompact (f g : C(α, β)) :
    dist (mkOfCompact f) (mkOfCompact g) = dist f g :=
  rfl


@[simp]
theorem _root_.BoundedContinuousFunction.dist_toContinuousMap (f g : α →ᵇ β) :
    dist f.toContinuousMap g.toContinuousMap = dist f g :=
  rfl


/-- The pointwise distance is controlled by the distance between functions, by definition. -/
theorem dist_apply_le_dist (x : α) : dist (f x) (g x) ≤ dist f g := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝² : TopologicalSpace α
    inst✝¹ : CompactSpace α
    inst✝ : PseudoMetricSpace β
    f g : ContinuousMap α β
    x : α
    ⊢ LE.le (Dist.dist (f x) (g x)) (Dist.dist f g)
  -/
  simp only [← dist_mkOfCompact, dist_coe_le_dist, ← mkOfCompact_apply]
  /-
    🎉 no goals
  -/


/-- The distance between two functions is controlled by the supremum of the pointwise distances. -/
theorem dist_le (C0 : (0 : ℝ) ≤ C) : dist f g ≤ C ↔ ∀ x : α, dist (f x) (g x) ≤ C := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝² : TopologicalSpace α
    inst✝¹ : CompactSpace α
    inst✝ : PseudoMetricSpace β
    f g : ContinuousMap α β
    C : Real
    C0 : LE.le 0 C
    ⊢ Iff (LE.le (Dist.dist f g) C) (∀ (x : α), LE.le (Dist.dist (f x) (g x)) C)
  -/
  simp only [← dist_mkOfCompact, BoundedContinuousFunction.dist_le C0, mkOfCompact_apply]
  /-
    🎉 no goals
  -/


theorem dist_le_iff_of_nonempty [Nonempty α] : dist f g ≤ C ↔ ∀ x, dist (f x) (g x) ≤ C := by
  simp only [← dist_mkOfCompact, BoundedContinuousFunction.dist_le_iff_of_nonempty,
    mkOfCompact_apply]


theorem dist_lt_iff_of_nonempty [Nonempty α] : dist f g < C ↔ ∀ x : α, dist (f x) (g x) < C := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝³ : TopologicalSpace α
    inst✝² : CompactSpace α
    inst✝¹ : PseudoMetricSpace β
    f g : ContinuousMap α β
    C : Real
    inst✝ : Nonempty α
    ⊢ Iff (LT.lt (Dist.dist f g) C) (∀ (x : α), LT.lt (Dist.dist (f x) (g x)) C)
  -/
  simp only [← dist_mkOfCompact, dist_lt_iff_of_nonempty_compact, mkOfCompact_apply]
  /-
    🎉 no goals
  -/


theorem dist_lt_of_nonempty [Nonempty α] (w : ∀ x : α, dist (f x) (g x) < C) : dist f g < C :=
  dist_lt_iff_of_nonempty.2 w


theorem dist_lt_iff (C0 : (0 : ℝ) < C) : dist f g < C ↔ ∀ x : α, dist (f x) (g x) < C := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝² : TopologicalSpace α
    inst✝¹ : CompactSpace α
    inst✝ : PseudoMetricSpace β
    f g : ContinuousMap α β
    C : Real
    C0 : LT.lt 0 C
    ⊢ Iff (LT.lt (Dist.dist f g) C) (∀ (x : α), LT.lt (Dist.dist (f x) (g x)) C)
  -/
  rw [← dist_mkOfCompact, dist_lt_iff_of_compact C0]
  /-
    α : Type u_1
    β : Type u_2
    inst✝² : TopologicalSpace α
    inst✝¹ : CompactSpace α
    inst✝ : PseudoMetricSpace β
    f g : ContinuousMap α β
    C : Real
    C0 : LT.lt 0 C
    ⊢ Iff (∀ (x : α), LT.lt (Dist.dist ((BoundedContinuousFunction.mkOfCompact f)  …
  -/
  simp only [mkOfCompact_apply]
  /-
    🎉 no goals
  -/


instance {R} [Zero R] [Zero β] [PseudoMetricSpace R] [SMul R β] [BoundedSMul R β] :
    BoundedSMul R C(α, β) where
  dist_smul_pair' r f g := by
    /-
      α : Type u_1
      β : Type u_2
      E : Type u_3
      inst✝⁸ : TopologicalSpace α
      inst✝⁷ : CompactSpace α
      inst✝⁶ : PseudoMetricSpace β
      inst✝⁵ : SeminormedAddCommGroup E
      f✝ g✝ : ContinuousMap α β
      C : Real
      R : Type u_4
      inst✝⁴ : Zero R
      inst✝³ : Zero β
      inst✝² : PseudoMetricSpace R
      inst✝¹ : SMul R β
      inst✝ : BoundedSMul R β
      r : R
      f g : ContinuousMap α β
      ⊢ LE.le (Dist.dist (HSMul.hSMul r f) (HSMul.hSMul r g)) (HMul.hMul (Dist.dist  …
    -/
    simpa only [← dist_mkOfCompact] using dist_smul_pair r (mkOfCompact f) (mkOfCompact g)
    /-
      🎉 no goals
    -/
  dist_pair_smul' r₁ r₂ f := by
    /-
      α : Type u_1
      β : Type u_2
      E : Type u_3
      inst✝⁸ : TopologicalSpace α
      inst✝⁷ : CompactSpace α
      inst✝⁶ : PseudoMetricSpace β
      inst✝⁵ : SeminormedAddCommGroup E
      f✝ g : ContinuousMap α β
      C : Real
      R : Type u_4
      inst✝⁴ : Zero R
      inst✝³ : Zero β
      inst✝² : PseudoMetricSpace R
      inst✝¹ : SMul R β
      inst✝ : BoundedSMul R β
      r₁ r₂ : R
      f : ContinuousMap α β
      ⊢ LE.le (Dist.dist (HSMul.hSMul r₁ f) (HSMul.hSMul r₂ f)) (HMul.hMul (Dist.dis …
    -/
    simpa only [← dist_mkOfCompact] using dist_pair_smul r₁ r₂ (mkOfCompact f)
    /-
      🎉 no goals
    -/


instance : Norm C(α, E) where norm x := dist x 0


@[simp]
theorem _root_.BoundedContinuousFunction.norm_mkOfCompact (f : C(α, E)) : ‖mkOfCompact f‖ = ‖f‖ :=
  rfl


@[simp]
theorem _root_.BoundedContinuousFunction.norm_toContinuousMap_eq (f : α →ᵇ E) :
    ‖f.toContinuousMap‖ = ‖f‖ :=
  rfl


instance : SeminormedAddCommGroup C(α, E) where
  __ := ContinuousMap.instPseudoMetricSpace _ _
  __ := ContinuousMap.instAddCommGroupContinuousMap
  dist_eq x y := by
    /-
      α : Type u_1
      β : Type u_2
      E : Type u_3
      inst✝³ : TopologicalSpace α
      inst✝² : CompactSpace α
      inst✝¹ : PseudoMetricSpace β
      inst✝ : SeminormedAddCommGroup E
      x y : ContinuousMap α E
      ⊢ Eq (Dist.dist x y) (Norm.norm (HSub.hSub x y))
    -/
    rw [← norm_mkOfCompact, ← dist_mkOfCompact, dist_eq_norm, mkOfCompact_sub]
    /-
      🎉 no goals
    -/
  dist := dist
  norm := norm


instance {E : Type*} [NormedAddCommGroup E] : NormedAddCommGroup C(α, E) where
  __ : SeminormedAddCommGroup C(α, E) := inferInstance
  __ : MetricSpace C(α, E) := inferInstance


instance [Nonempty α] [One E] [NormOneClass E] : NormOneClass C(α, E) where
                 /-
                   α : Type u_1
                   β : Type u_2
                   E : Type u_3
                   inst✝⁶ : TopologicalSpace α
                   inst✝⁵ : CompactSpace α
                   inst✝⁴ : PseudoMetricSpace β
                   inst✝³ : SeminormedAddCommGroup E
                   inst✝² : Nonempty α
                   inst✝¹ : One E
                   inst✝ : NormOneClass E
                   ⊢ Eq (Norm.norm 1) 1
                 -/
  norm_one := by simp only [← norm_mkOfCompact, mkOfCompact_one, norm_one]
                 /-
                   🎉 no goals
                 -/


theorem norm_coe_le_norm (x : α) : ‖f x‖ ≤ ‖f‖ :=
  (mkOfCompact f).norm_coe_le_norm x


/-- Distance between the images of any two points is at most twice the norm of the function. -/
theorem dist_le_two_norm (x y : α) : dist (f x) (f y) ≤ 2 * ‖f‖ :=
  (mkOfCompact f).dist_le_two_norm x y


/-- The norm of a function is controlled by the supremum of the pointwise norms. -/
theorem norm_le {C : ℝ} (C0 : (0 : ℝ) ≤ C) : ‖f‖ ≤ C ↔ ∀ x : α, ‖f x‖ ≤ C :=
  @BoundedContinuousFunction.norm_le _ _ _ _ (mkOfCompact f) _ C0


theorem norm_le_of_nonempty [Nonempty α] {M : ℝ} : ‖f‖ ≤ M ↔ ∀ x, ‖f x‖ ≤ M :=
  @BoundedContinuousFunction.norm_le_of_nonempty _ _ _ _ _ (mkOfCompact f) _


theorem norm_lt_iff {M : ℝ} (M0 : 0 < M) : ‖f‖ < M ↔ ∀ x, ‖f x‖ < M :=
  @BoundedContinuousFunction.norm_lt_iff_of_compact _ _ _ _ _ (mkOfCompact f) _ M0


theorem nnnorm_lt_iff {M : ℝ≥0} (M0 : 0 < M) : ‖f‖₊ < M ↔ ∀ x : α, ‖f x‖₊ < M :=
  f.norm_lt_iff M0


theorem norm_lt_iff_of_nonempty [Nonempty α] {M : ℝ} : ‖f‖ < M ↔ ∀ x, ‖f x‖ < M :=
  @BoundedContinuousFunction.norm_lt_iff_of_nonempty_compact _ _ _ _ _ _ (mkOfCompact f) _


theorem nnnorm_lt_iff_of_nonempty [Nonempty α] {M : ℝ≥0} : ‖f‖₊ < M ↔ ∀ x, ‖f x‖₊ < M :=
  f.norm_lt_iff_of_nonempty


theorem apply_le_norm (f : C(α, ℝ)) (x : α) : f x ≤ ‖f‖ :=
  le_trans (le_abs.mpr (Or.inl (le_refl (f x)))) (f.norm_coe_le_norm x)


theorem neg_norm_le_apply (f : C(α, ℝ)) (x : α) : -‖f‖ ≤ f x :=
  le_trans (neg_le_neg (f.norm_coe_le_norm x)) (neg_le.mp (neg_le_abs (f x)))


theorem nnnorm_eq_iSup_nnnorm : ‖f‖₊ = ⨆ x : α, ‖f x‖₊ :=
  (mkOfCompact f).nnnorm_eq_iSup_nnnorm


theorem norm_eq_iSup_norm : ‖f‖ = ⨆ x : α, ‖f x‖ :=
  (mkOfCompact f).norm_eq_iSup_norm

-- A version with better keys

instance {X : Type*} [TopologicalSpace X] (K : TopologicalSpace.Compacts X) :
    CompactSpace (K : Set X) :=
  TopologicalSpace.Compacts.instCompactSpaceSubtypeMem ..


theorem norm_restrict_mono_set {X : Type*} [TopologicalSpace X] (f : C(X, E))
    {K L : TopologicalSpace.Compacts X} (hKL : K ≤ L) : ‖f.restrict K‖ ≤ ‖f.restrict L‖ :=
  (norm_le _ (norm_nonneg _)).mpr fun x => norm_coe_le_norm (f.restrict L) <| Set.inclusion hKL x


instance [NonUnitalSeminormedRing R] : NonUnitalSeminormedRing C(α, R) where
  __ : SeminormedAddCommGroup C(α, R) := inferInstance
  __ : NonUnitalRing C(α, R) := inferInstance
  norm_mul f g := norm_mul_le (mkOfCompact f) (mkOfCompact g)


instance [NonUnitalSeminormedCommRing R] : NonUnitalSeminormedCommRing C(α, R) where
  __ : NonUnitalSeminormedRing C(α, R) := inferInstance
  __ : NonUnitalCommRing C(α, R) := inferInstance


instance [SeminormedRing R] : SeminormedRing C(α, R) where
  __ : NonUnitalSeminormedRing C(α, R) := inferInstance
  __ : Ring C(α, R) := inferInstance


instance [SeminormedCommRing R] : SeminormedCommRing C(α, R) where
  __ : SeminormedRing C(α, R) := inferInstance
  __ : CommRing C(α, R) := inferInstance


instance [NonUnitalNormedRing R] : NonUnitalNormedRing C(α, R) where
  __ : NormedAddCommGroup C(α, R) := inferInstance
  __ : NonUnitalSeminormedRing C(α, R) := inferInstance


instance [NonUnitalNormedCommRing R] : NonUnitalNormedCommRing C(α, R) where
  __ : NonUnitalNormedRing C(α, R) := inferInstance
  __ : NonUnitalCommRing C(α, R) := inferInstance


instance [NormedRing R] : NormedRing C(α, R) where
  __ : NormedAddCommGroup C(α, R) := inferInstance
  __ : SeminormedRing C(α, R) := inferInstance


instance [NormedCommRing R] : NormedCommRing C(α, R) where
  __ : NormedRing C(α, R) := inferInstance
  __ : CommRing C(α, R) := inferInstance


instance normedSpace : NormedSpace 𝕜 C(α, E) where
  norm_smul_le := norm_smul_le


/-- When `α` is compact and `𝕜` is a normed field,
the `𝕜`-algebra of bounded continuous maps `α →ᵇ β` is
`𝕜`-linearly isometric to `C(α, β)`.
-/
def linearIsometryBoundedOfCompact : C(α, E) ≃ₗᵢ[𝕜] α →ᵇ E :=
  { addEquivBoundedOfCompact α E with
    map_smul' := fun c f => by
      /-
        α : Type u_1
        β : Type u_2
        E : Type u_3
        inst✝⁵ : TopologicalSpace α
        inst✝⁴ : CompactSpace α
        inst✝³ : PseudoMetricSpace β
        inst✝² : SeminormedAddCommGroup E
        𝕜 : Type u_4
        inst✝¹ : NormedField 𝕜
        inst✝ : NormedSpace 𝕜 E
        c : 𝕜
        f : ContinuousMap α E
        ⊢ Eq ({ toFun := __src✝.toFun, map_add' := ⋯ }.toFun (HSMul.hSMul c f)) (HSMul …
      -/
      ext
      /-
        case h
        α : Type u_1
        β : Type u_2
        E : Type u_3
        inst✝⁵ : TopologicalSpace α
        inst✝⁴ : CompactSpace α
        inst✝³ : PseudoMetricSpace β
        inst✝² : SeminormedAddCommGroup E
        𝕜 : Type u_4
        inst✝¹ : NormedField 𝕜
        inst✝ : NormedSpace 𝕜 E
        c : 𝕜
        f : ContinuousMap α E
        x✝ : α
        ⊢ Eq (({ toFun := __src✝.toFun, map_add' := ⋯ }.toFun (HSMul.hSMul c f)) x✝) ( …
      -/
      norm_cast
      /-
        🎉 no goals
      -/
    norm_map' := fun _ => rfl }


@[simp]
theorem linearIsometryBoundedOfCompact_symm_apply (f : α →ᵇ E) :
    (linearIsometryBoundedOfCompact α E 𝕜).symm f = f.toContinuousMap :=
  rfl


@[simp]
theorem linearIsometryBoundedOfCompact_apply_apply (f : C(α, E)) (a : α) :
    (linearIsometryBoundedOfCompact α E 𝕜 f) a = f a :=
  rfl


@[simp]
theorem linearIsometryBoundedOfCompact_toIsometryEquiv :
    (linearIsometryBoundedOfCompact α E 𝕜).toIsometryEquiv = isometryEquivBoundedOfCompact α E :=
  rfl


@[simp]
theorem linearIsometryBoundedOfCompact_toAddEquiv :
    ((linearIsometryBoundedOfCompact α E 𝕜).toLinearEquiv : C(α, E) ≃+ (α →ᵇ E)) =
      addEquivBoundedOfCompact α E :=
  rfl


@[simp]
theorem linearIsometryBoundedOfCompact_of_compact_toEquiv :
    (linearIsometryBoundedOfCompact α E 𝕜).toLinearEquiv.toEquiv = equivBoundedOfCompact α E :=
  rfl


@[simp] lemma nnnorm_smul_const {R β : Type*} [NormedAddCommGroup β] [NormedDivisionRing R]
    [Module R β] [BoundedSMul R β] (f : C(α, R)) (b : β) :
    ‖f • const α b‖₊ = ‖f‖₊ * ‖b‖₊ := by
  /-
    α : Type u_1
    inst✝⁵ : TopologicalSpace α
    inst✝⁴ : CompactSpace α
    R : Type u_4
    β : Type u_5
    inst✝³ : NormedAddCommGroup β
    inst✝² : NormedDivisionRing R
    inst✝¹ : Module R β
    inst✝ : BoundedSMul R β
    f : ContinuousMap α R
    b : β
    ⊢ Eq (NNNorm.nnnorm (HSMul.hSMul f (ContinuousMap.const α b))) (HMul.hMul (NNN …
  -/
  simp only [nnnorm_eq_iSup_nnnorm, smul_apply', const_apply, nnnorm_smul, iSup_mul]
  /-
    🎉 no goals
  -/


@[simp] lemma norm_smul_const {R β : Type*} [NormedAddCommGroup β] [NormedDivisionRing R]
    [Module R β] [BoundedSMul R β] (f : C(α, R)) (b : β) :
    ‖f • const α b‖ = ‖f‖ * ‖b‖ := by
  /-
    α : Type u_1
    inst✝⁵ : TopologicalSpace α
    inst✝⁴ : CompactSpace α
    R : Type u_4
    β : Type u_5
    inst✝³ : NormedAddCommGroup β
    inst✝² : NormedDivisionRing R
    inst✝¹ : Module R β
    inst✝ : BoundedSMul R β
    f : ContinuousMap α R
    b : β
    ⊢ Eq (Norm.norm (HSMul.hSMul f (ContinuousMap.const α b))) (HMul.hMul (Norm.no …
  -/
  simp only [← coe_nnnorm, NNReal.coe_mul, nnnorm_smul_const]
  /-
    🎉 no goals
  -/


instance : NormedAlgebra 𝕜 C(α, γ) :=
  { ContinuousMap.normedSpace, ContinuousMap.algebra with }


theorem uniform_continuity (f : C(α, β)) (ε : ℝ) (h : 0 < ε) :
    ∃ δ > 0, ∀ {x y}, dist x y < δ → dist (f x) (f y) < ε :=
  Metric.uniformContinuous_iff.mp (CompactSpace.uniformContinuous_of_continuous f.continuous) ε h

-- This definition allows us to separate the choice of some `δ`,
-- and the corresponding use of `dist a b < δ → dist (f a) (f b) < ε`,
-- even across different declarations.

/-- An arbitrarily chosen modulus of uniform continuity for a given function `f` and `ε > 0`. -/
def modulus (f : C(α, β)) (ε : ℝ) (h : 0 < ε) : ℝ :=
  Classical.choose (uniform_continuity f ε h)


theorem modulus_pos (f : C(α, β)) {ε : ℝ} {h : 0 < ε} : 0 < f.modulus ε h :=
  (Classical.choose_spec (uniform_continuity f ε h)).1


theorem dist_lt_of_dist_lt_modulus (f : C(α, β)) (ε : ℝ) (h : 0 < ε) {a b : α}
    (w : dist a b < f.modulus ε h) : dist (f a) (f b) < ε :=
  (Classical.choose_spec (uniform_continuity f ε h)).2 w


/-- Postcomposition of continuous functions into a normed module by a continuous linear map is a
continuous linear map.
Transferred version of `ContinuousLinearMap.compLeftContinuousBounded`,
upgraded version of `ContinuousLinearMap.compLeftContinuous`,
similar to `LinearMap.compLeft`. -/
protected def ContinuousLinearMap.compLeftContinuousCompact (g : β →L[𝕜] γ) :
    C(X, β) →L[𝕜] C(X, γ) :=
  (linearIsometryBoundedOfCompact X γ 𝕜).symm.toLinearIsometry.toContinuousLinearMap.comp <|
    (g.compLeftContinuousBounded X).comp <|
      (linearIsometryBoundedOfCompact X β 𝕜).toLinearIsometry.toContinuousLinearMap


@[simp]
theorem ContinuousLinearMap.toLinear_compLeftContinuousCompact (g : β →L[𝕜] γ) :
    (g.compLeftContinuousCompact X : C(X, β) →ₗ[𝕜] C(X, γ)) = g.compLeftContinuous 𝕜 X := by
  /-
    X : Type u_1
    𝕜 : Type u_2
    β : Type u_3
    γ : Type u_4
    inst✝⁶ : TopologicalSpace X
    inst✝⁵ : CompactSpace X
    inst✝⁴ : NontriviallyNormedField 𝕜
    inst✝³ : SeminormedAddCommGroup β
    inst✝² : NormedSpace 𝕜 β
    inst✝¹ : SeminormedAddCommGroup γ
    inst✝ : NormedSpace 𝕜 γ
    g : ContinuousLinearMap (RingHom.id 𝕜) β γ
    ⊢ Eq (↑(ContinuousLinearMap.compLeftContinuousCompact X g)) (ContinuousLinearM …
  -/
  ext f
  /-
    case h.h
    X : Type u_1
    𝕜 : Type u_2
    β : Type u_3
    γ : Type u_4
    inst✝⁶ : TopologicalSpace X
    inst✝⁵ : CompactSpace X
    inst✝⁴ : NontriviallyNormedField 𝕜
    inst✝³ : SeminormedAddCommGroup β
    inst✝² : NormedSpace 𝕜 β
    inst✝¹ : SeminormedAddCommGroup γ
    inst✝ : NormedSpace 𝕜 γ
    g : ContinuousLinearMap (RingHom.id 𝕜) β γ
    f : ContinuousMap X β
    a✝ : X
    ⊢ Eq ((↑(ContinuousLinearMap.compLeftContinuousCompact X g) f) a✝) (((Continuo …
  -/
  rfl
  /-
    🎉 no goals
  -/


@[simp]
theorem ContinuousLinearMap.compLeftContinuousCompact_apply (g : β →L[𝕜] γ) (f : C(X, β)) (x : X) :
    g.compLeftContinuousCompact X f x = g (f x) :=
  rfl


theorem summable_of_locally_summable_norm {ι : Type*} {F : ι → C(X, E)}
    (hF : ∀ K : Compacts X, Summable fun i => ‖(F i).restrict K‖) : Summable F := by
  classical
  refine (ContinuousMap.exists_tendsto_compactOpen_iff_forall _).2 fun K hK => ?_
  lift K to Compacts X using hK
  have A : ∀ s : Finset ι, restrict (↑K) (∑ i ∈ s, F i) = ∑ i ∈ s, restrict K (F i) := by
    intro s
    ext1 x
    simp
    -- This used to be the end of the proof before https://github.com/leanprover/lean4/pull/2644
    erw [restrict_apply, restrict_apply, restrict_apply, restrict_apply]
    simp? says simp only [coe_sum, Finset.sum_apply]
    congr!
  simpa only [HasSum, A] using (hF K).of_norm


theorem _root_.BoundedContinuousFunction.mkOfCompact_star [CompactSpace α] (f : C(α, β)) :
    mkOfCompact (star f) = star (mkOfCompact f) :=
  rfl


instance [CompactSpace α] : NormedStarGroup C(α, β) where
  norm_star f := by
    rw [← BoundedContinuousFunction.norm_mkOfCompact, BoundedContinuousFunction.mkOfCompact_star,
      norm_star, BoundedContinuousFunction.norm_mkOfCompact]


instance [NonUnitalNormedRing β] [StarRing β] [CStarRing β] : CStarRing C(α, β) where
  norm_mul_self_le f := by
    rw [← sq, ← Real.le_sqrt (norm_nonneg _) (norm_nonneg _),
      ContinuousMap.norm_le _ (Real.sqrt_nonneg _)]
    /-
      α : Type u_1
      β : Type u_2
      inst✝⁴ : TopologicalSpace α
      inst✝³ : CompactSpace α
      inst✝² : NonUnitalNormedRing β
      inst✝¹ : StarRing β
      inst✝ : CStarRing β
      f : ContinuousMap α β
      ⊢ ∀ (x : α), LE.le (Norm.norm (f x)) (Norm.norm (HMul.hMul (Star.star f) f)).s …
    -/
    intro x
    /-
      α : Type u_1
      β : Type u_2
      inst✝⁴ : TopologicalSpace α
      inst✝³ : CompactSpace α
      inst✝² : NonUnitalNormedRing β
      inst✝¹ : StarRing β
      inst✝ : CStarRing β
      f : ContinuousMap α β
      x : α
      ⊢ LE.le (Norm.norm (f x)) (Norm.norm (HMul.hMul (Star.star f) f)).sqrt
    -/
    rw [Real.le_sqrt (norm_nonneg _) (norm_nonneg _), sq, ← CStarRing.norm_star_mul_self]
    /-
      α : Type u_1
      β : Type u_2
      inst✝⁴ : TopologicalSpace α
      inst✝³ : CompactSpace α
      inst✝² : NonUnitalNormedRing β
      inst✝¹ : StarRing β
      inst✝ : CStarRing β
      f : ContinuousMap α β
      x : α
      ⊢ LE.le (Norm.norm (HMul.hMul (Star.star (f x)) (f x))) (Norm.norm (HMul.hMul  …
    -/
    exact ContinuousMap.norm_coe_le_norm (star f * f) x
    /-
      🎉 no goals
    -/


