variable (F) in
/-- The linear map from `⨂[𝕜] i, Eᵢ` to `ContinuousMultilinearMap 𝕜 E F →L[𝕜] F` sending
`x` in `⨂[𝕜] i, Eᵢ` to the map `f ↦ f.lift x`.
-/
@[simps!]
noncomputable def toDualContinuousMultilinearMap : (⨂[𝕜] i, E i) →ₗ[𝕜]
    ContinuousMultilinearMap 𝕜 E F →L[𝕜] F where
  toFun x := LinearMap.mkContinuous
    ((LinearMap.flip (lift (R := 𝕜) (s := E) (E := F)).toLinearMap x) ∘ₗ
    ContinuousMultilinearMap.toMultilinearMapLinear)
    (projectiveSeminorm x)
    (fun _ ↦ by simp only [LinearMap.coe_comp, Function.comp_apply,
                  ContinuousMultilinearMap.toMultilinearMapLinear_apply, LinearMap.flip_apply,
                  LinearEquiv.coe_coe]
                /-
                  ι : Type uι
                  inst✝⁵ : Fintype ι
                  𝕜 : Type u𝕜
                  inst✝⁴ : NontriviallyNormedField 𝕜
                  E : ι → Type uE
                  inst✝³ : (i : ι) → SeminormedAddCommGroup (E i)
                  inst✝² : (i : ι) → NormedSpace 𝕜 (E i)
                  F : Type uF
                  inst✝¹ : SeminormedAddCommGroup F
                  inst✝ : NormedSpace 𝕜 F
                  x : PiTensorProduct 𝕜 fun i => E i
                  x✝ : ContinuousMultilinearMap 𝕜 E F
                  ⊢ LE.le (Norm.norm ((PiTensorProduct.lift x✝.toMultilinearMap) x)) (HMul.hMul  …
                -/
                exact norm_eval_le_projectiveSeminorm _ _ _)
                /-
                  🎉 no goals
                -/
  map_add' x y := by
    /-
      ι : Type uι
      inst✝⁵ : Fintype ι
      𝕜 : Type u𝕜
      inst✝⁴ : NontriviallyNormedField 𝕜
      E : ι → Type uE
      inst✝³ : (i : ι) → SeminormedAddCommGroup (E i)
      inst✝² : (i : ι) → NormedSpace 𝕜 (E i)
      F : Type uF
      inst✝¹ : SeminormedAddCommGroup F
      inst✝ : NormedSpace 𝕜 F
      x y : PiTensorProduct 𝕜 fun i => E i
      ⊢ Eq ((fun x => (((↑PiTensorProduct.lift).flip x).comp ContinuousMultilinearMa …
    -/
    ext _
    simp only [map_add, LinearMap.mkContinuous_apply, LinearMap.coe_comp, Function.comp_apply,
      ContinuousMultilinearMap.toMultilinearMapLinear_apply, LinearMap.add_apply,
      LinearMap.flip_apply, LinearEquiv.coe_coe, ContinuousLinearMap.add_apply]
  map_smul' a x := by
    /-
      ι : Type uι
      inst✝⁵ : Fintype ι
      𝕜 : Type u𝕜
      inst✝⁴ : NontriviallyNormedField 𝕜
      E : ι → Type uE
      inst✝³ : (i : ι) → SeminormedAddCommGroup (E i)
      inst✝² : (i : ι) → NormedSpace 𝕜 (E i)
      F : Type uF
      inst✝¹ : SeminormedAddCommGroup F
      inst✝ : NormedSpace 𝕜 F
      a : 𝕜
      x : PiTensorProduct 𝕜 fun i => E i
      ⊢ Eq ({ toFun := fun x => (((↑PiTensorProduct.lift).flip x).comp ContinuousMul …
    -/
    ext _
    simp only [map_smul, LinearMap.mkContinuous_apply, LinearMap.coe_comp, Function.comp_apply,
      ContinuousMultilinearMap.toMultilinearMapLinear_apply, LinearMap.smul_apply,
      LinearMap.flip_apply, LinearEquiv.coe_coe, RingHom.id_apply, ContinuousLinearMap.coe_smul',
      Pi.smul_apply]


theorem toDualContinuousMultilinearMap_le_projectiveSeminorm (x : ⨂[𝕜] i, E i) :
    ‖toDualContinuousMultilinearMap F x‖ ≤ projectiveSeminorm x := by
  /-
    ι : Type uι
    inst✝⁵ : Fintype ι
    𝕜 : Type u𝕜
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : ι → Type uE
    inst✝³ : (i : ι) → SeminormedAddCommGroup (E i)
    inst✝² : (i : ι) → NormedSpace 𝕜 (E i)
    F : Type uF
    inst✝¹ : SeminormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    x : PiTensorProduct 𝕜 fun i => E i
    ⊢ LE.le (Norm.norm ((PiTensorProduct.toDualContinuousMultilinearMap F) x)) (Pi …
  -/
  simp only [toDualContinuousMultilinearMap, LinearMap.coe_mk, AddHom.coe_mk]
  /-
    ι : Type uι
    inst✝⁵ : Fintype ι
    𝕜 : Type u𝕜
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : ι → Type uE
    inst✝³ : (i : ι) → SeminormedAddCommGroup (E i)
    inst✝² : (i : ι) → NormedSpace 𝕜 (E i)
    F : Type uF
    inst✝¹ : SeminormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    x : PiTensorProduct 𝕜 fun i => E i
    ⊢ LE.le (Norm.norm ((((↑PiTensorProduct.lift).flip x).comp ContinuousMultiline …
  -/
  apply LinearMap.mkContinuous_norm_le _ (apply_nonneg _ _)
  /-
    🎉 no goals
  -/


/-- The injective seminorm on `⨂[𝕜] i, Eᵢ`. Morally, it sends `x` in `⨂[𝕜] i, Eᵢ` to the
`sup` of the operator norms of the `PiTensorProduct.toDualContinuousMultilinearMap F x`, for all
normed vector spaces `F`. In fact, we only take in the same universe as `⨂[𝕜] i, Eᵢ`, and then
prove in `PiTensorProduct.norm_eval_le_injectiveSeminorm` that this gives the same result.
-/
noncomputable irreducible_def injectiveSeminorm : Seminorm 𝕜 (⨂[𝕜] i, E i) :=
  sSup {p | ∃ (G : Type (max uι u𝕜 uE)) (_ : SeminormedAddCommGroup G)
  (_ : NormedSpace 𝕜 G), p = Seminorm.comp (normSeminorm 𝕜 (ContinuousMultilinearMap 𝕜 E G →L[𝕜] G))
  (toDualContinuousMultilinearMap G (𝕜 := 𝕜) (E := E))}


lemma dualSeminorms_bounded : BddAbove {p | ∃ (G : Type (max uι u𝕜 uE))
    (_ : SeminormedAddCommGroup G) (_ : NormedSpace 𝕜 G),
    p = Seminorm.comp (normSeminorm 𝕜 (ContinuousMultilinearMap 𝕜 E G →L[𝕜] G))
    (toDualContinuousMultilinearMap G (𝕜 := 𝕜) (E := E))} := by
  /-
    ι : Type uι
    inst✝³ : Fintype ι
    𝕜 : Type u𝕜
    inst✝² : NontriviallyNormedField 𝕜
    E : ι → Type uE
    inst✝¹ : (i : ι) → SeminormedAddCommGroup (E i)
    inst✝ : (i : ι) → NormedSpace 𝕜 (E i)
    ⊢ BddAbove (setOf fun p => Exists fun G => Exists fun x => Exists fun x_1 => E …
  -/
  existsi projectiveSeminorm
  /-
    ι : Type uι
    inst✝³ : Fintype ι
    𝕜 : Type u𝕜
    inst✝² : NontriviallyNormedField 𝕜
    E : ι → Type uE
    inst✝¹ : (i : ι) → SeminormedAddCommGroup (E i)
    inst✝ : (i : ι) → NormedSpace 𝕜 (E i)
    ⊢ Membership.mem (upperBounds (setOf fun p => Exists fun G => Exists fun x =>  …
  -/
  rw [mem_upperBounds]
  /-
    ι : Type uι
    inst✝³ : Fintype ι
    𝕜 : Type u𝕜
    inst✝² : NontriviallyNormedField 𝕜
    E : ι → Type uE
    inst✝¹ : (i : ι) → SeminormedAddCommGroup (E i)
    inst✝ : (i : ι) → NormedSpace 𝕜 (E i)
    ⊢ ∀ (x : Seminorm 𝕜 (PiTensorProduct 𝕜 fun i => E i)), Membership.mem (setOf f …
  -/
  simp only [Set.mem_setOf_eq, forall_exists_index]
  /-
    ι : Type uι
    inst✝³ : Fintype ι
    𝕜 : Type u𝕜
    inst✝² : NontriviallyNormedField 𝕜
    E : ι → Type uE
    inst✝¹ : (i : ι) → SeminormedAddCommGroup (E i)
    inst✝ : (i : ι) → NormedSpace 𝕜 (E i)
    ⊢ ∀ (x : Seminorm 𝕜 (PiTensorProduct 𝕜 fun i => E i)) (x_1 : Type (max uι u𝕜 u …
  -/
  intro p G _ _ hp
  /-
    ι : Type uι
    inst✝³ : Fintype ι
    𝕜 : Type u𝕜
    inst✝² : NontriviallyNormedField 𝕜
    E : ι → Type uE
    inst✝¹ : (i : ι) → SeminormedAddCommGroup (E i)
    inst✝ : (i : ι) → NormedSpace 𝕜 (E i)
    p : Seminorm 𝕜 (PiTensorProduct 𝕜 fun i => E i)
    G : Type (max uι u𝕜 uE)
    x✝¹ : SeminormedAddCommGroup G
    x✝ : NormedSpace 𝕜 G
    hp : Eq p ((normSeminorm 𝕜 (ContinuousLinearMap (RingHom.id 𝕜) (ContinuousMult …
    ⊢ LE.le p PiTensorProduct.projectiveSeminorm
  -/
  rw [hp]
  /-
    ι : Type uι
    inst✝³ : Fintype ι
    𝕜 : Type u𝕜
    inst✝² : NontriviallyNormedField 𝕜
    E : ι → Type uE
    inst✝¹ : (i : ι) → SeminormedAddCommGroup (E i)
    inst✝ : (i : ι) → NormedSpace 𝕜 (E i)
    p : Seminorm 𝕜 (PiTensorProduct 𝕜 fun i => E i)
    G : Type (max uι u𝕜 uE)
    x✝¹ : SeminormedAddCommGroup G
    x✝ : NormedSpace 𝕜 G
    hp : Eq p ((normSeminorm 𝕜 (ContinuousLinearMap (RingHom.id 𝕜) (ContinuousMult …
    ⊢ LE.le ((normSeminorm 𝕜 (ContinuousLinearMap (RingHom.id 𝕜) (ContinuousMultil …
  -/
  intro x
  /-
    ι : Type uι
    inst✝³ : Fintype ι
    𝕜 : Type u𝕜
    inst✝² : NontriviallyNormedField 𝕜
    E : ι → Type uE
    inst✝¹ : (i : ι) → SeminormedAddCommGroup (E i)
    inst✝ : (i : ι) → NormedSpace 𝕜 (E i)
    p : Seminorm 𝕜 (PiTensorProduct 𝕜 fun i => E i)
    G : Type (max uι u𝕜 uE)
    x✝¹ : SeminormedAddCommGroup G
    x✝ : NormedSpace 𝕜 G
    hp : Eq p ((normSeminorm 𝕜 (ContinuousLinearMap (RingHom.id 𝕜) (ContinuousMult …
    x : PiTensorProduct 𝕜 fun i => E i
    ⊢ LE.le ((fun f => ⇑f) ((normSeminorm 𝕜 (ContinuousLinearMap (RingHom.id 𝕜) (C …
  -/
  simp only [Seminorm.comp_apply, coe_normSeminorm]
  /-
    ι : Type uι
    inst✝³ : Fintype ι
    𝕜 : Type u𝕜
    inst✝² : NontriviallyNormedField 𝕜
    E : ι → Type uE
    inst✝¹ : (i : ι) → SeminormedAddCommGroup (E i)
    inst✝ : (i : ι) → NormedSpace 𝕜 (E i)
    p : Seminorm 𝕜 (PiTensorProduct 𝕜 fun i => E i)
    G : Type (max uι u𝕜 uE)
    x✝¹ : SeminormedAddCommGroup G
    x✝ : NormedSpace 𝕜 G
    hp : Eq p ((normSeminorm 𝕜 (ContinuousLinearMap (RingHom.id 𝕜) (ContinuousMult …
    x : PiTensorProduct 𝕜 fun i => E i
    ⊢ LE.le (Norm.norm ((PiTensorProduct.toDualContinuousMultilinearMap G) x)) (Pi …
  -/
  exact toDualContinuousMultilinearMap_le_projectiveSeminorm _
  /-
    🎉 no goals
  -/


theorem injectiveSeminorm_apply (x : ⨂[𝕜] i, E i) :
    injectiveSeminorm x = ⨆ p : {p | ∃ (G : Type (max uι u𝕜 uE))
    (_ : SeminormedAddCommGroup G) (_ : NormedSpace 𝕜 G), p = Seminorm.comp (normSeminorm 𝕜
    (ContinuousMultilinearMap 𝕜 E G →L[𝕜] G))
    (toDualContinuousMultilinearMap G (𝕜 := 𝕜) (E := E))}, p.1 x := by
  simpa only [injectiveSeminorm, Set.coe_setOf, Set.mem_setOf_eq]
    using Seminorm.sSup_apply dualSeminorms_bounded


theorem norm_eval_le_injectiveSeminorm (f : ContinuousMultilinearMap 𝕜 E F) (x : ⨂[𝕜] i, E i) :
    ‖lift f.toMultilinearMap x‖ ≤ ‖f‖ * injectiveSeminorm x := by
    /- If `F` were in `Type (max uι u𝕜 uE)` (which is the type of `⨂[𝕜] i, E i`), then the
    property that we want to prove would hold by definition of `injectiveSeminorm`. This is
    not necessarily true, but we will show that there exists a normed vector space `G` in
    `Type (max uι u𝕜 uE)` and an injective isometry from `G` to `F` such that `f` factors
    through a continuous multilinear map `f'` from `E = Π i, E i` to `G`, to which we can apply
    the definition of `injectiveSeminorm`. The desired inequality for `f` then follows
    immediately.
    The idea is very simple: the multilinear map `f` corresponds by `PiTensorProduct.lift`
    to a linear map from `⨂[𝕜] i, E i` to `F`, say `l`. We want to take `G` to be the image of
    `l`, with the norm induced from that of `F`; to make sure that we are in the correct universe,
    it is actually more convenient to take `G` equal to the coimage of `l` (i.e. the quotient
    of `⨂[𝕜] i, E i` by the kernel of `l`), which is canonically isomorphic to its image by
    `LinearMap.quotKerEquivRange`. -/
  /-
    ι : Type uι
    inst✝⁵ : Fintype ι
    𝕜 : Type u𝕜
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : ι → Type uE
    inst✝³ : (i : ι) → SeminormedAddCommGroup (E i)
    inst✝² : (i : ι) → NormedSpace 𝕜 (E i)
    F : Type uF
    inst✝¹ : SeminormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    f : ContinuousMultilinearMap 𝕜 E F
    x : PiTensorProduct 𝕜 fun i => E i
    ⊢ LE.le (Norm.norm ((PiTensorProduct.lift f.toMultilinearMap) x)) (HMul.hMul ( …
  -/
  set G := (⨂[𝕜] i, E i) ⧸ LinearMap.ker (lift f.toMultilinearMap)
  /-
    ι : Type uι
    inst✝⁵ : Fintype ι
    𝕜 : Type u𝕜
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : ι → Type uE
    inst✝³ : (i : ι) → SeminormedAddCommGroup (E i)
    inst✝² : (i : ι) → NormedSpace 𝕜 (E i)
    F : Type uF
    inst✝¹ : SeminormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    f : ContinuousMultilinearMap 𝕜 E F
    x : PiTensorProduct 𝕜 fun i => E i
    G : Type (max (max (max uE u𝕜) uι) (max uE uι) u𝕜) := HasQuotient.Quotient (Pi …
    ⊢ LE.le (Norm.norm ((PiTensorProduct.lift f.toMultilinearMap) x)) (HMul.hMul ( …
  -/
  set G' := LinearMap.range (lift f.toMultilinearMap)
  /-
    ι : Type uι
    inst✝⁵ : Fintype ι
    𝕜 : Type u𝕜
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : ι → Type uE
    inst✝³ : (i : ι) → SeminormedAddCommGroup (E i)
    inst✝² : (i : ι) → NormedSpace 𝕜 (E i)
    F : Type uF
    inst✝¹ : SeminormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    f : ContinuousMultilinearMap 𝕜 E F
    x : PiTensorProduct 𝕜 fun i => E i
    G : Type (max (max (max uE u𝕜) uι) (max uE uι) u𝕜) := HasQuotient.Quotient (Pi …
    G' : Submodule 𝕜 F := LinearMap.range (PiTensorProduct.lift f.toMultilinearMap)
    ⊢ LE.le (Norm.norm ((PiTensorProduct.lift f.toMultilinearMap) x)) (HMul.hMul ( …
  -/
  set e := LinearMap.quotKerEquivRange (lift f.toMultilinearMap)
  /-
    ι : Type uι
    inst✝⁵ : Fintype ι
    𝕜 : Type u𝕜
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : ι → Type uE
    inst✝³ : (i : ι) → SeminormedAddCommGroup (E i)
    inst✝² : (i : ι) → NormedSpace 𝕜 (E i)
    F : Type uF
    inst✝¹ : SeminormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    f : ContinuousMultilinearMap 𝕜 E F
    x : PiTensorProduct 𝕜 fun i => E i
    G : Type (max (max (max uE u𝕜) uι) (max uE uι) u𝕜) := HasQuotient.Quotient (Pi …
    G' : Submodule 𝕜 F := LinearMap.range (PiTensorProduct.lift f.toMultilinearMap)
    e : LinearEquiv (RingHom.id 𝕜) (HasQuotient.Quotient (PiTensorProduct 𝕜 fun i  …
    ⊢ LE.le (Norm.norm ((PiTensorProduct.lift f.toMultilinearMap) x)) (HMul.hMul ( …
  -/
  letI := SeminormedAddCommGroup.induced G G' e
  /-
    ι : Type uι
    inst✝⁵ : Fintype ι
    𝕜 : Type u𝕜
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : ι → Type uE
    inst✝³ : (i : ι) → SeminormedAddCommGroup (E i)
    inst✝² : (i : ι) → NormedSpace 𝕜 (E i)
    F : Type uF
    inst✝¹ : SeminormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    f : ContinuousMultilinearMap 𝕜 E F
    x : PiTensorProduct 𝕜 fun i => E i
    G : Type (max (max (max uE u𝕜) uι) (max uE uι) u𝕜) := HasQuotient.Quotient (Pi …
    G' : Submodule 𝕜 F := LinearMap.range (PiTensorProduct.lift f.toMultilinearMap)
    e : LinearEquiv (RingHom.id 𝕜) (HasQuotient.Quotient (PiTensorProduct 𝕜 fun i  …
    this : SeminormedAddCommGroup G := SeminormedAddCommGroup.induced G (Subtype f …
    ⊢ LE.le (Norm.norm ((PiTensorProduct.lift f.toMultilinearMap) x)) (HMul.hMul ( …
  -/
  letI := NormedSpace.induced 𝕜 G G' e
  /-
    ι : Type uι
    inst✝⁵ : Fintype ι
    𝕜 : Type u𝕜
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : ι → Type uE
    inst✝³ : (i : ι) → SeminormedAddCommGroup (E i)
    inst✝² : (i : ι) → NormedSpace 𝕜 (E i)
    F : Type uF
    inst✝¹ : SeminormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    f : ContinuousMultilinearMap 𝕜 E F
    x : PiTensorProduct 𝕜 fun i => E i
    G : Type (max (max (max uE u𝕜) uι) (max uE uι) u𝕜) := HasQuotient.Quotient (Pi …
    G' : Submodule 𝕜 F := LinearMap.range (PiTensorProduct.lift f.toMultilinearMap)
    e : LinearEquiv (RingHom.id 𝕜) (HasQuotient.Quotient (PiTensorProduct 𝕜 fun i  …
    this✝ : SeminormedAddCommGroup G := SeminormedAddCommGroup.induced G (Subtype  …
    this : NormedSpace 𝕜 G := NormedSpace.induced 𝕜 G (Subtype fun x => Membership …
    ⊢ LE.le (Norm.norm ((PiTensorProduct.lift f.toMultilinearMap) x)) (HMul.hMul ( …
  -/
  set f'₀ := lift.symm (e.symm.toLinearMap ∘ₗ LinearMap.rangeRestrict (lift f.toMultilinearMap))
  have hf'₀ : ∀ (x : Π (i : ι), E i), ‖f'₀ x‖ ≤ ‖f‖ * ∏ i, ‖x i‖ := fun x ↦ by
    change ‖e (f'₀ x)‖ ≤ _
    simp only [lift_symm, LinearMap.compMultilinearMap_apply, LinearMap.coe_comp,
        LinearEquiv.coe_coe, Function.comp_apply, LinearEquiv.apply_symm_apply, Submodule.coe_norm,
        LinearMap.codRestrict_apply, lift.tprod, ContinuousMultilinearMap.coe_coe, e, f'₀]
    exact f.le_opNorm x
  /-
    ι : Type uι
    inst✝⁵ : Fintype ι
    𝕜 : Type u𝕜
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : ι → Type uE
    inst✝³ : (i : ι) → SeminormedAddCommGroup (E i)
    inst✝² : (i : ι) → NormedSpace 𝕜 (E i)
    F : Type uF
    inst✝¹ : SeminormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    f : ContinuousMultilinearMap 𝕜 E F
    x : PiTensorProduct 𝕜 fun i => E i
    G : Type (max (max (max uE u𝕜) uι) (max uE uι) u𝕜) := HasQuotient.Quotient (Pi …
    G' : Submodule 𝕜 F := LinearMap.range (PiTensorProduct.lift f.toMultilinearMap)
    e : LinearEquiv (RingHom.id 𝕜) (HasQuotient.Quotient (PiTensorProduct 𝕜 fun i  …
    this✝ : SeminormedAddCommGroup G := SeminormedAddCommGroup.induced G (Subtype  …
    this : NormedSpace 𝕜 G := NormedSpace.induced 𝕜 G (Subtype fun x => Membership …
    f'₀ : MultilinearMap 𝕜 E (HasQuotient.Quotient (PiTensorProduct 𝕜 fun i => E i …
    hf'₀ : ∀ (x : (i : ι) → E i), LE.le (Norm.norm (f'₀ x)) (HMul.hMul (Norm.norm  …
    ⊢ LE.le (Norm.norm ((PiTensorProduct.lift f.toMultilinearMap) x)) (HMul.hMul ( …
  -/
  set f' := MultilinearMap.mkContinuous f'₀ ‖f‖ hf'₀
  /-
    ι : Type uι
    inst✝⁵ : Fintype ι
    𝕜 : Type u𝕜
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : ι → Type uE
    inst✝³ : (i : ι) → SeminormedAddCommGroup (E i)
    inst✝² : (i : ι) → NormedSpace 𝕜 (E i)
    F : Type uF
    inst✝¹ : SeminormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    f : ContinuousMultilinearMap 𝕜 E F
    x : PiTensorProduct 𝕜 fun i => E i
    G : Type (max (max (max uE u𝕜) uι) (max uE uι) u𝕜) := HasQuotient.Quotient (Pi …
    G' : Submodule 𝕜 F := LinearMap.range (PiTensorProduct.lift f.toMultilinearMap)
    e : LinearEquiv (RingHom.id 𝕜) (HasQuotient.Quotient (PiTensorProduct 𝕜 fun i  …
    this✝ : SeminormedAddCommGroup G := SeminormedAddCommGroup.induced G (Subtype  …
    this : NormedSpace 𝕜 G := NormedSpace.induced 𝕜 G (Subtype fun x => Membership …
    f'₀ : MultilinearMap 𝕜 E (HasQuotient.Quotient (PiTensorProduct 𝕜 fun i => E i …
    hf'₀ : ∀ (x : (i : ι) → E i), LE.le (Norm.norm (f'₀ x)) (HMul.hMul (Norm.norm  …
    f' : ContinuousMultilinearMap 𝕜 E (HasQuotient.Quotient (PiTensorProduct 𝕜 fun …
    ⊢ LE.le (Norm.norm ((PiTensorProduct.lift f.toMultilinearMap) x)) (HMul.hMul ( …
  -/
  have hnorm : ‖f'‖ ≤ ‖f‖ := (f'.opNorm_le_iff (norm_nonneg f)).mpr hf'₀
  have heq : e (lift f'.toMultilinearMap x) = lift f.toMultilinearMap x := by
    induction x using PiTensorProduct.induction_on with
    | smul_tprod =>
      simp only [lift_symm, map_smul, lift.tprod, ContinuousMultilinearMap.coe_coe,
      MultilinearMap.coe_mkContinuous, LinearMap.compMultilinearMap_apply, LinearMap.coe_comp,
      LinearEquiv.coe_coe, Function.comp_apply, LinearEquiv.apply_symm_apply, SetLike.val_smul,
      LinearMap.codRestrict_apply, f', f'₀]
    | add _ _ hx hy => simp only [map_add, Submodule.coe_add, hx, hy]
  suffices h : ‖lift f'.toMultilinearMap x‖ ≤ ‖f'‖ * injectiveSeminorm x by
    change ‖(e (lift f'.toMultilinearMap x)).1‖ ≤ _ at h
    rw [heq] at h
    exact le_trans h (mul_le_mul_of_nonneg_right hnorm (apply_nonneg _ _))
  have hle : Seminorm.comp (normSeminorm 𝕜 (ContinuousMultilinearMap 𝕜 E G →L[𝕜] G))
      (toDualContinuousMultilinearMap G (𝕜 := 𝕜) (E := E)) ≤ injectiveSeminorm := by
    simp only [injectiveSeminorm]
    refine le_csSup dualSeminorms_bounded ?_
    rw [Set.mem_setOf]
    existsi G, inferInstance, inferInstance
    rfl
  /-
    ι : Type uι
    inst✝⁵ : Fintype ι
    𝕜 : Type u𝕜
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : ι → Type uE
    inst✝³ : (i : ι) → SeminormedAddCommGroup (E i)
    inst✝² : (i : ι) → NormedSpace 𝕜 (E i)
    F : Type uF
    inst✝¹ : SeminormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    f : ContinuousMultilinearMap 𝕜 E F
    x : PiTensorProduct 𝕜 fun i => E i
    G : Type (max (max (max uE u𝕜) uι) (max uE uι) u𝕜) := HasQuotient.Quotient (Pi …
    G' : Submodule 𝕜 F := LinearMap.range (PiTensorProduct.lift f.toMultilinearMap)
    e : LinearEquiv (RingHom.id 𝕜) (HasQuotient.Quotient (PiTensorProduct 𝕜 fun i  …
    this✝ : SeminormedAddCommGroup G := SeminormedAddCommGroup.induced G (Subtype  …
    this : NormedSpace 𝕜 G := NormedSpace.induced 𝕜 G (Subtype fun x => Membership …
    f'₀ : MultilinearMap 𝕜 E (HasQuotient.Quotient (PiTensorProduct 𝕜 fun i => E i …
    hf'₀ : ∀ (x : (i : ι) → E i), LE.le (Norm.norm (f'₀ x)) (HMul.hMul (Norm.norm  …
    f' : ContinuousMultilinearMap 𝕜 E (HasQuotient.Quotient (PiTensorProduct 𝕜 fun …
    hnorm : LE.le (Norm.norm f') (Norm.norm f)
    heq : Eq (↑(e ((PiTensorProduct.lift f'.toMultilinearMap) x))) ((PiTensorProdu …
    hle : LE.le ((normSeminorm 𝕜 (ContinuousLinearMap (RingHom.id 𝕜) (ContinuousMu …
    ⊢ LE.le (Norm.norm ((PiTensorProduct.lift f'.toMultilinearMap) x)) (HMul.hMul  …
  -/
  refine le_trans ?_ (mul_le_mul_of_nonneg_left (hle x) (norm_nonneg f'))
  /-
    ι : Type uι
    inst✝⁵ : Fintype ι
    𝕜 : Type u𝕜
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : ι → Type uE
    inst✝³ : (i : ι) → SeminormedAddCommGroup (E i)
    inst✝² : (i : ι) → NormedSpace 𝕜 (E i)
    F : Type uF
    inst✝¹ : SeminormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    f : ContinuousMultilinearMap 𝕜 E F
    x : PiTensorProduct 𝕜 fun i => E i
    G : Type (max (max (max uE u𝕜) uι) (max uE uι) u𝕜) := HasQuotient.Quotient (Pi …
    G' : Submodule 𝕜 F := LinearMap.range (PiTensorProduct.lift f.toMultilinearMap)
    e : LinearEquiv (RingHom.id 𝕜) (HasQuotient.Quotient (PiTensorProduct 𝕜 fun i  …
    this✝ : SeminormedAddCommGroup G := SeminormedAddCommGroup.induced G (Subtype  …
    this : NormedSpace 𝕜 G := NormedSpace.induced 𝕜 G (Subtype fun x => Membership …
    f'₀ : MultilinearMap 𝕜 E (HasQuotient.Quotient (PiTensorProduct 𝕜 fun i => E i …
    hf'₀ : ∀ (x : (i : ι) → E i), LE.le (Norm.norm (f'₀ x)) (HMul.hMul (Norm.norm  …
    f' : ContinuousMultilinearMap 𝕜 E (HasQuotient.Quotient (PiTensorProduct 𝕜 fun …
    hnorm : LE.le (Norm.norm f') (Norm.norm f)
    heq : Eq (↑(e ((PiTensorProduct.lift f'.toMultilinearMap) x))) ((PiTensorProdu …
    hle : LE.le ((normSeminorm 𝕜 (ContinuousLinearMap (RingHom.id 𝕜) (ContinuousMu …
    ⊢ LE.le (Norm.norm ((PiTensorProduct.lift f'.toMultilinearMap) x)) (HMul.hMul  …
  -/
  simp only [Seminorm.comp_apply, coe_normSeminorm, ← toDualContinuousMultilinearMap_apply_apply]
  /-
    ι : Type uι
    inst✝⁵ : Fintype ι
    𝕜 : Type u𝕜
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : ι → Type uE
    inst✝³ : (i : ι) → SeminormedAddCommGroup (E i)
    inst✝² : (i : ι) → NormedSpace 𝕜 (E i)
    F : Type uF
    inst✝¹ : SeminormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    f : ContinuousMultilinearMap 𝕜 E F
    x : PiTensorProduct 𝕜 fun i => E i
    G : Type (max (max (max uE u𝕜) uι) (max uE uι) u𝕜) := HasQuotient.Quotient (Pi …
    G' : Submodule 𝕜 F := LinearMap.range (PiTensorProduct.lift f.toMultilinearMap)
    e : LinearEquiv (RingHom.id 𝕜) (HasQuotient.Quotient (PiTensorProduct 𝕜 fun i  …
    this✝ : SeminormedAddCommGroup G := SeminormedAddCommGroup.induced G (Subtype  …
    this : NormedSpace 𝕜 G := NormedSpace.induced 𝕜 G (Subtype fun x => Membership …
    f'₀ : MultilinearMap 𝕜 E (HasQuotient.Quotient (PiTensorProduct 𝕜 fun i => E i …
    hf'₀ : ∀ (x : (i : ι) → E i), LE.le (Norm.norm (f'₀ x)) (HMul.hMul (Norm.norm  …
    f' : ContinuousMultilinearMap 𝕜 E (HasQuotient.Quotient (PiTensorProduct 𝕜 fun …
    hnorm : LE.le (Norm.norm f') (Norm.norm f)
    heq : Eq (↑(e ((PiTensorProduct.lift f'.toMultilinearMap) x))) ((PiTensorProdu …
    hle : LE.le ((normSeminorm 𝕜 (ContinuousLinearMap (RingHom.id 𝕜) (ContinuousMu …
    ⊢ LE.le (Norm.norm (((PiTensorProduct.toDualContinuousMultilinearMap (HasQuoti …
  -/
  rw [mul_comm]
  /-
    ι : Type uι
    inst✝⁵ : Fintype ι
    𝕜 : Type u𝕜
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : ι → Type uE
    inst✝³ : (i : ι) → SeminormedAddCommGroup (E i)
    inst✝² : (i : ι) → NormedSpace 𝕜 (E i)
    F : Type uF
    inst✝¹ : SeminormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    f : ContinuousMultilinearMap 𝕜 E F
    x : PiTensorProduct 𝕜 fun i => E i
    G : Type (max (max (max uE u𝕜) uι) (max uE uι) u𝕜) := HasQuotient.Quotient (Pi …
    G' : Submodule 𝕜 F := LinearMap.range (PiTensorProduct.lift f.toMultilinearMap)
    e : LinearEquiv (RingHom.id 𝕜) (HasQuotient.Quotient (PiTensorProduct 𝕜 fun i  …
    this✝ : SeminormedAddCommGroup G := SeminormedAddCommGroup.induced G (Subtype  …
    this : NormedSpace 𝕜 G := NormedSpace.induced 𝕜 G (Subtype fun x => Membership …
    f'₀ : MultilinearMap 𝕜 E (HasQuotient.Quotient (PiTensorProduct 𝕜 fun i => E i …
    hf'₀ : ∀ (x : (i : ι) → E i), LE.le (Norm.norm (f'₀ x)) (HMul.hMul (Norm.norm  …
    f' : ContinuousMultilinearMap 𝕜 E (HasQuotient.Quotient (PiTensorProduct 𝕜 fun …
    hnorm : LE.le (Norm.norm f') (Norm.norm f)
    heq : Eq (↑(e ((PiTensorProduct.lift f'.toMultilinearMap) x))) ((PiTensorProdu …
    hle : LE.le ((normSeminorm 𝕜 (ContinuousLinearMap (RingHom.id 𝕜) (ContinuousMu …
    ⊢ LE.le (Norm.norm (((PiTensorProduct.toDualContinuousMultilinearMap (HasQuoti …
  -/
  exact ContinuousLinearMap.le_opNorm _ _
  /-
    🎉 no goals
  -/


theorem injectiveSeminorm_le_projectiveSeminorm :
    injectiveSeminorm (𝕜 := 𝕜) (E := E) ≤ projectiveSeminorm := by
  /-
    ι : Type uι
    inst✝³ : Fintype ι
    𝕜 : Type u𝕜
    inst✝² : NontriviallyNormedField 𝕜
    E : ι → Type uE
    inst✝¹ : (i : ι) → SeminormedAddCommGroup (E i)
    inst✝ : (i : ι) → NormedSpace 𝕜 (E i)
    ⊢ LE.le PiTensorProduct.injectiveSeminorm PiTensorProduct.projectiveSeminorm
  -/
  rw [injectiveSeminorm]
  /-
    ι : Type uι
    inst✝³ : Fintype ι
    𝕜 : Type u𝕜
    inst✝² : NontriviallyNormedField 𝕜
    E : ι → Type uE
    inst✝¹ : (i : ι) → SeminormedAddCommGroup (E i)
    inst✝ : (i : ι) → NormedSpace 𝕜 (E i)
    ⊢ LE.le (SupSet.sSup (setOf fun p => Exists fun G => Exists fun x => Exists fu …
  -/
  refine csSup_le ?_ ?_
    /-
      case refine_1
      ι : Type uι
      inst✝³ : Fintype ι
      𝕜 : Type u𝕜
      inst✝² : NontriviallyNormedField 𝕜
      E : ι → Type uE
      inst✝¹ : (i : ι) → SeminormedAddCommGroup (E i)
      inst✝ : (i : ι) → NormedSpace 𝕜 (E i)
      ⊢ (setOf fun p => Exists fun G => Exists fun x => Exists fun x_1 => Eq p ((nor …
    -/
  · existsi 0
    /-
      case refine_1
      ι : Type uι
      inst✝³ : Fintype ι
      𝕜 : Type u𝕜
      inst✝² : NontriviallyNormedField 𝕜
      E : ι → Type uE
      inst✝¹ : (i : ι) → SeminormedAddCommGroup (E i)
      inst✝ : (i : ι) → NormedSpace 𝕜 (E i)
      ⊢ Membership.mem (setOf fun p => Exists fun G => Exists fun x => Exists fun x_ …
    -/
    simp only [Set.mem_setOf_eq]
    /-
      case refine_1
      ι : Type uι
      inst✝³ : Fintype ι
      𝕜 : Type u𝕜
      inst✝² : NontriviallyNormedField 𝕜
      E : ι → Type uE
      inst✝¹ : (i : ι) → SeminormedAddCommGroup (E i)
      inst✝ : (i : ι) → NormedSpace 𝕜 (E i)
      ⊢ Exists fun G => Exists fun x => Exists fun x_1 => Eq 0 ((normSeminorm 𝕜 (Con …
    -/
    existsi PUnit, inferInstance, inferInstance
    /-
      case refine_1
      ι : Type uι
      inst✝³ : Fintype ι
      𝕜 : Type u𝕜
      inst✝² : NontriviallyNormedField 𝕜
      E : ι → Type uE
      inst✝¹ : (i : ι) → SeminormedAddCommGroup (E i)
      inst✝ : (i : ι) → NormedSpace 𝕜 (E i)
      ⊢ Eq 0 ((normSeminorm 𝕜 (ContinuousLinearMap (RingHom.id 𝕜) (ContinuousMultili …
    -/
    ext x
    /-
      case refine_1.h
      ι : Type uι
      inst✝³ : Fintype ι
      𝕜 : Type u𝕜
      inst✝² : NontriviallyNormedField 𝕜
      E : ι → Type uE
      inst✝¹ : (i : ι) → SeminormedAddCommGroup (E i)
      inst✝ : (i : ι) → NormedSpace 𝕜 (E i)
      x : PiTensorProduct 𝕜 fun i => E i
      ⊢ Eq (0 x) (((normSeminorm 𝕜 (ContinuousLinearMap (RingHom.id 𝕜) (ContinuousMu …
    -/
    simp only [Seminorm.zero_apply, Seminorm.comp_apply, coe_normSeminorm]
    /-
      case refine_1.h
      ι : Type uι
      inst✝³ : Fintype ι
      𝕜 : Type u𝕜
      inst✝² : NontriviallyNormedField 𝕜
      E : ι → Type uE
      inst✝¹ : (i : ι) → SeminormedAddCommGroup (E i)
      inst✝ : (i : ι) → NormedSpace 𝕜 (E i)
      x : PiTensorProduct 𝕜 fun i => E i
      ⊢ Eq 0 (Norm.norm ((PiTensorProduct.toDualContinuousMultilinearMap PUnit.{max  …
    -/
    rw [Subsingleton.elim (toDualContinuousMultilinearMap PUnit x) 0, norm_zero]
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      ι : Type uι
      inst✝³ : Fintype ι
      𝕜 : Type u𝕜
      inst✝² : NontriviallyNormedField 𝕜
      E : ι → Type uE
      inst✝¹ : (i : ι) → SeminormedAddCommGroup (E i)
      inst✝ : (i : ι) → NormedSpace 𝕜 (E i)
      ⊢ ∀ (b : Seminorm 𝕜 (PiTensorProduct 𝕜 fun i => E i)), Membership.mem (setOf f …
    -/
  · intro p hp
    /-
      case refine_2
      ι : Type uι
      inst✝³ : Fintype ι
      𝕜 : Type u𝕜
      inst✝² : NontriviallyNormedField 𝕜
      E : ι → Type uE
      inst✝¹ : (i : ι) → SeminormedAddCommGroup (E i)
      inst✝ : (i : ι) → NormedSpace 𝕜 (E i)
      p : Seminorm 𝕜 (PiTensorProduct 𝕜 fun i => E i)
      hp : Membership.mem (setOf fun p => Exists fun G => Exists fun x => Exists fun …
      ⊢ LE.le p PiTensorProduct.projectiveSeminorm
    -/
    simp only [Set.mem_setOf_eq] at hp
    /-
      case refine_2
      ι : Type uι
      inst✝³ : Fintype ι
      𝕜 : Type u𝕜
      inst✝² : NontriviallyNormedField 𝕜
      E : ι → Type uE
      inst✝¹ : (i : ι) → SeminormedAddCommGroup (E i)
      inst✝ : (i : ι) → NormedSpace 𝕜 (E i)
      p : Seminorm 𝕜 (PiTensorProduct 𝕜 fun i => E i)
      hp : Exists fun G => Exists fun x => Exists fun x_1 => Eq p ((normSeminorm 𝕜 ( …
      ⊢ LE.le p PiTensorProduct.projectiveSeminorm
    -/
    obtain ⟨G, _, _, h⟩ := hp
    /-
      case refine_2.intro.intro.intro
      ι : Type uι
      inst✝³ : Fintype ι
      𝕜 : Type u𝕜
      inst✝² : NontriviallyNormedField 𝕜
      E : ι → Type uE
      inst✝¹ : (i : ι) → SeminormedAddCommGroup (E i)
      inst✝ : (i : ι) → NormedSpace 𝕜 (E i)
      p : Seminorm 𝕜 (PiTensorProduct 𝕜 fun i => E i)
      G : Type (max uι u𝕜 uE)
      w✝¹ : SeminormedAddCommGroup G
      w✝ : NormedSpace 𝕜 G
      h : Eq p ((normSeminorm 𝕜 (ContinuousLinearMap (RingHom.id 𝕜) (ContinuousMulti …
      ⊢ LE.le p PiTensorProduct.projectiveSeminorm
    -/
    rw [h]; intro x; simp only [Seminorm.comp_apply, coe_normSeminorm]
    /-
      case refine_2.intro.intro.intro
      ι : Type uι
      inst✝³ : Fintype ι
      𝕜 : Type u𝕜
      inst✝² : NontriviallyNormedField 𝕜
      E : ι → Type uE
      inst✝¹ : (i : ι) → SeminormedAddCommGroup (E i)
      inst✝ : (i : ι) → NormedSpace 𝕜 (E i)
      p : Seminorm 𝕜 (PiTensorProduct 𝕜 fun i => E i)
      G : Type (max uι u𝕜 uE)
      w✝¹ : SeminormedAddCommGroup G
      w✝ : NormedSpace 𝕜 G
      h : Eq p ((normSeminorm 𝕜 (ContinuousLinearMap (RingHom.id 𝕜) (ContinuousMulti …
      x : PiTensorProduct 𝕜 fun i => E i
      ⊢ LE.le (Norm.norm ((PiTensorProduct.toDualContinuousMultilinearMap G) x)) (Pi …
    -/
    exact toDualContinuousMultilinearMap_le_projectiveSeminorm _
    /-
      🎉 no goals
    -/


theorem injectiveSeminorm_tprod_le (m : Π (i : ι), E i) :
    injectiveSeminorm (⨂ₜ[𝕜] i, m i) ≤ ∏ i, ‖m i‖ :=
  le_trans (injectiveSeminorm_le_projectiveSeminorm _) (projectiveSeminorm_tprod_le m)


noncomputable instance : SeminormedAddCommGroup (⨂[𝕜] i, E i) :=
  AddGroupSeminorm.toSeminormedAddCommGroup injectiveSeminorm.toAddGroupSeminorm


noncomputable instance : NormedSpace 𝕜 (⨂[𝕜] i, E i) where
  norm_smul_le a x := by
    /-
      ι : Type uι
      inst✝⁵ : Fintype ι
      𝕜 : Type u𝕜
      inst✝⁴ : NontriviallyNormedField 𝕜
      E : ι → Type uE
      inst✝³ : (i : ι) → SeminormedAddCommGroup (E i)
      inst✝² : (i : ι) → NormedSpace 𝕜 (E i)
      F : Type uF
      inst✝¹ : SeminormedAddCommGroup F
      inst✝ : NormedSpace 𝕜 F
      a : 𝕜
      x : PiTensorProduct 𝕜 fun i => E i
      ⊢ LE.le (Norm.norm (HSMul.hSMul a x)) (HMul.hMul (Norm.norm a) (Norm.norm x))
    -/
    change injectiveSeminorm.toFun (a • x) ≤ _
    /-
      ι : Type uι
      inst✝⁵ : Fintype ι
      𝕜 : Type u𝕜
      inst✝⁴ : NontriviallyNormedField 𝕜
      E : ι → Type uE
      inst✝³ : (i : ι) → SeminormedAddCommGroup (E i)
      inst✝² : (i : ι) → NormedSpace 𝕜 (E i)
      F : Type uF
      inst✝¹ : SeminormedAddCommGroup F
      inst✝ : NormedSpace 𝕜 F
      a : 𝕜
      x : PiTensorProduct 𝕜 fun i => E i
      ⊢ LE.le (PiTensorProduct.injectiveSeminorm.toFun (HSMul.hSMul a x)) (HMul.hMul …
    -/
    rw [injectiveSeminorm.smul']
    /-
      ι : Type uι
      inst✝⁵ : Fintype ι
      𝕜 : Type u𝕜
      inst✝⁴ : NontriviallyNormedField 𝕜
      E : ι → Type uE
      inst✝³ : (i : ι) → SeminormedAddCommGroup (E i)
      inst✝² : (i : ι) → NormedSpace 𝕜 (E i)
      F : Type uF
      inst✝¹ : SeminormedAddCommGroup F
      inst✝ : NormedSpace 𝕜 F
      a : 𝕜
      x : PiTensorProduct 𝕜 fun i => E i
      ⊢ LE.le (HMul.hMul (Norm.norm a) (PiTensorProduct.injectiveSeminorm.toFun x))  …
    -/
    rfl
    /-
      🎉 no goals
    -/


/-- The linear equivalence between `ContinuousMultilinearMap 𝕜 E F` and `(⨂[𝕜] i, Eᵢ) →L[𝕜] F`
induced by `PiTensorProduct.lift`, for every normed space `F`.
-/
@[simps]
noncomputable def liftEquiv : ContinuousMultilinearMap 𝕜 E F ≃ₗ[𝕜] (⨂[𝕜] i, E i) →L[𝕜] F where
  toFun f := LinearMap.mkContinuous (lift f.toMultilinearMap) ‖f‖
    (fun x ↦ norm_eval_le_injectiveSeminorm f x)
                     /-
                       ι : Type uι
                       inst✝⁵ : Fintype ι
                       𝕜 : Type u𝕜
                       inst✝⁴ : NontriviallyNormedField 𝕜
                       E : ι → Type uE
                       inst✝³ : (i : ι) → SeminormedAddCommGroup (E i)
                       inst✝² : (i : ι) → NormedSpace 𝕜 (E i)
                       F : Type uF
                       inst✝¹ : SeminormedAddCommGroup F
                       inst✝ : NormedSpace 𝕜 F
                       f g : ContinuousMultilinearMap 𝕜 E F
                       ⊢ Eq ((fun f => (PiTensorProduct.lift f.toMultilinearMap).mkContinuous (Norm.n …
                     -/
  map_add' f g := by ext _; simp only [ContinuousMultilinearMap.toMultilinearMap_add, map_add,
    LinearMap.mkContinuous_apply, LinearMap.add_apply, ContinuousLinearMap.add_apply]
                      /-
                        ι : Type uι
                        inst✝⁵ : Fintype ι
                        𝕜 : Type u𝕜
                        inst✝⁴ : NontriviallyNormedField 𝕜
                        E : ι → Type uE
                        inst✝³ : (i : ι) → SeminormedAddCommGroup (E i)
                        inst✝² : (i : ι) → NormedSpace 𝕜 (E i)
                        F : Type uF
                        inst✝¹ : SeminormedAddCommGroup F
                        inst✝ : NormedSpace 𝕜 F
                        a : 𝕜
                        f : ContinuousMultilinearMap 𝕜 E F
                        ⊢ Eq ({ toFun := fun f => (PiTensorProduct.lift f.toMultilinearMap).mkContinuo …
                      -/
  map_smul' a f := by ext _; simp only [ContinuousMultilinearMap.toMultilinearMap_smul, map_smul,
    LinearMap.mkContinuous_apply, LinearMap.smul_apply, RingHom.id_apply,
    ContinuousLinearMap.coe_smul', Pi.smul_apply]
  invFun l := MultilinearMap.mkContinuous (lift.symm l.toLinearMap) ‖l‖ (fun x ↦ by
    /-
      ι : Type uι
      inst✝⁵ : Fintype ι
      𝕜 : Type u𝕜
      inst✝⁴ : NontriviallyNormedField 𝕜
      E : ι → Type uE
      inst✝³ : (i : ι) → SeminormedAddCommGroup (E i)
      inst✝² : (i : ι) → NormedSpace 𝕜 (E i)
      F : Type uF
      inst✝¹ : SeminormedAddCommGroup F
      inst✝ : NormedSpace 𝕜 F
      l : ContinuousLinearMap (RingHom.id 𝕜) (PiTensorProduct 𝕜 fun i => E i) F
      x : (i : ι) → E i
      ⊢ LE.le (Norm.norm ((PiTensorProduct.lift.symm ↑l) x)) (HMul.hMul (Norm.norm l …
    -/
    simp only [lift_symm, LinearMap.compMultilinearMap_apply, ContinuousLinearMap.coe_coe]
    refine le_trans (ContinuousLinearMap.le_opNorm _ _) (mul_le_mul_of_nonneg_left ?_
      (norm_nonneg l))
    /-
      ι : Type uι
      inst✝⁵ : Fintype ι
      𝕜 : Type u𝕜
      inst✝⁴ : NontriviallyNormedField 𝕜
      E : ι → Type uE
      inst✝³ : (i : ι) → SeminormedAddCommGroup (E i)
      inst✝² : (i : ι) → NormedSpace 𝕜 (E i)
      F : Type uF
      inst✝¹ : SeminormedAddCommGroup F
      inst✝ : NormedSpace 𝕜 F
      l : ContinuousLinearMap (RingHom.id 𝕜) (PiTensorProduct 𝕜 fun i => E i) F
      x : (i : ι) → E i
      ⊢ LE.le (Norm.norm ((PiTensorProduct.tprod 𝕜) x)) (Finset.univ.prod fun i => N …
    -/
    exact injectiveSeminorm_tprod_le x)
    /-
      🎉 no goals
    -/
                   /-
                     ι : Type uι
                     inst✝⁵ : Fintype ι
                     𝕜 : Type u𝕜
                     inst✝⁴ : NontriviallyNormedField 𝕜
                     E : ι → Type uE
                     inst✝³ : (i : ι) → SeminormedAddCommGroup (E i)
                     inst✝² : (i : ι) → NormedSpace 𝕜 (E i)
                     F : Type uF
                     inst✝¹ : SeminormedAddCommGroup F
                     inst✝ : NormedSpace 𝕜 F
                     f : ContinuousMultilinearMap 𝕜 E F
                     ⊢ Eq ((fun l => (PiTensorProduct.lift.symm ↑l).mkContinuous (Norm.norm l) ⋯) ( …
                   -/
  left_inv f := by ext x; simp only [LinearMap.mkContinuous_coe, LinearEquiv.symm_apply_apply,
      MultilinearMap.coe_mkContinuous, ContinuousMultilinearMap.coe_coe]
  right_inv l := by
    /-
      ι : Type uι
      inst✝⁵ : Fintype ι
      𝕜 : Type u𝕜
      inst✝⁴ : NontriviallyNormedField 𝕜
      E : ι → Type uE
      inst✝³ : (i : ι) → SeminormedAddCommGroup (E i)
      inst✝² : (i : ι) → NormedSpace 𝕜 (E i)
      F : Type uF
      inst✝¹ : SeminormedAddCommGroup F
      inst✝ : NormedSpace 𝕜 F
      l : ContinuousLinearMap (RingHom.id 𝕜) (PiTensorProduct 𝕜 fun i => E i) F
      ⊢ Eq ({ toFun := fun f => (PiTensorProduct.lift f.toMultilinearMap).mkContinuo …
    -/
    rw [← ContinuousLinearMap.coe_inj]
    /-
      ι : Type uι
      inst✝⁵ : Fintype ι
      𝕜 : Type u𝕜
      inst✝⁴ : NontriviallyNormedField 𝕜
      E : ι → Type uE
      inst✝³ : (i : ι) → SeminormedAddCommGroup (E i)
      inst✝² : (i : ι) → NormedSpace 𝕜 (E i)
      F : Type uF
      inst✝¹ : SeminormedAddCommGroup F
      inst✝ : NormedSpace 𝕜 F
      l : ContinuousLinearMap (RingHom.id 𝕜) (PiTensorProduct 𝕜 fun i => E i) F
      ⊢ Eq ↑({ toFun := fun f => (PiTensorProduct.lift f.toMultilinearMap).mkContinu …
    -/
    apply PiTensorProduct.ext; ext m
    simp only [lift_symm, LinearMap.mkContinuous_coe, LinearMap.compMultilinearMap_apply,
      lift.tprod, ContinuousMultilinearMap.coe_coe, MultilinearMap.coe_mkContinuous,
      ContinuousLinearMap.coe_coe]


/-- For a normed space `F`, we have constructed in `PiTensorProduct.liftEquiv` the canonical
linear equivalence between `ContinuousMultilinearMap 𝕜 E F` and `(⨂[𝕜] i, Eᵢ) →L[𝕜] F`
(induced by `PiTensorProduct.lift`). Here we give the upgrade of this equivalence to
an isometric linear equivalence; in particular, it is a continuous linear equivalence.
-/
noncomputable def liftIsometry : ContinuousMultilinearMap 𝕜 E F ≃ₗᵢ[𝕜] (⨂[𝕜] i, E i) →L[𝕜] F :=
  { liftEquiv 𝕜 E F with
    norm_map' := by
      /-
        ι : Type uι
        inst✝⁵ : Fintype ι
        𝕜 : Type u𝕜
        inst✝⁴ : NontriviallyNormedField 𝕜
        E : ι → Type uE
        inst✝³ : (i : ι) → SeminormedAddCommGroup (E i)
        inst✝² : (i : ι) → NormedSpace 𝕜 (E i)
        F : Type uF
        inst✝¹ : SeminormedAddCommGroup F
        inst✝ : NormedSpace 𝕜 F
        ⊢ ∀ (x : ContinuousMultilinearMap 𝕜 E F), Eq (Norm.norm (__src✝ x)) (Norm.norm …
      -/
      intro f
      /-
        ι : Type uι
        inst✝⁵ : Fintype ι
        𝕜 : Type u𝕜
        inst✝⁴ : NontriviallyNormedField 𝕜
        E : ι → Type uE
        inst✝³ : (i : ι) → SeminormedAddCommGroup (E i)
        inst✝² : (i : ι) → NormedSpace 𝕜 (E i)
        F : Type uF
        inst✝¹ : SeminormedAddCommGroup F
        inst✝ : NormedSpace 𝕜 F
        f : ContinuousMultilinearMap 𝕜 E F
        ⊢ Eq (Norm.norm (__src✝ f)) (Norm.norm f)
      -/
      refine le_antisymm ?_ ?_
        /-
          case refine_1
          ι : Type uι
          inst✝⁵ : Fintype ι
          𝕜 : Type u𝕜
          inst✝⁴ : NontriviallyNormedField 𝕜
          E : ι → Type uE
          inst✝³ : (i : ι) → SeminormedAddCommGroup (E i)
          inst✝² : (i : ι) → NormedSpace 𝕜 (E i)
          F : Type uF
          inst✝¹ : SeminormedAddCommGroup F
          inst✝ : NormedSpace 𝕜 F
          f : ContinuousMultilinearMap 𝕜 E F
          ⊢ LE.le (Norm.norm (__src✝ f)) (Norm.norm f)
        -/
      · simp only [liftEquiv, lift_symm, LinearEquiv.coe_mk]
        /-
          case refine_1
          ι : Type uι
          inst✝⁵ : Fintype ι
          𝕜 : Type u𝕜
          inst✝⁴ : NontriviallyNormedField 𝕜
          E : ι → Type uE
          inst✝³ : (i : ι) → SeminormedAddCommGroup (E i)
          inst✝² : (i : ι) → NormedSpace 𝕜 (E i)
          F : Type uF
          inst✝¹ : SeminormedAddCommGroup F
          inst✝ : NormedSpace 𝕜 F
          f : ContinuousMultilinearMap 𝕜 E F
          ⊢ LE.le (Norm.norm ((PiTensorProduct.lift f.toMultilinearMap).mkContinuous (No …
        -/
        exact LinearMap.mkContinuous_norm_le _ (norm_nonneg f) _
        /-
          🎉 no goals
        -/
        /-
          case refine_2
          ι : Type uι
          inst✝⁵ : Fintype ι
          𝕜 : Type u𝕜
          inst✝⁴ : NontriviallyNormedField 𝕜
          E : ι → Type uE
          inst✝³ : (i : ι) → SeminormedAddCommGroup (E i)
          inst✝² : (i : ι) → NormedSpace 𝕜 (E i)
          F : Type uF
          inst✝¹ : SeminormedAddCommGroup F
          inst✝ : NormedSpace 𝕜 F
          f : ContinuousMultilinearMap 𝕜 E F
          ⊢ LE.le (Norm.norm f) (Norm.norm (__src✝ f))
        -/
      · conv_lhs => rw [← (liftEquiv 𝕜 E F).left_inv f]
        simp only [liftEquiv, lift_symm, AddHom.toFun_eq_coe, AddHom.coe_mk,
          LinearEquiv.invFun_eq_symm, LinearEquiv.coe_symm_mk, LinearMap.mkContinuous_coe,
          LinearEquiv.coe_mk]
        /-
          case refine_2
          ι : Type uι
          inst✝⁵ : Fintype ι
          𝕜 : Type u𝕜
          inst✝⁴ : NontriviallyNormedField 𝕜
          E : ι → Type uE
          inst✝³ : (i : ι) → SeminormedAddCommGroup (E i)
          inst✝² : (i : ι) → NormedSpace 𝕜 (E i)
          F : Type uF
          inst✝¹ : SeminormedAddCommGroup F
          inst✝ : NormedSpace 𝕜 F
          f : ContinuousMultilinearMap 𝕜 E F
          ⊢ LE.le (Norm.norm (((PiTensorProduct.lift f.toMultilinearMap).compMultilinear …
        -/
        exact MultilinearMap.mkContinuous_norm_le _ (norm_nonneg _) _ }
        /-
          🎉 no goals
        -/


@[simp]
theorem liftIsometry_apply_apply (f : ContinuousMultilinearMap 𝕜 E F) (x : ⨂[𝕜] i, E i) :
    liftIsometry 𝕜 E F f x = lift f.toMultilinearMap x := by
  simp only [liftIsometry, LinearIsometryEquiv.coe_mk, liftEquiv_apply,
    LinearMap.mkContinuous_apply]


/-- The canonical continuous multilinear map from `E = Πᵢ Eᵢ` to `⨂[𝕜] i, Eᵢ`.
-/
@[simps!]
noncomputable def tprodL : ContinuousMultilinearMap 𝕜 E (⨂[𝕜] i, E i) :=
  (liftIsometry 𝕜 E _).symm (ContinuousLinearMap.id 𝕜 _)


@[simp]
theorem tprodL_coe : (tprodL 𝕜).toMultilinearMap = tprod 𝕜 (s := E) := by
  /-
    ι : Type uι
    inst✝³ : Fintype ι
    𝕜 : Type u𝕜
    inst✝² : NontriviallyNormedField 𝕜
    E : ι → Type uE
    inst✝¹ : (i : ι) → SeminormedAddCommGroup (E i)
    inst✝ : (i : ι) → NormedSpace 𝕜 (E i)
    ⊢ Eq (PiTensorProduct.tprodL 𝕜).toMultilinearMap (PiTensorProduct.tprod 𝕜)
  -/
  ext m
  /-
    case H
    ι : Type uι
    inst✝³ : Fintype ι
    𝕜 : Type u𝕜
    inst✝² : NontriviallyNormedField 𝕜
    E : ι → Type uE
    inst✝¹ : (i : ι) → SeminormedAddCommGroup (E i)
    inst✝ : (i : ι) → NormedSpace 𝕜 (E i)
    m : (i : ι) → E i
    ⊢ Eq ((PiTensorProduct.tprodL 𝕜).toMultilinearMap m) ((PiTensorProduct.tprod 𝕜 …
  -/
  simp only [ContinuousMultilinearMap.coe_coe, tprodL_toFun]
  /-
    🎉 no goals
  -/


@[simp]
theorem liftIsometry_symm_apply (l : (⨂[𝕜] i, E i) →L[𝕜] F) :
    (liftIsometry 𝕜 E F).symm l = l.compContinuousMultilinearMap (tprodL 𝕜) := by
  /-
    ι : Type uι
    inst✝⁵ : Fintype ι
    𝕜 : Type u𝕜
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : ι → Type uE
    inst✝³ : (i : ι) → SeminormedAddCommGroup (E i)
    inst✝² : (i : ι) → NormedSpace 𝕜 (E i)
    F : Type uF
    inst✝¹ : SeminormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    l : ContinuousLinearMap (RingHom.id 𝕜) (PiTensorProduct 𝕜 fun i => E i) F
    ⊢ Eq ((PiTensorProduct.liftIsometry 𝕜 E F).symm l) (l.compContinuousMultilinea …
  -/
  ext m
  /-
    case H
    ι : Type uι
    inst✝⁵ : Fintype ι
    𝕜 : Type u𝕜
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : ι → Type uE
    inst✝³ : (i : ι) → SeminormedAddCommGroup (E i)
    inst✝² : (i : ι) → NormedSpace 𝕜 (E i)
    F : Type uF
    inst✝¹ : SeminormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    l : ContinuousLinearMap (RingHom.id 𝕜) (PiTensorProduct 𝕜 fun i => E i) F
    m : (i : ι) → E i
    ⊢ Eq (((PiTensorProduct.liftIsometry 𝕜 E F).symm l) m) ((l.compContinuousMulti …
  -/
  change (liftEquiv 𝕜 E F).symm l m = _
  simp only [liftEquiv_symm_apply, lift_symm, MultilinearMap.coe_mkContinuous,
    LinearMap.compMultilinearMap_apply, ContinuousLinearMap.coe_coe,
    ContinuousLinearMap.compContinuousMultilinearMap_coe, Function.comp_apply, tprodL_toFun]


@[simp]
theorem liftIsometry_tprodL :
    liftIsometry 𝕜 E _ (tprodL 𝕜) = ContinuousLinearMap.id 𝕜 (⨂[𝕜] i, E i) := by
  /-
    ι : Type uι
    inst✝³ : Fintype ι
    𝕜 : Type u𝕜
    inst✝² : NontriviallyNormedField 𝕜
    E : ι → Type uE
    inst✝¹ : (i : ι) → SeminormedAddCommGroup (E i)
    inst✝ : (i : ι) → NormedSpace 𝕜 (E i)
    ⊢ Eq ((PiTensorProduct.liftIsometry 𝕜 E (PiTensorProduct 𝕜 fun i => E i)) (PiT …
  -/
  ext _
  simp only [liftIsometry_apply_apply, tprodL_coe, lift_tprod, LinearMap.id_coe, id_eq,
    ContinuousLinearMap.coe_id']


/--
Let `Eᵢ` and `E'ᵢ` be two families of normed `𝕜`-vector spaces.
Let `f` be a family of continuous `𝕜`-linear maps between `Eᵢ` and `E'ᵢ`, i.e.
`f : Πᵢ Eᵢ →L[𝕜] E'ᵢ`, then there is an induced continuous linear map
`⨂ᵢ Eᵢ → ⨂ᵢ E'ᵢ` by `⨂ aᵢ ↦ ⨂ fᵢ aᵢ`.
-/
noncomputable def mapL : (⨂[𝕜] i, E i) →L[𝕜] ⨂[𝕜] i, E' i :=
  liftIsometry 𝕜 E _ <| (tprodL 𝕜).compContinuousLinearMap f


@[simp]
theorem mapL_coe : (mapL f).toLinearMap = map (fun i ↦ (f i).toLinearMap) := by
  /-
    ι : Type uι
    inst✝⁵ : Fintype ι
    𝕜 : Type u𝕜
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : ι → Type uE
    inst✝³ : (i : ι) → SeminormedAddCommGroup (E i)
    inst✝² : (i : ι) → NormedSpace 𝕜 (E i)
    E' : ι → Type u_1
    inst✝¹ : (i : ι) → SeminormedAddCommGroup (E' i)
    inst✝ : (i : ι) → NormedSpace 𝕜 (E' i)
    f : (i : ι) → ContinuousLinearMap (RingHom.id 𝕜) (E i) (E' i)
    ⊢ Eq (↑(PiTensorProduct.mapL f)) (PiTensorProduct.map fun i => ↑(f i))
  -/
  ext
  simp only [mapL, LinearMap.compMultilinearMap_apply, ContinuousLinearMap.coe_coe,
    liftIsometry_apply_apply, lift.tprod, ContinuousMultilinearMap.coe_coe,
    ContinuousMultilinearMap.compContinuousLinearMap_apply, tprodL_toFun, map_tprod]


@[simp]
theorem mapL_apply (x : ⨂[𝕜] i, E i) : mapL f x = map (fun i ↦ (f i).toLinearMap) x := by
  induction x using PiTensorProduct.induction_on with
  | smul_tprod =>
    simp only [mapL, map_smul, liftIsometry_apply_apply, lift.tprod,
    ContinuousMultilinearMap.coe_coe, ContinuousMultilinearMap.compContinuousLinearMap_apply,
    tprodL_toFun, map_tprod, ContinuousLinearMap.coe_coe]
  | add _ _ hx hy => simp only [map_add, hx, hy]


/-- Given submodules `pᵢ ⊆ Eᵢ`, this is the natural map: `⨂[𝕜] i, pᵢ → ⨂[𝕜] i, Eᵢ`.
This is the continuous version of `PiTensorProduct.mapIncl`.
-/
@[simp]
noncomputable def mapLIncl (p : Π i, Submodule 𝕜 (E i)) : (⨂[𝕜] i, p i) →L[𝕜] ⨂[𝕜] i, E i :=
  mapL fun (i : ι) ↦ (p i).subtypeL


theorem mapL_comp : mapL (fun (i : ι) ↦ g i ∘L f i) = mapL g ∘L mapL f := by
  /-
    ι : Type uι
    inst✝⁷ : Fintype ι
    𝕜 : Type u𝕜
    inst✝⁶ : NontriviallyNormedField 𝕜
    E : ι → Type uE
    inst✝⁵ : (i : ι) → SeminormedAddCommGroup (E i)
    inst✝⁴ : (i : ι) → NormedSpace 𝕜 (E i)
    E' : ι → Type u_1
    E'' : ι → Type u_2
    inst✝³ : (i : ι) → SeminormedAddCommGroup (E' i)
    inst✝² : (i : ι) → NormedSpace 𝕜 (E' i)
    inst✝¹ : (i : ι) → SeminormedAddCommGroup (E'' i)
    inst✝ : (i : ι) → NormedSpace 𝕜 (E'' i)
    g : (i : ι) → ContinuousLinearMap (RingHom.id 𝕜) (E' i) (E'' i)
    f : (i : ι) → ContinuousLinearMap (RingHom.id 𝕜) (E i) (E' i)
    ⊢ Eq (PiTensorProduct.mapL fun i => (g i).comp (f i)) ((PiTensorProduct.mapL g …
  -/
  apply ContinuousLinearMap.coe_injective
  /-
    case a
    ι : Type uι
    inst✝⁷ : Fintype ι
    𝕜 : Type u𝕜
    inst✝⁶ : NontriviallyNormedField 𝕜
    E : ι → Type uE
    inst✝⁵ : (i : ι) → SeminormedAddCommGroup (E i)
    inst✝⁴ : (i : ι) → NormedSpace 𝕜 (E i)
    E' : ι → Type u_1
    E'' : ι → Type u_2
    inst✝³ : (i : ι) → SeminormedAddCommGroup (E' i)
    inst✝² : (i : ι) → NormedSpace 𝕜 (E' i)
    inst✝¹ : (i : ι) → SeminormedAddCommGroup (E'' i)
    inst✝ : (i : ι) → NormedSpace 𝕜 (E'' i)
    g : (i : ι) → ContinuousLinearMap (RingHom.id 𝕜) (E' i) (E'' i)
    f : (i : ι) → ContinuousLinearMap (RingHom.id 𝕜) (E i) (E' i)
    ⊢ Eq ↑(PiTensorProduct.mapL fun i => (g i).comp (f i)) ↑((PiTensorProduct.mapL …
  -/
  ext
  simp only [mapL_coe, ContinuousLinearMap.coe_comp, LinearMap.compMultilinearMap_apply, map_tprod,
    LinearMap.coe_comp, ContinuousLinearMap.coe_coe, Function.comp_apply]


theorem liftIsometry_comp_mapL (h : ContinuousMultilinearMap 𝕜 E' F) :
    liftIsometry 𝕜 E' F h ∘L mapL f = liftIsometry 𝕜 E F (h.compContinuousLinearMap f) := by
  /-
    ι : Type uι
    inst✝⁷ : Fintype ι
    𝕜 : Type u𝕜
    inst✝⁶ : NontriviallyNormedField 𝕜
    E : ι → Type uE
    inst✝⁵ : (i : ι) → SeminormedAddCommGroup (E i)
    inst✝⁴ : (i : ι) → NormedSpace 𝕜 (E i)
    F : Type uF
    inst✝³ : SeminormedAddCommGroup F
    inst✝² : NormedSpace 𝕜 F
    E' : ι → Type u_1
    inst✝¹ : (i : ι) → SeminormedAddCommGroup (E' i)
    inst✝ : (i : ι) → NormedSpace 𝕜 (E' i)
    f : (i : ι) → ContinuousLinearMap (RingHom.id 𝕜) (E i) (E' i)
    h : ContinuousMultilinearMap 𝕜 E' F
    ⊢ Eq (((PiTensorProduct.liftIsometry 𝕜 E' F) h).comp (PiTensorProduct.mapL f)) …
  -/
  apply ContinuousLinearMap.coe_injective
  /-
    case a
    ι : Type uι
    inst✝⁷ : Fintype ι
    𝕜 : Type u𝕜
    inst✝⁶ : NontriviallyNormedField 𝕜
    E : ι → Type uE
    inst✝⁵ : (i : ι) → SeminormedAddCommGroup (E i)
    inst✝⁴ : (i : ι) → NormedSpace 𝕜 (E i)
    F : Type uF
    inst✝³ : SeminormedAddCommGroup F
    inst✝² : NormedSpace 𝕜 F
    E' : ι → Type u_1
    inst✝¹ : (i : ι) → SeminormedAddCommGroup (E' i)
    inst✝ : (i : ι) → NormedSpace 𝕜 (E' i)
    f : (i : ι) → ContinuousLinearMap (RingHom.id 𝕜) (E i) (E' i)
    h : ContinuousMultilinearMap 𝕜 E' F
    ⊢ Eq ↑(((PiTensorProduct.liftIsometry 𝕜 E' F) h).comp (PiTensorProduct.mapL f) …
  -/
  ext
  simp only [ContinuousLinearMap.coe_comp, mapL_coe, LinearMap.compMultilinearMap_apply,
    LinearMap.coe_comp, ContinuousLinearMap.coe_coe, Function.comp_apply, map_tprod,
    liftIsometry_apply_apply, lift.tprod, ContinuousMultilinearMap.coe_coe,
    ContinuousMultilinearMap.compContinuousLinearMap_apply]


@[simp]
theorem mapL_id : mapL (fun i ↦ ContinuousLinearMap.id 𝕜 (E i)) = ContinuousLinearMap.id _ _ := by
  /-
    ι : Type uι
    inst✝³ : Fintype ι
    𝕜 : Type u𝕜
    inst✝² : NontriviallyNormedField 𝕜
    E : ι → Type uE
    inst✝¹ : (i : ι) → SeminormedAddCommGroup (E i)
    inst✝ : (i : ι) → NormedSpace 𝕜 (E i)
    ⊢ Eq (PiTensorProduct.mapL fun i => ContinuousLinearMap.id 𝕜 (E i)) (Continuou …
  -/
  apply ContinuousLinearMap.coe_injective
  /-
    case a
    ι : Type uι
    inst✝³ : Fintype ι
    𝕜 : Type u𝕜
    inst✝² : NontriviallyNormedField 𝕜
    E : ι → Type uE
    inst✝¹ : (i : ι) → SeminormedAddCommGroup (E i)
    inst✝ : (i : ι) → NormedSpace 𝕜 (E i)
    ⊢ Eq ↑(PiTensorProduct.mapL fun i => ContinuousLinearMap.id 𝕜 (E i)) ↑(Continu …
  -/
  ext
  simp only [mapL_coe, ContinuousLinearMap.coe_id, map_id, LinearMap.compMultilinearMap_apply,
    LinearMap.id_coe, id_eq]


@[simp]
theorem mapL_one : mapL (fun (i : ι) ↦ (1 : E i →L[𝕜] E i)) = 1 :=
  mapL_id


theorem mapL_mul (f₁ f₂ : Π i, E i →L[𝕜] E i) :
    mapL (fun i ↦ f₁ i * f₂ i) = mapL f₁ * mapL f₂ :=
  mapL_comp f₁ f₂


/-- Upgrading `PiTensorProduct.mapL` to a `MonoidHom` when `E = E'`. -/
@[simps]
noncomputable def mapLMonoidHom : (Π i, E i →L[𝕜] E i) →* ((⨂[𝕜] i, E i) →L[𝕜] ⨂[𝕜] i, E i) where
  toFun := mapL
  map_one' := mapL_one
  map_mul' := mapL_mul


@[simp]
protected theorem mapL_pow (f : Π i, E i →L[𝕜] E i) (n : ℕ) :
    mapL (f ^ n) = mapL f ^ n := MonoidHom.map_pow mapLMonoidHom _ _

-- We redeclare `ι` here, and later dependent arguments,
-- to avoid the `[Fintype ι]` assumption present throughout the rest of the file.

open Function in
private theorem mapL_add_smul_aux {ι : Type uι}
    {E : ι → Type uE} [(i : ι) → SeminormedAddCommGroup (E i)] [(i : ι) → NormedSpace 𝕜 (E i)]
    {E' : ι → Type u_1} [(i : ι) → SeminormedAddCommGroup (E' i)] [(i : ι) → NormedSpace 𝕜 (E' i)]
    (f : (i : ι) → E i →L[𝕜] E' i)
    [DecidableEq ι] (i : ι) (u : E i →L[𝕜] E' i) :
    (fun j ↦ (update f i u j).toLinearMap) =
      update (fun j ↦ (f j).toLinearMap) i u.toLinearMap := by
  /-
    𝕜 : Type u𝕜
    inst✝⁵ : NontriviallyNormedField 𝕜
    ι : Type uι
    E : ι → Type uE
    inst✝⁴ : (i : ι) → SeminormedAddCommGroup (E i)
    inst✝³ : (i : ι) → NormedSpace 𝕜 (E i)
    E' : ι → Type u_1
    inst✝² : (i : ι) → SeminormedAddCommGroup (E' i)
    inst✝¹ : (i : ι) → NormedSpace 𝕜 (E' i)
    f : (i : ι) → ContinuousLinearMap (RingHom.id 𝕜) (E i) (E' i)
    inst✝ : DecidableEq ι
    i : ι
    u : ContinuousLinearMap (RingHom.id 𝕜) (E i) (E' i)
    ⊢ Eq (fun j => ↑(Function.update f i u j)) (Function.update (fun j => ↑(f j))  …
  -/
  symm
  /-
    𝕜 : Type u𝕜
    inst✝⁵ : NontriviallyNormedField 𝕜
    ι : Type uι
    E : ι → Type uE
    inst✝⁴ : (i : ι) → SeminormedAddCommGroup (E i)
    inst✝³ : (i : ι) → NormedSpace 𝕜 (E i)
    E' : ι → Type u_1
    inst✝² : (i : ι) → SeminormedAddCommGroup (E' i)
    inst✝¹ : (i : ι) → NormedSpace 𝕜 (E' i)
    f : (i : ι) → ContinuousLinearMap (RingHom.id 𝕜) (E i) (E' i)
    inst✝ : DecidableEq ι
    i : ι
    u : ContinuousLinearMap (RingHom.id 𝕜) (E i) (E' i)
    ⊢ Eq (Function.update (fun j => ↑(f j)) i ↑u) fun j => ↑(Function.update f i u …
  -/
  rw [update_eq_iff]
  /-
    𝕜 : Type u𝕜
    inst✝⁵ : NontriviallyNormedField 𝕜
    ι : Type uι
    E : ι → Type uE
    inst✝⁴ : (i : ι) → SeminormedAddCommGroup (E i)
    inst✝³ : (i : ι) → NormedSpace 𝕜 (E i)
    E' : ι → Type u_1
    inst✝² : (i : ι) → SeminormedAddCommGroup (E' i)
    inst✝¹ : (i : ι) → NormedSpace 𝕜 (E' i)
    f : (i : ι) → ContinuousLinearMap (RingHom.id 𝕜) (E i) (E' i)
    inst✝ : DecidableEq ι
    i : ι
    u : ContinuousLinearMap (RingHom.id 𝕜) (E i) (E' i)
    ⊢ And (Eq ↑u ↑(Function.update f i u i)) (∀ (x : ι), Ne x i → Eq ↑(f x) ↑(Func …
  -/
  constructor
    /-
      case left
      𝕜 : Type u𝕜
      inst✝⁵ : NontriviallyNormedField 𝕜
      ι : Type uι
      E : ι → Type uE
      inst✝⁴ : (i : ι) → SeminormedAddCommGroup (E i)
      inst✝³ : (i : ι) → NormedSpace 𝕜 (E i)
      E' : ι → Type u_1
      inst✝² : (i : ι) → SeminormedAddCommGroup (E' i)
      inst✝¹ : (i : ι) → NormedSpace 𝕜 (E' i)
      f : (i : ι) → ContinuousLinearMap (RingHom.id 𝕜) (E i) (E' i)
      inst✝ : DecidableEq ι
      i : ι
      u : ContinuousLinearMap (RingHom.id 𝕜) (E i) (E' i)
      ⊢ Eq ↑u ↑(Function.update f i u i)
    -/
  · simp only [update_self]
    /-
      🎉 no goals
    -/
    /-
      case right
      𝕜 : Type u𝕜
      inst✝⁵ : NontriviallyNormedField 𝕜
      ι : Type uι
      E : ι → Type uE
      inst✝⁴ : (i : ι) → SeminormedAddCommGroup (E i)
      inst✝³ : (i : ι) → NormedSpace 𝕜 (E i)
      E' : ι → Type u_1
      inst✝² : (i : ι) → SeminormedAddCommGroup (E' i)
      inst✝¹ : (i : ι) → NormedSpace 𝕜 (E' i)
      f : (i : ι) → ContinuousLinearMap (RingHom.id 𝕜) (E i) (E' i)
      inst✝ : DecidableEq ι
      i : ι
      u : ContinuousLinearMap (RingHom.id 𝕜) (E i) (E' i)
      ⊢ ∀ (x : ι), Ne x i → Eq ↑(f x) ↑(Function.update f i u x)
    -/
  · exact fun _ h ↦ by simp only [ne_eq, h, not_false_eq_true, update_of_ne]
    /-
      🎉 no goals
    -/


open Function in
protected theorem mapL_add [DecidableEq ι] (i : ι) (u v : E i →L[𝕜] E' i) :
    mapL (update f i (u + v)) = mapL (update f i u) + mapL (update f i v) := by
  /-
    ι : Type uι
    inst✝⁶ : Fintype ι
    𝕜 : Type u𝕜
    inst✝⁵ : NontriviallyNormedField 𝕜
    E : ι → Type uE
    inst✝⁴ : (i : ι) → SeminormedAddCommGroup (E i)
    inst✝³ : (i : ι) → NormedSpace 𝕜 (E i)
    E' : ι → Type u_1
    inst✝² : (i : ι) → SeminormedAddCommGroup (E' i)
    inst✝¹ : (i : ι) → NormedSpace 𝕜 (E' i)
    f : (i : ι) → ContinuousLinearMap (RingHom.id 𝕜) (E i) (E' i)
    inst✝ : DecidableEq ι
    i : ι
    u v : ContinuousLinearMap (RingHom.id 𝕜) (E i) (E' i)
    ⊢ Eq (PiTensorProduct.mapL (Function.update f i (HAdd.hAdd u v))) (HAdd.hAdd ( …
  -/
  ext x
  simp only [mapL_apply, mapL_add_smul_aux, ContinuousLinearMap.coe_add,
    PiTensorProduct.map_update_add, LinearMap.add_apply, ContinuousLinearMap.add_apply]


open Function in
protected theorem mapL_smul [DecidableEq ι] (i : ι) (c : 𝕜) (u : E i →L[𝕜] E' i) :
    mapL (update f i (c • u)) = c • mapL (update f i u) := by
  /-
    ι : Type uι
    inst✝⁶ : Fintype ι
    𝕜 : Type u𝕜
    inst✝⁵ : NontriviallyNormedField 𝕜
    E : ι → Type uE
    inst✝⁴ : (i : ι) → SeminormedAddCommGroup (E i)
    inst✝³ : (i : ι) → NormedSpace 𝕜 (E i)
    E' : ι → Type u_1
    inst✝² : (i : ι) → SeminormedAddCommGroup (E' i)
    inst✝¹ : (i : ι) → NormedSpace 𝕜 (E' i)
    f : (i : ι) → ContinuousLinearMap (RingHom.id 𝕜) (E i) (E' i)
    inst✝ : DecidableEq ι
    i : ι
    c : 𝕜
    u : ContinuousLinearMap (RingHom.id 𝕜) (E i) (E' i)
    ⊢ Eq (PiTensorProduct.mapL (Function.update f i (HSMul.hSMul c u))) (HSMul.hSM …
  -/
  ext x
  simp only [mapL_apply, mapL_add_smul_aux, ContinuousLinearMap.coe_smul,
    PiTensorProduct.map_update_smul, LinearMap.smul_apply, ContinuousLinearMap.coe_smul',
    Pi.smul_apply]


theorem mapL_opNorm : ‖mapL f‖ ≤ ∏ i, ‖f i‖ := by
  /-
    ι : Type uι
    inst✝⁵ : Fintype ι
    𝕜 : Type u𝕜
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : ι → Type uE
    inst✝³ : (i : ι) → SeminormedAddCommGroup (E i)
    inst✝² : (i : ι) → NormedSpace 𝕜 (E i)
    E' : ι → Type u_1
    inst✝¹ : (i : ι) → SeminormedAddCommGroup (E' i)
    inst✝ : (i : ι) → NormedSpace 𝕜 (E' i)
    f : (i : ι) → ContinuousLinearMap (RingHom.id 𝕜) (E i) (E' i)
    ⊢ LE.le (Norm.norm (PiTensorProduct.mapL f)) (Finset.univ.prod fun i => Norm.n …
  -/
  rw [ContinuousLinearMap.opNorm_le_iff (by positivity)]
  /-
    ι : Type uι
    inst✝⁵ : Fintype ι
    𝕜 : Type u𝕜
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : ι → Type uE
    inst✝³ : (i : ι) → SeminormedAddCommGroup (E i)
    inst✝² : (i : ι) → NormedSpace 𝕜 (E i)
    E' : ι → Type u_1
    inst✝¹ : (i : ι) → SeminormedAddCommGroup (E' i)
    inst✝ : (i : ι) → NormedSpace 𝕜 (E' i)
    f : (i : ι) → ContinuousLinearMap (RingHom.id 𝕜) (E i) (E' i)
    ⊢ ∀ (x : PiTensorProduct 𝕜 fun i => E i), LE.le (Norm.norm ((PiTensorProduct.m …
  -/
  intro x
  /-
    ι : Type uι
    inst✝⁵ : Fintype ι
    𝕜 : Type u𝕜
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : ι → Type uE
    inst✝³ : (i : ι) → SeminormedAddCommGroup (E i)
    inst✝² : (i : ι) → NormedSpace 𝕜 (E i)
    E' : ι → Type u_1
    inst✝¹ : (i : ι) → SeminormedAddCommGroup (E' i)
    inst✝ : (i : ι) → NormedSpace 𝕜 (E' i)
    f : (i : ι) → ContinuousLinearMap (RingHom.id 𝕜) (E i) (E' i)
    x : PiTensorProduct 𝕜 fun i => E i
    ⊢ LE.le (Norm.norm ((PiTensorProduct.mapL f) x)) (HMul.hMul (Finset.univ.prod  …
  -/
  rw [mapL, liftIsometry]
  /-
    ι : Type uι
    inst✝⁵ : Fintype ι
    𝕜 : Type u𝕜
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : ι → Type uE
    inst✝³ : (i : ι) → SeminormedAddCommGroup (E i)
    inst✝² : (i : ι) → NormedSpace 𝕜 (E i)
    E' : ι → Type u_1
    inst✝¹ : (i : ι) → SeminormedAddCommGroup (E' i)
    inst✝ : (i : ι) → NormedSpace 𝕜 (E' i)
    f : (i : ι) → ContinuousLinearMap (RingHom.id 𝕜) (E i) (E' i)
    x : PiTensorProduct 𝕜 fun i => E i
    ⊢ LE.le (Norm.norm (({ toLinearEquiv := PiTensorProduct.liftEquiv 𝕜 E (PiTenso …
  -/
  simp only [LinearIsometryEquiv.coe_mk, liftEquiv_apply, LinearMap.mkContinuous_apply]
  refine le_trans (norm_eval_le_injectiveSeminorm _ _)
    (mul_le_mul_of_nonneg_right ?_ (norm_nonneg x))
  /-
    ι : Type uι
    inst✝⁵ : Fintype ι
    𝕜 : Type u𝕜
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : ι → Type uE
    inst✝³ : (i : ι) → SeminormedAddCommGroup (E i)
    inst✝² : (i : ι) → NormedSpace 𝕜 (E i)
    E' : ι → Type u_1
    inst✝¹ : (i : ι) → SeminormedAddCommGroup (E' i)
    inst✝ : (i : ι) → NormedSpace 𝕜 (E' i)
    f : (i : ι) → ContinuousLinearMap (RingHom.id 𝕜) (E i) (E' i)
    x : PiTensorProduct 𝕜 fun i => E i
    ⊢ LE.le (Norm.norm ((PiTensorProduct.tprodL 𝕜).compContinuousLinearMap f)) (Fi …
  -/
  rw [ContinuousMultilinearMap.opNorm_le_iff (Finset.prod_nonneg (fun _ _ ↦ norm_nonneg _))]
  /-
    ι : Type uι
    inst✝⁵ : Fintype ι
    𝕜 : Type u𝕜
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : ι → Type uE
    inst✝³ : (i : ι) → SeminormedAddCommGroup (E i)
    inst✝² : (i : ι) → NormedSpace 𝕜 (E i)
    E' : ι → Type u_1
    inst✝¹ : (i : ι) → SeminormedAddCommGroup (E' i)
    inst✝ : (i : ι) → NormedSpace 𝕜 (E' i)
    f : (i : ι) → ContinuousLinearMap (RingHom.id 𝕜) (E i) (E' i)
    x : PiTensorProduct 𝕜 fun i => E i
    ⊢ ∀ (m : (i : ι) → E i), LE.le (Norm.norm (((PiTensorProduct.tprodL 𝕜).compCon …
  -/
  intro m
  /-
    ι : Type uι
    inst✝⁵ : Fintype ι
    𝕜 : Type u𝕜
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : ι → Type uE
    inst✝³ : (i : ι) → SeminormedAddCommGroup (E i)
    inst✝² : (i : ι) → NormedSpace 𝕜 (E i)
    E' : ι → Type u_1
    inst✝¹ : (i : ι) → SeminormedAddCommGroup (E' i)
    inst✝ : (i : ι) → NormedSpace 𝕜 (E' i)
    f : (i : ι) → ContinuousLinearMap (RingHom.id 𝕜) (E i) (E' i)
    x : PiTensorProduct 𝕜 fun i => E i
    m : (i : ι) → E i
    ⊢ LE.le (Norm.norm (((PiTensorProduct.tprodL 𝕜).compContinuousLinearMap f) m)) …
  -/
  simp only [ContinuousMultilinearMap.compContinuousLinearMap_apply]
  /-
    ι : Type uι
    inst✝⁵ : Fintype ι
    𝕜 : Type u𝕜
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : ι → Type uE
    inst✝³ : (i : ι) → SeminormedAddCommGroup (E i)
    inst✝² : (i : ι) → NormedSpace 𝕜 (E i)
    E' : ι → Type u_1
    inst✝¹ : (i : ι) → SeminormedAddCommGroup (E' i)
    inst✝ : (i : ι) → NormedSpace 𝕜 (E' i)
    f : (i : ι) → ContinuousLinearMap (RingHom.id 𝕜) (E i) (E' i)
    x : PiTensorProduct 𝕜 fun i => E i
    m : (i : ι) → E i
    ⊢ LE.le (Norm.norm ((PiTensorProduct.tprodL 𝕜) fun i => (f i) (m i))) (HMul.hM …
  -/
  refine le_trans (injectiveSeminorm_tprod_le (fun i ↦ (f i) (m i))) ?_
  /-
    ι : Type uι
    inst✝⁵ : Fintype ι
    𝕜 : Type u𝕜
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : ι → Type uE
    inst✝³ : (i : ι) → SeminormedAddCommGroup (E i)
    inst✝² : (i : ι) → NormedSpace 𝕜 (E i)
    E' : ι → Type u_1
    inst✝¹ : (i : ι) → SeminormedAddCommGroup (E' i)
    inst✝ : (i : ι) → NormedSpace 𝕜 (E' i)
    f : (i : ι) → ContinuousLinearMap (RingHom.id 𝕜) (E i) (E' i)
    x : PiTensorProduct 𝕜 fun i => E i
    m : (i : ι) → E i
    ⊢ LE.le (Finset.univ.prod fun i => Norm.norm ((f i) (m i))) (HMul.hMul (Finset …
  -/
  rw [← Finset.prod_mul_distrib]
  /-
    ι : Type uι
    inst✝⁵ : Fintype ι
    𝕜 : Type u𝕜
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : ι → Type uE
    inst✝³ : (i : ι) → SeminormedAddCommGroup (E i)
    inst✝² : (i : ι) → NormedSpace 𝕜 (E i)
    E' : ι → Type u_1
    inst✝¹ : (i : ι) → SeminormedAddCommGroup (E' i)
    inst✝ : (i : ι) → NormedSpace 𝕜 (E' i)
    f : (i : ι) → ContinuousLinearMap (RingHom.id 𝕜) (E i) (E' i)
    x : PiTensorProduct 𝕜 fun i => E i
    m : (i : ι) → E i
    ⊢ LE.le (Finset.univ.prod fun i => Norm.norm ((f i) (m i))) (Finset.univ.prod  …
  -/
  exact Finset.prod_le_prod (fun _ _ ↦ norm_nonneg _) (fun _ _ ↦ ContinuousLinearMap.le_opNorm _ _ )
  /-
    🎉 no goals
  -/


/-- The tensor of a family of linear maps from `Eᵢ` to `E'ᵢ`, as a continuous multilinear map of
the family.
-/
@[simps!]
noncomputable def mapLMultilinear : ContinuousMultilinearMap 𝕜 (fun (i : ι) ↦ E i →L[𝕜] E' i)
    ((⨂[𝕜] i, E i) →L[𝕜] ⨂[𝕜] i, E' i) :=
  MultilinearMap.mkContinuous
  { toFun := mapL
    map_update_smul' := fun _ _ _ _ ↦ PiTensorProduct.mapL_smul _ _ _ _
    map_update_add' := fun _ _ _ _ ↦ PiTensorProduct.mapL_add _ _ _ _ }
                /-
                  ι : Type uι
                  inst✝⁹ : Fintype ι
                  𝕜 : Type u𝕜
                  inst✝⁸ : NontriviallyNormedField 𝕜
                  E : ι → Type uE
                  inst✝⁷ : (i : ι) → SeminormedAddCommGroup (E i)
                  inst✝⁶ : (i : ι) → NormedSpace 𝕜 (E i)
                  F : Type uF
                  inst✝⁵ : SeminormedAddCommGroup F
                  inst✝⁴ : NormedSpace 𝕜 F
                  E' : ι → Type u_1
                  E'' : ι → Type u_2
                  inst✝³ : (i : ι) → SeminormedAddCommGroup (E' i)
                  inst✝² : (i : ι) → NormedSpace 𝕜 (E' i)
                  inst✝¹ : (i : ι) → SeminormedAddCommGroup (E'' i)
                  inst✝ : (i : ι) → NormedSpace 𝕜 (E'' i)
                  g : (i : ι) → ContinuousLinearMap (RingHom.id 𝕜) (E' i) (E'' i)
                  f✝ f : (i : ι) → ContinuousLinearMap (RingHom.id 𝕜) (E i) (E' i)
                  ⊢ LE.le (Norm.norm ({ toFun := PiTensorProduct.mapL, map_update_add' := ⋯, map …
                -/
  1 (fun f ↦ by rw [one_mul]; exact mapL_opNorm f)
                              /-
                                🎉 no goals
                              -/


theorem mapLMultilinear_opNorm : ‖mapLMultilinear 𝕜 E E'‖ ≤ 1 :=
  MultilinearMap.mkContinuous_norm_le _ zero_le_one _


