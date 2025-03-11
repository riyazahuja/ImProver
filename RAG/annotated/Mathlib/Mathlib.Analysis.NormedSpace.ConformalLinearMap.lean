/-- A continuous linear map `f'` is said to be conformal if it's
    a nonzero multiple of a linear isometry. -/
def IsConformalMap {R : Type*} {X Y : Type*} [NormedField R] [SeminormedAddCommGroup X]
    [SeminormedAddCommGroup Y] [NormedSpace R X] [NormedSpace R Y] (f' : X →L[R] Y) :=
  ∃ c ≠ (0 : R), ∃ li : X →ₗᵢ[R] Y, f' = c • li.toContinuousLinearMap


theorem isConformalMap_id : IsConformalMap (id R M) :=
                          /-
                            R : Type u_1
                            M : Type u_2
                            inst✝² : NormedField R
                            inst✝¹ : SeminormedAddCommGroup M
                            inst✝ : NormedSpace R M
                            ⊢ Eq (ContinuousLinearMap.id R M) (HSMul.hSMul 1 LinearIsometry.id.toContinuou …
                          -/
  ⟨1, one_ne_zero, id, by simp⟩
                          /-
                            🎉 no goals
                          -/


theorem IsConformalMap.smul (hf : IsConformalMap f) {c : R} (hc : c ≠ 0) :
    IsConformalMap (c • f) := by
  /-
    R : Type u_1
    M : Type u_2
    N : Type u_3
    inst✝⁴ : NormedField R
    inst✝³ : SeminormedAddCommGroup M
    inst✝² : SeminormedAddCommGroup N
    inst✝¹ : NormedSpace R M
    inst✝ : NormedSpace R N
    f : ContinuousLinearMap (RingHom.id R) M N
    hf : IsConformalMap f
    c : R
    hc : Ne c 0
    ⊢ IsConformalMap (HSMul.hSMul c f)
  -/
  rcases hf with ⟨c', hc', li, rfl⟩
  /-
    case intro.intro.intro
    R : Type u_1
    M : Type u_2
    N : Type u_3
    inst✝⁴ : NormedField R
    inst✝³ : SeminormedAddCommGroup M
    inst✝² : SeminormedAddCommGroup N
    inst✝¹ : NormedSpace R M
    inst✝ : NormedSpace R N
    c : R
    hc : Ne c 0
    c' : R
    hc' : Ne c' 0
    li : LinearIsometry (RingHom.id R) M N
    ⊢ IsConformalMap (HSMul.hSMul c (HSMul.hSMul c' li.toContinuousLinearMap))
  -/
  exact ⟨c * c', mul_ne_zero hc hc', li, smul_smul _ _ _⟩
  /-
    🎉 no goals
  -/


theorem isConformalMap_const_smul (hc : c ≠ 0) : IsConformalMap (c • id R M) :=
  isConformalMap_id.smul hc


protected theorem LinearIsometry.isConformalMap (f' : M →ₗᵢ[R] N) :
    IsConformalMap f'.toContinuousLinearMap :=
  ⟨1, one_ne_zero, f', (one_smul _ _).symm⟩


@[nontriviality]
theorem isConformalMap_of_subsingleton [Subsingleton M] (f' : M →L[R] N) : IsConformalMap f' :=
                                   /-
                                     R : Type u_1
                                     M : Type u_2
                                     N : Type u_3
                                     inst✝⁵ : NormedField R
                                     inst✝⁴ : SeminormedAddCommGroup M
                                     inst✝³ : SeminormedAddCommGroup N
                                     inst✝² : NormedSpace R M
                                     inst✝¹ : NormedSpace R N
                                     inst✝ : Subsingleton M
                                     f' : ContinuousLinearMap (RingHom.id R) M N
                                     x : M
                                     ⊢ Eq (Norm.norm (0 x)) (Norm.norm x)
                                   -/
  ⟨1, one_ne_zero, ⟨0, fun x => by simp [Subsingleton.elim x 0]⟩, Subsingleton.elim _ _⟩
                                   /-
                                     🎉 no goals
                                   -/


theorem comp (hg : IsConformalMap g) (hf : IsConformalMap f) : IsConformalMap (g.comp f) := by
  /-
    R : Type u_1
    M : Type u_2
    N : Type u_3
    G : Type u_4
    inst✝⁶ : NormedField R
    inst✝⁵ : SeminormedAddCommGroup M
    inst✝⁴ : SeminormedAddCommGroup N
    inst✝³ : SeminormedAddCommGroup G
    inst✝² : NormedSpace R M
    inst✝¹ : NormedSpace R N
    inst✝ : NormedSpace R G
    f : ContinuousLinearMap (RingHom.id R) M N
    g : ContinuousLinearMap (RingHom.id R) N G
    hg : IsConformalMap g
    hf : IsConformalMap f
    ⊢ IsConformalMap (g.comp f)
  -/
  rcases hf with ⟨cf, hcf, lif, rfl⟩
  /-
    case intro.intro.intro
    R : Type u_1
    M : Type u_2
    N : Type u_3
    G : Type u_4
    inst✝⁶ : NormedField R
    inst✝⁵ : SeminormedAddCommGroup M
    inst✝⁴ : SeminormedAddCommGroup N
    inst✝³ : SeminormedAddCommGroup G
    inst✝² : NormedSpace R M
    inst✝¹ : NormedSpace R N
    inst✝ : NormedSpace R G
    g : ContinuousLinearMap (RingHom.id R) N G
    hg : IsConformalMap g
    cf : R
    hcf : Ne cf 0
    lif : LinearIsometry (RingHom.id R) M N
    ⊢ IsConformalMap (g.comp (HSMul.hSMul cf lif.toContinuousLinearMap))
  -/
  rcases hg with ⟨cg, hcg, lig, rfl⟩
  /-
    case intro.intro.intro.intro.intro.intro
    R : Type u_1
    M : Type u_2
    N : Type u_3
    G : Type u_4
    inst✝⁶ : NormedField R
    inst✝⁵ : SeminormedAddCommGroup M
    inst✝⁴ : SeminormedAddCommGroup N
    inst✝³ : SeminormedAddCommGroup G
    inst✝² : NormedSpace R M
    inst✝¹ : NormedSpace R N
    inst✝ : NormedSpace R G
    cf : R
    hcf : Ne cf 0
    lif : LinearIsometry (RingHom.id R) M N
    cg : R
    hcg : Ne cg 0
    lig : LinearIsometry (RingHom.id R) N G
    ⊢ IsConformalMap ((HSMul.hSMul cg lig.toContinuousLinearMap).comp (HSMul.hSMul …
  -/
  refine ⟨cg * cf, mul_ne_zero hcg hcf, lig.comp lif, ?_⟩
  /-
    case intro.intro.intro.intro.intro.intro
    R : Type u_1
    M : Type u_2
    N : Type u_3
    G : Type u_4
    inst✝⁶ : NormedField R
    inst✝⁵ : SeminormedAddCommGroup M
    inst✝⁴ : SeminormedAddCommGroup N
    inst✝³ : SeminormedAddCommGroup G
    inst✝² : NormedSpace R M
    inst✝¹ : NormedSpace R N
    inst✝ : NormedSpace R G
    cf : R
    hcf : Ne cf 0
    lif : LinearIsometry (RingHom.id R) M N
    cg : R
    hcg : Ne cg 0
    lig : LinearIsometry (RingHom.id R) N G
    ⊢ Eq ((HSMul.hSMul cg lig.toContinuousLinearMap).comp (HSMul.hSMul cf lif.toCo …
  -/
  rw [smul_comp, comp_smul, mul_smul]
  /-
    case intro.intro.intro.intro.intro.intro
    R : Type u_1
    M : Type u_2
    N : Type u_3
    G : Type u_4
    inst✝⁶ : NormedField R
    inst✝⁵ : SeminormedAddCommGroup M
    inst✝⁴ : SeminormedAddCommGroup N
    inst✝³ : SeminormedAddCommGroup G
    inst✝² : NormedSpace R M
    inst✝¹ : NormedSpace R N
    inst✝ : NormedSpace R G
    cf : R
    hcf : Ne cf 0
    lif : LinearIsometry (RingHom.id R) M N
    cg : R
    hcg : Ne cg 0
    lig : LinearIsometry (RingHom.id R) N G
    ⊢ Eq (HSMul.hSMul cg (HSMul.hSMul cf (lig.toContinuousLinearMap.comp lif.toCon …
  -/
  rfl
  /-
    🎉 no goals
  -/


protected theorem injective {f : M' →L[R] N} (h : IsConformalMap f) : Function.Injective f := by
  /-
    R : Type u_1
    N : Type u_3
    M' : Type u_5
    inst✝⁴ : NormedField R
    inst✝³ : SeminormedAddCommGroup N
    inst✝² : NormedSpace R N
    inst✝¹ : NormedAddCommGroup M'
    inst✝ : NormedSpace R M'
    f : ContinuousLinearMap (RingHom.id R) M' N
    h : IsConformalMap f
    ⊢ Function.Injective ⇑f
  -/
  rcases h with ⟨c, hc, li, rfl⟩
  /-
    case intro.intro.intro
    R : Type u_1
    N : Type u_3
    M' : Type u_5
    inst✝⁴ : NormedField R
    inst✝³ : SeminormedAddCommGroup N
    inst✝² : NormedSpace R N
    inst✝¹ : NormedAddCommGroup M'
    inst✝ : NormedSpace R M'
    c : R
    hc : Ne c 0
    li : LinearIsometry (RingHom.id R) M' N
    ⊢ Function.Injective ⇑(HSMul.hSMul c li.toContinuousLinearMap)
  -/
  exact (smul_right_injective _ hc).comp li.injective
  /-
    🎉 no goals
  -/


theorem ne_zero [Nontrivial M'] {f' : M' →L[R] N} (hf' : IsConformalMap f') : f' ≠ 0 := by
  /-
    R : Type u_1
    N : Type u_3
    M' : Type u_5
    inst✝⁵ : NormedField R
    inst✝⁴ : SeminormedAddCommGroup N
    inst✝³ : NormedSpace R N
    inst✝² : NormedAddCommGroup M'
    inst✝¹ : NormedSpace R M'
    inst✝ : Nontrivial M'
    f' : ContinuousLinearMap (RingHom.id R) M' N
    hf' : IsConformalMap f'
    ⊢ Ne f' 0
  -/
  rintro rfl
  /-
    R : Type u_1
    N : Type u_3
    M' : Type u_5
    inst✝⁵ : NormedField R
    inst✝⁴ : SeminormedAddCommGroup N
    inst✝³ : NormedSpace R N
    inst✝² : NormedAddCommGroup M'
    inst✝¹ : NormedSpace R M'
    inst✝ : Nontrivial M'
    hf' : IsConformalMap 0
    ⊢ False
  -/
  rcases exists_ne (0 : M') with ⟨a, ha⟩
  /-
    case intro
    R : Type u_1
    N : Type u_3
    M' : Type u_5
    inst✝⁵ : NormedField R
    inst✝⁴ : SeminormedAddCommGroup N
    inst✝³ : NormedSpace R N
    inst✝² : NormedAddCommGroup M'
    inst✝¹ : NormedSpace R M'
    inst✝ : Nontrivial M'
    hf' : IsConformalMap 0
    a : M'
    ha : Ne a 0
    ⊢ False
  -/
  exact ha (hf'.injective rfl)
  /-
    🎉 no goals
  -/


