/-- The operator norm of the first projection `E × F → E` is at most 1. (It is 0 if `E` is zero, so
the inequality cannot be improved without further assumptions.) -/
lemma norm_fst_le : ‖fst 𝕜 E F‖ ≤ 1 :=
                                                 /-
                                                   𝕜 : Type u_1
                                                   E : Type u_2
                                                   F : Type u_3
                                                   inst✝⁴ : NontriviallyNormedField 𝕜
                                                   inst✝³ : SeminormedAddCommGroup E
                                                   inst✝² : SeminormedAddCommGroup F
                                                   inst✝¹ : NormedSpace 𝕜 E
                                                   inst✝ : NormedSpace 𝕜 F
                                                   x✝ : Prod E F
                                                   e : E
                                                   f : F
                                                   ⊢ LE.le (Norm.norm ((ContinuousLinearMap.fst 𝕜 E F) { fst := e, snd := f })) ( …
                                                 -/
  opNorm_le_bound _ zero_le_one (fun ⟨e, f⟩ ↦ by simpa only [one_mul] using le_max_left ‖e‖ ‖f‖)
                                                 /-
                                                   🎉 no goals
                                                 -/


/-- The operator norm of the second projection `E × F → F` is at most 1. (It is 0 if `F` is zero, so
the inequality cannot be improved without further assumptions.) -/
lemma norm_snd_le : ‖snd 𝕜 E F‖ ≤ 1 :=
                                                 /-
                                                   𝕜 : Type u_1
                                                   E : Type u_2
                                                   F : Type u_3
                                                   inst✝⁴ : NontriviallyNormedField 𝕜
                                                   inst✝³ : SeminormedAddCommGroup E
                                                   inst✝² : SeminormedAddCommGroup F
                                                   inst✝¹ : NormedSpace 𝕜 E
                                                   inst✝ : NormedSpace 𝕜 F
                                                   x✝ : Prod E F
                                                   e : E
                                                   f : F
                                                   ⊢ LE.le (Norm.norm ((ContinuousLinearMap.snd 𝕜 E F) { fst := e, snd := f })) ( …
                                                 -/
  opNorm_le_bound _ zero_le_one (fun ⟨e, f⟩ ↦ by simpa only [one_mul] using le_max_right ‖e‖ ‖f‖)
                                                 /-
                                                   🎉 no goals
                                                 -/


@[simp]
theorem opNorm_prod (f : E →L[𝕜] F) (g : E →L[𝕜] G) : ‖f.prod g‖ = ‖(f, g)‖ :=
  le_antisymm
      (opNorm_le_bound _ (norm_nonneg _) fun x => by
        simpa only [prod_apply, Prod.norm_def, max_mul_of_nonneg, norm_nonneg] using
          max_le_max (le_opNorm f x) (le_opNorm g x)) <|
    max_le
      (opNorm_le_bound _ (norm_nonneg _) fun x =>
        (le_max_left _ _).trans ((f.prod g).le_opNorm x))
      (opNorm_le_bound _ (norm_nonneg _) fun x =>
        (le_max_right _ _).trans ((f.prod g).le_opNorm x))


@[deprecated (since := "2024-02-02")] alias op_norm_prod := opNorm_prod


@[simp]
theorem opNNNorm_prod (f : E →L[𝕜] F) (g : E →L[𝕜] G) : ‖f.prod g‖₊ = ‖(f, g)‖₊ :=
  Subtype.ext <| opNorm_prod f g


@[deprecated (since := "2024-02-02")] alias op_nnnorm_prod := opNNNorm_prod


/-- `ContinuousLinearMap.prod` as a `LinearIsometryEquiv`. -/
def prodₗᵢ (R : Type*) [Semiring R] [Module R F] [Module R G] [ContinuousConstSMul R F]
    [ContinuousConstSMul R G] [SMulCommClass 𝕜 R F] [SMulCommClass 𝕜 R G] :
    (E →L[𝕜] F) × (E →L[𝕜] G) ≃ₗᵢ[R] E →L[𝕜] F × G :=
  ⟨prodₗ R, fun ⟨f, g⟩ => opNorm_prod f g⟩


/-- `ContinuousLinearMap.prodMap` as a continuous linear map. -/
def prodMapL : (M₁ →L[𝕜] M₂) × (M₃ →L[𝕜] M₄) →L[𝕜] M₁ × M₃ →L[𝕜] M₂ × M₄ :=
  ContinuousLinearMap.copy
    (have Φ₁ : (M₁ →L[𝕜] M₂) →L[𝕜] M₁ →L[𝕜] M₂ × M₄ :=
      ContinuousLinearMap.compL 𝕜 M₁ M₂ (M₂ × M₄) (ContinuousLinearMap.inl 𝕜 M₂ M₄)
    have Φ₂ : (M₃ →L[𝕜] M₄) →L[𝕜] M₃ →L[𝕜] M₂ × M₄ :=
      ContinuousLinearMap.compL 𝕜 M₃ M₄ (M₂ × M₄) (ContinuousLinearMap.inr 𝕜 M₂ M₄)
    have Φ₁' :=
      (ContinuousLinearMap.compL 𝕜 (M₁ × M₃) M₁ (M₂ × M₄)).flip (ContinuousLinearMap.fst 𝕜 M₁ M₃)
    have Φ₂' :=
      (ContinuousLinearMap.compL 𝕜 (M₁ × M₃) M₃ (M₂ × M₄)).flip (ContinuousLinearMap.snd 𝕜 M₁ M₃)
    have Ψ₁ : (M₁ →L[𝕜] M₂) × (M₃ →L[𝕜] M₄) →L[𝕜] M₁ →L[𝕜] M₂ :=
      ContinuousLinearMap.fst 𝕜 (M₁ →L[𝕜] M₂) (M₃ →L[𝕜] M₄)
    have Ψ₂ : (M₁ →L[𝕜] M₂) × (M₃ →L[𝕜] M₄) →L[𝕜] M₃ →L[𝕜] M₄ :=
      ContinuousLinearMap.snd 𝕜 (M₁ →L[𝕜] M₂) (M₃ →L[𝕜] M₄)
    Φ₁' ∘L Φ₁ ∘L Ψ₁ + Φ₂' ∘L Φ₂ ∘L Ψ₂)
    (fun p : (M₁ →L[𝕜] M₂) × (M₃ →L[𝕜] M₄) => p.1.prodMap p.2) (by
      /-
        𝕜 : Type u_1
        E : Type u_2
        F : Type u_3
        G : Type u_4
        inst✝¹⁴ : NontriviallyNormedField 𝕜
        inst✝¹³ : SeminormedAddCommGroup E
        inst✝¹² : SeminormedAddCommGroup F
        inst✝¹¹ : SeminormedAddCommGroup G
        inst✝¹⁰ : NormedSpace 𝕜 E
        inst✝⁹ : NormedSpace 𝕜 F
        inst✝⁸ : NormedSpace 𝕜 G
        M₁ : Type u_5
        M₂ : Type u_6
        M₃ : Type u_7
        M₄ : Type u_8
        inst✝⁷ : SeminormedAddCommGroup M₁
        inst✝⁶ : NormedSpace 𝕜 M₁
        inst✝⁵ : SeminormedAddCommGroup M₂
        inst✝⁴ : NormedSpace 𝕜 M₂
        inst✝³ : SeminormedAddCommGroup M₃
        inst✝² : NormedSpace 𝕜 M₃
        inst✝¹ : SeminormedAddCommGroup M₄
        inst✝ : NormedSpace 𝕜 M₄
        ⊢ Eq (fun p => p.1.prodMap p.2) ⇑(letFun ((ContinuousLinearMap.compL 𝕜 M₁ M₂ ( …
      -/
      apply funext
      /-
        case h
        𝕜 : Type u_1
        E : Type u_2
        F : Type u_3
        G : Type u_4
        inst✝¹⁴ : NontriviallyNormedField 𝕜
        inst✝¹³ : SeminormedAddCommGroup E
        inst✝¹² : SeminormedAddCommGroup F
        inst✝¹¹ : SeminormedAddCommGroup G
        inst✝¹⁰ : NormedSpace 𝕜 E
        inst✝⁹ : NormedSpace 𝕜 F
        inst✝⁸ : NormedSpace 𝕜 G
        M₁ : Type u_5
        M₂ : Type u_6
        M₃ : Type u_7
        M₄ : Type u_8
        inst✝⁷ : SeminormedAddCommGroup M₁
        inst✝⁶ : NormedSpace 𝕜 M₁
        inst✝⁵ : SeminormedAddCommGroup M₂
        inst✝⁴ : NormedSpace 𝕜 M₂
        inst✝³ : SeminormedAddCommGroup M₃
        inst✝² : NormedSpace 𝕜 M₃
        inst✝¹ : SeminormedAddCommGroup M₄
        inst✝ : NormedSpace 𝕜 M₄
        ⊢ ∀ (x : Prod (ContinuousLinearMap (RingHom.id 𝕜) M₁ M₂) (ContinuousLinearMap  …
      -/
      rintro ⟨φ, ψ⟩
      /-
        case h.mk
        𝕜 : Type u_1
        E : Type u_2
        F : Type u_3
        G : Type u_4
        inst✝¹⁴ : NontriviallyNormedField 𝕜
        inst✝¹³ : SeminormedAddCommGroup E
        inst✝¹² : SeminormedAddCommGroup F
        inst✝¹¹ : SeminormedAddCommGroup G
        inst✝¹⁰ : NormedSpace 𝕜 E
        inst✝⁹ : NormedSpace 𝕜 F
        inst✝⁸ : NormedSpace 𝕜 G
        M₁ : Type u_5
        M₂ : Type u_6
        M₃ : Type u_7
        M₄ : Type u_8
        inst✝⁷ : SeminormedAddCommGroup M₁
        inst✝⁶ : NormedSpace 𝕜 M₁
        inst✝⁵ : SeminormedAddCommGroup M₂
        inst✝⁴ : NormedSpace 𝕜 M₂
        inst✝³ : SeminormedAddCommGroup M₃
        inst✝² : NormedSpace 𝕜 M₃
        inst✝¹ : SeminormedAddCommGroup M₄
        inst✝ : NormedSpace 𝕜 M₄
        φ : ContinuousLinearMap (RingHom.id 𝕜) M₁ M₂
        ψ : ContinuousLinearMap (RingHom.id 𝕜) M₃ M₄
        ⊢ Eq ({ fst := φ, snd := ψ }.1.prodMap { fst := φ, snd := ψ }.2) ((letFun ((Co …
      -/
      refine ContinuousLinearMap.ext fun ⟨x₁, x₂⟩ => ?_
      /-
        case h.mk
        𝕜 : Type u_1
        E : Type u_2
        F : Type u_3
        G : Type u_4
        inst✝¹⁴ : NontriviallyNormedField 𝕜
        inst✝¹³ : SeminormedAddCommGroup E
        inst✝¹² : SeminormedAddCommGroup F
        inst✝¹¹ : SeminormedAddCommGroup G
        inst✝¹⁰ : NormedSpace 𝕜 E
        inst✝⁹ : NormedSpace 𝕜 F
        inst✝⁸ : NormedSpace 𝕜 G
        M₁ : Type u_5
        M₂ : Type u_6
        M₃ : Type u_7
        M₄ : Type u_8
        inst✝⁷ : SeminormedAddCommGroup M₁
        inst✝⁶ : NormedSpace 𝕜 M₁
        inst✝⁵ : SeminormedAddCommGroup M₂
        inst✝⁴ : NormedSpace 𝕜 M₂
        inst✝³ : SeminormedAddCommGroup M₃
        inst✝² : NormedSpace 𝕜 M₃
        inst✝¹ : SeminormedAddCommGroup M₄
        inst✝ : NormedSpace 𝕜 M₄
        φ : ContinuousLinearMap (RingHom.id 𝕜) M₁ M₂
        ψ : ContinuousLinearMap (RingHom.id 𝕜) M₃ M₄
        x✝ : Prod M₁ M₃
        x₁ : M₁
        x₂ : M₃
        ⊢ Eq (({ fst := φ, snd := ψ }.1.prodMap { fst := φ, snd := ψ }.2) { fst := x₁, …
      -/
      dsimp
      /-
        case h.mk
        𝕜 : Type u_1
        E : Type u_2
        F : Type u_3
        G : Type u_4
        inst✝¹⁴ : NontriviallyNormedField 𝕜
        inst✝¹³ : SeminormedAddCommGroup E
        inst✝¹² : SeminormedAddCommGroup F
        inst✝¹¹ : SeminormedAddCommGroup G
        inst✝¹⁰ : NormedSpace 𝕜 E
        inst✝⁹ : NormedSpace 𝕜 F
        inst✝⁸ : NormedSpace 𝕜 G
        M₁ : Type u_5
        M₂ : Type u_6
        M₃ : Type u_7
        M₄ : Type u_8
        inst✝⁷ : SeminormedAddCommGroup M₁
        inst✝⁶ : NormedSpace 𝕜 M₁
        inst✝⁵ : SeminormedAddCommGroup M₂
        inst✝⁴ : NormedSpace 𝕜 M₂
        inst✝³ : SeminormedAddCommGroup M₃
        inst✝² : NormedSpace 𝕜 M₃
        inst✝¹ : SeminormedAddCommGroup M₄
        inst✝ : NormedSpace 𝕜 M₄
        φ : ContinuousLinearMap (RingHom.id 𝕜) M₁ M₂
        ψ : ContinuousLinearMap (RingHom.id 𝕜) M₃ M₄
        x✝ : Prod M₁ M₃
        x₁ : M₁
        x₂ : M₃
        ⊢ Eq { fst := φ x₁, snd := ψ x₂ } { fst := HAdd.hAdd (φ x₁) 0, snd := HAdd.hAd …
      -/
      simp)
      /-
        🎉 no goals
      -/


@[simp]
theorem prodMapL_apply (p : (M₁ →L[𝕜] M₂) × (M₃ →L[𝕜] M₄)) :
    ContinuousLinearMap.prodMapL 𝕜 M₁ M₂ M₃ M₄ p = p.1.prodMap p.2 :=
  rfl


theorem _root_.Continuous.prod_mapL {f : X → M₁ →L[𝕜] M₂} {g : X → M₃ →L[𝕜] M₄} (hf : Continuous f)
    (hg : Continuous g) : Continuous fun x => (f x).prodMap (g x) :=
  (prodMapL 𝕜 M₁ M₂ M₃ M₄).continuous.comp (hf.prod_mk hg)


theorem _root_.Continuous.prod_map_equivL {f : X → M₁ ≃L[𝕜] M₂} {g : X → M₃ ≃L[𝕜] M₄}
    (hf : Continuous fun x => (f x : M₁ →L[𝕜] M₂)) (hg : Continuous fun x => (g x : M₃ →L[𝕜] M₄)) :
    Continuous fun x => ((f x).prod (g x) : M₁ × M₃ →L[𝕜] M₂ × M₄) :=
  (prodMapL 𝕜 M₁ M₂ M₃ M₄).continuous.comp (hf.prod_mk hg)


theorem _root_.ContinuousOn.prod_mapL {f : X → M₁ →L[𝕜] M₂} {g : X → M₃ →L[𝕜] M₄} {s : Set X}
    (hf : ContinuousOn f s) (hg : ContinuousOn g s) :
    ContinuousOn (fun x => (f x).prodMap (g x)) s :=
  ((prodMapL 𝕜 M₁ M₂ M₃ M₄).continuous.comp_continuousOn (hf.prod hg) : _)


theorem _root_.ContinuousOn.prod_map_equivL {f : X → M₁ ≃L[𝕜] M₂} {g : X → M₃ ≃L[𝕜] M₄} {s : Set X}
    (hf : ContinuousOn (fun x => (f x : M₁ →L[𝕜] M₂)) s)
    (hg : ContinuousOn (fun x => (g x : M₃ →L[𝕜] M₄)) s) :
    ContinuousOn (fun x => ((f x).prod (g x) : M₁ × M₃ →L[𝕜] M₂ × M₄)) s :=
  (prodMapL 𝕜 M₁ M₂ M₃ M₄).continuous.comp_continuousOn (hf.prod hg)


/-- The operator norm of the first projection `E × F → E` is exactly 1 if `E` is nontrivial. -/
@[simp] lemma norm_fst [NormedAddCommGroup E] [NormedSpace 𝕜 E]
    [SeminormedAddCommGroup F] [NormedSpace 𝕜 F] [Nontrivial E] :
    ‖fst 𝕜 E F‖ = 1 := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝⁵ : NontriviallyNormedField 𝕜
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace 𝕜 E
    inst✝² : SeminormedAddCommGroup F
    inst✝¹ : NormedSpace 𝕜 F
    inst✝ : Nontrivial E
    ⊢ Eq (Norm.norm (ContinuousLinearMap.fst 𝕜 E F)) 1
  -/
  refine le_antisymm (norm_fst_le ..) ?_
  /-
    𝕜 : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝⁵ : NontriviallyNormedField 𝕜
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace 𝕜 E
    inst✝² : SeminormedAddCommGroup F
    inst✝¹ : NormedSpace 𝕜 F
    inst✝ : Nontrivial E
    ⊢ LE.le 1 (Norm.norm (ContinuousLinearMap.fst 𝕜 E F))
  -/
  let ⟨e, he⟩ := exists_ne (0 : E)
  /-
    𝕜 : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝⁵ : NontriviallyNormedField 𝕜
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace 𝕜 E
    inst✝² : SeminormedAddCommGroup F
    inst✝¹ : NormedSpace 𝕜 F
    inst✝ : Nontrivial E
    e : E
    he : Ne e 0
    ⊢ LE.le 1 (Norm.norm (ContinuousLinearMap.fst 𝕜 E F))
  -/
  have : ‖e‖ ≤ _ * max ‖e‖ ‖(0 : F)‖ := (fst 𝕜 E F).le_opNorm (e, 0)
  /-
    𝕜 : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝⁵ : NontriviallyNormedField 𝕜
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace 𝕜 E
    inst✝² : SeminormedAddCommGroup F
    inst✝¹ : NormedSpace 𝕜 F
    inst✝ : Nontrivial E
    e : E
    he : Ne e 0
    this : LE.le (Norm.norm e) (HMul.hMul (Norm.norm (ContinuousLinearMap.fst 𝕜 E  …
    ⊢ LE.le 1 (Norm.norm (ContinuousLinearMap.fst 𝕜 E F))
  -/
  rw [norm_zero, max_eq_left (norm_nonneg e)] at this
  /-
    𝕜 : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝⁵ : NontriviallyNormedField 𝕜
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace 𝕜 E
    inst✝² : SeminormedAddCommGroup F
    inst✝¹ : NormedSpace 𝕜 F
    inst✝ : Nontrivial E
    e : E
    he : Ne e 0
    this : LE.le (Norm.norm e) (HMul.hMul (Norm.norm (ContinuousLinearMap.fst 𝕜 E  …
    ⊢ LE.le 1 (Norm.norm (ContinuousLinearMap.fst 𝕜 E F))
  -/
  rwa [← mul_le_mul_iff_of_pos_right (norm_pos_iff.mpr he), one_mul]
  /-
    🎉 no goals
  -/


/-- The operator norm of the second projection `E × F → F` is exactly 1 if `F` is nontrivial. -/
@[simp] lemma norm_snd [SeminormedAddCommGroup E] [NormedSpace 𝕜 E]
    [NormedAddCommGroup F] [NormedSpace 𝕜 F] [Nontrivial F]  :
    ‖snd 𝕜 E F‖ = 1 := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝⁵ : NontriviallyNormedField 𝕜
    inst✝⁴ : SeminormedAddCommGroup E
    inst✝³ : NormedSpace 𝕜 E
    inst✝² : NormedAddCommGroup F
    inst✝¹ : NormedSpace 𝕜 F
    inst✝ : Nontrivial F
    ⊢ Eq (Norm.norm (ContinuousLinearMap.snd 𝕜 E F)) 1
  -/
  refine le_antisymm (norm_snd_le ..) ?_
  /-
    𝕜 : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝⁵ : NontriviallyNormedField 𝕜
    inst✝⁴ : SeminormedAddCommGroup E
    inst✝³ : NormedSpace 𝕜 E
    inst✝² : NormedAddCommGroup F
    inst✝¹ : NormedSpace 𝕜 F
    inst✝ : Nontrivial F
    ⊢ LE.le 1 (Norm.norm (ContinuousLinearMap.snd 𝕜 E F))
  -/
  let ⟨f, hf⟩ := exists_ne (0 : F)
  /-
    𝕜 : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝⁵ : NontriviallyNormedField 𝕜
    inst✝⁴ : SeminormedAddCommGroup E
    inst✝³ : NormedSpace 𝕜 E
    inst✝² : NormedAddCommGroup F
    inst✝¹ : NormedSpace 𝕜 F
    inst✝ : Nontrivial F
    f : F
    hf : Ne f 0
    ⊢ LE.le 1 (Norm.norm (ContinuousLinearMap.snd 𝕜 E F))
  -/
  have : ‖f‖ ≤ _ * max ‖(0 : E)‖ ‖f‖ := (snd 𝕜 E F).le_opNorm (0, f)
  /-
    𝕜 : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝⁵ : NontriviallyNormedField 𝕜
    inst✝⁴ : SeminormedAddCommGroup E
    inst✝³ : NormedSpace 𝕜 E
    inst✝² : NormedAddCommGroup F
    inst✝¹ : NormedSpace 𝕜 F
    inst✝ : Nontrivial F
    f : F
    hf : Ne f 0
    this : LE.le (Norm.norm f) (HMul.hMul (Norm.norm (ContinuousLinearMap.snd 𝕜 E  …
    ⊢ LE.le 1 (Norm.norm (ContinuousLinearMap.snd 𝕜 E F))
  -/
  rw [norm_zero, max_eq_right (norm_nonneg f)] at this
  /-
    𝕜 : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝⁵ : NontriviallyNormedField 𝕜
    inst✝⁴ : SeminormedAddCommGroup E
    inst✝³ : NormedSpace 𝕜 E
    inst✝² : NormedAddCommGroup F
    inst✝¹ : NormedSpace 𝕜 F
    inst✝ : Nontrivial F
    f : F
    hf : Ne f 0
    this : LE.le (Norm.norm f) (HMul.hMul (Norm.norm (ContinuousLinearMap.snd 𝕜 E  …
    ⊢ LE.le 1 (Norm.norm (ContinuousLinearMap.snd 𝕜 E F))
  -/
  rwa [← mul_le_mul_iff_of_pos_right (norm_pos_iff.mpr hf), one_mul]
  /-
    🎉 no goals
  -/


