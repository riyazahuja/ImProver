theorem opNorm_ext [RingHomIsometric σ₁₃] (f : E →SL[σ₁₂] F) (g : E →SL[σ₁₃] G)
    (h : ∀ x, ‖f x‖ = ‖g x‖) : ‖f‖ = ‖g‖ :=
  opNorm_eq_of_bounds (norm_nonneg _)
    (fun x => by
      /-
        𝕜 : Type u_1
        𝕜₂ : Type u_2
        𝕜₃ : Type u_3
        E : Type u_4
        F : Type u_6
        G : Type u_8
        inst✝⁹ : SeminormedAddCommGroup E
        inst✝⁸ : SeminormedAddCommGroup F
        inst✝⁷ : SeminormedAddCommGroup G
        inst✝⁶ : NontriviallyNormedField 𝕜
        inst✝⁵ : NontriviallyNormedField 𝕜₂
        inst✝⁴ : NontriviallyNormedField 𝕜₃
        inst✝³ : NormedSpace 𝕜 E
        inst✝² : NormedSpace 𝕜₂ F
        inst✝¹ : NormedSpace 𝕜₃ G
        σ₁₂ : RingHom 𝕜 𝕜₂
        σ₁₃ : RingHom 𝕜 𝕜₃
        inst✝ : RingHomIsometric σ₁₃
        f : ContinuousLinearMap σ₁₂ E F
        g : ContinuousLinearMap σ₁₃ E G
        h : ∀ (x : E), Eq (Norm.norm (f x)) (Norm.norm (g x))
        x : E
        ⊢ LE.le (Norm.norm (f x)) (HMul.hMul (Norm.norm g) (Norm.norm x))
      -/
      rw [h x]
      /-
        𝕜 : Type u_1
        𝕜₂ : Type u_2
        𝕜₃ : Type u_3
        E : Type u_4
        F : Type u_6
        G : Type u_8
        inst✝⁹ : SeminormedAddCommGroup E
        inst✝⁸ : SeminormedAddCommGroup F
        inst✝⁷ : SeminormedAddCommGroup G
        inst✝⁶ : NontriviallyNormedField 𝕜
        inst✝⁵ : NontriviallyNormedField 𝕜₂
        inst✝⁴ : NontriviallyNormedField 𝕜₃
        inst✝³ : NormedSpace 𝕜 E
        inst✝² : NormedSpace 𝕜₂ F
        inst✝¹ : NormedSpace 𝕜₃ G
        σ₁₂ : RingHom 𝕜 𝕜₂
        σ₁₃ : RingHom 𝕜 𝕜₃
        inst✝ : RingHomIsometric σ₁₃
        f : ContinuousLinearMap σ₁₂ E F
        g : ContinuousLinearMap σ₁₃ E G
        h : ∀ (x : E), Eq (Norm.norm (f x)) (Norm.norm (g x))
        x : E
        ⊢ LE.le (Norm.norm (g x)) (HMul.hMul (Norm.norm g) (Norm.norm x))
      -/
      exact le_opNorm _ _)
      /-
        🎉 no goals
      -/
    fun c hc h₂ =>
    opNorm_le_bound _ hc fun z => by
      /-
        𝕜 : Type u_1
        𝕜₂ : Type u_2
        𝕜₃ : Type u_3
        E : Type u_4
        F : Type u_6
        G : Type u_8
        inst✝⁹ : SeminormedAddCommGroup E
        inst✝⁸ : SeminormedAddCommGroup F
        inst✝⁷ : SeminormedAddCommGroup G
        inst✝⁶ : NontriviallyNormedField 𝕜
        inst✝⁵ : NontriviallyNormedField 𝕜₂
        inst✝⁴ : NontriviallyNormedField 𝕜₃
        inst✝³ : NormedSpace 𝕜 E
        inst✝² : NormedSpace 𝕜₂ F
        inst✝¹ : NormedSpace 𝕜₃ G
        σ₁₂ : RingHom 𝕜 𝕜₂
        σ₁₃ : RingHom 𝕜 𝕜₃
        inst✝ : RingHomIsometric σ₁₃
        f : ContinuousLinearMap σ₁₂ E F
        g : ContinuousLinearMap σ₁₃ E G
        h : ∀ (x : E), Eq (Norm.norm (f x)) (Norm.norm (g x))
        c : Real
        hc : GE.ge c 0
        h₂ : ∀ (x : E), LE.le (Norm.norm (f x)) (HMul.hMul c (Norm.norm x))
        z : E
        ⊢ LE.le (Norm.norm (g z)) (HMul.hMul c (Norm.norm z))
      -/
      rw [← h z]
      /-
        𝕜 : Type u_1
        𝕜₂ : Type u_2
        𝕜₃ : Type u_3
        E : Type u_4
        F : Type u_6
        G : Type u_8
        inst✝⁹ : SeminormedAddCommGroup E
        inst✝⁸ : SeminormedAddCommGroup F
        inst✝⁷ : SeminormedAddCommGroup G
        inst✝⁶ : NontriviallyNormedField 𝕜
        inst✝⁵ : NontriviallyNormedField 𝕜₂
        inst✝⁴ : NontriviallyNormedField 𝕜₃
        inst✝³ : NormedSpace 𝕜 E
        inst✝² : NormedSpace 𝕜₂ F
        inst✝¹ : NormedSpace 𝕜₃ G
        σ₁₂ : RingHom 𝕜 𝕜₂
        σ₁₃ : RingHom 𝕜 𝕜₃
        inst✝ : RingHomIsometric σ₁₃
        f : ContinuousLinearMap σ₁₂ E F
        g : ContinuousLinearMap σ₁₃ E G
        h : ∀ (x : E), Eq (Norm.norm (f x)) (Norm.norm (g x))
        c : Real
        hc : GE.ge c 0
        h₂ : ∀ (x : E), LE.le (Norm.norm (f x)) (HMul.hMul c (Norm.norm x))
        z : E
        ⊢ LE.le (Norm.norm (f z)) (HMul.hMul c (Norm.norm z))
      -/
      exact h₂ z
      /-
        🎉 no goals
      -/


@[deprecated (since := "2024-02-02")] alias op_norm_ext := opNorm_ext


theorem opNorm_le_bound₂ (f : E →SL[σ₁₃] F →SL[σ₂₃] G) {C : ℝ} (h0 : 0 ≤ C)
    (hC : ∀ x y, ‖f x y‖ ≤ C * ‖x‖ * ‖y‖) : ‖f‖ ≤ C :=
  f.opNorm_le_bound h0 fun x => (f x).opNorm_le_bound (mul_nonneg h0 (norm_nonneg _)) <| hC x


@[deprecated (since := "2024-02-02")] alias op_norm_le_bound₂ := opNorm_le_bound₂


theorem le_opNorm₂ [RingHomIsometric σ₁₃] (f : E →SL[σ₁₃] F →SL[σ₂₃] G) (x : E) (y : F) :
    ‖f x y‖ ≤ ‖f‖ * ‖x‖ * ‖y‖ :=
  (f x).le_of_opNorm_le (f.le_opNorm x) y


@[deprecated (since := "2024-02-02")] alias le_op_norm₂ := le_opNorm₂


theorem le_of_opNorm₂_le_of_le [RingHomIsometric σ₁₃] (f : E →SL[σ₁₃] F →SL[σ₂₃] G) {x : E} {y : F}
    {a b c : ℝ} (hf : ‖f‖ ≤ a) (hx : ‖x‖ ≤ b) (hy : ‖y‖ ≤ c) :
    ‖f x y‖ ≤ a * b * c :=
  (f x).le_of_opNorm_le_of_le (f.le_of_opNorm_le_of_le hf hx) hy


@[deprecated (since := "2024-02-02")] alias le_of_op_norm₂_le_of_le := le_of_opNorm₂_le_of_le


lemma norm_mkContinuous₂_aux (f : E →ₛₗ[σ₁₃] F →ₛₗ[σ₂₃] G) (C : ℝ)
    (h : ∀ x y, ‖f x y‖ ≤ C * ‖x‖ * ‖y‖) (x : E) :
    ‖(f x).mkContinuous (C * ‖x‖) (h x)‖ ≤ max C 0 * ‖x‖ :=
  (mkContinuous_norm_le' (f x) (h x)).trans_eq <| by
    /-
      𝕜 : Type u_1
      𝕜₂ : Type u_2
      𝕜₃ : Type u_3
      E : Type u_4
      F : Type u_6
      G : Type u_8
      inst✝⁸ : SeminormedAddCommGroup E
      inst✝⁷ : SeminormedAddCommGroup F
      inst✝⁶ : SeminormedAddCommGroup G
      inst✝⁵ : NontriviallyNormedField 𝕜
      inst✝⁴ : NontriviallyNormedField 𝕜₂
      inst✝³ : NontriviallyNormedField 𝕜₃
      inst✝² : NormedSpace 𝕜 E
      inst✝¹ : NormedSpace 𝕜₂ F
      inst✝ : NormedSpace 𝕜₃ G
      σ₂₃ : RingHom 𝕜₂ 𝕜₃
      σ₁₃ : RingHom 𝕜 𝕜₃
      f : LinearMap σ₁₃ E (LinearMap σ₂₃ F G)
      C : Real
      h : ∀ (x : E) (y : F), LE.le (Norm.norm ((f x) y)) (HMul.hMul (HMul.hMul C (No …
      x : E
      ⊢ Eq (Max.max (HMul.hMul C (Norm.norm x)) 0) (HMul.hMul (Max.max C 0) (Norm.no …
    -/
    rw [max_mul_of_nonneg _ _ (norm_nonneg x), zero_mul]
    /-
      🎉 no goals
    -/


/-- Create a bilinear map (represented as a map `E →L[𝕜] F →L[𝕜] G`) from the corresponding linear
map and existence of a bound on the norm of the image. The linear map can be constructed using
`LinearMap.mk₂`.

If you have an explicit bound, use `LinearMap.mkContinuous₂` instead, as a norm estimate will
follow automatically in `LinearMap.mkContinuous₂_norm_le`. -/
def mkContinuousOfExistsBound₂ (f : E →ₛₗ[σ₁₃] F →ₛₗ[σ₂₃] G)
    (h : ∃ C, ∀ x y, ‖f x y‖ ≤ C * ‖x‖ * ‖y‖) : E →SL[σ₁₃] F →SL[σ₂₃] G :=
  LinearMap.mkContinuousOfExistsBound
    { toFun := fun x => (f x).mkContinuousOfExistsBound <| let ⟨C, hC⟩ := h; ⟨C * ‖x‖, hC x⟩
      map_add' := fun x y => by
        /-
          𝕜 : Type u_1
          𝕜₂ : Type u_2
          𝕜₃ : Type u_3
          E : Type u_4
          Eₗ : Type u_5
          F : Type u_6
          Fₗ : Type u_7
          G : Type u_8
          Gₗ : Type u_9
          𝓕 : Type u_10
          inst✝¹⁷ : SeminormedAddCommGroup E
          inst✝¹⁶ : SeminormedAddCommGroup Eₗ
          inst✝¹⁵ : SeminormedAddCommGroup F
          inst✝¹⁴ : SeminormedAddCommGroup Fₗ
          inst✝¹³ : SeminormedAddCommGroup G
          inst✝¹² : SeminormedAddCommGroup Gₗ
          inst✝¹¹ : NontriviallyNormedField 𝕜
          inst✝¹⁰ : NontriviallyNormedField 𝕜₂
          inst✝⁹ : NontriviallyNormedField 𝕜₃
          inst✝⁸ : NormedSpace 𝕜 E
          inst✝⁷ : NormedSpace 𝕜 Eₗ
          inst✝⁶ : NormedSpace 𝕜₂ F
          inst✝⁵ : NormedSpace 𝕜 Fₗ
          inst✝⁴ : NormedSpace 𝕜₃ G
          inst✝³ : NormedSpace 𝕜 Gₗ
          σ₁₂ : RingHom 𝕜 𝕜₂
          σ₂₃ : RingHom 𝕜₂ 𝕜₃
          σ₁₃ : RingHom 𝕜 𝕜₃
          inst✝² : RingHomCompTriple σ₁₂ σ₂₃ σ₁₃
          inst✝¹ : FunLike 𝓕 E F
          inst✝ : RingHomIsometric σ₂₃
          f : LinearMap σ₁₃ E (LinearMap σ₂₃ F G)
          h : Exists fun C => ∀ (x : E) (y : F), LE.le (Norm.norm ((f x) y)) (HMul.hMul  …
          x y : E
          ⊢ Eq ((fun x => (f x).mkContinuousOfExistsBound ⋯) (HAdd.hAdd x y)) (HAdd.hAdd …
        -/
        ext z
        /-
          case h
          𝕜 : Type u_1
          𝕜₂ : Type u_2
          𝕜₃ : Type u_3
          E : Type u_4
          Eₗ : Type u_5
          F : Type u_6
          Fₗ : Type u_7
          G : Type u_8
          Gₗ : Type u_9
          𝓕 : Type u_10
          inst✝¹⁷ : SeminormedAddCommGroup E
          inst✝¹⁶ : SeminormedAddCommGroup Eₗ
          inst✝¹⁵ : SeminormedAddCommGroup F
          inst✝¹⁴ : SeminormedAddCommGroup Fₗ
          inst✝¹³ : SeminormedAddCommGroup G
          inst✝¹² : SeminormedAddCommGroup Gₗ
          inst✝¹¹ : NontriviallyNormedField 𝕜
          inst✝¹⁰ : NontriviallyNormedField 𝕜₂
          inst✝⁹ : NontriviallyNormedField 𝕜₃
          inst✝⁸ : NormedSpace 𝕜 E
          inst✝⁷ : NormedSpace 𝕜 Eₗ
          inst✝⁶ : NormedSpace 𝕜₂ F
          inst✝⁵ : NormedSpace 𝕜 Fₗ
          inst✝⁴ : NormedSpace 𝕜₃ G
          inst✝³ : NormedSpace 𝕜 Gₗ
          σ₁₂ : RingHom 𝕜 𝕜₂
          σ₂₃ : RingHom 𝕜₂ 𝕜₃
          σ₁₃ : RingHom 𝕜 𝕜₃
          inst✝² : RingHomCompTriple σ₁₂ σ₂₃ σ₁₃
          inst✝¹ : FunLike 𝓕 E F
          inst✝ : RingHomIsometric σ₂₃
          f : LinearMap σ₁₃ E (LinearMap σ₂₃ F G)
          h : Exists fun C => ∀ (x : E) (y : F), LE.le (Norm.norm ((f x) y)) (HMul.hMul  …
          x y : E
          z : F
          ⊢ Eq (((fun x => (f x).mkContinuousOfExistsBound ⋯) (HAdd.hAdd x y)) z) ((HAdd …
        -/
        simp
        /-
          🎉 no goals
        -/
      map_smul' := fun c x => by
        /-
          𝕜 : Type u_1
          𝕜₂ : Type u_2
          𝕜₃ : Type u_3
          E : Type u_4
          Eₗ : Type u_5
          F : Type u_6
          Fₗ : Type u_7
          G : Type u_8
          Gₗ : Type u_9
          𝓕 : Type u_10
          inst✝¹⁷ : SeminormedAddCommGroup E
          inst✝¹⁶ : SeminormedAddCommGroup Eₗ
          inst✝¹⁵ : SeminormedAddCommGroup F
          inst✝¹⁴ : SeminormedAddCommGroup Fₗ
          inst✝¹³ : SeminormedAddCommGroup G
          inst✝¹² : SeminormedAddCommGroup Gₗ
          inst✝¹¹ : NontriviallyNormedField 𝕜
          inst✝¹⁰ : NontriviallyNormedField 𝕜₂
          inst✝⁹ : NontriviallyNormedField 𝕜₃
          inst✝⁸ : NormedSpace 𝕜 E
          inst✝⁷ : NormedSpace 𝕜 Eₗ
          inst✝⁶ : NormedSpace 𝕜₂ F
          inst✝⁵ : NormedSpace 𝕜 Fₗ
          inst✝⁴ : NormedSpace 𝕜₃ G
          inst✝³ : NormedSpace 𝕜 Gₗ
          σ₁₂ : RingHom 𝕜 𝕜₂
          σ₂₃ : RingHom 𝕜₂ 𝕜₃
          σ₁₃ : RingHom 𝕜 𝕜₃
          inst✝² : RingHomCompTriple σ₁₂ σ₂₃ σ₁₃
          inst✝¹ : FunLike 𝓕 E F
          inst✝ : RingHomIsometric σ₂₃
          f : LinearMap σ₁₃ E (LinearMap σ₂₃ F G)
          h : Exists fun C => ∀ (x : E) (y : F), LE.le (Norm.norm ((f x) y)) (HMul.hMul  …
          c : 𝕜
          x : E
          ⊢ Eq ({ toFun := fun x => (f x).mkContinuousOfExistsBound ⋯, map_add' := ⋯ }.t …
        -/
        ext z
        /-
          case h
          𝕜 : Type u_1
          𝕜₂ : Type u_2
          𝕜₃ : Type u_3
          E : Type u_4
          Eₗ : Type u_5
          F : Type u_6
          Fₗ : Type u_7
          G : Type u_8
          Gₗ : Type u_9
          𝓕 : Type u_10
          inst✝¹⁷ : SeminormedAddCommGroup E
          inst✝¹⁶ : SeminormedAddCommGroup Eₗ
          inst✝¹⁵ : SeminormedAddCommGroup F
          inst✝¹⁴ : SeminormedAddCommGroup Fₗ
          inst✝¹³ : SeminormedAddCommGroup G
          inst✝¹² : SeminormedAddCommGroup Gₗ
          inst✝¹¹ : NontriviallyNormedField 𝕜
          inst✝¹⁰ : NontriviallyNormedField 𝕜₂
          inst✝⁹ : NontriviallyNormedField 𝕜₃
          inst✝⁸ : NormedSpace 𝕜 E
          inst✝⁷ : NormedSpace 𝕜 Eₗ
          inst✝⁶ : NormedSpace 𝕜₂ F
          inst✝⁵ : NormedSpace 𝕜 Fₗ
          inst✝⁴ : NormedSpace 𝕜₃ G
          inst✝³ : NormedSpace 𝕜 Gₗ
          σ₁₂ : RingHom 𝕜 𝕜₂
          σ₂₃ : RingHom 𝕜₂ 𝕜₃
          σ₁₃ : RingHom 𝕜 𝕜₃
          inst✝² : RingHomCompTriple σ₁₂ σ₂₃ σ₁₃
          inst✝¹ : FunLike 𝓕 E F
          inst✝ : RingHomIsometric σ₂₃
          f : LinearMap σ₁₃ E (LinearMap σ₂₃ F G)
          h : Exists fun C => ∀ (x : E) (y : F), LE.le (Norm.norm ((f x) y)) (HMul.hMul  …
          c : 𝕜
          x : E
          z : F
          ⊢ Eq (({ toFun := fun x => (f x).mkContinuousOfExistsBound ⋯, map_add' := ⋯ }. …
        -/
        simp } <|
        /-
          🎉 no goals
        -/
    let ⟨C, hC⟩ := h; ⟨max C 0, norm_mkContinuous₂_aux f C hC⟩


/-- Create a bilinear map (represented as a map `E →L[𝕜] F →L[𝕜] G`) from the corresponding linear
map and a bound on the norm of the image. The linear map can be constructed using
`LinearMap.mk₂`. Lemmas `LinearMap.mkContinuous₂_norm_le'` and `LinearMap.mkContinuous₂_norm_le`
provide estimates on the norm of an operator constructed using this function. -/
def mkContinuous₂ (f : E →ₛₗ[σ₁₃] F →ₛₗ[σ₂₃] G) (C : ℝ) (hC : ∀ x y, ‖f x y‖ ≤ C * ‖x‖ * ‖y‖) :
    E →SL[σ₁₃] F →SL[σ₂₃] G :=
  mkContinuousOfExistsBound₂ f ⟨C, hC⟩


@[simp]
theorem mkContinuous₂_apply (f : E →ₛₗ[σ₁₃] F →ₛₗ[σ₂₃] G) {C : ℝ}
    (hC : ∀ x y, ‖f x y‖ ≤ C * ‖x‖ * ‖y‖) (x : E) (y : F) : f.mkContinuous₂ C hC x y = f x y :=
  rfl


theorem mkContinuous₂_norm_le' (f : E →ₛₗ[σ₁₃] F →ₛₗ[σ₂₃] G) {C : ℝ}
    (hC : ∀ x y, ‖f x y‖ ≤ C * ‖x‖ * ‖y‖) : ‖f.mkContinuous₂ C hC‖ ≤ max C 0 :=
  mkContinuous_norm_le _ (le_max_iff.2 <| Or.inr le_rfl) (norm_mkContinuous₂_aux f C hC)


theorem mkContinuous₂_norm_le (f : E →ₛₗ[σ₁₃] F →ₛₗ[σ₂₃] G) {C : ℝ} (h0 : 0 ≤ C)
    (hC : ∀ x y, ‖f x y‖ ≤ C * ‖x‖ * ‖y‖) : ‖f.mkContinuous₂ C hC‖ ≤ C :=
  (f.mkContinuous₂_norm_le' hC).trans_eq <| max_eq_left h0


/-- Flip the order of arguments of a continuous bilinear map.
For a version bundled as `LinearIsometryEquiv`, see
`ContinuousLinearMap.flipL`. -/
def flip (f : E →SL[σ₁₃] F →SL[σ₂₃] G) : F →SL[σ₂₃] E →SL[σ₁₃] G :=
  LinearMap.mkContinuous₂
    -- Porting note: the `simp only`s below used to be `rw`.
    -- Now that doesn't work as we need to do some beta reduction along the way.
    (LinearMap.mk₂'ₛₗ σ₂₃ σ₁₃ (fun y x => f x y) (fun x y z => (f z).map_add x y)
                                                           /-
                                                             𝕜 : Type u_1
                                                             𝕜₂ : Type u_2
                                                             𝕜₃ : Type u_3
                                                             E : Type u_4
                                                             Eₗ : Type u_5
                                                             F : Type u_6
                                                             Fₗ : Type u_7
                                                             G : Type u_8
                                                             Gₗ : Type u_9
                                                             𝓕 : Type u_10
                                                             inst✝¹⁸ : SeminormedAddCommGroup E
                                                             inst✝¹⁷ : SeminormedAddCommGroup Eₗ
                                                             inst✝¹⁶ : SeminormedAddCommGroup F
                                                             inst✝¹⁵ : SeminormedAddCommGroup Fₗ
                                                             inst✝¹⁴ : SeminormedAddCommGroup G
                                                             inst✝¹³ : SeminormedAddCommGroup Gₗ
                                                             inst✝¹² : NontriviallyNormedField 𝕜
                                                             inst✝¹¹ : NontriviallyNormedField 𝕜₂
                                                             inst✝¹⁰ : NontriviallyNormedField 𝕜₃
                                                             inst✝⁹ : NormedSpace 𝕜 E
                                                             inst✝⁸ : NormedSpace 𝕜 Eₗ
                                                             inst✝⁷ : NormedSpace 𝕜₂ F
                                                             inst✝⁶ : NormedSpace 𝕜 Fₗ
                                                             inst✝⁵ : NormedSpace 𝕜₃ G
                                                             inst✝⁴ : NormedSpace 𝕜 Gₗ
                                                             σ₁₂ : RingHom 𝕜 𝕜₂
                                                             σ₂₃ : RingHom 𝕜₂ 𝕜₃
                                                             σ₁₃ : RingHom 𝕜 𝕜₃
                                                             inst✝³ : RingHomCompTriple σ₁₂ σ₂₃ σ₁₃
                                                             inst✝² : FunLike 𝓕 E F
                                                             inst✝¹ : RingHomIsometric σ₂₃
                                                             inst✝ : RingHomIsometric σ₁₃
                                                             f : ContinuousLinearMap σ₁₃ E (ContinuousLinearMap σ₂₃ F G)
                                                             z : F
                                                             x y : E
                                                             ⊢ Eq ((fun y x => (f x) y) z (HAdd.hAdd x y)) (HAdd.hAdd ((fun y x => (f x) y) …
                                                           -/
      (fun c y x => (f x).map_smulₛₗ c y) (fun z x y => by simp only [f.map_add, add_apply])
                                                           /-
                                                             🎉 no goals
                                                           -/
                         /-
                           𝕜 : Type u_1
                           𝕜₂ : Type u_2
                           𝕜₃ : Type u_3
                           E : Type u_4
                           Eₗ : Type u_5
                           F : Type u_6
                           Fₗ : Type u_7
                           G : Type u_8
                           Gₗ : Type u_9
                           𝓕 : Type u_10
                           inst✝¹⁸ : SeminormedAddCommGroup E
                           inst✝¹⁷ : SeminormedAddCommGroup Eₗ
                           inst✝¹⁶ : SeminormedAddCommGroup F
                           inst✝¹⁵ : SeminormedAddCommGroup Fₗ
                           inst✝¹⁴ : SeminormedAddCommGroup G
                           inst✝¹³ : SeminormedAddCommGroup Gₗ
                           inst✝¹² : NontriviallyNormedField 𝕜
                           inst✝¹¹ : NontriviallyNormedField 𝕜₂
                           inst✝¹⁰ : NontriviallyNormedField 𝕜₃
                           inst✝⁹ : NormedSpace 𝕜 E
                           inst✝⁸ : NormedSpace 𝕜 Eₗ
                           inst✝⁷ : NormedSpace 𝕜₂ F
                           inst✝⁶ : NormedSpace 𝕜 Fₗ
                           inst✝⁵ : NormedSpace 𝕜₃ G
                           inst✝⁴ : NormedSpace 𝕜 Gₗ
                           σ₁₂ : RingHom 𝕜 𝕜₂
                           σ₂₃ : RingHom 𝕜₂ 𝕜₃
                           σ₁₃ : RingHom 𝕜 𝕜₃
                           inst✝³ : RingHomCompTriple σ₁₂ σ₂₃ σ₁₃
                           inst✝² : FunLike 𝓕 E F
                           inst✝¹ : RingHomIsometric σ₂₃
                           inst✝ : RingHomIsometric σ₁₃
                           f : ContinuousLinearMap σ₁₃ E (ContinuousLinearMap σ₂₃ F G)
                           c : 𝕜
                           y : F
                           x : E
                           ⊢ Eq ((fun y x => (f x) y) y (HSMul.hSMul c x)) (HSMul.hSMul (σ₁₃ c) ((fun y x …
                         -/
        (fun c y x => by simp only [f.map_smulₛₗ, smul_apply]))
                         /-
                           🎉 no goals
                         -/
                                                     /-
                                                       𝕜 : Type u_1
                                                       𝕜₂ : Type u_2
                                                       𝕜₃ : Type u_3
                                                       E : Type u_4
                                                       Eₗ : Type u_5
                                                       F : Type u_6
                                                       Fₗ : Type u_7
                                                       G : Type u_8
                                                       Gₗ : Type u_9
                                                       𝓕 : Type u_10
                                                       inst✝¹⁸ : SeminormedAddCommGroup E
                                                       inst✝¹⁷ : SeminormedAddCommGroup Eₗ
                                                       inst✝¹⁶ : SeminormedAddCommGroup F
                                                       inst✝¹⁵ : SeminormedAddCommGroup Fₗ
                                                       inst✝¹⁴ : SeminormedAddCommGroup G
                                                       inst✝¹³ : SeminormedAddCommGroup Gₗ
                                                       inst✝¹² : NontriviallyNormedField 𝕜
                                                       inst✝¹¹ : NontriviallyNormedField 𝕜₂
                                                       inst✝¹⁰ : NontriviallyNormedField 𝕜₃
                                                       inst✝⁹ : NormedSpace 𝕜 E
                                                       inst✝⁸ : NormedSpace 𝕜 Eₗ
                                                       inst✝⁷ : NormedSpace 𝕜₂ F
                                                       inst✝⁶ : NormedSpace 𝕜 Fₗ
                                                       inst✝⁵ : NormedSpace 𝕜₃ G
                                                       inst✝⁴ : NormedSpace 𝕜 Gₗ
                                                       σ₁₂ : RingHom 𝕜 𝕜₂
                                                       σ₂₃ : RingHom 𝕜₂ 𝕜₃
                                                       σ₁₃ : RingHom 𝕜 𝕜₃
                                                       inst✝³ : RingHomCompTriple σ₁₂ σ₂₃ σ₁₃
                                                       inst✝² : FunLike 𝓕 E F
                                                       inst✝¹ : RingHomIsometric σ₂₃
                                                       inst✝ : RingHomIsometric σ₁₃
                                                       f : ContinuousLinearMap σ₁₃ E (ContinuousLinearMap σ₂₃ F G)
                                                       y : F
                                                       x : E
                                                       ⊢ Eq (HMul.hMul (HMul.hMul (Norm.norm f) (Norm.norm x)) (Norm.norm y)) (HMul.h …
                                                     -/
    ‖f‖ fun y x => (f.le_opNorm₂ x y).trans_eq <| by simp only [mul_right_comm]
                                                     /-
                                                       🎉 no goals
                                                     -/


private theorem le_norm_flip (f : E →SL[σ₁₃] F →SL[σ₂₃] G) : ‖f‖ ≤ ‖flip f‖ :=
  #adaptation_note
  /--
  After https://github.com/leanprover/lean4/pull/4119 we either need
  to specify the `f.flip` argument, or use `set_option maxSynthPendingDepth 2 in`.
  -/
  f.opNorm_le_bound₂ (norm_nonneg f.flip) fun x y => by
    /-
      𝕜 : Type u_1
      𝕜₂ : Type u_2
      𝕜₃ : Type u_3
      E : Type u_4
      F : Type u_6
      G : Type u_8
      inst✝¹⁰ : SeminormedAddCommGroup E
      inst✝⁹ : SeminormedAddCommGroup F
      inst✝⁸ : SeminormedAddCommGroup G
      inst✝⁷ : NontriviallyNormedField 𝕜
      inst✝⁶ : NontriviallyNormedField 𝕜₂
      inst✝⁵ : NontriviallyNormedField 𝕜₃
      inst✝⁴ : NormedSpace 𝕜 E
      inst✝³ : NormedSpace 𝕜₂ F
      inst✝² : NormedSpace 𝕜₃ G
      σ₂₃ : RingHom 𝕜₂ 𝕜₃
      σ₁₃ : RingHom 𝕜 𝕜₃
      inst✝¹ : RingHomIsometric σ₂₃
      inst✝ : RingHomIsometric σ₁₃
      f : ContinuousLinearMap σ₁₃ E (ContinuousLinearMap σ₂₃ F G)
      x : E
      y : F
      ⊢ LE.le (Norm.norm ((f x) y)) (HMul.hMul (HMul.hMul (Norm.norm f.flip) (Norm.n …
    -/
    rw [mul_right_comm]
    /-
      𝕜 : Type u_1
      𝕜₂ : Type u_2
      𝕜₃ : Type u_3
      E : Type u_4
      F : Type u_6
      G : Type u_8
      inst✝¹⁰ : SeminormedAddCommGroup E
      inst✝⁹ : SeminormedAddCommGroup F
      inst✝⁸ : SeminormedAddCommGroup G
      inst✝⁷ : NontriviallyNormedField 𝕜
      inst✝⁶ : NontriviallyNormedField 𝕜₂
      inst✝⁵ : NontriviallyNormedField 𝕜₃
      inst✝⁴ : NormedSpace 𝕜 E
      inst✝³ : NormedSpace 𝕜₂ F
      inst✝² : NormedSpace 𝕜₃ G
      σ₂₃ : RingHom 𝕜₂ 𝕜₃
      σ₁₃ : RingHom 𝕜 𝕜₃
      inst✝¹ : RingHomIsometric σ₂₃
      inst✝ : RingHomIsometric σ₁₃
      f : ContinuousLinearMap σ₁₃ E (ContinuousLinearMap σ₂₃ F G)
      x : E
      y : F
      ⊢ LE.le (Norm.norm ((f x) y)) (HMul.hMul (HMul.hMul (Norm.norm f.flip) (Norm.n …
    -/
    exact (flip f).le_opNorm₂ y x
    /-
      🎉 no goals
    -/


@[simp]
theorem flip_apply (f : E →SL[σ₁₃] F →SL[σ₂₃] G) (x : E) (y : F) : f.flip y x = f x y :=
  rfl


@[simp]
theorem flip_flip (f : E →SL[σ₁₃] F →SL[σ₂₃] G) : f.flip.flip = f := by
  /-
    𝕜 : Type u_1
    𝕜₂ : Type u_2
    𝕜₃ : Type u_3
    E : Type u_4
    F : Type u_6
    G : Type u_8
    inst✝¹⁰ : SeminormedAddCommGroup E
    inst✝⁹ : SeminormedAddCommGroup F
    inst✝⁸ : SeminormedAddCommGroup G
    inst✝⁷ : NontriviallyNormedField 𝕜
    inst✝⁶ : NontriviallyNormedField 𝕜₂
    inst✝⁵ : NontriviallyNormedField 𝕜₃
    inst✝⁴ : NormedSpace 𝕜 E
    inst✝³ : NormedSpace 𝕜₂ F
    inst✝² : NormedSpace 𝕜₃ G
    σ₂₃ : RingHom 𝕜₂ 𝕜₃
    σ₁₃ : RingHom 𝕜 𝕜₃
    inst✝¹ : RingHomIsometric σ₂₃
    inst✝ : RingHomIsometric σ₁₃
    f : ContinuousLinearMap σ₁₃ E (ContinuousLinearMap σ₂₃ F G)
    ⊢ Eq f.flip.flip f
  -/
  ext
  /-
    case h.h
    𝕜 : Type u_1
    𝕜₂ : Type u_2
    𝕜₃ : Type u_3
    E : Type u_4
    F : Type u_6
    G : Type u_8
    inst✝¹⁰ : SeminormedAddCommGroup E
    inst✝⁹ : SeminormedAddCommGroup F
    inst✝⁸ : SeminormedAddCommGroup G
    inst✝⁷ : NontriviallyNormedField 𝕜
    inst✝⁶ : NontriviallyNormedField 𝕜₂
    inst✝⁵ : NontriviallyNormedField 𝕜₃
    inst✝⁴ : NormedSpace 𝕜 E
    inst✝³ : NormedSpace 𝕜₂ F
    inst✝² : NormedSpace 𝕜₃ G
    σ₂₃ : RingHom 𝕜₂ 𝕜₃
    σ₁₃ : RingHom 𝕜 𝕜₃
    inst✝¹ : RingHomIsometric σ₂₃
    inst✝ : RingHomIsometric σ₁₃
    f : ContinuousLinearMap σ₁₃ E (ContinuousLinearMap σ₂₃ F G)
    x✝¹ : E
    x✝ : F
    ⊢ Eq ((f.flip.flip x✝¹) x✝) ((f x✝¹) x✝)
  -/
  rfl
  /-
    🎉 no goals
  -/


@[simp]
theorem opNorm_flip (f : E →SL[σ₁₃] F →SL[σ₂₃] G) : ‖f.flip‖ = ‖f‖ :=
                  /-
                    𝕜 : Type u_1
                    𝕜₂ : Type u_2
                    𝕜₃ : Type u_3
                    E : Type u_4
                    F : Type u_6
                    G : Type u_8
                    inst✝¹⁰ : SeminormedAddCommGroup E
                    inst✝⁹ : SeminormedAddCommGroup F
                    inst✝⁸ : SeminormedAddCommGroup G
                    inst✝⁷ : NontriviallyNormedField 𝕜
                    inst✝⁶ : NontriviallyNormedField 𝕜₂
                    inst✝⁵ : NontriviallyNormedField 𝕜₃
                    inst✝⁴ : NormedSpace 𝕜 E
                    inst✝³ : NormedSpace 𝕜₂ F
                    inst✝² : NormedSpace 𝕜₃ G
                    σ₂₃ : RingHom 𝕜₂ 𝕜₃
                    σ₁₃ : RingHom 𝕜 𝕜₃
                    inst✝¹ : RingHomIsometric σ₂₃
                    inst✝ : RingHomIsometric σ₁₃
                    f : ContinuousLinearMap σ₁₃ E (ContinuousLinearMap σ₂₃ F G)
                    ⊢ LE.le (Norm.norm f.flip) (Norm.norm f)
                  -/
  le_antisymm (by simpa only [flip_flip] using le_norm_flip f.flip) (le_norm_flip f)
                  /-
                    🎉 no goals
                  -/


@[deprecated (since := "2024-02-02")] alias op_norm_flip := opNorm_flip


@[simp]
theorem flip_add (f g : E →SL[σ₁₃] F →SL[σ₂₃] G) : (f + g).flip = f.flip + g.flip :=
  rfl


@[simp]
theorem flip_smul (c : 𝕜₃) (f : E →SL[σ₁₃] F →SL[σ₂₃] G) : (c • f).flip = c • f.flip :=
  rfl


/-- Flip the order of arguments of a continuous bilinear map.
This is a version bundled as a `LinearIsometryEquiv`.
For an unbundled version see `ContinuousLinearMap.flip`. -/
def flipₗᵢ' : (E →SL[σ₁₃] F →SL[σ₂₃] G) ≃ₗᵢ[𝕜₃] F →SL[σ₂₃] E →SL[σ₁₃] G where
  toFun := flip
  invFun := flip
  map_add' := flip_add
  map_smul' := flip_smul
  left_inv := flip_flip
  right_inv := flip_flip
  norm_map' := opNorm_flip


@[simp]
theorem flipₗᵢ'_symm : (flipₗᵢ' E F G σ₂₃ σ₁₃).symm = flipₗᵢ' F E G σ₁₃ σ₂₃ :=
  rfl


@[simp]
theorem coe_flipₗᵢ' : ⇑(flipₗᵢ' E F G σ₂₃ σ₁₃) = flip :=
  rfl


/-- Flip the order of arguments of a continuous bilinear map.
This is a version bundled as a `LinearIsometryEquiv`.
For an unbundled version see `ContinuousLinearMap.flip`. -/
def flipₗᵢ : (E →L[𝕜] Fₗ →L[𝕜] Gₗ) ≃ₗᵢ[𝕜] Fₗ →L[𝕜] E →L[𝕜] Gₗ where
  toFun := flip
  invFun := flip
  map_add' := flip_add
  map_smul' := flip_smul
  left_inv := flip_flip
  right_inv := flip_flip
  norm_map' := opNorm_flip


@[simp]
theorem flipₗᵢ_symm : (flipₗᵢ 𝕜 E Fₗ Gₗ).symm = flipₗᵢ 𝕜 Fₗ E Gₗ :=
  rfl


@[simp]
theorem coe_flipₗᵢ : ⇑(flipₗᵢ 𝕜 E Fₗ Gₗ) = flip :=
  rfl


/-- The continuous semilinear map obtained by applying a continuous semilinear map at a given
vector.

This is the continuous version of `LinearMap.applyₗ`. -/
def apply' : E →SL[σ₁₂] (E →SL[σ₁₂] F) →L[𝕜₂] F :=
  flip (id 𝕜₂ (E →SL[σ₁₂] F))


@[simp]
theorem apply_apply' (v : E) (f : E →SL[σ₁₂] F) : apply' F σ₁₂ v f = f v :=
  rfl


/-- The continuous semilinear map obtained by applying a continuous semilinear map at a given
vector.

This is the continuous version of `LinearMap.applyₗ`. -/
def apply : E →L[𝕜] (E →L[𝕜] Fₗ) →L[𝕜] Fₗ :=
  flip (id 𝕜 (E →L[𝕜] Fₗ))


@[simp]
theorem apply_apply (v : E) (f : E →L[𝕜] Fₗ) : apply 𝕜 Fₗ v f = f v :=
  rfl


/-- Composition of continuous semilinear maps as a continuous semibilinear map. -/
def compSL : (F →SL[σ₂₃] G) →L[𝕜₃] (E →SL[σ₁₂] F) →SL[σ₂₃] E →SL[σ₁₃] G :=
  LinearMap.mkContinuous₂
    (LinearMap.mk₂'ₛₗ (RingHom.id 𝕜₃) σ₂₃ comp add_comp smul_comp comp_add fun c f g => by
      /-
        𝕜 : Type u_1
        𝕜₂ : Type u_2
        𝕜₃ : Type u_3
        E : Type u_4
        Eₗ : Type u_5
        F : Type u_6
        Fₗ : Type u_7
        G : Type u_8
        Gₗ : Type u_9
        𝓕 : Type u_10
        inst✝¹⁹ : SeminormedAddCommGroup E
        inst✝¹⁸ : SeminormedAddCommGroup Eₗ
        inst✝¹⁷ : SeminormedAddCommGroup F
        inst✝¹⁶ : SeminormedAddCommGroup Fₗ
        inst✝¹⁵ : SeminormedAddCommGroup G
        inst✝¹⁴ : SeminormedAddCommGroup Gₗ
        inst✝¹³ : NontriviallyNormedField 𝕜
        inst✝¹² : NontriviallyNormedField 𝕜₂
        inst✝¹¹ : NontriviallyNormedField 𝕜₃
        inst✝¹⁰ : NormedSpace 𝕜 E
        inst✝⁹ : NormedSpace 𝕜 Eₗ
        inst✝⁸ : NormedSpace 𝕜₂ F
        inst✝⁷ : NormedSpace 𝕜 Fₗ
        inst✝⁶ : NormedSpace 𝕜₃ G
        inst✝⁵ : NormedSpace 𝕜 Gₗ
        σ₁₂ : RingHom 𝕜 𝕜₂
        σ₂₃ : RingHom 𝕜₂ 𝕜₃
        σ₁₃ : RingHom 𝕜 𝕜₃
        inst✝⁴ : RingHomCompTriple σ₁₂ σ₂₃ σ₁₃
        inst✝³ : FunLike 𝓕 E F
        inst✝² : RingHomIsometric σ₂₃
        inst✝¹ : RingHomIsometric σ₁₃
        inst✝ : RingHomIsometric σ₁₂
        c : 𝕜₂
        f : ContinuousLinearMap σ₂₃ F G
        g : ContinuousLinearMap σ₁₂ E F
        ⊢ Eq (f.comp (HSMul.hSMul c g)) (HSMul.hSMul (σ₂₃ c) (f.comp g))
      -/
      ext
      simp only [ContinuousLinearMap.map_smulₛₗ, coe_smul', coe_comp', Function.comp_apply,
        Pi.smul_apply])
                    /-
                      𝕜 : Type u_1
                      𝕜₂ : Type u_2
                      𝕜₃ : Type u_3
                      E : Type u_4
                      Eₗ : Type u_5
                      F : Type u_6
                      Fₗ : Type u_7
                      G : Type u_8
                      Gₗ : Type u_9
                      𝓕 : Type u_10
                      inst✝¹⁹ : SeminormedAddCommGroup E
                      inst✝¹⁸ : SeminormedAddCommGroup Eₗ
                      inst✝¹⁷ : SeminormedAddCommGroup F
                      inst✝¹⁶ : SeminormedAddCommGroup Fₗ
                      inst✝¹⁵ : SeminormedAddCommGroup G
                      inst✝¹⁴ : SeminormedAddCommGroup Gₗ
                      inst✝¹³ : NontriviallyNormedField 𝕜
                      inst✝¹² : NontriviallyNormedField 𝕜₂
                      inst✝¹¹ : NontriviallyNormedField 𝕜₃
                      inst✝¹⁰ : NormedSpace 𝕜 E
                      inst✝⁹ : NormedSpace 𝕜 Eₗ
                      inst✝⁸ : NormedSpace 𝕜₂ F
                      inst✝⁷ : NormedSpace 𝕜 Fₗ
                      inst✝⁶ : NormedSpace 𝕜₃ G
                      inst✝⁵ : NormedSpace 𝕜 Gₗ
                      σ₁₂ : RingHom 𝕜 𝕜₂
                      σ₂₃ : RingHom 𝕜₂ 𝕜₃
                      σ₁₃ : RingHom 𝕜 𝕜₃
                      inst✝⁴ : RingHomCompTriple σ₁₂ σ₂₃ σ₁₃
                      inst✝³ : FunLike 𝓕 E F
                      inst✝² : RingHomIsometric σ₂₃
                      inst✝¹ : RingHomIsometric σ₁₃
                      inst✝ : RingHomIsometric σ₁₂
                      f : ContinuousLinearMap σ₂₃ F G
                      g : ContinuousLinearMap σ₁₂ E F
                      ⊢ LE.le (Norm.norm (((LinearMap.mk₂'ₛₗ (RingHom.id 𝕜₃) σ₂₃ ContinuousLinearMap …
                    -/
    1 fun f g => by simpa only [one_mul] using opNorm_comp_le f g
                    /-
                      🎉 no goals
                    -/


set_option maxSynthPendingDepth 2 in
theorem norm_compSL_le : ‖compSL E F G σ₁₂ σ₂₃‖ ≤ 1 :=
  LinearMap.mkContinuous₂_norm_le _ zero_le_one _


@[simp]
theorem compSL_apply (f : F →SL[σ₂₃] G) (g : E →SL[σ₁₂] F) : compSL E F G σ₁₂ σ₂₃ f g = f.comp g :=
  rfl


theorem _root_.Continuous.const_clm_comp {X} [TopologicalSpace X] {f : X → E →SL[σ₁₂] F}
    (hf : Continuous f) (g : F →SL[σ₂₃] G) :
    Continuous (fun x => g.comp (f x) : X → E →SL[σ₁₃] G) :=
  (compSL E F G σ₁₂ σ₂₃ g).continuous.comp hf

-- Giving the implicit argument speeds up elaboration significantly

theorem _root_.Continuous.clm_comp_const {X} [TopologicalSpace X] {g : X → F →SL[σ₂₃] G}
    (hg : Continuous g) (f : E →SL[σ₁₂] F) :
    Continuous (fun x => (g x).comp f : X → E →SL[σ₁₃] G) :=
  (@ContinuousLinearMap.flip _ _ _ _ _ (E →SL[σ₁₃] G) _ _ _ _ _ _ _ _ _ _ _ _ _
    (compSL E F G σ₁₂ σ₂₃) f).continuous.comp hg


/-- Composition of continuous linear maps as a continuous bilinear map. -/
def compL : (Fₗ →L[𝕜] Gₗ) →L[𝕜] (E →L[𝕜] Fₗ) →L[𝕜] E →L[𝕜] Gₗ :=
  compSL E Fₗ Gₗ (RingHom.id 𝕜) (RingHom.id 𝕜)


set_option maxSynthPendingDepth 2 in
theorem norm_compL_le : ‖compL 𝕜 E Fₗ Gₗ‖ ≤ 1 :=
  norm_compSL_le _ _ _ _ _


@[simp]
theorem compL_apply (f : Fₗ →L[𝕜] Gₗ) (g : E →L[𝕜] Fₗ) : compL 𝕜 E Fₗ Gₗ f g = f.comp g :=
  rfl


/-- Apply `L(x,-)` pointwise to bilinear maps, as a continuous bilinear map -/
@[simps! apply]
def precompR (L : E →L[𝕜] Fₗ →L[𝕜] Gₗ) : E →L[𝕜] (Eₗ →L[𝕜] Fₗ) →L[𝕜] Eₗ →L[𝕜] Gₗ :=
  (compL 𝕜 Eₗ Fₗ Gₗ).comp L


/-- Apply `L(-,y)` pointwise to bilinear maps, as a continuous bilinear map -/
def precompL (L : E →L[𝕜] Fₗ →L[𝕜] Gₗ) : (Eₗ →L[𝕜] E) →L[𝕜] Fₗ →L[𝕜] Eₗ →L[𝕜] Gₗ :=
  (precompR Eₗ (flip L)).flip


@[simp] lemma precompL_apply (L : E →L[𝕜] Fₗ →L[𝕜] Gₗ) (u : Eₗ →L[𝕜] E) (f : Fₗ) (g : Eₗ) :
    precompL Eₗ L u f g = L (u g) f := rfl


set_option maxSynthPendingDepth 2 in
theorem norm_precompR_le (L : E →L[𝕜] Fₗ →L[𝕜] Gₗ) : ‖precompR Eₗ L‖ ≤ ‖L‖ :=
  calc
    ‖precompR Eₗ L‖ ≤ ‖compL 𝕜 Eₗ Fₗ Gₗ‖ * ‖L‖ := opNorm_comp_le _ _
    _ ≤ 1 * ‖L‖ := mul_le_mul_of_nonneg_right (norm_compL_le _ _ _ _) (norm_nonneg L)
                  /-
                    𝕜 : Type u_1
                    E : Type u_4
                    Eₗ : Type u_5
                    Fₗ : Type u_7
                    Gₗ : Type u_9
                    inst✝⁸ : SeminormedAddCommGroup E
                    inst✝⁷ : SeminormedAddCommGroup Eₗ
                    inst✝⁶ : SeminormedAddCommGroup Fₗ
                    inst✝⁵ : SeminormedAddCommGroup Gₗ
                    inst✝⁴ : NontriviallyNormedField 𝕜
                    inst✝³ : NormedSpace 𝕜 E
                    inst✝² : NormedSpace 𝕜 Eₗ
                    inst✝¹ : NormedSpace 𝕜 Fₗ
                    inst✝ : NormedSpace 𝕜 Gₗ
                    L : ContinuousLinearMap (RingHom.id 𝕜) E (ContinuousLinearMap (RingHom.id 𝕜) F …
                    ⊢ Eq (HMul.hMul 1 (Norm.norm L)) (Norm.norm L)
                  -/
    _ = ‖L‖ := by rw [one_mul]
                  /-
                    🎉 no goals
                  -/


set_option maxSynthPendingDepth 2 in
theorem norm_precompL_le (L : E →L[𝕜] Fₗ →L[𝕜] Gₗ) : ‖precompL Eₗ L‖ ≤ ‖L‖ := by
  /-
    𝕜 : Type u_1
    E : Type u_4
    Eₗ : Type u_5
    Fₗ : Type u_7
    Gₗ : Type u_9
    inst✝⁸ : SeminormedAddCommGroup E
    inst✝⁷ : SeminormedAddCommGroup Eₗ
    inst✝⁶ : SeminormedAddCommGroup Fₗ
    inst✝⁵ : SeminormedAddCommGroup Gₗ
    inst✝⁴ : NontriviallyNormedField 𝕜
    inst✝³ : NormedSpace 𝕜 E
    inst✝² : NormedSpace 𝕜 Eₗ
    inst✝¹ : NormedSpace 𝕜 Fₗ
    inst✝ : NormedSpace 𝕜 Gₗ
    L : ContinuousLinearMap (RingHom.id 𝕜) E (ContinuousLinearMap (RingHom.id 𝕜) F …
    ⊢ LE.le (Norm.norm (ContinuousLinearMap.precompL Eₗ L)) (Norm.norm L)
  -/
  rw [precompL, opNorm_flip, ← opNorm_flip L]
  /-
    𝕜 : Type u_1
    E : Type u_4
    Eₗ : Type u_5
    Fₗ : Type u_7
    Gₗ : Type u_9
    inst✝⁸ : SeminormedAddCommGroup E
    inst✝⁷ : SeminormedAddCommGroup Eₗ
    inst✝⁶ : SeminormedAddCommGroup Fₗ
    inst✝⁵ : SeminormedAddCommGroup Gₗ
    inst✝⁴ : NontriviallyNormedField 𝕜
    inst✝³ : NormedSpace 𝕜 E
    inst✝² : NormedSpace 𝕜 Eₗ
    inst✝¹ : NormedSpace 𝕜 Fₗ
    inst✝ : NormedSpace 𝕜 Gₗ
    L : ContinuousLinearMap (RingHom.id 𝕜) E (ContinuousLinearMap (RingHom.id 𝕜) F …
    ⊢ LE.le (Norm.norm (ContinuousLinearMap.precompR Eₗ L.flip)) (Norm.norm L.flip)
  -/
  exact norm_precompR_le _ L.flip
  /-
    🎉 no goals
  -/


/-- Compose a bilinear map `E →SL[σ₁₃] F →SL[σ₂₃] G` with two linear maps
`E' →SL[σ₁'] E` and `F' →SL[σ₂'] F`. -/
def bilinearComp (f : E →SL[σ₁₃] F →SL[σ₂₃] G) (gE : E' →SL[σ₁'] E) (gF : F' →SL[σ₂'] F) :
    E' →SL[σ₁₃'] F' →SL[σ₂₃'] G :=
  ((f.comp gE).flip.comp gF).flip


@[simp]
theorem bilinearComp_apply (f : E →SL[σ₁₃] F →SL[σ₂₃] G) (gE : E' →SL[σ₁'] E) (gF : F' →SL[σ₂'] F)
    (x : E') (y : F') : f.bilinearComp gE gF x y = f (gE x) (gF y) :=
  rfl


/-- Derivative of a continuous bilinear map `f : E →L[𝕜] F →L[𝕜] G` interpreted as a map `E × F → G`
at point `p : E × F` evaluated at `q : E × F`, as a continuous bilinear map. -/
def deriv₂ (f : E →L[𝕜] Fₗ →L[𝕜] Gₗ) : E × Fₗ →L[𝕜] E × Fₗ →L[𝕜] Gₗ :=
  f.bilinearComp (fst _ _ _) (snd _ _ _) + f.flip.bilinearComp (snd _ _ _) (fst _ _ _)


@[simp]
theorem coe_deriv₂ (f : E →L[𝕜] Fₗ →L[𝕜] Gₗ) (p : E × Fₗ) :
    ⇑(f.deriv₂ p) = fun q : E × Fₗ => f p.1 q.2 + f q.1 p.2 :=
  rfl


theorem map_add_add (f : E →L[𝕜] Fₗ →L[𝕜] Gₗ) (x x' : E) (y y' : Fₗ) :
    f (x + x') (y + y') = f x y + f.deriv₂ (x, y) (x', y') + f x' y' := by
  /-
    𝕜 : Type u_1
    E : Type u_4
    Fₗ : Type u_7
    Gₗ : Type u_9
    inst✝⁶ : SeminormedAddCommGroup E
    inst✝⁵ : SeminormedAddCommGroup Fₗ
    inst✝⁴ : SeminormedAddCommGroup Gₗ
    inst✝³ : NontriviallyNormedField 𝕜
    inst✝² : NormedSpace 𝕜 E
    inst✝¹ : NormedSpace 𝕜 Fₗ
    inst✝ : NormedSpace 𝕜 Gₗ
    f : ContinuousLinearMap (RingHom.id 𝕜) E (ContinuousLinearMap (RingHom.id 𝕜) F …
    x x' : E
    y y' : Fₗ
    ⊢ Eq ((f (HAdd.hAdd x x')) (HAdd.hAdd y y')) (HAdd.hAdd (HAdd.hAdd ((f x) y) ( …
  -/
  simp only [map_add, add_apply, coe_deriv₂, add_assoc]
  /-
    𝕜 : Type u_1
    E : Type u_4
    Fₗ : Type u_7
    Gₗ : Type u_9
    inst✝⁶ : SeminormedAddCommGroup E
    inst✝⁵ : SeminormedAddCommGroup Fₗ
    inst✝⁴ : SeminormedAddCommGroup Gₗ
    inst✝³ : NontriviallyNormedField 𝕜
    inst✝² : NormedSpace 𝕜 E
    inst✝¹ : NormedSpace 𝕜 Fₗ
    inst✝ : NormedSpace 𝕜 Gₗ
    f : ContinuousLinearMap (RingHom.id 𝕜) E (ContinuousLinearMap (RingHom.id 𝕜) F …
    x x' : E
    y y' : Fₗ
    ⊢ Eq (HAdd.hAdd ((f x) y) (HAdd.hAdd ((f x') y) (HAdd.hAdd ((f x) y') ((f x')  …
  -/
  /-
    🎉 no goals
  -/
  abel
  /-
    🎉 no goals
  -/


/-- The norm of the tensor product of a scalar linear map and of an element of a normed space
is the product of the norms. -/
@[simp]
theorem norm_smulRight_apply (c : E →L[𝕜] 𝕜) (f : Fₗ) : ‖smulRight c f‖ = ‖c‖ * ‖f‖ := by
  /-
    𝕜 : Type u_1
    E : Type u_4
    Fₗ : Type u_7
    inst✝⁴ : SeminormedAddCommGroup E
    inst✝³ : SeminormedAddCommGroup Fₗ
    inst✝² : NontriviallyNormedField 𝕜
    inst✝¹ : NormedSpace 𝕜 E
    inst✝ : NormedSpace 𝕜 Fₗ
    c : ContinuousLinearMap (RingHom.id 𝕜) E 𝕜
    f : Fₗ
    ⊢ Eq (Norm.norm (c.smulRight f)) (HMul.hMul (Norm.norm c) (Norm.norm f))
  -/
  refine le_antisymm ?_ ?_
    /-
      case refine_1
      𝕜 : Type u_1
      E : Type u_4
      Fₗ : Type u_7
      inst✝⁴ : SeminormedAddCommGroup E
      inst✝³ : SeminormedAddCommGroup Fₗ
      inst✝² : NontriviallyNormedField 𝕜
      inst✝¹ : NormedSpace 𝕜 E
      inst✝ : NormedSpace 𝕜 Fₗ
      c : ContinuousLinearMap (RingHom.id 𝕜) E 𝕜
      f : Fₗ
      ⊢ LE.le (Norm.norm (c.smulRight f)) (HMul.hMul (Norm.norm c) (Norm.norm f))
    -/
  · refine opNorm_le_bound _ (mul_nonneg (norm_nonneg _) (norm_nonneg _)) fun x => ?_
    calc
      ‖c x • f‖ = ‖c x‖ * ‖f‖ := norm_smul _ _
      _ ≤ ‖c‖ * ‖x‖ * ‖f‖ := mul_le_mul_of_nonneg_right (le_opNorm _ _) (norm_nonneg _)
      _ = ‖c‖ * ‖f‖ * ‖x‖ := by ring
    /-
      case refine_2
      𝕜 : Type u_1
      E : Type u_4
      Fₗ : Type u_7
      inst✝⁴ : SeminormedAddCommGroup E
      inst✝³ : SeminormedAddCommGroup Fₗ
      inst✝² : NontriviallyNormedField 𝕜
      inst✝¹ : NormedSpace 𝕜 E
      inst✝ : NormedSpace 𝕜 Fₗ
      c : ContinuousLinearMap (RingHom.id 𝕜) E 𝕜
      f : Fₗ
      ⊢ LE.le (HMul.hMul (Norm.norm c) (Norm.norm f)) (Norm.norm (c.smulRight f))
    -/
  · obtain hf | hf := (norm_nonneg f).eq_or_gt
      /-
        case refine_2.inl
        𝕜 : Type u_1
        E : Type u_4
        Fₗ : Type u_7
        inst✝⁴ : SeminormedAddCommGroup E
        inst✝³ : SeminormedAddCommGroup Fₗ
        inst✝² : NontriviallyNormedField 𝕜
        inst✝¹ : NormedSpace 𝕜 E
        inst✝ : NormedSpace 𝕜 Fₗ
        c : ContinuousLinearMap (RingHom.id 𝕜) E 𝕜
        f : Fₗ
        hf : Eq (Norm.norm f) 0
        ⊢ LE.le (HMul.hMul (Norm.norm c) (Norm.norm f)) (Norm.norm (c.smulRight f))
      -/
    · simp [hf]
      /-
        🎉 no goals
      -/
      /-
        case refine_2.inr
        𝕜 : Type u_1
        E : Type u_4
        Fₗ : Type u_7
        inst✝⁴ : SeminormedAddCommGroup E
        inst✝³ : SeminormedAddCommGroup Fₗ
        inst✝² : NontriviallyNormedField 𝕜
        inst✝¹ : NormedSpace 𝕜 E
        inst✝ : NormedSpace 𝕜 Fₗ
        c : ContinuousLinearMap (RingHom.id 𝕜) E 𝕜
        f : Fₗ
        hf : LT.lt 0 (Norm.norm f)
        ⊢ LE.le (HMul.hMul (Norm.norm c) (Norm.norm f)) (Norm.norm (c.smulRight f))
      -/
    · rw [← le_div_iff₀ hf]
      /-
        case refine_2.inr
        𝕜 : Type u_1
        E : Type u_4
        Fₗ : Type u_7
        inst✝⁴ : SeminormedAddCommGroup E
        inst✝³ : SeminormedAddCommGroup Fₗ
        inst✝² : NontriviallyNormedField 𝕜
        inst✝¹ : NormedSpace 𝕜 E
        inst✝ : NormedSpace 𝕜 Fₗ
        c : ContinuousLinearMap (RingHom.id 𝕜) E 𝕜
        f : Fₗ
        hf : LT.lt 0 (Norm.norm f)
        ⊢ LE.le (Norm.norm c) (HDiv.hDiv (Norm.norm (c.smulRight f)) (Norm.norm f))
      -/
      refine opNorm_le_bound _ (div_nonneg (norm_nonneg _) (norm_nonneg f)) fun x => ?_
      /-
        case refine_2.inr
        𝕜 : Type u_1
        E : Type u_4
        Fₗ : Type u_7
        inst✝⁴ : SeminormedAddCommGroup E
        inst✝³ : SeminormedAddCommGroup Fₗ
        inst✝² : NontriviallyNormedField 𝕜
        inst✝¹ : NormedSpace 𝕜 E
        inst✝ : NormedSpace 𝕜 Fₗ
        c : ContinuousLinearMap (RingHom.id 𝕜) E 𝕜
        f : Fₗ
        hf : LT.lt 0 (Norm.norm f)
        x : E
        ⊢ LE.le (Norm.norm (c x)) (HMul.hMul (HDiv.hDiv (Norm.norm (c.smulRight f)) (N …
      -/
      rw [div_mul_eq_mul_div, le_div_iff₀ hf]
      calc
        ‖c x‖ * ‖f‖ = ‖c x • f‖ := (norm_smul _ _).symm
        _ = ‖smulRight c f x‖ := rfl
        _ ≤ ‖smulRight c f‖ * ‖x‖ := le_opNorm _ _


/-- The non-negative norm of the tensor product of a scalar linear map and of an element of a normed
space is the product of the non-negative norms. -/
@[simp]
theorem nnnorm_smulRight_apply (c : E →L[𝕜] 𝕜) (f : Fₗ) : ‖smulRight c f‖₊ = ‖c‖₊ * ‖f‖₊ :=
  NNReal.eq <| c.norm_smulRight_apply f


variable (𝕜 E Fₗ) in
/-- `ContinuousLinearMap.smulRight` as a continuous trilinear map:
`smulRightL (c : E →L[𝕜] 𝕜) (f : F) (x : E) = c x • f`. -/
def smulRightL : (E →L[𝕜] 𝕜) →L[𝕜] Fₗ →L[𝕜] E →L[𝕜] Fₗ :=
  LinearMap.mkContinuous₂
    { toFun := smulRightₗ
      map_add' := fun c₁ c₂ => by
        /-
          𝕜 : Type u_1
          𝕜₂ : Type u_2
          𝕜₃ : Type u_3
          E : Type u_4
          Eₗ : Type u_5
          F : Type u_6
          Fₗ : Type u_7
          G : Type u_8
          Gₗ : Type u_9
          𝓕 : Type u_10
          inst✝³² : SeminormedAddCommGroup E
          inst✝³¹ : SeminormedAddCommGroup Eₗ
          inst✝³⁰ : SeminormedAddCommGroup F
          inst✝²⁹ : SeminormedAddCommGroup Fₗ
          inst✝²⁸ : SeminormedAddCommGroup G
          inst✝²⁷ : SeminormedAddCommGroup Gₗ
          inst✝²⁶ : NontriviallyNormedField 𝕜
          inst✝²⁵ : NontriviallyNormedField 𝕜₂
          inst✝²⁴ : NontriviallyNormedField 𝕜₃
          inst✝²³ : NormedSpace 𝕜 E
          inst✝²² : NormedSpace 𝕜 Eₗ
          inst✝²¹ : NormedSpace 𝕜₂ F
          inst✝²⁰ : NormedSpace 𝕜 Fₗ
          inst✝¹⁹ : NormedSpace 𝕜₃ G
          inst✝¹⁸ : NormedSpace 𝕜 Gₗ
          σ₁₂ : RingHom 𝕜 𝕜₂
          σ₂₃ : RingHom 𝕜₂ 𝕜₃
          σ₁₃ : RingHom 𝕜 𝕜₃
          inst✝¹⁷ : RingHomCompTriple σ₁₂ σ₂₃ σ₁₃
          inst✝¹⁶ : FunLike 𝓕 E F
          σ₂₁ : RingHom 𝕜₂ 𝕜
          inst✝¹⁵ : RingHomInvPair σ₁₂ σ₂₁
          inst✝¹⁴ : RingHomInvPair σ₂₁ σ₁₂
          E' : Type u_11
          F' : Type u_12
          inst✝¹³ : SeminormedAddCommGroup E'
          inst✝¹² : SeminormedAddCommGroup F'
          𝕜₁' : Type u_13
          𝕜₂' : Type u_14
          inst✝¹¹ : NontriviallyNormedField 𝕜₁'
          inst✝¹⁰ : NontriviallyNormedField 𝕜₂'
          inst✝⁹ : NormedSpace 𝕜₁' E'
          inst✝⁸ : NormedSpace 𝕜₂' F'
          σ₁' : RingHom 𝕜₁' 𝕜
          σ₁₃' : RingHom 𝕜₁' 𝕜₃
          σ₂' : RingHom 𝕜₂' 𝕜₂
          σ₂₃' : RingHom 𝕜₂' 𝕜₃
          inst✝⁷ : RingHomCompTriple σ₁' σ₁₃ σ₁₃'
          inst✝⁶ : RingHomCompTriple σ₂' σ₂₃ σ₂₃'
          inst✝⁵ : RingHomIsometric σ₂₃
          inst✝⁴ : RingHomIsometric σ₁₃'
          inst✝³ : RingHomIsometric σ₂₃'
          inst✝² : RingHomIsometric σ₁₃
          inst✝¹ : RingHomIsometric σ₁'
          inst✝ : RingHomIsometric σ₂'
          c₁ c₂ : ContinuousLinearMap (RingHom.id 𝕜) E 𝕜
          ⊢ Eq (HAdd.hAdd c₁ c₂).smulRightₗ (HAdd.hAdd c₁.smulRightₗ c₂.smulRightₗ)
        -/
        ext x
        /-
          case h.h
          𝕜 : Type u_1
          𝕜₂ : Type u_2
          𝕜₃ : Type u_3
          E : Type u_4
          Eₗ : Type u_5
          F : Type u_6
          Fₗ : Type u_7
          G : Type u_8
          Gₗ : Type u_9
          𝓕 : Type u_10
          inst✝³² : SeminormedAddCommGroup E
          inst✝³¹ : SeminormedAddCommGroup Eₗ
          inst✝³⁰ : SeminormedAddCommGroup F
          inst✝²⁹ : SeminormedAddCommGroup Fₗ
          inst✝²⁸ : SeminormedAddCommGroup G
          inst✝²⁷ : SeminormedAddCommGroup Gₗ
          inst✝²⁶ : NontriviallyNormedField 𝕜
          inst✝²⁵ : NontriviallyNormedField 𝕜₂
          inst✝²⁴ : NontriviallyNormedField 𝕜₃
          inst✝²³ : NormedSpace 𝕜 E
          inst✝²² : NormedSpace 𝕜 Eₗ
          inst✝²¹ : NormedSpace 𝕜₂ F
          inst✝²⁰ : NormedSpace 𝕜 Fₗ
          inst✝¹⁹ : NormedSpace 𝕜₃ G
          inst✝¹⁸ : NormedSpace 𝕜 Gₗ
          σ₁₂ : RingHom 𝕜 𝕜₂
          σ₂₃ : RingHom 𝕜₂ 𝕜₃
          σ₁₃ : RingHom 𝕜 𝕜₃
          inst✝¹⁷ : RingHomCompTriple σ₁₂ σ₂₃ σ₁₃
          inst✝¹⁶ : FunLike 𝓕 E F
          σ₂₁ : RingHom 𝕜₂ 𝕜
          inst✝¹⁵ : RingHomInvPair σ₁₂ σ₂₁
          inst✝¹⁴ : RingHomInvPair σ₂₁ σ₁₂
          E' : Type u_11
          F' : Type u_12
          inst✝¹³ : SeminormedAddCommGroup E'
          inst✝¹² : SeminormedAddCommGroup F'
          𝕜₁' : Type u_13
          𝕜₂' : Type u_14
          inst✝¹¹ : NontriviallyNormedField 𝕜₁'
          inst✝¹⁰ : NontriviallyNormedField 𝕜₂'
          inst✝⁹ : NormedSpace 𝕜₁' E'
          inst✝⁸ : NormedSpace 𝕜₂' F'
          σ₁' : RingHom 𝕜₁' 𝕜
          σ₁₃' : RingHom 𝕜₁' 𝕜₃
          σ₂' : RingHom 𝕜₂' 𝕜₂
          σ₂₃' : RingHom 𝕜₂' 𝕜₃
          inst✝⁷ : RingHomCompTriple σ₁' σ₁₃ σ₁₃'
          inst✝⁶ : RingHomCompTriple σ₂' σ₂₃ σ₂₃'
          inst✝⁵ : RingHomIsometric σ₂₃
          inst✝⁴ : RingHomIsometric σ₁₃'
          inst✝³ : RingHomIsometric σ₂₃'
          inst✝² : RingHomIsometric σ₁₃
          inst✝¹ : RingHomIsometric σ₁'
          inst✝ : RingHomIsometric σ₂'
          c₁ c₂ : ContinuousLinearMap (RingHom.id 𝕜) E 𝕜
          x : Fₗ
          x✝ : E
          ⊢ Eq (((HAdd.hAdd c₁ c₂).smulRightₗ x) x✝) (((HAdd.hAdd c₁.smulRightₗ c₂.smulR …
        -/
        simp only [add_smul, coe_smulRightₗ, add_apply, smulRight_apply, LinearMap.add_apply]
        /-
          🎉 no goals
        -/
      map_smul' := fun m c => by
        /-
          𝕜 : Type u_1
          𝕜₂ : Type u_2
          𝕜₃ : Type u_3
          E : Type u_4
          Eₗ : Type u_5
          F : Type u_6
          Fₗ : Type u_7
          G : Type u_8
          Gₗ : Type u_9
          𝓕 : Type u_10
          inst✝³² : SeminormedAddCommGroup E
          inst✝³¹ : SeminormedAddCommGroup Eₗ
          inst✝³⁰ : SeminormedAddCommGroup F
          inst✝²⁹ : SeminormedAddCommGroup Fₗ
          inst✝²⁸ : SeminormedAddCommGroup G
          inst✝²⁷ : SeminormedAddCommGroup Gₗ
          inst✝²⁶ : NontriviallyNormedField 𝕜
          inst✝²⁵ : NontriviallyNormedField 𝕜₂
          inst✝²⁴ : NontriviallyNormedField 𝕜₃
          inst✝²³ : NormedSpace 𝕜 E
          inst✝²² : NormedSpace 𝕜 Eₗ
          inst✝²¹ : NormedSpace 𝕜₂ F
          inst✝²⁰ : NormedSpace 𝕜 Fₗ
          inst✝¹⁹ : NormedSpace 𝕜₃ G
          inst✝¹⁸ : NormedSpace 𝕜 Gₗ
          σ₁₂ : RingHom 𝕜 𝕜₂
          σ₂₃ : RingHom 𝕜₂ 𝕜₃
          σ₁₃ : RingHom 𝕜 𝕜₃
          inst✝¹⁷ : RingHomCompTriple σ₁₂ σ₂₃ σ₁₃
          inst✝¹⁶ : FunLike 𝓕 E F
          σ₂₁ : RingHom 𝕜₂ 𝕜
          inst✝¹⁵ : RingHomInvPair σ₁₂ σ₂₁
          inst✝¹⁴ : RingHomInvPair σ₂₁ σ₁₂
          E' : Type u_11
          F' : Type u_12
          inst✝¹³ : SeminormedAddCommGroup E'
          inst✝¹² : SeminormedAddCommGroup F'
          𝕜₁' : Type u_13
          𝕜₂' : Type u_14
          inst✝¹¹ : NontriviallyNormedField 𝕜₁'
          inst✝¹⁰ : NontriviallyNormedField 𝕜₂'
          inst✝⁹ : NormedSpace 𝕜₁' E'
          inst✝⁸ : NormedSpace 𝕜₂' F'
          σ₁' : RingHom 𝕜₁' 𝕜
          σ₁₃' : RingHom 𝕜₁' 𝕜₃
          σ₂' : RingHom 𝕜₂' 𝕜₂
          σ₂₃' : RingHom 𝕜₂' 𝕜₃
          inst✝⁷ : RingHomCompTriple σ₁' σ₁₃ σ₁₃'
          inst✝⁶ : RingHomCompTriple σ₂' σ₂₃ σ₂₃'
          inst✝⁵ : RingHomIsometric σ₂₃
          inst✝⁴ : RingHomIsometric σ₁₃'
          inst✝³ : RingHomIsometric σ₂₃'
          inst✝² : RingHomIsometric σ₁₃
          inst✝¹ : RingHomIsometric σ₁'
          inst✝ : RingHomIsometric σ₂'
          m : 𝕜
          c : ContinuousLinearMap (RingHom.id 𝕜) E 𝕜
          ⊢ Eq ({ toFun := ContinuousLinearMap.smulRightₗ, map_add' := ⋯ }.toFun (HSMul. …
        -/
        ext x
        /-
          case h.h
          𝕜 : Type u_1
          𝕜₂ : Type u_2
          𝕜₃ : Type u_3
          E : Type u_4
          Eₗ : Type u_5
          F : Type u_6
          Fₗ : Type u_7
          G : Type u_8
          Gₗ : Type u_9
          𝓕 : Type u_10
          inst✝³² : SeminormedAddCommGroup E
          inst✝³¹ : SeminormedAddCommGroup Eₗ
          inst✝³⁰ : SeminormedAddCommGroup F
          inst✝²⁹ : SeminormedAddCommGroup Fₗ
          inst✝²⁸ : SeminormedAddCommGroup G
          inst✝²⁷ : SeminormedAddCommGroup Gₗ
          inst✝²⁶ : NontriviallyNormedField 𝕜
          inst✝²⁵ : NontriviallyNormedField 𝕜₂
          inst✝²⁴ : NontriviallyNormedField 𝕜₃
          inst✝²³ : NormedSpace 𝕜 E
          inst✝²² : NormedSpace 𝕜 Eₗ
          inst✝²¹ : NormedSpace 𝕜₂ F
          inst✝²⁰ : NormedSpace 𝕜 Fₗ
          inst✝¹⁹ : NormedSpace 𝕜₃ G
          inst✝¹⁸ : NormedSpace 𝕜 Gₗ
          σ₁₂ : RingHom 𝕜 𝕜₂
          σ₂₃ : RingHom 𝕜₂ 𝕜₃
          σ₁₃ : RingHom 𝕜 𝕜₃
          inst✝¹⁷ : RingHomCompTriple σ₁₂ σ₂₃ σ₁₃
          inst✝¹⁶ : FunLike 𝓕 E F
          σ₂₁ : RingHom 𝕜₂ 𝕜
          inst✝¹⁵ : RingHomInvPair σ₁₂ σ₂₁
          inst✝¹⁴ : RingHomInvPair σ₂₁ σ₁₂
          E' : Type u_11
          F' : Type u_12
          inst✝¹³ : SeminormedAddCommGroup E'
          inst✝¹² : SeminormedAddCommGroup F'
          𝕜₁' : Type u_13
          𝕜₂' : Type u_14
          inst✝¹¹ : NontriviallyNormedField 𝕜₁'
          inst✝¹⁰ : NontriviallyNormedField 𝕜₂'
          inst✝⁹ : NormedSpace 𝕜₁' E'
          inst✝⁸ : NormedSpace 𝕜₂' F'
          σ₁' : RingHom 𝕜₁' 𝕜
          σ₁₃' : RingHom 𝕜₁' 𝕜₃
          σ₂' : RingHom 𝕜₂' 𝕜₂
          σ₂₃' : RingHom 𝕜₂' 𝕜₃
          inst✝⁷ : RingHomCompTriple σ₁' σ₁₃ σ₁₃'
          inst✝⁶ : RingHomCompTriple σ₂' σ₂₃ σ₂₃'
          inst✝⁵ : RingHomIsometric σ₂₃
          inst✝⁴ : RingHomIsometric σ₁₃'
          inst✝³ : RingHomIsometric σ₂₃'
          inst✝² : RingHomIsometric σ₁₃
          inst✝¹ : RingHomIsometric σ₁'
          inst✝ : RingHomIsometric σ₂'
          m : 𝕜
          c : ContinuousLinearMap (RingHom.id 𝕜) E 𝕜
          x : Fₗ
          x✝ : E
          ⊢ Eq ((({ toFun := ContinuousLinearMap.smulRightₗ, map_add' := ⋯ }.toFun (HSMu …
        -/
        dsimp
        /-
          case h.h
          𝕜 : Type u_1
          𝕜₂ : Type u_2
          𝕜₃ : Type u_3
          E : Type u_4
          Eₗ : Type u_5
          F : Type u_6
          Fₗ : Type u_7
          G : Type u_8
          Gₗ : Type u_9
          𝓕 : Type u_10
          inst✝³² : SeminormedAddCommGroup E
          inst✝³¹ : SeminormedAddCommGroup Eₗ
          inst✝³⁰ : SeminormedAddCommGroup F
          inst✝²⁹ : SeminormedAddCommGroup Fₗ
          inst✝²⁸ : SeminormedAddCommGroup G
          inst✝²⁷ : SeminormedAddCommGroup Gₗ
          inst✝²⁶ : NontriviallyNormedField 𝕜
          inst✝²⁵ : NontriviallyNormedField 𝕜₂
          inst✝²⁴ : NontriviallyNormedField 𝕜₃
          inst✝²³ : NormedSpace 𝕜 E
          inst✝²² : NormedSpace 𝕜 Eₗ
          inst✝²¹ : NormedSpace 𝕜₂ F
          inst✝²⁰ : NormedSpace 𝕜 Fₗ
          inst✝¹⁹ : NormedSpace 𝕜₃ G
          inst✝¹⁸ : NormedSpace 𝕜 Gₗ
          σ₁₂ : RingHom 𝕜 𝕜₂
          σ₂₃ : RingHom 𝕜₂ 𝕜₃
          σ₁₃ : RingHom 𝕜 𝕜₃
          inst✝¹⁷ : RingHomCompTriple σ₁₂ σ₂₃ σ₁₃
          inst✝¹⁶ : FunLike 𝓕 E F
          σ₂₁ : RingHom 𝕜₂ 𝕜
          inst✝¹⁵ : RingHomInvPair σ₁₂ σ₂₁
          inst✝¹⁴ : RingHomInvPair σ₂₁ σ₁₂
          E' : Type u_11
          F' : Type u_12
          inst✝¹³ : SeminormedAddCommGroup E'
          inst✝¹² : SeminormedAddCommGroup F'
          𝕜₁' : Type u_13
          𝕜₂' : Type u_14
          inst✝¹¹ : NontriviallyNormedField 𝕜₁'
          inst✝¹⁰ : NontriviallyNormedField 𝕜₂'
          inst✝⁹ : NormedSpace 𝕜₁' E'
          inst✝⁸ : NormedSpace 𝕜₂' F'
          σ₁' : RingHom 𝕜₁' 𝕜
          σ₁₃' : RingHom 𝕜₁' 𝕜₃
          σ₂' : RingHom 𝕜₂' 𝕜₂
          σ₂₃' : RingHom 𝕜₂' 𝕜₃
          inst✝⁷ : RingHomCompTriple σ₁' σ₁₃ σ₁₃'
          inst✝⁶ : RingHomCompTriple σ₂' σ₂₃ σ₂₃'
          inst✝⁵ : RingHomIsometric σ₂₃
          inst✝⁴ : RingHomIsometric σ₁₃'
          inst✝³ : RingHomIsometric σ₂₃'
          inst✝² : RingHomIsometric σ₁₃
          inst✝¹ : RingHomIsometric σ₁'
          inst✝ : RingHomIsometric σ₂'
          m : 𝕜
          c : ContinuousLinearMap (RingHom.id 𝕜) E 𝕜
          x : Fₗ
          x✝ : E
          ⊢ Eq (HSMul.hSMul (HMul.hMul m (c x✝)) x) (HSMul.hSMul m (HSMul.hSMul (c x✝) x))
        -/
        rw [smul_smul] }
        /-
          🎉 no goals
        -/
    1 fun c x => by
      simp only [coe_smulRightₗ, one_mul, norm_smulRight_apply, LinearMap.coe_mk, AddHom.coe_mk,
        le_refl]



@[simp]
theorem norm_smulRightL_apply (c : E →L[𝕜] 𝕜) (f : Fₗ) : ‖smulRightL 𝕜 E Fₗ c f‖ = ‖c‖ * ‖f‖ :=
  norm_smulRight_apply c f


variable (𝕜) in
/-- Convenience function for restricting the linearity of a bilinear map. -/
def bilinearRestrictScalars (B : E →L[𝕜'] F →L[𝕜'] G) : E →L[𝕜] F →L[𝕜] G :=
  (restrictScalarsL 𝕜' F G 𝕜 𝕜).comp (B.restrictScalars 𝕜)


theorem bilinearRestrictScalars_eq_restrictScalarsL_comp_restrictScalars :
    B.bilinearRestrictScalars 𝕜 = (restrictScalarsL 𝕜' F G 𝕜 𝕜).comp (B.restrictScalars 𝕜) := rfl


theorem bilinearRestrictScalars_eq_restrictScalars_restrictScalarsL_comp :
    B.bilinearRestrictScalars 𝕜 = restrictScalars 𝕜 ((restrictScalarsL 𝕜' F G 𝕜 𝕜').comp B) := rfl


variable (𝕜) in
@[simp]
theorem bilinearRestrictScalars_apply_apply : (B.bilinearRestrictScalars 𝕜) x y = B x y := rfl


@[simp]
theorem norm_bilinearRestrictScalars : ‖B.bilinearRestrictScalars 𝕜‖ = ‖B‖ := rfl


