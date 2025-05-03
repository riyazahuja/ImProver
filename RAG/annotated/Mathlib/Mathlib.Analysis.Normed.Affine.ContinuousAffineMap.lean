/-- The linear map underlying a continuous affine map is continuous. -/
def contLinear (f : P →ᴬ[R] Q) : V →L[R] W :=
  { f.linear with
    toFun := f.linear
               /-
                 𝕜 : Type u_1
                 R : Type u_2
                 V : Type u_3
                 W : Type u_4
                 W₂ : Type u_5
                 P : Type u_6
                 Q : Type u_7
                 Q₂ : Type u_8
                 inst✝¹⁶ : NormedAddCommGroup V
                 inst✝¹⁵ : MetricSpace P
                 inst✝¹⁴ : NormedAddTorsor V P
                 inst✝¹³ : NormedAddCommGroup W
                 inst✝¹² : MetricSpace Q
                 inst✝¹¹ : NormedAddTorsor W Q
                 inst✝¹⁰ : NormedAddCommGroup W₂
                 inst✝⁹ : MetricSpace Q₂
                 inst✝⁸ : NormedAddTorsor W₂ Q₂
                 inst✝⁷ : NormedField R
                 inst✝⁶ : NormedSpace R V
                 inst✝⁵ : NormedSpace R W
                 inst✝⁴ : NormedSpace R W₂
                 inst✝³ : NontriviallyNormedField 𝕜
                 inst✝² : NormedSpace 𝕜 V
                 inst✝¹ : NormedSpace 𝕜 W
                 inst✝ : NormedSpace 𝕜 W₂
                 f : ContinuousAffineMap R P Q
                 ⊢ Continuous { toFun := ⇑f.linear, map_add' := ⋯, map_smul' := ⋯ }.toFun
               -/
    cont := by rw [AffineMap.continuous_linear_iff]; exact f.cont }
                                                     /-
                                                       🎉 no goals
                                                     -/


@[simp]
theorem coe_contLinear (f : P →ᴬ[R] Q) : (f.contLinear : V → W) = f.linear :=
  rfl


@[simp]
theorem coe_contLinear_eq_linear (f : P →ᴬ[R] Q) :
                                                              /-
                                                                R : Type u_2
                                                                V : Type u_3
                                                                W : Type u_4
                                                                P : Type u_6
                                                                Q : Type u_7
                                                                inst✝⁸ : NormedAddCommGroup V
                                                                inst✝⁷ : MetricSpace P
                                                                inst✝⁶ : NormedAddTorsor V P
                                                                inst✝⁵ : NormedAddCommGroup W
                                                                inst✝⁴ : MetricSpace Q
                                                                inst✝³ : NormedAddTorsor W Q
                                                                inst✝² : NormedField R
                                                                inst✝¹ : NormedSpace R V
                                                                inst✝ : NormedSpace R W
                                                                f : ContinuousAffineMap R P Q
                                                                ⊢ Eq (↑f.contLinear) f.linear
                                                              -/
    (f.contLinear : V →ₗ[R] W) = (f : P →ᵃ[R] Q).linear := by ext; rfl
                                                                   /-
                                                                     🎉 no goals
                                                                   -/


@[simp]
theorem coe_mk_const_linear_eq_linear (f : P →ᵃ[R] Q) (h) :
    ((⟨f, h⟩ : P →ᴬ[R] Q).contLinear : V → W) = f.linear :=
  rfl


theorem coe_linear_eq_coe_contLinear (f : P →ᴬ[R] Q) :
    ((f : P →ᵃ[R] Q).linear : V → W) = (⇑f.contLinear : V → W) :=
  rfl


@[simp]
theorem comp_contLinear (f : P →ᴬ[R] Q) (g : Q →ᴬ[R] Q₂) :
    (g.comp f).contLinear = g.contLinear.comp f.contLinear :=
  rfl


@[simp]
theorem map_vadd (f : P →ᴬ[R] Q) (p : P) (v : V) : f (v +ᵥ p) = f.contLinear v +ᵥ f p :=
  f.map_vadd' p v


@[simp]
theorem contLinear_map_vsub (f : P →ᴬ[R] Q) (p₁ p₂ : P) : f.contLinear (p₁ -ᵥ p₂) = f p₁ -ᵥ f p₂ :=
  f.toAffineMap.linearMap_vsub p₁ p₂


@[simp]
theorem const_contLinear (q : Q) : (const R P q).contLinear = 0 :=
  rfl


theorem contLinear_eq_zero_iff_exists_const (f : P →ᴬ[R] Q) :
    f.contLinear = 0 ↔ ∃ q, f = const R P q := by
  have h₁ : f.contLinear = 0 ↔ (f : P →ᵃ[R] Q).linear = 0 := by
    refine ⟨fun h => ?_, fun h => ?_⟩ <;> ext
    · rw [← coe_contLinear_eq_linear, h]; rfl
    · rw [← coe_linear_eq_coe_contLinear, h]; rfl
  have h₂ : ∀ q : Q, f = const R P q ↔ (f : P →ᵃ[R] Q) = AffineMap.const R P q := by
    intro q
    refine ⟨fun h => ?_, fun h => ?_⟩ <;> ext
    · rw [h]; rfl
    · rw [← coe_to_affineMap, h]; rfl
  /-
    R : Type u_2
    V : Type u_3
    W : Type u_4
    P : Type u_6
    Q : Type u_7
    inst✝⁸ : NormedAddCommGroup V
    inst✝⁷ : MetricSpace P
    inst✝⁶ : NormedAddTorsor V P
    inst✝⁵ : NormedAddCommGroup W
    inst✝⁴ : MetricSpace Q
    inst✝³ : NormedAddTorsor W Q
    inst✝² : NormedField R
    inst✝¹ : NormedSpace R V
    inst✝ : NormedSpace R W
    f : ContinuousAffineMap R P Q
    h₁ : Iff (Eq f.contLinear 0) (Eq f.linear 0)
    h₂ : ∀ (q : Q), Iff (Eq f (ContinuousAffineMap.const R P q)) (Eq f.toAffineMap …
    ⊢ Iff (Eq f.contLinear 0) (Exists fun q => Eq f (ContinuousAffineMap.const R P …
  -/
  simp_rw [h₁, h₂]
  /-
    R : Type u_2
    V : Type u_3
    W : Type u_4
    P : Type u_6
    Q : Type u_7
    inst✝⁸ : NormedAddCommGroup V
    inst✝⁷ : MetricSpace P
    inst✝⁶ : NormedAddTorsor V P
    inst✝⁵ : NormedAddCommGroup W
    inst✝⁴ : MetricSpace Q
    inst✝³ : NormedAddTorsor W Q
    inst✝² : NormedField R
    inst✝¹ : NormedSpace R V
    inst✝ : NormedSpace R W
    f : ContinuousAffineMap R P Q
    h₁ : Iff (Eq f.contLinear 0) (Eq f.linear 0)
    h₂ : ∀ (q : Q), Iff (Eq f (ContinuousAffineMap.const R P q)) (Eq f.toAffineMap …
    ⊢ Iff (Eq f.linear 0) (Exists fun q => Eq f.toAffineMap (AffineMap.const R P q))
  -/
  exact (f : P →ᵃ[R] Q).linear_eq_zero_iff_exists_const
  /-
    🎉 no goals
  -/


@[simp]
theorem to_affine_map_contLinear (f : V →L[R] W) : f.toContinuousAffineMap.contLinear = f := by
  /-
    R : Type u_2
    V : Type u_3
    W : Type u_4
    inst✝⁴ : NormedAddCommGroup V
    inst✝³ : NormedAddCommGroup W
    inst✝² : NormedField R
    inst✝¹ : NormedSpace R V
    inst✝ : NormedSpace R W
    f : ContinuousLinearMap (RingHom.id R) V W
    ⊢ Eq f.toContinuousAffineMap.contLinear f
  -/
  ext
  /-
    case h
    R : Type u_2
    V : Type u_3
    W : Type u_4
    inst✝⁴ : NormedAddCommGroup V
    inst✝³ : NormedAddCommGroup W
    inst✝² : NormedField R
    inst✝¹ : NormedSpace R V
    inst✝ : NormedSpace R W
    f : ContinuousLinearMap (RingHom.id R) V W
    x✝ : V
    ⊢ Eq (f.toContinuousAffineMap.contLinear x✝) (f x✝)
  -/
  rfl
  /-
    🎉 no goals
  -/


@[simp]
theorem zero_contLinear : (0 : P →ᴬ[R] W).contLinear = 0 :=
  rfl


@[simp]
theorem add_contLinear (f g : P →ᴬ[R] W) : (f + g).contLinear = f.contLinear + g.contLinear :=
  rfl


@[simp]
theorem sub_contLinear (f g : P →ᴬ[R] W) : (f - g).contLinear = f.contLinear - g.contLinear :=
  rfl


@[simp]
theorem neg_contLinear (f : P →ᴬ[R] W) : (-f).contLinear = -f.contLinear :=
  rfl


@[simp]
theorem smul_contLinear (t : R) (f : P →ᴬ[R] W) : (t • f).contLinear = t • f.contLinear :=
  rfl


theorem decomp (f : V →ᴬ[R] W) : (f : V → W) = f.contLinear + Function.const V (f 0) := by
  /-
    R : Type u_2
    V : Type u_3
    W : Type u_4
    inst✝⁴ : NormedAddCommGroup V
    inst✝³ : NormedAddCommGroup W
    inst✝² : NormedField R
    inst✝¹ : NormedSpace R V
    inst✝ : NormedSpace R W
    f : ContinuousAffineMap R V W
    ⊢ Eq (⇑f) (HAdd.hAdd (⇑f.contLinear) (Function.const V (f 0)))
  -/
  rcases f with ⟨f, h⟩
  rw [coe_mk_const_linear_eq_linear, coe_mk, f.decomp, Pi.add_apply, LinearMap.map_zero, zero_add,
    ← Function.const_def]


/-- Note that unlike the operator norm for linear maps, this norm is _not_ submultiplicative:
we do _not_ necessarily have `‖f.comp g‖ ≤ ‖f‖ * ‖g‖`. See `norm_comp_le` for what we can say. -/
noncomputable instance hasNorm : Norm (V →ᴬ[𝕜] W) :=
  ⟨fun f => max ‖f 0‖ ‖f.contLinear‖⟩


theorem norm_def : ‖f‖ = max ‖f 0‖ ‖f.contLinear‖ :=
  rfl


theorem norm_contLinear_le : ‖f.contLinear‖ ≤ ‖f‖ :=
  le_max_right _ _


theorem norm_image_zero_le : ‖f 0‖ ≤ ‖f‖ :=
  le_max_left _ _


@[simp]
theorem norm_eq (h : f 0 = 0) : ‖f‖ = ‖f.contLinear‖ :=
  calc
                                         /-
                                           𝕜 : Type u_1
                                           V : Type u_3
                                           W : Type u_4
                                           inst✝⁴ : NormedAddCommGroup V
                                           inst✝³ : NormedAddCommGroup W
                                           inst✝² : NontriviallyNormedField 𝕜
                                           inst✝¹ : NormedSpace 𝕜 V
                                           inst✝ : NormedSpace 𝕜 W
                                           f : ContinuousAffineMap 𝕜 V W
                                           h : Eq (f 0) 0
                                           ⊢ Eq (Norm.norm f) (Max.max (Norm.norm (f 0)) (Norm.norm f.contLinear))
                                         -/
    ‖f‖ = max ‖f 0‖ ‖f.contLinear‖ := by rw [norm_def]
                                         /-
                                           🎉 no goals
                                         -/
                                   /-
                                     𝕜 : Type u_1
                                     V : Type u_3
                                     W : Type u_4
                                     inst✝⁴ : NormedAddCommGroup V
                                     inst✝³ : NormedAddCommGroup W
                                     inst✝² : NontriviallyNormedField 𝕜
                                     inst✝¹ : NormedSpace 𝕜 V
                                     inst✝ : NormedSpace 𝕜 W
                                     f : ContinuousAffineMap 𝕜 V W
                                     h : Eq (f 0) 0
                                     ⊢ Eq (Max.max (Norm.norm (f 0)) (Norm.norm f.contLinear)) (Max.max 0 (Norm.nor …
                                   -/
    _ = max 0 ‖f.contLinear‖ := by rw [h, norm_zero]
                                   /-
                                     🎉 no goals
                                   -/
    _ = ‖f.contLinear‖ := max_eq_right (norm_nonneg _)


noncomputable instance : NormedAddCommGroup (V →ᴬ[𝕜] W) :=
  AddGroupNorm.toNormedAddCommGroup
    { toFun := fun f => max ‖f 0‖ ‖f.contLinear‖
                      /-
                        𝕜 : Type u_1
                        R : Type u_2
                        V : Type u_3
                        W : Type u_4
                        W₂ : Type u_5
                        P : Type u_6
                        Q : Type u_7
                        Q₂ : Type u_8
                        inst✝¹⁶ : NormedAddCommGroup V
                        inst✝¹⁵ : MetricSpace P
                        inst✝¹⁴ : NormedAddTorsor V P
                        inst✝¹³ : NormedAddCommGroup W
                        inst✝¹² : MetricSpace Q
                        inst✝¹¹ : NormedAddTorsor W Q
                        inst✝¹⁰ : NormedAddCommGroup W₂
                        inst✝⁹ : MetricSpace Q₂
                        inst✝⁸ : NormedAddTorsor W₂ Q₂
                        inst✝⁷ : NormedField R
                        inst✝⁶ : NormedSpace R V
                        inst✝⁵ : NormedSpace R W
                        inst✝⁴ : NormedSpace R W₂
                        inst✝³ : NontriviallyNormedField 𝕜
                        inst✝² : NormedSpace 𝕜 V
                        inst✝¹ : NormedSpace 𝕜 W
                        inst✝ : NormedSpace 𝕜 W₂
                        f : ContinuousAffineMap 𝕜 V W
                        ⊢ Eq ((fun f => Max.max (Norm.norm (f 0)) (Norm.norm f.contLinear)) 0) 0
                      -/
      map_zero' := by simp [(ContinuousAffineMap.zero_apply)]
                      /-
                        🎉 no goals
                      -/
      neg' := fun f => by
        /-
          𝕜 : Type u_1
          R : Type u_2
          V : Type u_3
          W : Type u_4
          W₂ : Type u_5
          P : Type u_6
          Q : Type u_7
          Q₂ : Type u_8
          inst✝¹⁶ : NormedAddCommGroup V
          inst✝¹⁵ : MetricSpace P
          inst✝¹⁴ : NormedAddTorsor V P
          inst✝¹³ : NormedAddCommGroup W
          inst✝¹² : MetricSpace Q
          inst✝¹¹ : NormedAddTorsor W Q
          inst✝¹⁰ : NormedAddCommGroup W₂
          inst✝⁹ : MetricSpace Q₂
          inst✝⁸ : NormedAddTorsor W₂ Q₂
          inst✝⁷ : NormedField R
          inst✝⁶ : NormedSpace R V
          inst✝⁵ : NormedSpace R W
          inst✝⁴ : NormedSpace R W₂
          inst✝³ : NontriviallyNormedField 𝕜
          inst✝² : NormedSpace 𝕜 V
          inst✝¹ : NormedSpace 𝕜 W
          inst✝ : NormedSpace 𝕜 W₂
          f✝ f : ContinuousAffineMap 𝕜 V W
          ⊢ Eq ((fun f => Max.max (Norm.norm (f 0)) (Norm.norm f.contLinear)) (Neg.neg f …
        -/
        simp [(ContinuousAffineMap.neg_apply)]
        /-
          𝕜 : Type u_1
          R : Type u_2
          V : Type u_3
          W : Type u_4
          W₂ : Type u_5
          P : Type u_6
          Q : Type u_7
          Q₂ : Type u_8
          inst✝¹⁶ : NormedAddCommGroup V
          inst✝¹⁵ : MetricSpace P
          inst✝¹⁴ : NormedAddTorsor V P
          inst✝¹³ : NormedAddCommGroup W
          inst✝¹² : MetricSpace Q
          inst✝¹¹ : NormedAddTorsor W Q
          inst✝¹⁰ : NormedAddCommGroup W₂
          inst✝⁹ : MetricSpace Q₂
          inst✝⁸ : NormedAddTorsor W₂ Q₂
          inst✝⁷ : NormedField R
          inst✝⁶ : NormedSpace R V
          inst✝⁵ : NormedSpace R W
          inst✝⁴ : NormedSpace R W₂
          inst✝³ : NontriviallyNormedField 𝕜
          inst✝² : NormedSpace 𝕜 V
          inst✝¹ : NormedSpace 𝕜 W
          inst✝ : NormedSpace 𝕜 W₂
          f✝ f g : ContinuousAffineMap 𝕜 V W
          ⊢ LE.le ((fun f => Max.max (Norm.norm (f 0)) (Norm.norm f.contLinear)) (HAdd.h …
        -/
        /-
          🎉 no goals
        -/
      add_le' := fun f g => by
        simp only [coe_add, max_le_iff, Pi.add_apply, add_contLinear]
        exact
          ⟨(norm_add_le _ _).trans (add_le_add (le_max_left _ _) (le_max_left _ _)),
            (norm_add_le _ _).trans (add_le_add (le_max_right _ _) (le_max_right _ _))⟩
      eq_zero_of_map_eq_zero' := fun f h₀ => by
        /-
          𝕜 : Type u_1
          R : Type u_2
          V : Type u_3
          W : Type u_4
          W₂ : Type u_5
          P : Type u_6
          Q : Type u_7
          Q₂ : Type u_8
          inst✝¹⁶ : NormedAddCommGroup V
          inst✝¹⁵ : MetricSpace P
          inst✝¹⁴ : NormedAddTorsor V P
          inst✝¹³ : NormedAddCommGroup W
          inst✝¹² : MetricSpace Q
          inst✝¹¹ : NormedAddTorsor W Q
          inst✝¹⁰ : NormedAddCommGroup W₂
          inst✝⁹ : MetricSpace Q₂
          inst✝⁸ : NormedAddTorsor W₂ Q₂
          inst✝⁷ : NormedField R
          inst✝⁶ : NormedSpace R V
          inst✝⁵ : NormedSpace R W
          inst✝⁴ : NormedSpace R W₂
          inst✝³ : NontriviallyNormedField 𝕜
          inst✝² : NormedSpace 𝕜 V
          inst✝¹ : NormedSpace 𝕜 W
          inst✝ : NormedSpace 𝕜 W₂
          f✝ f : ContinuousAffineMap 𝕜 V W
          h₀ : Eq ({ toFun := fun f => Max.max (Norm.norm (f 0)) (Norm.norm f.contLinear …
          ⊢ Eq f 0
        -/
        rcases max_eq_iff.mp h₀ with (⟨h₁, h₂⟩ | ⟨h₁, h₂⟩) <;> rw [h₁] at h₂
          /-
            case inl.intro
            𝕜 : Type u_1
            R : Type u_2
            V : Type u_3
            W : Type u_4
            W₂ : Type u_5
            P : Type u_6
            Q : Type u_7
            Q₂ : Type u_8
            inst✝¹⁶ : NormedAddCommGroup V
            inst✝¹⁵ : MetricSpace P
            inst✝¹⁴ : NormedAddTorsor V P
            inst✝¹³ : NormedAddCommGroup W
            inst✝¹² : MetricSpace Q
            inst✝¹¹ : NormedAddTorsor W Q
            inst✝¹⁰ : NormedAddCommGroup W₂
            inst✝⁹ : MetricSpace Q₂
            inst✝⁸ : NormedAddTorsor W₂ Q₂
            inst✝⁷ : NormedField R
            inst✝⁶ : NormedSpace R V
            inst✝⁵ : NormedSpace R W
            inst✝⁴ : NormedSpace R W₂
            inst✝³ : NontriviallyNormedField 𝕜
            inst✝² : NormedSpace 𝕜 V
            inst✝¹ : NormedSpace 𝕜 W
            inst✝ : NormedSpace 𝕜 W₂
            f✝ f : ContinuousAffineMap 𝕜 V W
            h₀ : Eq ({ toFun := fun f => Max.max (Norm.norm (f 0)) (Norm.norm f.contLinear …
            h₁ : Eq (Norm.norm (f 0)) 0
            h₂ : LE.le (Norm.norm f.contLinear) 0
            ⊢ Eq f 0
          -/
        · rw [norm_le_zero_iff, contLinear_eq_zero_iff_exists_const] at h₂
          /-
            case inl.intro
            𝕜 : Type u_1
            R : Type u_2
            V : Type u_3
            W : Type u_4
            W₂ : Type u_5
            P : Type u_6
            Q : Type u_7
            Q₂ : Type u_8
            inst✝¹⁶ : NormedAddCommGroup V
            inst✝¹⁵ : MetricSpace P
            inst✝¹⁴ : NormedAddTorsor V P
            inst✝¹³ : NormedAddCommGroup W
            inst✝¹² : MetricSpace Q
            inst✝¹¹ : NormedAddTorsor W Q
            inst✝¹⁰ : NormedAddCommGroup W₂
            inst✝⁹ : MetricSpace Q₂
            inst✝⁸ : NormedAddTorsor W₂ Q₂
            inst✝⁷ : NormedField R
            inst✝⁶ : NormedSpace R V
            inst✝⁵ : NormedSpace R W
            inst✝⁴ : NormedSpace R W₂
            inst✝³ : NontriviallyNormedField 𝕜
            inst✝² : NormedSpace 𝕜 V
            inst✝¹ : NormedSpace 𝕜 W
            inst✝ : NormedSpace 𝕜 W₂
            f✝ f : ContinuousAffineMap 𝕜 V W
            h₀ : Eq ({ toFun := fun f => Max.max (Norm.norm (f 0)) (Norm.norm f.contLinear …
            h₁ : Eq (Norm.norm (f 0)) 0
            h₂ : Exists fun q => Eq f (ContinuousAffineMap.const 𝕜 V q)
            ⊢ Eq f 0
          -/
          obtain ⟨q, rfl⟩ := h₂
          /-
            case inl.intro.intro
            𝕜 : Type u_1
            R : Type u_2
            V : Type u_3
            W : Type u_4
            W₂ : Type u_5
            P : Type u_6
            Q : Type u_7
            Q₂ : Type u_8
            inst✝¹⁶ : NormedAddCommGroup V
            inst✝¹⁵ : MetricSpace P
            inst✝¹⁴ : NormedAddTorsor V P
            inst✝¹³ : NormedAddCommGroup W
            inst✝¹² : MetricSpace Q
            inst✝¹¹ : NormedAddTorsor W Q
            inst✝¹⁰ : NormedAddCommGroup W₂
            inst✝⁹ : MetricSpace Q₂
            inst✝⁸ : NormedAddTorsor W₂ Q₂
            inst✝⁷ : NormedField R
            inst✝⁶ : NormedSpace R V
            inst✝⁵ : NormedSpace R W
            inst✝⁴ : NormedSpace R W₂
            inst✝³ : NontriviallyNormedField 𝕜
            inst✝² : NormedSpace 𝕜 V
            inst✝¹ : NormedSpace 𝕜 W
            inst✝ : NormedSpace 𝕜 W₂
            f : ContinuousAffineMap 𝕜 V W
            q : W
            h₀ : Eq ({ toFun := fun f => Max.max (Norm.norm (f 0)) (Norm.norm f.contLinear …
            h₁ : Eq (Norm.norm ((ContinuousAffineMap.const 𝕜 V q) 0)) 0
            ⊢ Eq (ContinuousAffineMap.const 𝕜 V q) 0
          -/
          simp only [norm_eq_zero, coe_const, Function.const_apply] at h₁
          /-
            case inl.intro.intro
            𝕜 : Type u_1
            R : Type u_2
            V : Type u_3
            W : Type u_4
            W₂ : Type u_5
            P : Type u_6
            Q : Type u_7
            Q₂ : Type u_8
            inst✝¹⁶ : NormedAddCommGroup V
            inst✝¹⁵ : MetricSpace P
            inst✝¹⁴ : NormedAddTorsor V P
            inst✝¹³ : NormedAddCommGroup W
            inst✝¹² : MetricSpace Q
            inst✝¹¹ : NormedAddTorsor W Q
            inst✝¹⁰ : NormedAddCommGroup W₂
            inst✝⁹ : MetricSpace Q₂
            inst✝⁸ : NormedAddTorsor W₂ Q₂
            inst✝⁷ : NormedField R
            inst✝⁶ : NormedSpace R V
            inst✝⁵ : NormedSpace R W
            inst✝⁴ : NormedSpace R W₂
            inst✝³ : NontriviallyNormedField 𝕜
            inst✝² : NormedSpace 𝕜 V
            inst✝¹ : NormedSpace 𝕜 W
            inst✝ : NormedSpace 𝕜 W₂
            f : ContinuousAffineMap 𝕜 V W
            q : W
            h₀ : Eq ({ toFun := fun f => Max.max (Norm.norm (f 0)) (Norm.norm f.contLinear …
            h₁ : Eq q 0
            ⊢ Eq (ContinuousAffineMap.const 𝕜 V q) 0
          -/
          rw [h₁]
          /-
            case inl.intro.intro
            𝕜 : Type u_1
            R : Type u_2
            V : Type u_3
            W : Type u_4
            W₂ : Type u_5
            P : Type u_6
            Q : Type u_7
            Q₂ : Type u_8
            inst✝¹⁶ : NormedAddCommGroup V
            inst✝¹⁵ : MetricSpace P
            inst✝¹⁴ : NormedAddTorsor V P
            inst✝¹³ : NormedAddCommGroup W
            inst✝¹² : MetricSpace Q
            inst✝¹¹ : NormedAddTorsor W Q
            inst✝¹⁰ : NormedAddCommGroup W₂
            inst✝⁹ : MetricSpace Q₂
            inst✝⁸ : NormedAddTorsor W₂ Q₂
            inst✝⁷ : NormedField R
            inst✝⁶ : NormedSpace R V
            inst✝⁵ : NormedSpace R W
            inst✝⁴ : NormedSpace R W₂
            inst✝³ : NontriviallyNormedField 𝕜
            inst✝² : NormedSpace 𝕜 V
            inst✝¹ : NormedSpace 𝕜 W
            inst✝ : NormedSpace 𝕜 W₂
            f : ContinuousAffineMap 𝕜 V W
            q : W
            h₀ : Eq ({ toFun := fun f => Max.max (Norm.norm (f 0)) (Norm.norm f.contLinear …
            h₁ : Eq q 0
            ⊢ Eq (ContinuousAffineMap.const 𝕜 V 0) 0
          -/
          rfl
          /-
            🎉 no goals
          -/
          /-
            case inr.intro
            𝕜 : Type u_1
            R : Type u_2
            V : Type u_3
            W : Type u_4
            W₂ : Type u_5
            P : Type u_6
            Q : Type u_7
            Q₂ : Type u_8
            inst✝¹⁶ : NormedAddCommGroup V
            inst✝¹⁵ : MetricSpace P
            inst✝¹⁴ : NormedAddTorsor V P
            inst✝¹³ : NormedAddCommGroup W
            inst✝¹² : MetricSpace Q
            inst✝¹¹ : NormedAddTorsor W Q
            inst✝¹⁰ : NormedAddCommGroup W₂
            inst✝⁹ : MetricSpace Q₂
            inst✝⁸ : NormedAddTorsor W₂ Q₂
            inst✝⁷ : NormedField R
            inst✝⁶ : NormedSpace R V
            inst✝⁵ : NormedSpace R W
            inst✝⁴ : NormedSpace R W₂
            inst✝³ : NontriviallyNormedField 𝕜
            inst✝² : NormedSpace 𝕜 V
            inst✝¹ : NormedSpace 𝕜 W
            inst✝ : NormedSpace 𝕜 W₂
            f✝ f : ContinuousAffineMap 𝕜 V W
            h₀ : Eq ({ toFun := fun f => Max.max (Norm.norm (f 0)) (Norm.norm f.contLinear …
            h₁ : Eq (Norm.norm f.contLinear) 0
            h₂ : LE.le (Norm.norm (f 0)) 0
            ⊢ Eq f 0
          -/
        · rw [norm_eq_zero, contLinear_eq_zero_iff_exists_const] at h₁
          /-
            case inr.intro
            𝕜 : Type u_1
            R : Type u_2
            V : Type u_3
            W : Type u_4
            W₂ : Type u_5
            P : Type u_6
            Q : Type u_7
            Q₂ : Type u_8
            inst✝¹⁶ : NormedAddCommGroup V
            inst✝¹⁵ : MetricSpace P
            inst✝¹⁴ : NormedAddTorsor V P
            inst✝¹³ : NormedAddCommGroup W
            inst✝¹² : MetricSpace Q
            inst✝¹¹ : NormedAddTorsor W Q
            inst✝¹⁰ : NormedAddCommGroup W₂
            inst✝⁹ : MetricSpace Q₂
            inst✝⁸ : NormedAddTorsor W₂ Q₂
            inst✝⁷ : NormedField R
            inst✝⁶ : NormedSpace R V
            inst✝⁵ : NormedSpace R W
            inst✝⁴ : NormedSpace R W₂
            inst✝³ : NontriviallyNormedField 𝕜
            inst✝² : NormedSpace 𝕜 V
            inst✝¹ : NormedSpace 𝕜 W
            inst✝ : NormedSpace 𝕜 W₂
            f✝ f : ContinuousAffineMap 𝕜 V W
            h₀ : Eq ({ toFun := fun f => Max.max (Norm.norm (f 0)) (Norm.norm f.contLinear …
            h₁ : Exists fun q => Eq f (ContinuousAffineMap.const 𝕜 V q)
            h₂ : LE.le (Norm.norm (f 0)) 0
            ⊢ Eq f 0
          -/
          obtain ⟨q, rfl⟩ := h₁
          /-
            case inr.intro.intro
            𝕜 : Type u_1
            R : Type u_2
            V : Type u_3
            W : Type u_4
            W₂ : Type u_5
            P : Type u_6
            Q : Type u_7
            Q₂ : Type u_8
            inst✝¹⁶ : NormedAddCommGroup V
            inst✝¹⁵ : MetricSpace P
            inst✝¹⁴ : NormedAddTorsor V P
            inst✝¹³ : NormedAddCommGroup W
            inst✝¹² : MetricSpace Q
            inst✝¹¹ : NormedAddTorsor W Q
            inst✝¹⁰ : NormedAddCommGroup W₂
            inst✝⁹ : MetricSpace Q₂
            inst✝⁸ : NormedAddTorsor W₂ Q₂
            inst✝⁷ : NormedField R
            inst✝⁶ : NormedSpace R V
            inst✝⁵ : NormedSpace R W
            inst✝⁴ : NormedSpace R W₂
            inst✝³ : NontriviallyNormedField 𝕜
            inst✝² : NormedSpace 𝕜 V
            inst✝¹ : NormedSpace 𝕜 W
            inst✝ : NormedSpace 𝕜 W₂
            f : ContinuousAffineMap 𝕜 V W
            q : W
            h₀ : Eq ({ toFun := fun f => Max.max (Norm.norm (f 0)) (Norm.norm f.contLinear …
            h₂ : LE.le (Norm.norm ((ContinuousAffineMap.const 𝕜 V q) 0)) 0
            ⊢ Eq (ContinuousAffineMap.const 𝕜 V q) 0
          -/
          simp only [norm_le_zero_iff, coe_const, Function.const_apply] at h₂
          /-
            case inr.intro.intro
            𝕜 : Type u_1
            R : Type u_2
            V : Type u_3
            W : Type u_4
            W₂ : Type u_5
            P : Type u_6
            Q : Type u_7
            Q₂ : Type u_8
            inst✝¹⁶ : NormedAddCommGroup V
            inst✝¹⁵ : MetricSpace P
            inst✝¹⁴ : NormedAddTorsor V P
            inst✝¹³ : NormedAddCommGroup W
            inst✝¹² : MetricSpace Q
            inst✝¹¹ : NormedAddTorsor W Q
            inst✝¹⁰ : NormedAddCommGroup W₂
            inst✝⁹ : MetricSpace Q₂
            inst✝⁸ : NormedAddTorsor W₂ Q₂
            inst✝⁷ : NormedField R
            inst✝⁶ : NormedSpace R V
            inst✝⁵ : NormedSpace R W
            inst✝⁴ : NormedSpace R W₂
            inst✝³ : NontriviallyNormedField 𝕜
            inst✝² : NormedSpace 𝕜 V
            inst✝¹ : NormedSpace 𝕜 W
            inst✝ : NormedSpace 𝕜 W₂
            f : ContinuousAffineMap 𝕜 V W
            q : W
            h₀ : Eq ({ toFun := fun f => Max.max (Norm.norm (f 0)) (Norm.norm f.contLinear …
            h₂ : Eq q 0
            ⊢ Eq (ContinuousAffineMap.const 𝕜 V q) 0
          -/
          rw [h₂]
          /-
            case inr.intro.intro
            𝕜 : Type u_1
            R : Type u_2
            V : Type u_3
            W : Type u_4
            W₂ : Type u_5
            P : Type u_6
            Q : Type u_7
            Q₂ : Type u_8
            inst✝¹⁶ : NormedAddCommGroup V
            inst✝¹⁵ : MetricSpace P
            inst✝¹⁴ : NormedAddTorsor V P
            inst✝¹³ : NormedAddCommGroup W
            inst✝¹² : MetricSpace Q
            inst✝¹¹ : NormedAddTorsor W Q
            inst✝¹⁰ : NormedAddCommGroup W₂
            inst✝⁹ : MetricSpace Q₂
            inst✝⁸ : NormedAddTorsor W₂ Q₂
            inst✝⁷ : NormedField R
            inst✝⁶ : NormedSpace R V
            inst✝⁵ : NormedSpace R W
            inst✝⁴ : NormedSpace R W₂
            inst✝³ : NontriviallyNormedField 𝕜
            inst✝² : NormedSpace 𝕜 V
            inst✝¹ : NormedSpace 𝕜 W
            inst✝ : NormedSpace 𝕜 W₂
            f : ContinuousAffineMap 𝕜 V W
            q : W
            h₀ : Eq ({ toFun := fun f => Max.max (Norm.norm (f 0)) (Norm.norm f.contLinear …
            h₂ : Eq q 0
            ⊢ Eq (ContinuousAffineMap.const 𝕜 V 0) 0
          -/
          rfl }
          /-
            🎉 no goals
          -/


set_option maxSynthPendingDepth 2 in
instance : NormedSpace 𝕜 (V →ᴬ[𝕜] W) where
  norm_smul_le t f := by
    simp only [norm_def, coe_smul, Pi.smul_apply, norm_smul, smul_contLinear,
      ← mul_max_of_nonneg _ _ (norm_nonneg t), le_refl]


theorem norm_comp_le (g : W₂ →ᴬ[𝕜] V) : ‖f.comp g‖ ≤ ‖f‖ * ‖g‖ + ‖f 0‖ := by
  /-
    𝕜 : Type u_1
    V : Type u_3
    W : Type u_4
    W₂ : Type u_5
    inst✝⁶ : NormedAddCommGroup V
    inst✝⁵ : NormedAddCommGroup W
    inst✝⁴ : NormedAddCommGroup W₂
    inst✝³ : NontriviallyNormedField 𝕜
    inst✝² : NormedSpace 𝕜 V
    inst✝¹ : NormedSpace 𝕜 W
    inst✝ : NormedSpace 𝕜 W₂
    f : ContinuousAffineMap 𝕜 V W
    g : ContinuousAffineMap 𝕜 W₂ V
    ⊢ LE.le (Norm.norm (f.comp g)) (HAdd.hAdd (HMul.hMul (Norm.norm f) (Norm.norm  …
  -/
  rw [norm_def, max_le_iff]
  /-
    𝕜 : Type u_1
    V : Type u_3
    W : Type u_4
    W₂ : Type u_5
    inst✝⁶ : NormedAddCommGroup V
    inst✝⁵ : NormedAddCommGroup W
    inst✝⁴ : NormedAddCommGroup W₂
    inst✝³ : NontriviallyNormedField 𝕜
    inst✝² : NormedSpace 𝕜 V
    inst✝¹ : NormedSpace 𝕜 W
    inst✝ : NormedSpace 𝕜 W₂
    f : ContinuousAffineMap 𝕜 V W
    g : ContinuousAffineMap 𝕜 W₂ V
    ⊢ And (LE.le (Norm.norm ((f.comp g) 0)) (HAdd.hAdd (HMul.hMul (Norm.norm f) (N …
  -/
  constructor
  · calc
      ‖f.comp g 0‖ = ‖f (g 0)‖ := by simp
      _ = ‖f.contLinear (g 0) + f 0‖ := by rw [f.decomp]; simp
      _ ≤ ‖f.contLinear‖ * ‖g 0‖ + ‖f 0‖ :=
        ((norm_add_le _ _).trans (add_le_add_right (f.contLinear.le_opNorm _) _))
      _ ≤ ‖f‖ * ‖g‖ + ‖f 0‖ :=
        add_le_add_right
          (mul_le_mul f.norm_contLinear_le g.norm_image_zero_le (norm_nonneg _) (norm_nonneg _)) _
  · calc
      ‖(f.comp g).contLinear‖ ≤ ‖f.contLinear‖ * ‖g.contLinear‖ :=
        (g.comp_contLinear f).symm ▸ f.contLinear.opNorm_comp_le _
      _ ≤ ‖f‖ * ‖g‖ :=
        (mul_le_mul f.norm_contLinear_le g.norm_contLinear_le (norm_nonneg _) (norm_nonneg _))
      _ ≤ ‖f‖ * ‖g‖ + ‖f 0‖ := by rw [le_add_iff_nonneg_right]; apply norm_nonneg


/-- The space of affine maps between two normed spaces is linearly isometric to the product of the
codomain with the space of linear maps, by taking the value of the affine map at `(0 : V)` and the
linear part. -/
def toConstProdContinuousLinearMap : (V →ᴬ[𝕜] W) ≃ₗᵢ[𝕜] W × (V →L[𝕜] W) where
  toFun f := ⟨f 0, f.contLinear⟩
  invFun p := p.2.toContinuousAffineMap + const 𝕜 V p.1
  left_inv f := by
    /-
      𝕜 : Type u_1
      R : Type u_2
      V : Type u_3
      W : Type u_4
      W₂ : Type u_5
      P : Type u_6
      Q : Type u_7
      Q₂ : Type u_8
      inst✝¹⁶ : NormedAddCommGroup V
      inst✝¹⁵ : MetricSpace P
      inst✝¹⁴ : NormedAddTorsor V P
      inst✝¹³ : NormedAddCommGroup W
      inst✝¹² : MetricSpace Q
      inst✝¹¹ : NormedAddTorsor W Q
      inst✝¹⁰ : NormedAddCommGroup W₂
      inst✝⁹ : MetricSpace Q₂
      inst✝⁸ : NormedAddTorsor W₂ Q₂
      inst✝⁷ : NormedField R
      inst✝⁶ : NormedSpace R V
      inst✝⁵ : NormedSpace R W
      inst✝⁴ : NormedSpace R W₂
      inst✝³ : NontriviallyNormedField 𝕜
      inst✝² : NormedSpace 𝕜 V
      inst✝¹ : NormedSpace 𝕜 W
      inst✝ : NormedSpace 𝕜 W₂
      f✝ f : ContinuousAffineMap 𝕜 V W
      ⊢ Eq ((fun p => HAdd.hAdd p.2.toContinuousAffineMap (ContinuousAffineMap.const …
    -/
    ext
    /-
      case h
      𝕜 : Type u_1
      R : Type u_2
      V : Type u_3
      W : Type u_4
      W₂ : Type u_5
      P : Type u_6
      Q : Type u_7
      Q₂ : Type u_8
      inst✝¹⁶ : NormedAddCommGroup V
      inst✝¹⁵ : MetricSpace P
      inst✝¹⁴ : NormedAddTorsor V P
      inst✝¹³ : NormedAddCommGroup W
      inst✝¹² : MetricSpace Q
      inst✝¹¹ : NormedAddTorsor W Q
      inst✝¹⁰ : NormedAddCommGroup W₂
      inst✝⁹ : MetricSpace Q₂
      inst✝⁸ : NormedAddTorsor W₂ Q₂
      inst✝⁷ : NormedField R
      inst✝⁶ : NormedSpace R V
      inst✝⁵ : NormedSpace R W
      inst✝⁴ : NormedSpace R W₂
      inst✝³ : NontriviallyNormedField 𝕜
      inst✝² : NormedSpace 𝕜 V
      inst✝¹ : NormedSpace 𝕜 W
      inst✝ : NormedSpace 𝕜 W₂
      f✝ f : ContinuousAffineMap 𝕜 V W
      x✝ : V
      ⊢ Eq (((fun p => HAdd.hAdd p.2.toContinuousAffineMap (ContinuousAffineMap.cons …
    -/
    rw [f.decomp]
    /-
      case h
      𝕜 : Type u_1
      R : Type u_2
      V : Type u_3
      W : Type u_4
      W₂ : Type u_5
      P : Type u_6
      Q : Type u_7
      Q₂ : Type u_8
      inst✝¹⁶ : NormedAddCommGroup V
      inst✝¹⁵ : MetricSpace P
      inst✝¹⁴ : NormedAddTorsor V P
      inst✝¹³ : NormedAddCommGroup W
      inst✝¹² : MetricSpace Q
      inst✝¹¹ : NormedAddTorsor W Q
      inst✝¹⁰ : NormedAddCommGroup W₂
      inst✝⁹ : MetricSpace Q₂
      inst✝⁸ : NormedAddTorsor W₂ Q₂
      inst✝⁷ : NormedField R
      inst✝⁶ : NormedSpace R V
      inst✝⁵ : NormedSpace R W
      inst✝⁴ : NormedSpace R W₂
      inst✝³ : NontriviallyNormedField 𝕜
      inst✝² : NormedSpace 𝕜 V
      inst✝¹ : NormedSpace 𝕜 W
      inst✝ : NormedSpace 𝕜 W₂
      f✝ f : ContinuousAffineMap 𝕜 V W
      x✝ : V
      ⊢ Eq (((fun p => HAdd.hAdd p.2.toContinuousAffineMap (ContinuousAffineMap.cons …
    -/
    simp only [coe_add, ContinuousLinearMap.coe_toContinuousAffineMap, Pi.add_apply, coe_const]
    /-
      🎉 no goals
    -/
                  /-
                    𝕜 : Type u_1
                    R : Type u_2
                    V : Type u_3
                    W : Type u_4
                    W₂ : Type u_5
                    P : Type u_6
                    Q : Type u_7
                    Q₂ : Type u_8
                    inst✝¹⁶ : NormedAddCommGroup V
                    inst✝¹⁵ : MetricSpace P
                    inst✝¹⁴ : NormedAddTorsor V P
                    inst✝¹³ : NormedAddCommGroup W
                    inst✝¹² : MetricSpace Q
                    inst✝¹¹ : NormedAddTorsor W Q
                    inst✝¹⁰ : NormedAddCommGroup W₂
                    inst✝⁹ : MetricSpace Q₂
                    inst✝⁸ : NormedAddTorsor W₂ Q₂
                    inst✝⁷ : NormedField R
                    inst✝⁶ : NormedSpace R V
                    inst✝⁵ : NormedSpace R W
                    inst✝⁴ : NormedSpace R W₂
                    inst✝³ : NontriviallyNormedField 𝕜
                    inst✝² : NormedSpace 𝕜 V
                    inst✝¹ : NormedSpace 𝕜 W
                    inst✝ : NormedSpace 𝕜 W₂
                    f : ContinuousAffineMap 𝕜 V W
                    ⊢ Function.RightInverse (fun p => HAdd.hAdd p.2.toContinuousAffineMap (Continu …
                  -/
                                         /-
                                           🎉 no goals
                                         -/
  right_inv := by rintro ⟨v, f⟩; ext <;> simp
                                         /-
                                           🎉 no goals
                                         -/
  map_add' _ _ := rfl
  map_smul' _ _ := rfl
  norm_map' _ := rfl


@[simp]
theorem toConstProdContinuousLinearMap_fst (f : V →ᴬ[𝕜] W) :
    (toConstProdContinuousLinearMap 𝕜 V W f).fst = f 0 :=
  rfl


@[simp]
theorem toConstProdContinuousLinearMap_snd (f : V →ᴬ[𝕜] W) :
    (toConstProdContinuousLinearMap 𝕜 V W f).snd = f.contLinear :=
  rfl


