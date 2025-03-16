theorem toSpanSingleton_homothety (x : E) (c : 𝕜) :
    ‖LinearMap.toSpanSingleton 𝕜 E x c‖ = ‖x‖ * ‖c‖ := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝³ : NormedDivisionRing 𝕜
    inst✝² : SeminormedAddCommGroup E
    inst✝¹ : Module 𝕜 E
    inst✝ : BoundedSMul 𝕜 E
    x : E
    c : 𝕜
    ⊢ Eq (Norm.norm ((LinearMap.toSpanSingleton 𝕜 E x) c)) (HMul.hMul (Norm.norm x …
  -/
  rw [mul_comm]
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝³ : NormedDivisionRing 𝕜
    inst✝² : SeminormedAddCommGroup E
    inst✝¹ : Module 𝕜 E
    inst✝ : BoundedSMul 𝕜 E
    x : E
    c : 𝕜
    ⊢ Eq (Norm.norm ((LinearMap.toSpanSingleton 𝕜 E x) c)) (HMul.hMul (Norm.norm c …
  -/
  exact norm_smul _ _
  /-
    🎉 no goals
  -/


theorem _root_.LinearEquiv.toSpanNonzeroSingleton_homothety (x : E) (h : x ≠ 0) (c : 𝕜) :
    ‖LinearEquiv.toSpanNonzeroSingleton 𝕜 E x h c‖ = ‖x‖ * ‖c‖ :=
  LinearMap.toSpanSingleton_homothety _ _ _


/-- Given a nonzero element `x` of a normed space `E₁` over a field `𝕜`, the natural
    continuous linear equivalence from `E₁` to the span of `x`. -/
noncomputable def toSpanNonzeroSingleton (x : E) (h : x ≠ 0) : 𝕜 ≃L[𝕜] 𝕜 ∙ x :=
  ofHomothety (LinearEquiv.toSpanNonzeroSingleton 𝕜 E x h) ‖x‖ (norm_pos_iff.mpr h)
    (LinearEquiv.toSpanNonzeroSingleton_homothety 𝕜 x h)


/-- Given a nonzero element `x` of a normed space `E₁` over a field `𝕜`, the natural continuous
    linear map from the span of `x` to `𝕜`. -/
noncomputable def coord (x : E) (h : x ≠ 0) : (𝕜 ∙ x) →L[𝕜] 𝕜 :=
  (toSpanNonzeroSingleton 𝕜 x h).symm


@[simp]
theorem coe_toSpanNonzeroSingleton_symm {x : E} (h : x ≠ 0) :
    ⇑(toSpanNonzeroSingleton 𝕜 x h).symm = coord 𝕜 x h :=
  rfl


@[simp]
theorem coord_toSpanNonzeroSingleton {x : E} (h : x ≠ 0) (c : 𝕜) :
    coord 𝕜 x h (toSpanNonzeroSingleton 𝕜 x h c) = c :=
  (toSpanNonzeroSingleton 𝕜 x h).symm_apply_apply c


@[simp]
theorem toSpanNonzeroSingleton_coord {x : E} (h : x ≠ 0) (y : 𝕜 ∙ x) :
    toSpanNonzeroSingleton 𝕜 x h (coord 𝕜 x h y) = y :=
  (toSpanNonzeroSingleton 𝕜 x h).apply_symm_apply y


@[simp]
theorem coord_self (x : E) (h : x ≠ 0) :
    (coord 𝕜 x h) (⟨x, Submodule.mem_span_singleton_self x⟩ : 𝕜 ∙ x) = 1 :=
  LinearEquiv.coord_self 𝕜 E x h


/-- Given a unit element `x` of a normed space `E` over a field `𝕜`, the natural
    linear isometry equivalence from `E` to the span of `x`. -/
noncomputable def toSpanUnitSingleton (x : E) (hx : ‖x‖ = 1) :
    𝕜 ≃ₗᵢ[𝕜] 𝕜 ∙ x where
                                                                /-
                                                                  𝕜 : Type u_1
                                                                  E : Type u_2
                                                                  inst✝³ : NormedDivisionRing 𝕜
                                                                  inst✝² : SeminormedAddCommGroup E
                                                                  inst✝¹ : Module 𝕜 E
                                                                  inst✝ : BoundedSMul 𝕜 E
                                                                  x : E
                                                                  hx : Eq (Norm.norm x) 1
                                                                  ⊢ Ne x 0
                                                                -/
  toLinearEquiv := LinearEquiv.toSpanNonzeroSingleton 𝕜 E x (by aesop)
                                                                /-
                                                                  🎉 no goals
                                                                -/
  norm_map' := by
    /-
      𝕜 : Type u_1
      E : Type u_2
      inst✝³ : NormedDivisionRing 𝕜
      inst✝² : SeminormedAddCommGroup E
      inst✝¹ : Module 𝕜 E
      inst✝ : BoundedSMul 𝕜 E
      x : E
      hx : Eq (Norm.norm x) 1
      ⊢ ∀ (x_1 : 𝕜), Eq (Norm.norm ((LinearEquiv.toSpanNonzeroSingleton 𝕜 E x ⋯) x_1 …
    -/
    intro
    /-
      𝕜 : Type u_1
      E : Type u_2
      inst✝³ : NormedDivisionRing 𝕜
      inst✝² : SeminormedAddCommGroup E
      inst✝¹ : Module 𝕜 E
      inst✝ : BoundedSMul 𝕜 E
      x : E
      hx : Eq (Norm.norm x) 1
      x✝ : 𝕜
      ⊢ Eq (Norm.norm ((LinearEquiv.toSpanNonzeroSingleton 𝕜 E x ⋯) x✝)) (Norm.norm  …
    -/
    rw [LinearEquiv.toSpanNonzeroSingleton_homothety, hx, one_mul]
    /-
      🎉 no goals
    -/


@[simp] theorem toSpanUnitSingleton_apply (x : E) (hx : ‖x‖ = 1) (r : 𝕜) :
                                             /-
                                               𝕜 : Type u_1
                                               E : Type u_2
                                               inst✝³ : NormedDivisionRing 𝕜
                                               inst✝² : SeminormedAddCommGroup E
                                               inst✝¹ : Module 𝕜 E
                                               inst✝ : BoundedSMul 𝕜 E
                                               x : E
                                               hx : Eq (Norm.norm x) 1
                                               r : 𝕜
                                               ⊢ Membership.mem (Submodule.span 𝕜 (Singleton.singleton x)) (HSMul.hSMul r x)
                                             -/
    toSpanUnitSingleton x hx r = (⟨r • x, by aesop⟩ : 𝕜 ∙ x) := by
                                             /-
                                               🎉 no goals
                                             -/
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝³ : NormedDivisionRing 𝕜
    inst✝² : SeminormedAddCommGroup E
    inst✝¹ : Module 𝕜 E
    inst✝ : BoundedSMul 𝕜 E
    x : E
    hx : Eq (Norm.norm x) 1
    r : 𝕜
    ⊢ Eq ((LinearIsometryEquiv.toSpanUnitSingleton x hx) r) ⟨HSMul.hSMul r x, ⋯⟩
  -/
  rfl
  /-
    🎉 no goals
  -/


