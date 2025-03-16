local notation "∞" => (⊤ : ℕ∞)


/-- Left-invariant global derivations.

A global derivation is left-invariant if it is equal to its pullback along left multiplication by
an arbitrary element of `G`.
-/
structure LeftInvariantDerivation extends Derivation 𝕜 C^∞⟮I, G; 𝕜⟯ C^∞⟮I, G; 𝕜⟯ where
  left_invariant'' :
    ∀ g, 𝒅ₕ (smoothLeftMul_one I g) (Derivation.evalAt 1 toDerivation) =
      Derivation.evalAt g toDerivation


instance : Coe (LeftInvariantDerivation I G) (Derivation 𝕜 C^∞⟮I, G; 𝕜⟯ C^∞⟮I, G; 𝕜⟯) :=
  ⟨toDerivation⟩


theorem toDerivation_injective :
    Function.Injective (toDerivation : LeftInvariantDerivation I G → _) :=
                  /-
                    𝕜 : Type u_1
                    inst✝⁷ : NontriviallyNormedField 𝕜
                    E : Type u_2
                    inst✝⁶ : NormedAddCommGroup E
                    inst✝⁵ : NormedSpace 𝕜 E
                    H : Type u_3
                    inst✝⁴ : TopologicalSpace H
                    I : ModelWithCorners 𝕜 E H
                    G : Type u_4
                    inst✝³ : TopologicalSpace G
                    inst✝² : ChartedSpace H G
                    inst✝¹ : Monoid G
                    inst✝ : SmoothMul I G
                    X Y : LeftInvariantDerivation I G
                    h : Eq ↑X ↑Y
                    ⊢ Eq X Y
                  -/
  fun X Y h => by cases X; cases Y; congr
                                    /-
                                      🎉 no goals
                                    -/


instance : FunLike (LeftInvariantDerivation I G) C^∞⟮I, G; 𝕜⟯ C^∞⟮I, G; 𝕜⟯ where
  coe f := f.toDerivation
  coe_injective' _ _ h := toDerivation_injective <| DFunLike.ext' h


instance : LinearMapClass (LeftInvariantDerivation I G) 𝕜 C^∞⟮I, G; 𝕜⟯ C^∞⟮I, G; 𝕜⟯ where
  map_add f := map_add f.1
  map_smulₛₗ f := map_smul f.1.1


theorem toFun_eq_coe : X.toFun = ⇑X :=
  rfl

-- Porting note: now LHS is the same as RHS


theorem coe_injective :
    @Function.Injective (LeftInvariantDerivation I G) (_ → C^∞⟮I, G; 𝕜⟯) DFunLike.coe :=
  DFunLike.coe_injective


@[ext]
theorem ext (h : ∀ f, X f = Y f) : X = Y := DFunLike.ext _ _ h


theorem coe_derivation :
    ⇑(X : Derivation 𝕜 C^∞⟮I, G; 𝕜⟯ C^∞⟮I, G; 𝕜⟯) = (X : C^∞⟮I, G; 𝕜⟯ → C^∞⟮I, G; 𝕜⟯) :=
  rfl


/-- Premature version of the lemma. Prefer using `left_invariant` instead. -/
theorem left_invariant' :
    𝒅ₕ (smoothLeftMul_one I g) (Derivation.evalAt (1 : G) ↑X) = Derivation.evalAt g ↑X :=
  left_invariant'' X g

-- Porting note: was `@[simp]` but `_root_.map_add` can prove it now

protected theorem map_add : X (f + f') = X f + X f' := map_add X f f'

-- Porting note: was `@[simp]` but `_root_.map_zero` can prove it now

protected theorem map_zero : X 0 = 0 := map_zero X

-- Porting note: was `@[simp]` but `_root_.map_neg` can prove it now

protected theorem map_neg : X (-f) = -X f := map_neg X f

-- Porting note: was `@[simp]` but `_root_.map_sub` can prove it now

protected theorem map_sub : X (f - f') = X f - X f' := map_sub X f f'

-- Porting note: was `@[simp]` but `_root_.map_smul` can prove it now

protected theorem map_smul : X (r • f) = r • X f := map_smul X r f


@[simp]
theorem leibniz : X (f * f') = f • X f' + f' • X f :=
  X.leibniz' _ _


instance : Zero (LeftInvariantDerivation I G) :=
                   /-
                     𝕜 : Type u_1
                     inst✝⁷ : NontriviallyNormedField 𝕜
                     E : Type u_2
                     inst✝⁶ : NormedAddCommGroup E
                     inst✝⁵ : NormedSpace 𝕜 E
                     H : Type u_3
                     inst✝⁴ : TopologicalSpace H
                     I : ModelWithCorners 𝕜 E H
                     G : Type u_4
                     inst✝³ : TopologicalSpace G
                     inst✝² : ChartedSpace H G
                     inst✝¹ : Monoid G
                     inst✝ : SmoothMul I G
                     g✝ h : G
                     r : 𝕜
                     X Y : LeftInvariantDerivation I G
                     f f' : ContMDiffMap I (modelWithCornersSelf 𝕜 𝕜) G 𝕜 Top.top
                     g : G
                     ⊢ Eq ((hfdifferential ⋯) ((Derivation.evalAt 1) 0)) ((Derivation.evalAt g) 0)
                   -/
  ⟨⟨0, fun g => by simp only [_root_.map_zero]⟩⟩
                   /-
                     🎉 no goals
                   -/


instance : Inhabited (LeftInvariantDerivation I G) :=
  ⟨0⟩


instance : Add (LeftInvariantDerivation I G) where
  add X Y :=
    ⟨X + Y, fun g => by
      /-
        𝕜 : Type u_1
        inst✝⁷ : NontriviallyNormedField 𝕜
        E : Type u_2
        inst✝⁶ : NormedAddCommGroup E
        inst✝⁵ : NormedSpace 𝕜 E
        H : Type u_3
        inst✝⁴ : TopologicalSpace H
        I : ModelWithCorners 𝕜 E H
        G : Type u_4
        inst✝³ : TopologicalSpace G
        inst✝² : ChartedSpace H G
        inst✝¹ : Monoid G
        inst✝ : SmoothMul I G
        g✝ h : G
        r : 𝕜
        X✝ Y✝ : LeftInvariantDerivation I G
        f f' : ContMDiffMap I (modelWithCornersSelf 𝕜 𝕜) G 𝕜 Top.top
        X Y : LeftInvariantDerivation I G
        g : G
        ⊢ Eq ((hfdifferential ⋯) ((Derivation.evalAt 1) (HAdd.hAdd ↑X ↑Y))) ((Derivati …
      -/
      simp only [map_add, Derivation.coe_add, left_invariant', Pi.add_apply]⟩
      /-
        🎉 no goals
      -/


instance : Neg (LeftInvariantDerivation I G) where
                            /-
                              𝕜 : Type u_1
                              inst✝⁷ : NontriviallyNormedField 𝕜
                              E : Type u_2
                              inst✝⁶ : NormedAddCommGroup E
                              inst✝⁵ : NormedSpace 𝕜 E
                              H : Type u_3
                              inst✝⁴ : TopologicalSpace H
                              I : ModelWithCorners 𝕜 E H
                              G : Type u_4
                              inst✝³ : TopologicalSpace G
                              inst✝² : ChartedSpace H G
                              inst✝¹ : Monoid G
                              inst✝ : SmoothMul I G
                              g✝ h : G
                              r : 𝕜
                              X✝ Y : LeftInvariantDerivation I G
                              f f' : ContMDiffMap I (modelWithCornersSelf 𝕜 𝕜) G 𝕜 Top.top
                              X : LeftInvariantDerivation I G
                              g : G
                              ⊢ Eq ((hfdifferential ⋯) ((Derivation.evalAt 1) (Neg.neg ↑X))) ((Derivation.ev …
                            -/
  neg X := ⟨-X, fun g => by simp [left_invariant']⟩
                            /-
                              🎉 no goals
                            -/


instance : Sub (LeftInvariantDerivation I G) where
                                 /-
                                   𝕜 : Type u_1
                                   inst✝⁷ : NontriviallyNormedField 𝕜
                                   E : Type u_2
                                   inst✝⁶ : NormedAddCommGroup E
                                   inst✝⁵ : NormedSpace 𝕜 E
                                   H : Type u_3
                                   inst✝⁴ : TopologicalSpace H
                                   I : ModelWithCorners 𝕜 E H
                                   G : Type u_4
                                   inst✝³ : TopologicalSpace G
                                   inst✝² : ChartedSpace H G
                                   inst✝¹ : Monoid G
                                   inst✝ : SmoothMul I G
                                   g✝ h : G
                                   r : 𝕜
                                   X✝ Y✝ : LeftInvariantDerivation I G
                                   f f' : ContMDiffMap I (modelWithCornersSelf 𝕜 𝕜) G 𝕜 Top.top
                                   X Y : LeftInvariantDerivation I G
                                   g : G
                                   ⊢ Eq ((hfdifferential ⋯) ((Derivation.evalAt 1) (HSub.hSub ↑X ↑Y))) ((Derivati …
                                 -/
  sub X Y := ⟨X - Y, fun g => by simp [left_invariant']⟩
                                 /-
                                   🎉 no goals
                                 -/


@[simp]
theorem coe_add : ⇑(X + Y) = X + Y :=
  rfl


@[simp]
theorem coe_zero : ⇑(0 : LeftInvariantDerivation I G) = 0 :=
  rfl


@[simp]
theorem coe_neg : ⇑(-X) = -X :=
  rfl


@[simp]
theorem coe_sub : ⇑(X - Y) = X - Y :=
  rfl


@[simp, norm_cast]
theorem lift_add : (↑(X + Y) : Derivation 𝕜 C^∞⟮I, G; 𝕜⟯ C^∞⟮I, G; 𝕜⟯) = X + Y :=
  rfl


@[simp, norm_cast]
theorem lift_zero :
    (↑(0 : LeftInvariantDerivation I G) : Derivation 𝕜 C^∞⟮I, G; 𝕜⟯ C^∞⟮I, G; 𝕜⟯) = 0 :=
  rfl


instance hasNatScalar : SMul ℕ (LeftInvariantDerivation I G) where
                                    /-
                                      𝕜 : Type u_1
                                      inst✝⁷ : NontriviallyNormedField 𝕜
                                      E : Type u_2
                                      inst✝⁶ : NormedAddCommGroup E
                                      inst✝⁵ : NormedSpace 𝕜 E
                                      H : Type u_3
                                      inst✝⁴ : TopologicalSpace H
                                      I : ModelWithCorners 𝕜 E H
                                      G : Type u_4
                                      inst✝³ : TopologicalSpace G
                                      inst✝² : ChartedSpace H G
                                      inst✝¹ : Monoid G
                                      inst✝ : SmoothMul I G
                                      g✝ h : G
                                      r✝ : 𝕜
                                      X✝ Y : LeftInvariantDerivation I G
                                      f f' : ContMDiffMap I (modelWithCornersSelf 𝕜 𝕜) G 𝕜 Top.top
                                      r : Nat
                                      X : LeftInvariantDerivation I G
                                      g : G
                                      ⊢ Eq ((hfdifferential ⋯) ((Derivation.evalAt 1) (HSMul.hSMul r ↑X))) ((Derivat …
                                    -/
  smul r X := ⟨r • X.1, fun g => by simp_rw [LinearMap.map_smul_of_tower _ r, left_invariant']⟩
                                    /-
                                      🎉 no goals
                                    -/


instance hasIntScalar : SMul ℤ (LeftInvariantDerivation I G) where
                                    /-
                                      𝕜 : Type u_1
                                      inst✝⁷ : NontriviallyNormedField 𝕜
                                      E : Type u_2
                                      inst✝⁶ : NormedAddCommGroup E
                                      inst✝⁵ : NormedSpace 𝕜 E
                                      H : Type u_3
                                      inst✝⁴ : TopologicalSpace H
                                      I : ModelWithCorners 𝕜 E H
                                      G : Type u_4
                                      inst✝³ : TopologicalSpace G
                                      inst✝² : ChartedSpace H G
                                      inst✝¹ : Monoid G
                                      inst✝ : SmoothMul I G
                                      g✝ h : G
                                      r✝ : 𝕜
                                      X✝ Y : LeftInvariantDerivation I G
                                      f f' : ContMDiffMap I (modelWithCornersSelf 𝕜 𝕜) G 𝕜 Top.top
                                      r : Int
                                      X : LeftInvariantDerivation I G
                                      g : G
                                      ⊢ Eq ((hfdifferential ⋯) ((Derivation.evalAt 1) (HSMul.hSMul r ↑X))) ((Derivat …
                                    -/
  smul r X := ⟨r • X.1, fun g => by simp_rw [LinearMap.map_smul_of_tower _ r, left_invariant']⟩
                                    /-
                                      🎉 no goals
                                    -/


instance : AddCommGroup (LeftInvariantDerivation I G) :=
  coe_injective.addCommGroup _ coe_zero coe_add coe_neg coe_sub (fun _ _ => rfl) fun _ _ => rfl


instance : SMul 𝕜 (LeftInvariantDerivation I G) where
                                    /-
                                      𝕜 : Type u_1
                                      inst✝⁷ : NontriviallyNormedField 𝕜
                                      E : Type u_2
                                      inst✝⁶ : NormedAddCommGroup E
                                      inst✝⁵ : NormedSpace 𝕜 E
                                      H : Type u_3
                                      inst✝⁴ : TopologicalSpace H
                                      I : ModelWithCorners 𝕜 E H
                                      G : Type u_4
                                      inst✝³ : TopologicalSpace G
                                      inst✝² : ChartedSpace H G
                                      inst✝¹ : Monoid G
                                      inst✝ : SmoothMul I G
                                      g✝ h : G
                                      r✝ : 𝕜
                                      X✝ Y : LeftInvariantDerivation I G
                                      f f' : ContMDiffMap I (modelWithCornersSelf 𝕜 𝕜) G 𝕜 Top.top
                                      r : 𝕜
                                      X : LeftInvariantDerivation I G
                                      g : G
                                      ⊢ Eq ((hfdifferential ⋯) ((Derivation.evalAt 1) (HSMul.hSMul r ↑X))) ((Derivat …
                                    -/
  smul r X := ⟨r • X.1, fun g => by simp_rw [LinearMap.map_smul, left_invariant']⟩
                                    /-
                                      🎉 no goals
                                    -/


@[simp]
theorem coe_smul : ⇑(r • X) = r • ⇑X :=
  rfl


@[simp]
theorem lift_smul (k : 𝕜) : (k • X).1 = k • X.1 :=
  rfl


/-- The coercion to function is a monoid homomorphism. -/
@[simps]
def coeFnAddMonoidHom : LeftInvariantDerivation I G →+ C^∞⟮I, G; 𝕜⟯ → C^∞⟮I, G; 𝕜⟯ :=
  ⟨⟨DFunLike.coe, coe_zero⟩, coe_add⟩


instance : Module 𝕜 (LeftInvariantDerivation I G) :=
  coe_injective.module _ (coeFnAddMonoidHom I G) coe_smul


/-- Evaluation at a point for left invariant derivation. Same thing as for generic global
derivations (`Derivation.evalAt`). -/
def evalAt : LeftInvariantDerivation I G →ₗ[𝕜] PointDerivation I g where
  toFun X := Derivation.evalAt g X.1
  map_add' _ _ := rfl
  map_smul' _ _ := rfl


theorem evalAt_apply : evalAt g X f = (X f) g :=
  rfl


@[simp]
theorem evalAt_coe : Derivation.evalAt g ↑X = evalAt g X :=
  rfl


theorem left_invariant : 𝒅ₕ (smoothLeftMul_one I g) (evalAt (1 : G) X) = evalAt g X :=
  X.left_invariant'' g


theorem evalAt_mul : evalAt (g * h) X = 𝒅ₕ (L_apply I g h) (evalAt h X) := by
  /-
    𝕜 : Type u_1
    inst✝⁷ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝⁶ : NormedAddCommGroup E
    inst✝⁵ : NormedSpace 𝕜 E
    H : Type u_3
    inst✝⁴ : TopologicalSpace H
    I : ModelWithCorners 𝕜 E H
    G : Type u_4
    inst✝³ : TopologicalSpace G
    inst✝² : ChartedSpace H G
    inst✝¹ : Monoid G
    inst✝ : SmoothMul I G
    g h : G
    X : LeftInvariantDerivation I G
    ⊢ Eq ((LeftInvariantDerivation.evalAt (HMul.hMul g h)) X) ((hfdifferential ⋯)  …
  -/
  ext f
  rw [← left_invariant, hfdifferential_apply, hfdifferential_apply, L_mul, fdifferential_comp,
    fdifferential_apply]
  -- Porting note: more aggressive here
  /-
    case H
    𝕜 : Type u_1
    inst✝⁷ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝⁶ : NormedAddCommGroup E
    inst✝⁵ : NormedSpace 𝕜 E
    H : Type u_3
    inst✝⁴ : TopologicalSpace H
    I : ModelWithCorners 𝕜 E H
    G : Type u_4
    inst✝³ : TopologicalSpace G
    inst✝² : ChartedSpace H G
    inst✝¹ : Monoid G
    inst✝ : SmoothMul I G
    g h : G
    X : LeftInvariantDerivation I G
    f : PointedSmoothMap 𝕜 I G Top.top (HMul.hMul g h)
    ⊢ Eq ((((fdifferential (smoothLeftMul I g) ((smoothLeftMul I h) 1)).comp (fdif …
  -/
  erw [LinearMap.comp_apply]
  -- This used to be `rw`, but we need `erw` after https://github.com/leanprover/lean4/pull/2644
  /-
    case H
    𝕜 : Type u_1
    inst✝⁷ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝⁶ : NormedAddCommGroup E
    inst✝⁵ : NormedSpace 𝕜 E
    H : Type u_3
    inst✝⁴ : TopologicalSpace H
    I : ModelWithCorners 𝕜 E H
    G : Type u_4
    inst✝³ : TopologicalSpace G
    inst✝² : ChartedSpace H G
    inst✝¹ : Monoid G
    inst✝ : SmoothMul I G
    g h : G
    X : LeftInvariantDerivation I G
    f : PointedSmoothMap 𝕜 I G Top.top (HMul.hMul g h)
    ⊢ Eq (((fdifferential (smoothLeftMul I g) ((smoothLeftMul I h) 1)) ((fdifferen …
  -/
  erw [fdifferential_apply, ← hfdifferential_apply, left_invariant]
  /-
    🎉 no goals
  -/


theorem comp_L : (X f).comp (𝑳 I g) = X (f.comp (𝑳 I g)) := by
  /-
    𝕜 : Type u_1
    inst✝⁷ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝⁶ : NormedAddCommGroup E
    inst✝⁵ : NormedSpace 𝕜 E
    H : Type u_3
    inst✝⁴ : TopologicalSpace H
    I : ModelWithCorners 𝕜 E H
    G : Type u_4
    inst✝³ : TopologicalSpace G
    inst✝² : ChartedSpace H G
    inst✝¹ : Monoid G
    inst✝ : SmoothMul I G
    g : G
    X : LeftInvariantDerivation I G
    f : ContMDiffMap I (modelWithCornersSelf 𝕜 𝕜) G 𝕜 Top.top
    ⊢ Eq ((X f).comp (smoothLeftMul I g)) (X (f.comp (smoothLeftMul I g)))
  -/
  ext h
  rw [ContMDiffMap.comp_apply, L_apply, ← evalAt_apply, evalAt_mul, hfdifferential_apply,
    fdifferential_apply, evalAt_apply]


instance : Bracket (LeftInvariantDerivation I G) (LeftInvariantDerivation I G) where
  bracket X Y :=
    ⟨⁅(X : Derivation 𝕜 C^∞⟮I, G; 𝕜⟯ C^∞⟮I, G; 𝕜⟯), Y⁆, fun g => by
      /-
        𝕜 : Type u_1
        inst✝⁷ : NontriviallyNormedField 𝕜
        E : Type u_2
        inst✝⁶ : NormedAddCommGroup E
        inst✝⁵ : NormedSpace 𝕜 E
        H : Type u_3
        inst✝⁴ : TopologicalSpace H
        I : ModelWithCorners 𝕜 E H
        G : Type u_4
        inst✝³ : TopologicalSpace G
        inst✝² : ChartedSpace H G
        inst✝¹ : Monoid G
        inst✝ : SmoothMul I G
        g✝ h : G
        r : 𝕜
        X✝ Y✝ : LeftInvariantDerivation I G
        f f' : ContMDiffMap I (modelWithCornersSelf 𝕜 𝕜) G 𝕜 Top.top
        X Y : LeftInvariantDerivation I G
        g : G
        ⊢ Eq ((hfdifferential ⋯) ((Derivation.evalAt 1) (Bracket.bracket ↑X ↑Y))) ((De …
      -/
      ext f
      /-
        case H
        𝕜 : Type u_1
        inst✝⁷ : NontriviallyNormedField 𝕜
        E : Type u_2
        inst✝⁶ : NormedAddCommGroup E
        inst✝⁵ : NormedSpace 𝕜 E
        H : Type u_3
        inst✝⁴ : TopologicalSpace H
        I : ModelWithCorners 𝕜 E H
        G : Type u_4
        inst✝³ : TopologicalSpace G
        inst✝² : ChartedSpace H G
        inst✝¹ : Monoid G
        inst✝ : SmoothMul I G
        g✝ h : G
        r : 𝕜
        X✝ Y✝ : LeftInvariantDerivation I G
        f✝ f' : ContMDiffMap I (modelWithCornersSelf 𝕜 𝕜) G 𝕜 Top.top
        X Y : LeftInvariantDerivation I G
        g : G
        f : PointedSmoothMap 𝕜 I G Top.top g
        ⊢ Eq (((hfdifferential ⋯) ((Derivation.evalAt 1) (Bracket.bracket ↑X ↑Y))) f)  …
      -/
      have hX := Derivation.congr_fun (left_invariant' g X) (Y f)
      /-
        case H
        𝕜 : Type u_1
        inst✝⁷ : NontriviallyNormedField 𝕜
        E : Type u_2
        inst✝⁶ : NormedAddCommGroup E
        inst✝⁵ : NormedSpace 𝕜 E
        H : Type u_3
        inst✝⁴ : TopologicalSpace H
        I : ModelWithCorners 𝕜 E H
        G : Type u_4
        inst✝³ : TopologicalSpace G
        inst✝² : ChartedSpace H G
        inst✝¹ : Monoid G
        inst✝ : SmoothMul I G
        g✝ h : G
        r : 𝕜
        X✝ Y✝ : LeftInvariantDerivation I G
        f✝ f' : ContMDiffMap I (modelWithCornersSelf 𝕜 𝕜) G 𝕜 Top.top
        X Y : LeftInvariantDerivation I G
        g : G
        f : PointedSmoothMap 𝕜 I G Top.top g
        hX : Eq (((hfdifferential ⋯) ((Derivation.evalAt 1) ↑X)) (Y f)) (((Derivation. …
        ⊢ Eq (((hfdifferential ⋯) ((Derivation.evalAt 1) (Bracket.bracket ↑X ↑Y))) f)  …
      -/
      have hY := Derivation.congr_fun (left_invariant' g Y) (X f)
      /-
        case H
        𝕜 : Type u_1
        inst✝⁷ : NontriviallyNormedField 𝕜
        E : Type u_2
        inst✝⁶ : NormedAddCommGroup E
        inst✝⁵ : NormedSpace 𝕜 E
        H : Type u_3
        inst✝⁴ : TopologicalSpace H
        I : ModelWithCorners 𝕜 E H
        G : Type u_4
        inst✝³ : TopologicalSpace G
        inst✝² : ChartedSpace H G
        inst✝¹ : Monoid G
        inst✝ : SmoothMul I G
        g✝ h : G
        r : 𝕜
        X✝ Y✝ : LeftInvariantDerivation I G
        f✝ f' : ContMDiffMap I (modelWithCornersSelf 𝕜 𝕜) G 𝕜 Top.top
        X Y : LeftInvariantDerivation I G
        g : G
        f : PointedSmoothMap 𝕜 I G Top.top g
        hX : Eq (((hfdifferential ⋯) ((Derivation.evalAt 1) ↑X)) (Y f)) (((Derivation. …
        hY : Eq (((hfdifferential ⋯) ((Derivation.evalAt 1) ↑Y)) (X f)) (((Derivation. …
        ⊢ Eq (((hfdifferential ⋯) ((Derivation.evalAt 1) (Bracket.bracket ↑X ↑Y))) f)  …
      -/
      rw [hfdifferential_apply, fdifferential_apply, Derivation.evalAt_apply] at hX hY ⊢
      /-
        case H
        𝕜 : Type u_1
        inst✝⁷ : NontriviallyNormedField 𝕜
        E : Type u_2
        inst✝⁶ : NormedAddCommGroup E
        inst✝⁵ : NormedSpace 𝕜 E
        H : Type u_3
        inst✝⁴ : TopologicalSpace H
        I : ModelWithCorners 𝕜 E H
        G : Type u_4
        inst✝³ : TopologicalSpace G
        inst✝² : ChartedSpace H G
        inst✝¹ : Monoid G
        inst✝ : SmoothMul I G
        g✝ h : G
        r : 𝕜
        X✝ Y✝ : LeftInvariantDerivation I G
        f✝ f' : ContMDiffMap I (modelWithCornersSelf 𝕜 𝕜) G 𝕜 Top.top
        X Y : LeftInvariantDerivation I G
        g : G
        f : PointedSmoothMap 𝕜 I G Top.top g
        hX : Eq ((↑X ((Y f).comp (smoothLeftMul I g))) 1) (((Derivation.evalAt g) ↑X)  …
        hY : Eq ((↑Y ((X f).comp (smoothLeftMul I g))) 1) (((Derivation.evalAt g) ↑Y)  …
        ⊢ Eq (((Bracket.bracket ↑X ↑Y) (ContMDiffMap.comp f (smoothLeftMul I g))) 1) ( …
      -/
      rw [comp_L] at hX hY
      /-
        case H
        𝕜 : Type u_1
        inst✝⁷ : NontriviallyNormedField 𝕜
        E : Type u_2
        inst✝⁶ : NormedAddCommGroup E
        inst✝⁵ : NormedSpace 𝕜 E
        H : Type u_3
        inst✝⁴ : TopologicalSpace H
        I : ModelWithCorners 𝕜 E H
        G : Type u_4
        inst✝³ : TopologicalSpace G
        inst✝² : ChartedSpace H G
        inst✝¹ : Monoid G
        inst✝ : SmoothMul I G
        g✝ h : G
        r : 𝕜
        X✝ Y✝ : LeftInvariantDerivation I G
        f✝ f' : ContMDiffMap I (modelWithCornersSelf 𝕜 𝕜) G 𝕜 Top.top
        X Y : LeftInvariantDerivation I G
        g : G
        f : PointedSmoothMap 𝕜 I G Top.top g
        hX : Eq ((↑X (Y (ContMDiffMap.comp f (smoothLeftMul I g)))) 1) (((Derivation.e …
        hY : Eq ((↑Y (X (ContMDiffMap.comp f (smoothLeftMul I g)))) 1) (((Derivation.e …
        ⊢ Eq (((Bracket.bracket ↑X ↑Y) (ContMDiffMap.comp f (smoothLeftMul I g))) 1) ( …
      -/
      rw [Derivation.commutator_apply, SmoothMap.coe_sub, Pi.sub_apply, coe_derivation]
      /-
        case H
        𝕜 : Type u_1
        inst✝⁷ : NontriviallyNormedField 𝕜
        E : Type u_2
        inst✝⁶ : NormedAddCommGroup E
        inst✝⁵ : NormedSpace 𝕜 E
        H : Type u_3
        inst✝⁴ : TopologicalSpace H
        I : ModelWithCorners 𝕜 E H
        G : Type u_4
        inst✝³ : TopologicalSpace G
        inst✝² : ChartedSpace H G
        inst✝¹ : Monoid G
        inst✝ : SmoothMul I G
        g✝ h : G
        r : 𝕜
        X✝ Y✝ : LeftInvariantDerivation I G
        f✝ f' : ContMDiffMap I (modelWithCornersSelf 𝕜 𝕜) G 𝕜 Top.top
        X Y : LeftInvariantDerivation I G
        g : G
        f : PointedSmoothMap 𝕜 I G Top.top g
        hX : Eq ((↑X (Y (ContMDiffMap.comp f (smoothLeftMul I g)))) 1) (((Derivation.e …
        hY : Eq ((↑Y (X (ContMDiffMap.comp f (smoothLeftMul I g)))) 1) (((Derivation.e …
        ⊢ Eq (HSub.hSub ((X (↑Y (ContMDiffMap.comp f (smoothLeftMul I g)))) 1) ((↑Y (X …
      -/
      rw [coe_derivation] at hX hY ⊢
      /-
        case H
        𝕜 : Type u_1
        inst✝⁷ : NontriviallyNormedField 𝕜
        E : Type u_2
        inst✝⁶ : NormedAddCommGroup E
        inst✝⁵ : NormedSpace 𝕜 E
        H : Type u_3
        inst✝⁴ : TopologicalSpace H
        I : ModelWithCorners 𝕜 E H
        G : Type u_4
        inst✝³ : TopologicalSpace G
        inst✝² : ChartedSpace H G
        inst✝¹ : Monoid G
        inst✝ : SmoothMul I G
        g✝ h : G
        r : 𝕜
        X✝ Y✝ : LeftInvariantDerivation I G
        f✝ f' : ContMDiffMap I (modelWithCornersSelf 𝕜 𝕜) G 𝕜 Top.top
        X Y : LeftInvariantDerivation I G
        g : G
        f : PointedSmoothMap 𝕜 I G Top.top g
        hX : Eq ((X (Y (ContMDiffMap.comp f (smoothLeftMul I g)))) 1) (((Derivation.ev …
        hY : Eq ((Y (X (ContMDiffMap.comp f (smoothLeftMul I g)))) 1) (((Derivation.ev …
        ⊢ Eq (HSub.hSub ((X (Y (ContMDiffMap.comp f (smoothLeftMul I g)))) 1) ((Y (X ( …
      -/
      rw [hX, hY]
      /-
        case H
        𝕜 : Type u_1
        inst✝⁷ : NontriviallyNormedField 𝕜
        E : Type u_2
        inst✝⁶ : NormedAddCommGroup E
        inst✝⁵ : NormedSpace 𝕜 E
        H : Type u_3
        inst✝⁴ : TopologicalSpace H
        I : ModelWithCorners 𝕜 E H
        G : Type u_4
        inst✝³ : TopologicalSpace G
        inst✝² : ChartedSpace H G
        inst✝¹ : Monoid G
        inst✝ : SmoothMul I G
        g✝ h : G
        r : 𝕜
        X✝ Y✝ : LeftInvariantDerivation I G
        f✝ f' : ContMDiffMap I (modelWithCornersSelf 𝕜 𝕜) G 𝕜 Top.top
        X Y : LeftInvariantDerivation I G
        g : G
        f : PointedSmoothMap 𝕜 I G Top.top g
        hX : Eq ((X (Y (ContMDiffMap.comp f (smoothLeftMul I g)))) 1) (((Derivation.ev …
        hY : Eq ((Y (X (ContMDiffMap.comp f (smoothLeftMul I g)))) 1) (((Derivation.ev …
        ⊢ Eq (HSub.hSub (((Derivation.evalAt g) ↑X) (Y f)) (((Derivation.evalAt g) ↑Y) …
      -/
      rfl⟩
      /-
        🎉 no goals
      -/


@[simp]
theorem commutator_coe_derivation :
    ⇑⁅X, Y⁆ =
      (⁅(X : Derivation 𝕜 C^∞⟮I, G; 𝕜⟯ C^∞⟮I, G; 𝕜⟯), Y⁆ :
        Derivation 𝕜 C^∞⟮I, G; 𝕜⟯ C^∞⟮I, G; 𝕜⟯) :=
  rfl


theorem commutator_apply : ⁅X, Y⁆ f = X (Y f) - Y (X f) :=
  rfl


instance : LieRing (LeftInvariantDerivation I G) where
  add_lie X Y Z := by
    /-
      𝕜 : Type u_1
      inst✝⁷ : NontriviallyNormedField 𝕜
      E : Type u_2
      inst✝⁶ : NormedAddCommGroup E
      inst✝⁵ : NormedSpace 𝕜 E
      H : Type u_3
      inst✝⁴ : TopologicalSpace H
      I : ModelWithCorners 𝕜 E H
      G : Type u_4
      inst✝³ : TopologicalSpace G
      inst✝² : ChartedSpace H G
      inst✝¹ : Monoid G
      inst✝ : SmoothMul I G
      g h : G
      r : 𝕜
      X✝ Y✝ : LeftInvariantDerivation I G
      f f' : ContMDiffMap I (modelWithCornersSelf 𝕜 𝕜) G 𝕜 Top.top
      X Y Z : LeftInvariantDerivation I G
      ⊢ Eq (Bracket.bracket (HAdd.hAdd X Y) Z) (HAdd.hAdd (Bracket.bracket X Z) (Bra …
    -/
    ext1
    /-
      case h
      𝕜 : Type u_1
      inst✝⁷ : NontriviallyNormedField 𝕜
      E : Type u_2
      inst✝⁶ : NormedAddCommGroup E
      inst✝⁵ : NormedSpace 𝕜 E
      H : Type u_3
      inst✝⁴ : TopologicalSpace H
      I : ModelWithCorners 𝕜 E H
      G : Type u_4
      inst✝³ : TopologicalSpace G
      inst✝² : ChartedSpace H G
      inst✝¹ : Monoid G
      inst✝ : SmoothMul I G
      g h : G
      r : 𝕜
      X✝ Y✝ : LeftInvariantDerivation I G
      f f' : ContMDiffMap I (modelWithCornersSelf 𝕜 𝕜) G 𝕜 Top.top
      X Y Z : LeftInvariantDerivation I G
      f✝ : ContMDiffMap I (modelWithCornersSelf 𝕜 𝕜) G 𝕜 Top.top
      ⊢ Eq ((Bracket.bracket (HAdd.hAdd X Y) Z) f✝) ((HAdd.hAdd (Bracket.bracket X Z …
    -/
    simp only [commutator_apply, coe_add, Pi.add_apply, map_add]
    /-
      case h
      𝕜 : Type u_1
      inst✝⁷ : NontriviallyNormedField 𝕜
      E : Type u_2
      inst✝⁶ : NormedAddCommGroup E
      inst✝⁵ : NormedSpace 𝕜 E
      H : Type u_3
      inst✝⁴ : TopologicalSpace H
      I : ModelWithCorners 𝕜 E H
      G : Type u_4
      inst✝³ : TopologicalSpace G
      inst✝² : ChartedSpace H G
      inst✝¹ : Monoid G
      inst✝ : SmoothMul I G
      g h : G
      r : 𝕜
      X✝ Y✝ : LeftInvariantDerivation I G
      f f' : ContMDiffMap I (modelWithCornersSelf 𝕜 𝕜) G 𝕜 Top.top
      X Y Z : LeftInvariantDerivation I G
      f✝ : ContMDiffMap I (modelWithCornersSelf 𝕜 𝕜) G 𝕜 Top.top
      ⊢ Eq (HSub.hSub (HAdd.hAdd (X (Z f✝)) (Y (Z f✝))) (HAdd.hAdd (Z (X f✝)) (Z (Y  …
    -/
    ring
    /-
      🎉 no goals
    -/
  lie_add X Y Z := by
    /-
      𝕜 : Type u_1
      inst✝⁷ : NontriviallyNormedField 𝕜
      E : Type u_2
      inst✝⁶ : NormedAddCommGroup E
      inst✝⁵ : NormedSpace 𝕜 E
      H : Type u_3
      inst✝⁴ : TopologicalSpace H
      I : ModelWithCorners 𝕜 E H
      G : Type u_4
      inst✝³ : TopologicalSpace G
      inst✝² : ChartedSpace H G
      inst✝¹ : Monoid G
      inst✝ : SmoothMul I G
      g h : G
      r : 𝕜
      X✝ Y✝ : LeftInvariantDerivation I G
      f f' : ContMDiffMap I (modelWithCornersSelf 𝕜 𝕜) G 𝕜 Top.top
      X Y Z : LeftInvariantDerivation I G
      ⊢ Eq (Bracket.bracket X (HAdd.hAdd Y Z)) (HAdd.hAdd (Bracket.bracket X Y) (Bra …
    -/
    ext1
    /-
      case h
      𝕜 : Type u_1
      inst✝⁷ : NontriviallyNormedField 𝕜
      E : Type u_2
      inst✝⁶ : NormedAddCommGroup E
      inst✝⁵ : NormedSpace 𝕜 E
      H : Type u_3
      inst✝⁴ : TopologicalSpace H
      I : ModelWithCorners 𝕜 E H
      G : Type u_4
      inst✝³ : TopologicalSpace G
      inst✝² : ChartedSpace H G
      inst✝¹ : Monoid G
      inst✝ : SmoothMul I G
      g h : G
      r : 𝕜
      X✝ Y✝ : LeftInvariantDerivation I G
      f f' : ContMDiffMap I (modelWithCornersSelf 𝕜 𝕜) G 𝕜 Top.top
      X Y Z : LeftInvariantDerivation I G
      f✝ : ContMDiffMap I (modelWithCornersSelf 𝕜 𝕜) G 𝕜 Top.top
      ⊢ Eq ((Bracket.bracket X (HAdd.hAdd Y Z)) f✝) ((HAdd.hAdd (Bracket.bracket X Y …
    -/
    simp only [commutator_apply, coe_add, Pi.add_apply, map_add]
    /-
      case h
      𝕜 : Type u_1
      inst✝⁷ : NontriviallyNormedField 𝕜
      E : Type u_2
      inst✝⁶ : NormedAddCommGroup E
      inst✝⁵ : NormedSpace 𝕜 E
      H : Type u_3
      inst✝⁴ : TopologicalSpace H
      I : ModelWithCorners 𝕜 E H
      G : Type u_4
      inst✝³ : TopologicalSpace G
      inst✝² : ChartedSpace H G
      inst✝¹ : Monoid G
      inst✝ : SmoothMul I G
      g h : G
      r : 𝕜
      X✝ Y✝ : LeftInvariantDerivation I G
      f f' : ContMDiffMap I (modelWithCornersSelf 𝕜 𝕜) G 𝕜 Top.top
      X Y Z : LeftInvariantDerivation I G
      f✝ : ContMDiffMap I (modelWithCornersSelf 𝕜 𝕜) G 𝕜 Top.top
      ⊢ Eq (HSub.hSub (HAdd.hAdd (X (Y f✝)) (X (Z f✝))) (HAdd.hAdd (Y (X f✝)) (Z (X  …
    -/
    ring
    /-
      🎉 no goals
    -/
                   /-
                     𝕜 : Type u_1
                     inst✝⁷ : NontriviallyNormedField 𝕜
                     E : Type u_2
                     inst✝⁶ : NormedAddCommGroup E
                     inst✝⁵ : NormedSpace 𝕜 E
                     H : Type u_3
                     inst✝⁴ : TopologicalSpace H
                     I : ModelWithCorners 𝕜 E H
                     G : Type u_4
                     inst✝³ : TopologicalSpace G
                     inst✝² : ChartedSpace H G
                     inst✝¹ : Monoid G
                     inst✝ : SmoothMul I G
                     g h : G
                     r : 𝕜
                     X✝ Y : LeftInvariantDerivation I G
                     f f' : ContMDiffMap I (modelWithCornersSelf 𝕜 𝕜) G 𝕜 Top.top
                     X : LeftInvariantDerivation I G
                     ⊢ Eq (Bracket.bracket X X) 0
                   -/
  lie_self X := by ext1; simp only [commutator_apply, sub_self]; rfl
                                                                 /-
                                                                   🎉 no goals
                                                                 -/
  leibniz_lie X Y Z := by
    /-
      𝕜 : Type u_1
      inst✝⁷ : NontriviallyNormedField 𝕜
      E : Type u_2
      inst✝⁶ : NormedAddCommGroup E
      inst✝⁵ : NormedSpace 𝕜 E
      H : Type u_3
      inst✝⁴ : TopologicalSpace H
      I : ModelWithCorners 𝕜 E H
      G : Type u_4
      inst✝³ : TopologicalSpace G
      inst✝² : ChartedSpace H G
      inst✝¹ : Monoid G
      inst✝ : SmoothMul I G
      g h : G
      r : 𝕜
      X✝ Y✝ : LeftInvariantDerivation I G
      f f' : ContMDiffMap I (modelWithCornersSelf 𝕜 𝕜) G 𝕜 Top.top
      X Y Z : LeftInvariantDerivation I G
      ⊢ Eq (Bracket.bracket X (Bracket.bracket Y Z)) (HAdd.hAdd (Bracket.bracket (Br …
    -/
    ext1
    /-
      case h
      𝕜 : Type u_1
      inst✝⁷ : NontriviallyNormedField 𝕜
      E : Type u_2
      inst✝⁶ : NormedAddCommGroup E
      inst✝⁵ : NormedSpace 𝕜 E
      H : Type u_3
      inst✝⁴ : TopologicalSpace H
      I : ModelWithCorners 𝕜 E H
      G : Type u_4
      inst✝³ : TopologicalSpace G
      inst✝² : ChartedSpace H G
      inst✝¹ : Monoid G
      inst✝ : SmoothMul I G
      g h : G
      r : 𝕜
      X✝ Y✝ : LeftInvariantDerivation I G
      f f' : ContMDiffMap I (modelWithCornersSelf 𝕜 𝕜) G 𝕜 Top.top
      X Y Z : LeftInvariantDerivation I G
      f✝ : ContMDiffMap I (modelWithCornersSelf 𝕜 𝕜) G 𝕜 Top.top
      ⊢ Eq ((Bracket.bracket X (Bracket.bracket Y Z)) f✝) ((HAdd.hAdd (Bracket.brack …
    -/
    simp only [commutator_apply, coe_add, coe_sub, map_sub, Pi.add_apply]
    /-
      case h
      𝕜 : Type u_1
      inst✝⁷ : NontriviallyNormedField 𝕜
      E : Type u_2
      inst✝⁶ : NormedAddCommGroup E
      inst✝⁵ : NormedSpace 𝕜 E
      H : Type u_3
      inst✝⁴ : TopologicalSpace H
      I : ModelWithCorners 𝕜 E H
      G : Type u_4
      inst✝³ : TopologicalSpace G
      inst✝² : ChartedSpace H G
      inst✝¹ : Monoid G
      inst✝ : SmoothMul I G
      g h : G
      r : 𝕜
      X✝ Y✝ : LeftInvariantDerivation I G
      f f' : ContMDiffMap I (modelWithCornersSelf 𝕜 𝕜) G 𝕜 Top.top
      X Y Z : LeftInvariantDerivation I G
      f✝ : ContMDiffMap I (modelWithCornersSelf 𝕜 𝕜) G 𝕜 Top.top
      ⊢ Eq (HSub.hSub (HSub.hSub (X (Y (Z f✝))) (X (Z (Y f✝)))) (HSub.hSub (Y (Z (X  …
    -/
    ring
    /-
      🎉 no goals
    -/


instance : LieAlgebra 𝕜 (LeftInvariantDerivation I G) where
  lie_smul r Y Z := by
    /-
      𝕜 : Type u_1
      inst✝⁷ : NontriviallyNormedField 𝕜
      E : Type u_2
      inst✝⁶ : NormedAddCommGroup E
      inst✝⁵ : NormedSpace 𝕜 E
      H : Type u_3
      inst✝⁴ : TopologicalSpace H
      I : ModelWithCorners 𝕜 E H
      G : Type u_4
      inst✝³ : TopologicalSpace G
      inst✝² : ChartedSpace H G
      inst✝¹ : Monoid G
      inst✝ : SmoothMul I G
      g h : G
      r✝ : 𝕜
      X Y✝ : LeftInvariantDerivation I G
      f f' : ContMDiffMap I (modelWithCornersSelf 𝕜 𝕜) G 𝕜 Top.top
      r : 𝕜
      Y Z : LeftInvariantDerivation I G
      ⊢ Eq (Bracket.bracket Y (HSMul.hSMul r Z)) (HSMul.hSMul r (Bracket.bracket Y Z))
    -/
    ext1
    /-
      case h
      𝕜 : Type u_1
      inst✝⁷ : NontriviallyNormedField 𝕜
      E : Type u_2
      inst✝⁶ : NormedAddCommGroup E
      inst✝⁵ : NormedSpace 𝕜 E
      H : Type u_3
      inst✝⁴ : TopologicalSpace H
      I : ModelWithCorners 𝕜 E H
      G : Type u_4
      inst✝³ : TopologicalSpace G
      inst✝² : ChartedSpace H G
      inst✝¹ : Monoid G
      inst✝ : SmoothMul I G
      g h : G
      r✝ : 𝕜
      X Y✝ : LeftInvariantDerivation I G
      f f' : ContMDiffMap I (modelWithCornersSelf 𝕜 𝕜) G 𝕜 Top.top
      r : 𝕜
      Y Z : LeftInvariantDerivation I G
      f✝ : ContMDiffMap I (modelWithCornersSelf 𝕜 𝕜) G 𝕜 Top.top
      ⊢ Eq ((Bracket.bracket Y (HSMul.hSMul r Z)) f✝) ((HSMul.hSMul r (Bracket.brack …
    -/
    simp only [commutator_apply, map_smul, smul_sub, coe_smul, Pi.smul_apply]
    /-
      🎉 no goals
    -/


