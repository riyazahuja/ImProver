/-- Typeclass saying that `F` is a type of bundled equivalences such that all `e : F` are
dilations. -/
class DilationEquivClass [EquivLike F X Y] : Prop where
  edist_eq' : ∀ f : F, ∃ r : ℝ≥0, r ≠ 0 ∧ ∀ x y : X, edist (f x) (f y) = r * edist x y


instance (priority := 100) [EquivLike F X Y] [DilationEquivClass F X Y] : DilationClass F X Y :=
  { inferInstanceAs (FunLike F X Y), ‹DilationEquivClass F X Y› with }


/-- Type of equivalences `X ≃ Y` such that `∀ x y, edist (f x) (f y) = r * edist x y` for some
`r : ℝ≥0`, `r ≠ 0`. -/
structure DilationEquiv (X Y : Type*) [PseudoEMetricSpace X] [PseudoEMetricSpace Y]
    extends X ≃ Y, Dilation X Y


@[inherit_doc] infixl:25 " ≃ᵈ " => DilationEquiv


instance : EquivLike (X ≃ᵈ Y) X Y where
  coe f := f.1
  inv f := f.1.symm
  left_inv f := f.left_inv'
  right_inv f := f.right_inv'
                       /-
                         X : Type u_1
                         Y : Type u_2
                         Z : Type u_3
                         inst✝² : PseudoEMetricSpace X
                         inst✝¹ : PseudoEMetricSpace Y
                         inst✝ : PseudoEMetricSpace Z
                         ⊢ ∀ (e g : DilationEquiv X Y), Eq ((fun f => ⇑f.toEquiv) e) ((fun f => ⇑f.toEq …
                       -/
  coe_injective' := by rintro ⟨⟩ ⟨⟩ h -; congr; exact DFunLike.ext' h
                                                /-
                                                  🎉 no goals
                                                -/


instance : DilationEquivClass (X ≃ᵈ Y) X Y where
  edist_eq' f := f.edist_eq'


@[simp] theorem coe_toEquiv (e : X ≃ᵈ Y) : ⇑e.toEquiv = e := rfl


@[ext]
protected theorem ext {e e' : X ≃ᵈ Y} (h : ∀ x, e x = e' x) : e = e' :=
  DFunLike.ext _ _ h


/-- Inverse `DilationEquiv`. -/
def symm (e : X ≃ᵈ Y) : Y ≃ᵈ X where
  toEquiv := e.1.symm
  edist_eq' := by
    /-
      X : Type u_1
      Y : Type u_2
      Z : Type u_3
      inst✝² : PseudoEMetricSpace X
      inst✝¹ : PseudoEMetricSpace Y
      inst✝ : PseudoEMetricSpace Z
      e : DilationEquiv X Y
      ⊢ Exists fun r => And (Ne r 0) (∀ (x y : Y), Eq (EDist.edist (e.symm.toFun x)  …
    -/
    refine ⟨(ratio e)⁻¹, inv_ne_zero <| ratio_ne_zero e, e.surjective.forall₂.2 fun x y ↦ ?_⟩
    /-
      X : Type u_1
      Y : Type u_2
      Z : Type u_3
      inst✝² : PseudoEMetricSpace X
      inst✝¹ : PseudoEMetricSpace Y
      inst✝ : PseudoEMetricSpace Z
      e : DilationEquiv X Y
      x y : X
      ⊢ Eq (EDist.edist (e.symm.toFun (e.toEquiv x)) (e.symm.toFun (e.toEquiv y))) ( …
    -/
    simp_rw [Equiv.toFun_as_coe, Equiv.symm_apply_apply, coe_toEquiv, edist_eq]
    rw [← mul_assoc, ← ENNReal.coe_mul, inv_mul_cancel₀ (ratio_ne_zero e),
      ENNReal.coe_one, one_mul]


@[simp] theorem symm_symm (e : X ≃ᵈ Y) : e.symm.symm = e := rfl


theorem symm_bijective : Function.Bijective (DilationEquiv.symm : (X ≃ᵈ Y) → Y ≃ᵈ X) :=
  Function.bijective_iff_has_inverse.mpr ⟨_, symm_symm, symm_symm⟩


@[simp] theorem apply_symm_apply (e : X ≃ᵈ Y) (x : Y) : e (e.symm x) = x := e.right_inv x

@[simp] theorem symm_apply_apply (e : X ≃ᵈ Y) (x : X) : e.symm (e x) = x := e.left_inv x


/-- See Note [custom simps projection]. -/
def Simps.symm_apply (e : X ≃ᵈ Y) : Y → X := e.symm


lemma ratio_toDilation (e : X ≃ᵈ Y) : ratio e.toDilation = ratio e := rfl


/-- Identity map as a `DilationEquiv`. -/
@[simps! (config := .asFn) apply]
def refl (X : Type*) [PseudoEMetricSpace X] : X ≃ᵈ X where
  toEquiv := .refl X
                                             /-
                                               X✝ : Type u_1
                                               Y : Type u_2
                                               Z : Type u_3
                                               inst✝³ : PseudoEMetricSpace X✝
                                               inst✝² : PseudoEMetricSpace Y
                                               inst✝¹ : PseudoEMetricSpace Z
                                               X : Type u_4
                                               inst✝ : PseudoEMetricSpace X
                                               x✝¹ x✝ : X
                                               ⊢ Eq (EDist.edist ((Equiv.refl X).toFun x✝¹) ((Equiv.refl X).toFun x✝)) (HMul. …
                                             -/
  edist_eq' := ⟨1, one_ne_zero, fun _ _ ↦ by simp⟩
                                             /-
                                               🎉 no goals
                                             -/


@[simp] theorem refl_symm : (refl X).symm = refl X := rfl

@[simp] theorem ratio_refl : ratio (refl X) = 1 := Dilation.ratio_id


/-- Composition of `DilationEquiv`s. -/
@[simps! (config := .asFn) apply]
def trans (e₁ : X ≃ᵈ Y) (e₂ : Y ≃ᵈ Z) : X ≃ᵈ Z where
  toEquiv := e₁.1.trans e₂.1
  __ := e₂.toDilation.comp e₁.toDilation


@[simp] theorem refl_trans (e : X ≃ᵈ Y) : (refl X).trans e = e := rfl

@[simp] theorem trans_refl (e : X ≃ᵈ Y) : e.trans (refl Y) = e := rfl


@[simp] theorem symm_trans_self (e : X ≃ᵈ Y) : e.symm.trans e = refl Y :=
  DilationEquiv.ext e.apply_symm_apply


@[simp] theorem self_trans_symm (e : X ≃ᵈ Y) : e.trans e.symm = refl X :=
  DilationEquiv.ext e.symm_apply_apply


protected theorem surjective (e : X ≃ᵈ Y) : Surjective e := e.1.surjective

protected theorem bijective (e : X ≃ᵈ Y) : Bijective e := e.1.bijective

protected theorem injective (e : X ≃ᵈ Y) : Injective e := e.1.injective


@[simp]
theorem ratio_trans (e : X ≃ᵈ Y) (e' : Y ≃ᵈ Z) : ratio (e.trans e') = ratio e * ratio e' := by
  -- If `X` is trivial, then so is `Y`, otherwise we apply `Dilation.ratio_comp'`
  /-
    X : Type u_1
    Y : Type u_2
    Z : Type u_3
    inst✝² : PseudoEMetricSpace X
    inst✝¹ : PseudoEMetricSpace Y
    inst✝ : PseudoEMetricSpace Z
    e : DilationEquiv X Y
    e' : DilationEquiv Y Z
    ⊢ Eq (Dilation.ratio (e.trans e')) (HMul.hMul (Dilation.ratio e) (Dilation.rat …
  -/
  by_cases hX : ∀ x y : X, edist x y = 0 ∨ edist x y = ∞
  · have hY : ∀ x y : Y, edist x y = 0 ∨ edist x y = ∞ := e.surjective.forall₂.2 fun x y ↦ by
      refine (hX x y).imp (fun h ↦ ?_) fun h ↦ ?_ <;> simp [*, Dilation.ratio_ne_zero]
    /-
      case pos
      X : Type u_1
      Y : Type u_2
      Z : Type u_3
      inst✝² : PseudoEMetricSpace X
      inst✝¹ : PseudoEMetricSpace Y
      inst✝ : PseudoEMetricSpace Z
      e : DilationEquiv X Y
      e' : DilationEquiv Y Z
      hX : ∀ (x y : X), Or (Eq (EDist.edist x y) 0) (Eq (EDist.edist x y) Top.top)
      hY : ∀ (x y : Y), Or (Eq (EDist.edist x y) 0) (Eq (EDist.edist x y) Top.top)
      ⊢ Eq (Dilation.ratio (e.trans e')) (HMul.hMul (Dilation.ratio e) (Dilation.rat …
    -/
    simp [Dilation.ratio_of_trivial, *]
    /-
      🎉 no goals
    -/
  /-
    case neg
    X : Type u_1
    Y : Type u_2
    Z : Type u_3
    inst✝² : PseudoEMetricSpace X
    inst✝¹ : PseudoEMetricSpace Y
    inst✝ : PseudoEMetricSpace Z
    e : DilationEquiv X Y
    e' : DilationEquiv Y Z
    hX : Not (∀ (x y : X), Or (Eq (EDist.edist x y) 0) (Eq (EDist.edist x y) Top.t …
    ⊢ Eq (Dilation.ratio (e.trans e')) (HMul.hMul (Dilation.ratio e) (Dilation.rat …
  -/
  push_neg at hX
  /-
    case neg
    X : Type u_1
    Y : Type u_2
    Z : Type u_3
    inst✝² : PseudoEMetricSpace X
    inst✝¹ : PseudoEMetricSpace Y
    inst✝ : PseudoEMetricSpace Z
    e : DilationEquiv X Y
    e' : DilationEquiv Y Z
    hX : Exists fun x => Exists fun y => And (Ne (EDist.edist x y) 0) (Ne (EDist.e …
    ⊢ Eq (Dilation.ratio (e.trans e')) (HMul.hMul (Dilation.ratio e) (Dilation.rat …
  -/
  exact (Dilation.ratio_comp' (g := e'.toDilation) (f := e.toDilation) hX).trans (mul_comm _ _)
  /-
    🎉 no goals
  -/


@[simp]
theorem ratio_symm (e : X ≃ᵈ Y) : ratio e.symm = (ratio e)⁻¹ :=
                                  /-
                                    X : Type u_1
                                    Y : Type u_2
                                    inst✝¹ : PseudoEMetricSpace X
                                    inst✝ : PseudoEMetricSpace Y
                                    e : DilationEquiv X Y
                                    ⊢ Eq (HMul.hMul (Dilation.ratio e.symm) (Dilation.ratio e)) 1
                                  -/
  eq_inv_of_mul_eq_one_left <| by rw [← ratio_trans, symm_trans_self, ratio_refl]
                                  /-
                                    🎉 no goals
                                  -/


instance : Group (X ≃ᵈ X) where
  mul e e' := e'.trans e
  mul_assoc _ _ _ := rfl
  one := refl _
  one_mul _ := rfl
  mul_one _ := rfl
  inv := symm
  inv_mul_cancel := self_trans_symm


theorem mul_def (e e' : X ≃ᵈ X) : e * e' = e'.trans e := rfl

theorem one_def : (1 : X ≃ᵈ X) = refl X := rfl

theorem inv_def (e : X ≃ᵈ X) : e⁻¹ = e.symm := rfl


@[simp] theorem coe_mul (e e' : X ≃ᵈ X) : ⇑(e * e') = e ∘ e' := rfl

@[simp] theorem coe_one : ⇑(1 : X ≃ᵈ X) = id := rfl

theorem coe_inv (e : X ≃ᵈ X) : ⇑(e⁻¹) = e.symm := rfl


/-- `Dilation.ratio` as a monoid homomorphism. -/
noncomputable def ratioHom : (X ≃ᵈ X) →* ℝ≥0 where
  toFun := Dilation.ratio
  map_one' := ratio_refl
  map_mul' _ _ := (ratio_trans _ _).trans (mul_comm _ _)


@[simp]
theorem ratio_inv (e : X ≃ᵈ X) : ratio (e⁻¹) = (ratio e)⁻¹ := ratio_symm e


@[simp]
theorem ratio_pow (e : X ≃ᵈ X) (n : ℕ) : ratio (e ^ n) = ratio e ^ n :=
  ratioHom.map_pow _ _


@[simp]
theorem ratio_zpow (e : X ≃ᵈ X) (n : ℤ) : ratio (e ^ n) = ratio e ^ n :=
  ratioHom.map_zpow _ _


/-- `DilationEquiv.toEquiv` as a monoid homomorphism. -/
@[simps]
def toPerm : (X ≃ᵈ X) →* Equiv.Perm X where
  toFun e := e.1
  map_mul' _ _ := rfl
  map_one' := rfl


@[norm_cast]
theorem coe_pow (e : X ≃ᵈ X) (n : ℕ) : ⇑(e ^ n) = e^[n] := by
  /-
    X : Type u_1
    inst✝ : PseudoEMetricSpace X
    e : DilationEquiv X X
    n : Nat
    ⊢ Eq (⇑(HPow.hPow e n)) (Nat.iterate (⇑e) n)
  -/
  rw [← coe_toEquiv, ← toPerm_apply, map_pow, Equiv.Perm.coe_pow]; rfl
                                                                   /-
                                                                     🎉 no goals
                                                                   -/

-- TODO: Once `IsometryEquiv` follows the `*EquivClass` pattern, replace this with an instance
-- of `DilationEquivClass` assuming `IsometryEquivClass`.

/-- Every isometry equivalence is a dilation equivalence of ratio `1`. -/
def _root_.IsometryEquiv.toDilationEquiv (e : X ≃ᵢ Y) : X ≃ᵈ Y where
                                   /-
                                     X : Type u_1
                                     Y : Type u_2
                                     Z : Type u_3
                                     inst✝² : PseudoEMetricSpace X
                                     inst✝¹ : PseudoEMetricSpace Y
                                     inst✝ : PseudoEMetricSpace Z
                                     e : IsometryEquiv X Y
                                     ⊢ ∀ (x y : X), Eq (EDist.edist (__spread✝⁻⁰.toFun x) (__spread✝⁻⁰.toFun y)) (H …
                                   -/
  edist_eq' := ⟨1, one_ne_zero, by simpa using e.isometry⟩
                                   /-
                                     🎉 no goals
                                   -/
  __ := e.toEquiv


@[simp]
lemma _root_.IsometryEquiv.toDilationEquiv_apply (e : X ≃ᵢ Y) (x : X) :
    e.toDilationEquiv x = e x :=
  rfl


@[simp]
lemma _root_.IsometryEquiv.toDilationEquiv_symm (e : X ≃ᵢ Y) :
    e.toDilationEquiv.symm = e.symm.toDilationEquiv :=
  rfl


@[simp]
lemma _root_.IsometryEquiv.toDilationEquiv_toDilation (e : X ≃ᵢ Y) :
    (e.toDilationEquiv.toDilation : X →ᵈ Y) = e.isometry.toDilation :=
  rfl


@[simp]
lemma _root_.IsometryEquiv.toDilationEquiv_ratio (e : X ≃ᵢ Y) : ratio e.toDilationEquiv = 1 := by
  /-
    X : Type u_1
    Y : Type u_2
    inst✝¹ : PseudoEMetricSpace X
    inst✝ : PseudoEMetricSpace Y
    e : IsometryEquiv X Y
    ⊢ Eq (Dilation.ratio e.toDilationEquiv) 1
  -/
  rw [← ratio_toDilation, IsometryEquiv.toDilationEquiv_toDilation, Isometry.toDilation_ratio]
  /-
    🎉 no goals
  -/


/-- Reinterpret a `DilationEquiv` as a homeomorphism. -/
def toHomeomorph (e : X ≃ᵈ Y) : X ≃ₜ Y where
  continuous_toFun := Dilation.toContinuous e
  continuous_invFun := Dilation.toContinuous e.symm
  __ := e.toEquiv


@[simp]
lemma coe_toHomeomorph (e : X ≃ᵈ Y) : ⇑e.toHomeomorph = e :=
  rfl


@[simp]
lemma toHomeomorph_symm (e : X ≃ᵈ Y) : e.toHomeomorph.symm = e.symm.toHomeomorph :=
  rfl


@[simp]
lemma map_cobounded (e : F) : map e (cobounded X) = cobounded Y := by
  /-
    X : Type u_1
    Y : Type u_2
    F : Type u_3
    inst✝³ : PseudoMetricSpace X
    inst✝² : PseudoMetricSpace Y
    inst✝¹ : EquivLike F X Y
    inst✝ : DilationEquivClass F X Y
    e : F
    ⊢ Eq (Filter.map (⇑e) (Bornology.cobounded X)) (Bornology.cobounded Y)
  -/
  rw [← Dilation.comap_cobounded e, map_comap_of_surjective (EquivLike.surjective e)]
  /-
    🎉 no goals
  -/


