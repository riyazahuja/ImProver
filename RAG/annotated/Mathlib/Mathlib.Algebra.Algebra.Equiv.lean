/-- An equivalence of algebras is an equivalence of rings commuting with the actions of scalars. -/
structure AlgEquiv (R : Type u) (A : Type v) (B : Type w) [CommSemiring R] [Semiring A] [Semiring B]
  [Algebra R A] [Algebra R B] extends A ≃ B, A ≃* B, A ≃+ B, A ≃+* B where
  /-- An equivalence of algebras commutes with the action of scalars. -/
  protected commutes' : ∀ r : R, toFun (algebraMap R A r) = algebraMap R B r


@[inherit_doc]
notation:50 A " ≃ₐ[" R "] " A' => AlgEquiv R A A'


/-- `AlgEquivClass F R A B` states that `F` is a type of algebra structure preserving
  equivalences. You should extend this class when you extend `AlgEquiv`. -/
class AlgEquivClass (F : Type*) (R A B : outParam Type*) [CommSemiring R] [Semiring A]
    [Semiring B] [Algebra R A] [Algebra R B] [EquivLike F A B]
    extends RingEquivClass F A B : Prop where
  /-- An equivalence of algebras commutes with the action of scalars. -/
  commutes : ∀ (f : F) (r : R), f (algebraMap R A r) = algebraMap R B r


instance (priority := 100) toAlgHomClass (F R A B : Type*) [CommSemiring R] [Semiring A]
    [Semiring B] [Algebra R A] [Algebra R B] [EquivLike F A B] [h : AlgEquivClass F R A B] :
    AlgHomClass F R A B :=
  { h with }


instance (priority := 100) toLinearEquivClass (F R A B : Type*) [CommSemiring R]
    [Semiring A] [Semiring B] [Algebra R A] [Algebra R B]
    [EquivLike F A B] [h : AlgEquivClass F R A B] : LinearEquivClass F R A B :=
  { h with map_smulₛₗ := fun f => map_smulₛₗ f }


/-- Turn an element of a type `F` satisfying `AlgEquivClass F R A B` into an actual `AlgEquiv`.
This is declared as the default coercion from `F` to `A ≃ₐ[R] B`. -/
@[coe]
def toAlgEquiv {F R A B : Type*} [CommSemiring R] [Semiring A] [Semiring B] [Algebra R A]
    [Algebra R B] [EquivLike F A B] [AlgEquivClass F R A B] (f : F) : A ≃ₐ[R] B :=
  { (f : A ≃ B), (f : A ≃+* B) with commutes' := commutes f }


instance (F R A B : Type*) [CommSemiring R] [Semiring A] [Semiring B] [Algebra R A] [Algebra R B]
    [EquivLike F A B] [AlgEquivClass F R A B] : CoeTC F (A ≃ₐ[R] B) :=
  ⟨toAlgEquiv⟩

instance : EquivLike (A₁ ≃ₐ[R] A₂) A₁ A₂ where
  coe f := f.toFun
  inv f := f.invFun
  left_inv f := f.left_inv
  right_inv f := f.right_inv
  coe_injective' f g h₁ h₂ := by
    /-
      R : Type uR
      A₁ : Type uA₁
      A₂ : Type uA₂
      A₃ : Type uA₃
      A₁' : Type uA₁'
      A₂' : Type uA₂'
      A₃' : Type uA₃'
      inst✝¹² : CommSemiring R
      inst✝¹¹ : Semiring A₁
      inst✝¹⁰ : Semiring A₂
      inst✝⁹ : Semiring A₃
      inst✝⁸ : Semiring A₁'
      inst✝⁷ : Semiring A₂'
      inst✝⁶ : Semiring A₃'
      inst✝⁵ : Algebra R A₁
      inst✝⁴ : Algebra R A₂
      inst✝³ : Algebra R A₃
      inst✝² : Algebra R A₁'
      inst✝¹ : Algebra R A₂'
      inst✝ : Algebra R A₃'
      e f g : AlgEquiv R A₁ A₂
      h₁ : Eq ((fun f => f.toFun) f) ((fun f => f.toFun) g)
      h₂ : Eq ((fun f => f.invFun) f) ((fun f => f.invFun) g)
      ⊢ Eq f g
    -/
    obtain ⟨⟨f,_⟩,_⟩ := f
    /-
      case mk.mk
      R : Type uR
      A₁ : Type uA₁
      A₂ : Type uA₂
      A₃ : Type uA₃
      A₁' : Type uA₁'
      A₂' : Type uA₂'
      A₃' : Type uA₃'
      inst✝¹² : CommSemiring R
      inst✝¹¹ : Semiring A₁
      inst✝¹⁰ : Semiring A₂
      inst✝⁹ : Semiring A₃
      inst✝⁸ : Semiring A₁'
      inst✝⁷ : Semiring A₂'
      inst✝⁶ : Semiring A₃'
      inst✝⁵ : Algebra R A₁
      inst✝⁴ : Algebra R A₂
      inst✝³ : Algebra R A₃
      inst✝² : Algebra R A₁'
      inst✝¹ : Algebra R A₂'
      inst✝ : Algebra R A₃'
      e g : AlgEquiv R A₁ A₂
      f : A₁ → A₂
      invFun✝ : A₂ → A₁
      left_inv✝ : Function.LeftInverse invFun✝ f
      right_inv✝ : Function.RightInverse invFun✝ f
      map_mul'✝ : ∀ (x y : A₁), Eq ({ toFun := f, invFun := invFun✝, left_inv := lef …
      map_add'✝ : ∀ (x y : A₁), Eq ({ toFun := f, invFun := invFun✝, left_inv := lef …
      commutes'✝ : ∀ (r : R), Eq ({ toFun := f, invFun := invFun✝, left_inv := left_ …
      h₁ : Eq ((fun f => f.toFun) { toFun := f, invFun := invFun✝, left_inv := left_ …
      h₂ : Eq ((fun f => f.invFun) { toFun := f, invFun := invFun✝, left_inv := left …
      ⊢ Eq { toFun := f, invFun := invFun✝, left_inv := left_inv✝, right_inv := righ …
    -/
    obtain ⟨⟨g,_⟩,_⟩ := g
    /-
      case mk.mk.mk.mk
      R : Type uR
      A₁ : Type uA₁
      A₂ : Type uA₂
      A₃ : Type uA₃
      A₁' : Type uA₁'
      A₂' : Type uA₂'
      A₃' : Type uA₃'
      inst✝¹² : CommSemiring R
      inst✝¹¹ : Semiring A₁
      inst✝¹⁰ : Semiring A₂
      inst✝⁹ : Semiring A₃
      inst✝⁸ : Semiring A₁'
      inst✝⁷ : Semiring A₂'
      inst✝⁶ : Semiring A₃'
      inst✝⁵ : Algebra R A₁
      inst✝⁴ : Algebra R A₂
      inst✝³ : Algebra R A₃
      inst✝² : Algebra R A₁'
      inst✝¹ : Algebra R A₂'
      inst✝ : Algebra R A₃'
      e : AlgEquiv R A₁ A₂
      f : A₁ → A₂
      invFun✝¹ : A₂ → A₁
      left_inv✝¹ : Function.LeftInverse invFun✝¹ f
      right_inv✝¹ : Function.RightInverse invFun✝¹ f
      map_mul'✝¹ : ∀ (x y : A₁), Eq ({ toFun := f, invFun := invFun✝¹, left_inv := l …
      map_add'✝¹ : ∀ (x y : A₁), Eq ({ toFun := f, invFun := invFun✝¹, left_inv := l …
      commutes'✝¹ : ∀ (r : R), Eq ({ toFun := f, invFun := invFun✝¹, left_inv := lef …
      g : A₁ → A₂
      invFun✝ : A₂ → A₁
      left_inv✝ : Function.LeftInverse invFun✝ g
      right_inv✝ : Function.RightInverse invFun✝ g
      map_mul'✝ : ∀ (x y : A₁), Eq ({ toFun := g, invFun := invFun✝, left_inv := lef …
      map_add'✝ : ∀ (x y : A₁), Eq ({ toFun := g, invFun := invFun✝, left_inv := lef …
      commutes'✝ : ∀ (r : R), Eq ({ toFun := g, invFun := invFun✝, left_inv := left_ …
      h₁ : Eq ((fun f => f.toFun) { toFun := f, invFun := invFun✝¹, left_inv := left …
      h₂ : Eq ((fun f => f.invFun) { toFun := f, invFun := invFun✝¹, left_inv := lef …
      ⊢ Eq { toFun := f, invFun := invFun✝¹, left_inv := left_inv✝¹, right_inv := ri …
    -/
    congr
    /-
      🎉 no goals
    -/


/-- Helper instance since the coercion is not always found. -/
instance : FunLike (A₁ ≃ₐ[R] A₂) A₁ A₂ where
  coe := DFunLike.coe
  coe_injective' := DFunLike.coe_injective'


instance : AlgEquivClass (A₁ ≃ₐ[R] A₂) R A₁ A₂ where
  map_add f := f.map_add'
  map_mul f := f.map_mul'
  commutes f := f.commutes'


@[ext]
theorem ext {f g : A₁ ≃ₐ[R] A₂} (h : ∀ a, f a = g a) : f = g :=
  DFunLike.ext f g h


protected theorem congr_arg {f : A₁ ≃ₐ[R] A₂} {x x' : A₁} : x = x' → f x = f x' :=
  DFunLike.congr_arg f


protected theorem congr_fun {f g : A₁ ≃ₐ[R] A₂} (h : f = g) (x : A₁) : f x = g x :=
  DFunLike.congr_fun h x


@[simp]
theorem coe_mk {toEquiv map_mul map_add commutes} :
    ⇑(⟨toEquiv, map_mul, map_add, commutes⟩ : A₁ ≃ₐ[R] A₂) = toEquiv :=
  rfl


@[simp]
theorem mk_coe (e : A₁ ≃ₐ[R] A₂) (e' h₁ h₂ h₃ h₄ h₅) :
    (⟨⟨e, e', h₁, h₂⟩, h₃, h₄, h₅⟩ : A₁ ≃ₐ[R] A₂) = e :=
  ext fun _ => rfl


@[simp]
theorem toEquiv_eq_coe : e.toEquiv = e :=
  rfl


@[simp]
protected theorem coe_coe {F : Type*} [EquivLike F A₁ A₂] [AlgEquivClass F R A₁ A₂] (f : F) :
    ⇑(f : A₁ ≃ₐ[R] A₂) = f :=
  rfl


theorem coe_fun_injective : @Function.Injective (A₁ ≃ₐ[R] A₂) (A₁ → A₂) fun e => (e : A₁ → A₂) :=
  DFunLike.coe_injective


instance hasCoeToRingEquiv : CoeOut (A₁ ≃ₐ[R] A₂) (A₁ ≃+* A₂) :=
  ⟨AlgEquiv.toRingEquiv⟩

-- Porting note: `toFun_eq_coe` no longer needed in Lean4


@[simp]
theorem toRingEquiv_eq_coe : e.toRingEquiv = e :=
  rfl


@[simp, norm_cast]
lemma toRingEquiv_toRingHom : ((e : A₁ ≃+* A₂) : A₁ →+* A₂) = e :=
  rfl


@[simp, norm_cast]
theorem coe_ringEquiv : ((e : A₁ ≃+* A₂) : A₁ → A₂) = e :=
  rfl


theorem coe_ringEquiv' : (e.toRingEquiv : A₁ → A₂) = e :=
  rfl


theorem coe_ringEquiv_injective : Function.Injective ((↑) : (A₁ ≃ₐ[R] A₂) → A₁ ≃+* A₂) :=
  fun _ _ h => ext <| RingEquiv.congr_fun h

-- Porting note: Added [coe] attribute

/-- Interpret an algebra equivalence as an algebra homomorphism.

This definition is included for symmetry with the other `to*Hom` projections.
The `simp` normal form is to use the coercion of the `AlgHomClass.coeTC` instance. -/
@[coe]
def toAlgHom : A₁ →ₐ[R] A₂ :=
  { e with
    map_one' := map_one e
    map_zero' := map_zero e }


@[simp]
theorem toAlgHom_eq_coe : e.toAlgHom = e :=
  rfl


@[simp, norm_cast]
theorem coe_algHom : DFunLike.coe (e.toAlgHom) = DFunLike.coe e :=
  rfl


theorem coe_algHom_injective : Function.Injective ((↑) : (A₁ ≃ₐ[R] A₂) → A₁ →ₐ[R] A₂) :=
  fun _ _ h => ext <| AlgHom.congr_fun h


@[simp, norm_cast]
lemma toAlgHom_toRingHom : ((e : A₁ →ₐ[R] A₂) : A₁ →+* A₂) = e :=
  rfl


/-- The two paths coercion can take to a `RingHom` are equivalent -/
theorem coe_ringHom_commutes : ((e : A₁ →ₐ[R] A₂) : A₁ →+* A₂) = ((e : A₁ ≃+* A₂) : A₁ →+* A₂) :=
  rfl


@[simp]
theorem commutes : ∀ r : R, e (algebraMap R A₁ r) = algebraMap R A₂ r :=
  e.commutes'


@[deprecated map_add (since := "2024-06-20")]
protected theorem map_add : ∀ x y, e (x + y) = e x + e y :=
  map_add e


@[deprecated map_zero (since := "2024-06-20")]
protected theorem map_zero : e 0 = 0 :=
  map_zero e


@[deprecated map_mul (since := "2024-06-20")]
protected theorem map_mul : ∀ x y, e (x * y) = e x * e y :=
  map_mul e


@[deprecated map_one (since := "2024-06-20")]
protected theorem map_one : e 1 = 1 :=
  map_one e


@[deprecated map_smul (since := "2024-06-20")]
protected theorem map_smul (r : R) (x : A₁) : e (r • x) = r • e x :=
  map_smul _ _ _


@[deprecated map_pow (since := "2024-06-20")]
protected theorem map_pow : ∀ (x : A₁) (n : ℕ), e (x ^ n) = e x ^ n :=
  map_pow _


protected theorem bijective : Function.Bijective e :=
  EquivLike.bijective e


protected theorem injective : Function.Injective e :=
  EquivLike.injective e


protected theorem surjective : Function.Surjective e :=
  EquivLike.surjective e


/-- Algebra equivalences are reflexive. -/
@[refl]
def refl : A₁ ≃ₐ[R] A₁ :=
  { (.refl _ : A₁ ≃+* A₁) with commutes' := fun _ => rfl }


instance : Inhabited (A₁ ≃ₐ[R] A₁) :=
  ⟨refl⟩


@[simp]
theorem refl_toAlgHom : ↑(refl : A₁ ≃ₐ[R] A₁) = AlgHom.id R A₁ :=
  rfl


@[simp]
theorem coe_refl : ⇑(refl : A₁ ≃ₐ[R] A₁) = id :=
  rfl


/-- Algebra equivalences are symmetric. -/
@[symm]
def symm (e : A₁ ≃ₐ[R] A₂) : A₂ ≃ₐ[R] A₁ :=
  { e.toRingEquiv.symm with
    commutes' := fun r => by
      /-
        R : Type uR
        A₁ : Type uA₁
        A₂ : Type uA₂
        A₃ : Type uA₃
        A₁' : Type uA₁'
        A₂' : Type uA₂'
        A₃' : Type uA₃'
        inst✝¹² : CommSemiring R
        inst✝¹¹ : Semiring A₁
        inst✝¹⁰ : Semiring A₂
        inst✝⁹ : Semiring A₃
        inst✝⁸ : Semiring A₁'
        inst✝⁷ : Semiring A₂'
        inst✝⁶ : Semiring A₃'
        inst✝⁵ : Algebra R A₁
        inst✝⁴ : Algebra R A₂
        inst✝³ : Algebra R A₃
        inst✝² : Algebra R A₁'
        inst✝¹ : Algebra R A₂'
        inst✝ : Algebra R A₃'
        e✝ e : AlgEquiv R A₁ A₂
        r : R
        ⊢ Eq (__src✝.toFun ((algebraMap R A₂) r)) ((algebraMap R A₁) r)
      -/
      rw [← e.toRingEquiv.symm_apply_apply (algebraMap R A₁ r)]
      /-
        R : Type uR
        A₁ : Type uA₁
        A₂ : Type uA₂
        A₃ : Type uA₃
        A₁' : Type uA₁'
        A₂' : Type uA₂'
        A₃' : Type uA₃'
        inst✝¹² : CommSemiring R
        inst✝¹¹ : Semiring A₁
        inst✝¹⁰ : Semiring A₂
        inst✝⁹ : Semiring A₃
        inst✝⁸ : Semiring A₁'
        inst✝⁷ : Semiring A₂'
        inst✝⁶ : Semiring A₃'
        inst✝⁵ : Algebra R A₁
        inst✝⁴ : Algebra R A₂
        inst✝³ : Algebra R A₃
        inst✝² : Algebra R A₁'
        inst✝¹ : Algebra R A₂'
        inst✝ : Algebra R A₃'
        e✝ e : AlgEquiv R A₁ A₂
        r : R
        ⊢ Eq (__src✝.toFun ((algebraMap R A₂) r)) (e.toRingEquiv.symm (e.toRingEquiv ( …
      -/
      congr
      /-
        case e_a
        R : Type uR
        A₁ : Type uA₁
        A₂ : Type uA₂
        A₃ : Type uA₃
        A₁' : Type uA₁'
        A₂' : Type uA₂'
        A₃' : Type uA₃'
        inst✝¹² : CommSemiring R
        inst✝¹¹ : Semiring A₁
        inst✝¹⁰ : Semiring A₂
        inst✝⁹ : Semiring A₃
        inst✝⁸ : Semiring A₁'
        inst✝⁷ : Semiring A₂'
        inst✝⁶ : Semiring A₃'
        inst✝⁵ : Algebra R A₁
        inst✝⁴ : Algebra R A₂
        inst✝³ : Algebra R A₃
        inst✝² : Algebra R A₁'
        inst✝¹ : Algebra R A₂'
        inst✝ : Algebra R A₃'
        e✝ e : AlgEquiv R A₁ A₂
        r : R
        ⊢ Eq ((algebraMap R A₂) r) (e.toRingEquiv ((algebraMap R A₁) r))
      -/
      change _ = e _
      /-
        case e_a
        R : Type uR
        A₁ : Type uA₁
        A₂ : Type uA₂
        A₃ : Type uA₃
        A₁' : Type uA₁'
        A₂' : Type uA₂'
        A₃' : Type uA₃'
        inst✝¹² : CommSemiring R
        inst✝¹¹ : Semiring A₁
        inst✝¹⁰ : Semiring A₂
        inst✝⁹ : Semiring A₃
        inst✝⁸ : Semiring A₁'
        inst✝⁷ : Semiring A₂'
        inst✝⁶ : Semiring A₃'
        inst✝⁵ : Algebra R A₁
        inst✝⁴ : Algebra R A₂
        inst✝³ : Algebra R A₃
        inst✝² : Algebra R A₁'
        inst✝¹ : Algebra R A₂'
        inst✝ : Algebra R A₃'
        e✝ e : AlgEquiv R A₁ A₂
        r : R
        ⊢ Eq ((algebraMap R A₂) r) (e ((algebraMap R A₁) r))
      -/
      rw [e.commutes] }
      /-
        🎉 no goals
      -/


theorem invFun_eq_symm {e : A₁ ≃ₐ[R] A₂} : e.invFun = e.symm :=
  rfl


@[simp]
theorem coe_apply_coe_coe_symm_apply {F : Type*} [EquivLike F A₁ A₂] [AlgEquivClass F R A₁ A₂]
    (f : F) (x : A₂) :
    f ((f : A₁ ≃ₐ[R] A₂).symm x) = x :=
  EquivLike.right_inv f x


@[simp]
theorem coe_coe_symm_apply_coe_apply {F : Type*} [EquivLike F A₁ A₂] [AlgEquivClass F R A₁ A₂]
    (f : F) (x : A₁) :
    (f : A₁ ≃ₐ[R] A₂).symm (f x) = x :=
  EquivLike.left_inv f x

-- Porting note: `simp` normal form of `invFun_eq_symm`

@[simp]
theorem symm_toEquiv_eq_symm {e : A₁ ≃ₐ[R] A₂} : (e : A₁ ≃ A₂).symm = e.symm :=
  rfl


@[simp]
theorem symm_symm (e : A₁ ≃ₐ[R] A₂) : e.symm.symm = e := rfl


theorem symm_bijective : Function.Bijective (symm : (A₁ ≃ₐ[R] A₂) → A₂ ≃ₐ[R] A₁) :=
  Function.bijective_iff_has_inverse.mpr ⟨_, symm_symm, symm_symm⟩


@[simp]
theorem mk_coe' (e : A₁ ≃ₐ[R] A₂) (f h₁ h₂ h₃ h₄ h₅) :
    (⟨⟨f, e, h₁, h₂⟩, h₃, h₄, h₅⟩ : A₂ ≃ₐ[R] A₁) = e.symm :=
  symm_bijective.injective <| ext fun _ => rfl


/-- Auxiliary definition to avoid looping in `dsimp` with `AlgEquiv.symm_mk`. -/
protected def symm_mk.aux (f f') (h₁ h₂ h₃ h₄ h₅) :=
  (⟨⟨f, f', h₁, h₂⟩, h₃, h₄, h₅⟩ : A₁ ≃ₐ[R] A₂).symm


@[simp]
theorem symm_mk (f f') (h₁ h₂ h₃ h₄ h₅) :
    (⟨⟨f, f', h₁, h₂⟩, h₃, h₄, h₅⟩ : A₁ ≃ₐ[R] A₂).symm =
      { symm_mk.aux f f' h₁ h₂ h₃ h₄ h₅ with
        toFun := f'
        invFun := f } :=
  rfl


@[simp]
theorem refl_symm : (AlgEquiv.refl : A₁ ≃ₐ[R] A₁).symm = AlgEquiv.refl :=
  rfl

--this should be a simp lemma but causes a lint timeout

theorem toRingEquiv_symm (f : A₁ ≃ₐ[R] A₁) : (f : A₁ ≃+* A₁).symm = f.symm :=
  rfl


@[simp]
theorem symm_toRingEquiv : (e.symm : A₂ ≃+* A₁) = (e : A₁ ≃+* A₂).symm :=
  rfl


@[simp]
theorem apply_symm_apply (e : A₁ ≃ₐ[R] A₂) : ∀ x, e (e.symm x) = x :=
  e.toEquiv.apply_symm_apply


@[simp]
theorem symm_apply_apply (e : A₁ ≃ₐ[R] A₂) : ∀ x, e.symm (e x) = x :=
  e.toEquiv.symm_apply_apply


theorem symm_apply_eq (e : A₁ ≃ₐ[R] A₂) {x y} : e.symm x = y ↔ x = e y :=
  e.toEquiv.symm_apply_eq


theorem eq_symm_apply (e : A₁ ≃ₐ[R] A₂) {x y} : y = e.symm x ↔ e y = x :=
  e.toEquiv.eq_symm_apply


@[simp]
theorem comp_symm (e : A₁ ≃ₐ[R] A₂) : AlgHom.comp (e : A₁ →ₐ[R] A₂) ↑e.symm = AlgHom.id R A₂ := by
  /-
    R : Type uR
    A₁ : Type uA₁
    A₂ : Type uA₂
    inst✝⁴ : CommSemiring R
    inst✝³ : Semiring A₁
    inst✝² : Semiring A₂
    inst✝¹ : Algebra R A₁
    inst✝ : Algebra R A₂
    e : AlgEquiv R A₁ A₂
    ⊢ Eq ((↑e).comp ↑e.symm) (AlgHom.id R A₂)
  -/
  ext
  /-
    case H
    R : Type uR
    A₁ : Type uA₁
    A₂ : Type uA₂
    inst✝⁴ : CommSemiring R
    inst✝³ : Semiring A₁
    inst✝² : Semiring A₂
    inst✝¹ : Algebra R A₁
    inst✝ : Algebra R A₂
    e : AlgEquiv R A₁ A₂
    x✝ : A₂
    ⊢ Eq (((↑e).comp ↑e.symm) x✝) ((AlgHom.id R A₂) x✝)
  -/
  simp
  /-
    🎉 no goals
  -/


@[simp]
theorem symm_comp (e : A₁ ≃ₐ[R] A₂) : AlgHom.comp ↑e.symm (e : A₁ →ₐ[R] A₂) = AlgHom.id R A₁ := by
  /-
    R : Type uR
    A₁ : Type uA₁
    A₂ : Type uA₂
    inst✝⁴ : CommSemiring R
    inst✝³ : Semiring A₁
    inst✝² : Semiring A₂
    inst✝¹ : Algebra R A₁
    inst✝ : Algebra R A₂
    e : AlgEquiv R A₁ A₂
    ⊢ Eq ((↑e.symm).comp ↑e) (AlgHom.id R A₁)
  -/
  ext
  /-
    case H
    R : Type uR
    A₁ : Type uA₁
    A₂ : Type uA₂
    inst✝⁴ : CommSemiring R
    inst✝³ : Semiring A₁
    inst✝² : Semiring A₂
    inst✝¹ : Algebra R A₁
    inst✝ : Algebra R A₂
    e : AlgEquiv R A₁ A₂
    x✝ : A₁
    ⊢ Eq (((↑e.symm).comp ↑e) x✝) ((AlgHom.id R A₁) x✝)
  -/
  simp
  /-
    🎉 no goals
  -/


theorem leftInverse_symm (e : A₁ ≃ₐ[R] A₂) : Function.LeftInverse e.symm e :=
  e.left_inv


theorem rightInverse_symm (e : A₁ ≃ₐ[R] A₂) : Function.RightInverse e.symm e :=
  e.right_inv


/-- See Note [custom simps projection] -/
def Simps.apply (e : A₁ ≃ₐ[R] A₂) : A₁ → A₂ :=
  e

-- Porting note: the default simps projection was `e.toEquiv`, it should be `EquivLike.toEquiv`

/-- See Note [custom simps projection] -/
def Simps.toEquiv (e : A₁ ≃ₐ[R] A₂) : A₁ ≃ A₂ :=
  e


/-- See Note [custom simps projection] -/
def Simps.symm_apply (e : A₁ ≃ₐ[R] A₂) : A₂ → A₁ :=
  e.symm


/-- Algebra equivalences are transitive. -/
@[trans]
def trans (e₁ : A₁ ≃ₐ[R] A₂) (e₂ : A₂ ≃ₐ[R] A₃) : A₁ ≃ₐ[R] A₃ :=
  { e₁.toRingEquiv.trans e₂.toRingEquiv with
                                                            /-
                                                              R : Type uR
                                                              A₁ : Type uA₁
                                                              A₂ : Type uA₂
                                                              A₃ : Type uA₃
                                                              A₁' : Type uA₁'
                                                              A₂' : Type uA₂'
                                                              A₃' : Type uA₃'
                                                              inst✝¹² : CommSemiring R
                                                              inst✝¹¹ : Semiring A₁
                                                              inst✝¹⁰ : Semiring A₂
                                                              inst✝⁹ : Semiring A₃
                                                              inst✝⁸ : Semiring A₁'
                                                              inst✝⁷ : Semiring A₂'
                                                              inst✝⁶ : Semiring A₃'
                                                              inst✝⁵ : Algebra R A₁
                                                              inst✝⁴ : Algebra R A₂
                                                              inst✝³ : Algebra R A₃
                                                              inst✝² : Algebra R A₁'
                                                              inst✝¹ : Algebra R A₂'
                                                              inst✝ : Algebra R A₃'
                                                              e e₁ : AlgEquiv R A₁ A₂
                                                              e₂ : AlgEquiv R A₂ A₃
                                                              r : R
                                                              ⊢ Eq (e₂.toFun (e₁.toFun ((algebraMap R A₁) r))) ((algebraMap R A₃) r)
                                                            -/
    commutes' := fun r => show e₂.toFun (e₁.toFun _) = _ by rw [e₁.commutes', e₂.commutes'] }
                                                            /-
                                                              🎉 no goals
                                                            -/


@[simp]
theorem coe_trans (e₁ : A₁ ≃ₐ[R] A₂) (e₂ : A₂ ≃ₐ[R] A₃) : ⇑(e₁.trans e₂) = e₂ ∘ e₁ :=
  rfl


@[simp]
theorem trans_apply (e₁ : A₁ ≃ₐ[R] A₂) (e₂ : A₂ ≃ₐ[R] A₃) (x : A₁) : (e₁.trans e₂) x = e₂ (e₁ x) :=
  rfl


@[simp]
theorem symm_trans_apply (e₁ : A₁ ≃ₐ[R] A₂) (e₂ : A₂ ≃ₐ[R] A₃) (x : A₃) :
    (e₁.trans e₂).symm x = e₁.symm (e₂.symm x) :=
  rfl


/-- If `A₁` is equivalent to `A₁'` and `A₂` is equivalent to `A₂'`, then the type of maps
`A₁ →ₐ[R] A₂` is equivalent to the type of maps `A₁' →ₐ[R] A₂'`. -/
@[simps apply]
def arrowCongr (e₁ : A₁ ≃ₐ[R] A₁') (e₂ : A₂ ≃ₐ[R] A₂') : (A₁ →ₐ[R] A₂) ≃ (A₁' →ₐ[R] A₂') where
  toFun f := (e₂.toAlgHom.comp f).comp e₁.symm.toAlgHom
  invFun f := (e₂.symm.toAlgHom.comp f).comp e₁.toAlgHom
  left_inv f := by
    /-
      R : Type uR
      A₁ : Type uA₁
      A₂ : Type uA₂
      A₃ : Type uA₃
      A₁' : Type uA₁'
      A₂' : Type uA₂'
      A₃' : Type uA₃'
      inst✝¹² : CommSemiring R
      inst✝¹¹ : Semiring A₁
      inst✝¹⁰ : Semiring A₂
      inst✝⁹ : Semiring A₃
      inst✝⁸ : Semiring A₁'
      inst✝⁷ : Semiring A₂'
      inst✝⁶ : Semiring A₃'
      inst✝⁵ : Algebra R A₁
      inst✝⁴ : Algebra R A₂
      inst✝³ : Algebra R A₃
      inst✝² : Algebra R A₁'
      inst✝¹ : Algebra R A₂'
      inst✝ : Algebra R A₃'
      e : AlgEquiv R A₁ A₂
      e₁ : AlgEquiv R A₁ A₁'
      e₂ : AlgEquiv R A₂ A₂'
      f : AlgHom R A₁ A₂
      ⊢ Eq ((fun f => ((↑e₂.symm).comp f).comp ↑e₁) ((fun f => ((↑e₂).comp f).comp ↑ …
    -/
    simp only [AlgHom.comp_assoc, toAlgHom_eq_coe, symm_comp]
    /-
      R : Type uR
      A₁ : Type uA₁
      A₂ : Type uA₂
      A₃ : Type uA₃
      A₁' : Type uA₁'
      A₂' : Type uA₂'
      A₃' : Type uA₃'
      inst✝¹² : CommSemiring R
      inst✝¹¹ : Semiring A₁
      inst✝¹⁰ : Semiring A₂
      inst✝⁹ : Semiring A₃
      inst✝⁸ : Semiring A₁'
      inst✝⁷ : Semiring A₂'
      inst✝⁶ : Semiring A₃'
      inst✝⁵ : Algebra R A₁
      inst✝⁴ : Algebra R A₂
      inst✝³ : Algebra R A₃
      inst✝² : Algebra R A₁'
      inst✝¹ : Algebra R A₂'
      inst✝ : Algebra R A₃'
      e : AlgEquiv R A₁ A₂
      e₁ : AlgEquiv R A₁ A₁'
      e₂ : AlgEquiv R A₂ A₂'
      f : AlgHom R A₁ A₂
      ⊢ Eq ((↑e₂.symm).comp ((↑e₂).comp (f.comp (AlgHom.id R A₁)))) f
    -/
    simp only [← AlgHom.comp_assoc, symm_comp, AlgHom.id_comp, AlgHom.comp_id]
    /-
      🎉 no goals
    -/
  right_inv f := by
    /-
      R : Type uR
      A₁ : Type uA₁
      A₂ : Type uA₂
      A₃ : Type uA₃
      A₁' : Type uA₁'
      A₂' : Type uA₂'
      A₃' : Type uA₃'
      inst✝¹² : CommSemiring R
      inst✝¹¹ : Semiring A₁
      inst✝¹⁰ : Semiring A₂
      inst✝⁹ : Semiring A₃
      inst✝⁸ : Semiring A₁'
      inst✝⁷ : Semiring A₂'
      inst✝⁶ : Semiring A₃'
      inst✝⁵ : Algebra R A₁
      inst✝⁴ : Algebra R A₂
      inst✝³ : Algebra R A₃
      inst✝² : Algebra R A₁'
      inst✝¹ : Algebra R A₂'
      inst✝ : Algebra R A₃'
      e : AlgEquiv R A₁ A₂
      e₁ : AlgEquiv R A₁ A₁'
      e₂ : AlgEquiv R A₂ A₂'
      f : AlgHom R A₁' A₂'
      ⊢ Eq ((fun f => ((↑e₂).comp f).comp ↑e₁.symm) ((fun f => ((↑e₂.symm).comp f).c …
    -/
    simp only [AlgHom.comp_assoc, toAlgHom_eq_coe, comp_symm]
    /-
      R : Type uR
      A₁ : Type uA₁
      A₂ : Type uA₂
      A₃ : Type uA₃
      A₁' : Type uA₁'
      A₂' : Type uA₂'
      A₃' : Type uA₃'
      inst✝¹² : CommSemiring R
      inst✝¹¹ : Semiring A₁
      inst✝¹⁰ : Semiring A₂
      inst✝⁹ : Semiring A₃
      inst✝⁸ : Semiring A₁'
      inst✝⁷ : Semiring A₂'
      inst✝⁶ : Semiring A₃'
      inst✝⁵ : Algebra R A₁
      inst✝⁴ : Algebra R A₂
      inst✝³ : Algebra R A₃
      inst✝² : Algebra R A₁'
      inst✝¹ : Algebra R A₂'
      inst✝ : Algebra R A₃'
      e : AlgEquiv R A₁ A₂
      e₁ : AlgEquiv R A₁ A₁'
      e₂ : AlgEquiv R A₂ A₂'
      f : AlgHom R A₁' A₂'
      ⊢ Eq ((↑e₂).comp ((↑e₂.symm).comp (f.comp (AlgHom.id R A₁')))) f
    -/
    simp only [← AlgHom.comp_assoc, comp_symm, AlgHom.id_comp, AlgHom.comp_id]
    /-
      🎉 no goals
    -/


theorem arrowCongr_comp (e₁ : A₁ ≃ₐ[R] A₁') (e₂ : A₂ ≃ₐ[R] A₂')
    (e₃ : A₃ ≃ₐ[R] A₃') (f : A₁ →ₐ[R] A₂) (g : A₂ →ₐ[R] A₃) :
    arrowCongr e₁ e₃ (g.comp f) = (arrowCongr e₂ e₃ g).comp (arrowCongr e₁ e₂ f) := by
  /-
    R : Type uR
    A₁ : Type uA₁
    A₂ : Type uA₂
    A₃ : Type uA₃
    A₁' : Type uA₁'
    A₂' : Type uA₂'
    A₃' : Type uA₃'
    inst✝¹² : CommSemiring R
    inst✝¹¹ : Semiring A₁
    inst✝¹⁰ : Semiring A₂
    inst✝⁹ : Semiring A₃
    inst✝⁸ : Semiring A₁'
    inst✝⁷ : Semiring A₂'
    inst✝⁶ : Semiring A₃'
    inst✝⁵ : Algebra R A₁
    inst✝⁴ : Algebra R A₂
    inst✝³ : Algebra R A₃
    inst✝² : Algebra R A₁'
    inst✝¹ : Algebra R A₂'
    inst✝ : Algebra R A₃'
    e₁ : AlgEquiv R A₁ A₁'
    e₂ : AlgEquiv R A₂ A₂'
    e₃ : AlgEquiv R A₃ A₃'
    f : AlgHom R A₁ A₂
    g : AlgHom R A₂ A₃
    ⊢ Eq ((e₁.arrowCongr e₃) (g.comp f)) (((e₂.arrowCongr e₃) g).comp ((e₁.arrowCo …
  -/
  ext
  /-
    case H
    R : Type uR
    A₁ : Type uA₁
    A₂ : Type uA₂
    A₃ : Type uA₃
    A₁' : Type uA₁'
    A₂' : Type uA₂'
    A₃' : Type uA₃'
    inst✝¹² : CommSemiring R
    inst✝¹¹ : Semiring A₁
    inst✝¹⁰ : Semiring A₂
    inst✝⁹ : Semiring A₃
    inst✝⁸ : Semiring A₁'
    inst✝⁷ : Semiring A₂'
    inst✝⁶ : Semiring A₃'
    inst✝⁵ : Algebra R A₁
    inst✝⁴ : Algebra R A₂
    inst✝³ : Algebra R A₃
    inst✝² : Algebra R A₁'
    inst✝¹ : Algebra R A₂'
    inst✝ : Algebra R A₃'
    e₁ : AlgEquiv R A₁ A₁'
    e₂ : AlgEquiv R A₂ A₂'
    e₃ : AlgEquiv R A₃ A₃'
    f : AlgHom R A₁ A₂
    g : AlgHom R A₂ A₃
    x✝ : A₁'
    ⊢ Eq (((e₁.arrowCongr e₃) (g.comp f)) x✝) ((((e₂.arrowCongr e₃) g).comp ((e₁.a …
  -/
  simp only [arrowCongr, Equiv.coe_fn_mk, AlgHom.comp_apply]
  /-
    case H
    R : Type uR
    A₁ : Type uA₁
    A₂ : Type uA₂
    A₃ : Type uA₃
    A₁' : Type uA₁'
    A₂' : Type uA₂'
    A₃' : Type uA₃'
    inst✝¹² : CommSemiring R
    inst✝¹¹ : Semiring A₁
    inst✝¹⁰ : Semiring A₂
    inst✝⁹ : Semiring A₃
    inst✝⁸ : Semiring A₁'
    inst✝⁷ : Semiring A₂'
    inst✝⁶ : Semiring A₃'
    inst✝⁵ : Algebra R A₁
    inst✝⁴ : Algebra R A₂
    inst✝³ : Algebra R A₃
    inst✝² : Algebra R A₁'
    inst✝¹ : Algebra R A₂'
    inst✝ : Algebra R A₃'
    e₁ : AlgEquiv R A₁ A₁'
    e₂ : AlgEquiv R A₂ A₂'
    e₃ : AlgEquiv R A₃ A₃'
    f : AlgHom R A₁ A₂
    g : AlgHom R A₂ A₃
    x✝ : A₁'
    ⊢ Eq (↑e₃ (g (f (↑e₁.symm x✝)))) (↑e₃ (g (↑e₂.symm (↑e₂ (f (↑e₁.symm x✝))))))
  -/
  congr
  /-
    case H.h.e_6.h.h.e_6.h
    R : Type uR
    A₁ : Type uA₁
    A₂ : Type uA₂
    A₃ : Type uA₃
    A₁' : Type uA₁'
    A₂' : Type uA₂'
    A₃' : Type uA₃'
    inst✝¹² : CommSemiring R
    inst✝¹¹ : Semiring A₁
    inst✝¹⁰ : Semiring A₂
    inst✝⁹ : Semiring A₃
    inst✝⁸ : Semiring A₁'
    inst✝⁷ : Semiring A₂'
    inst✝⁶ : Semiring A₃'
    inst✝⁵ : Algebra R A₁
    inst✝⁴ : Algebra R A₂
    inst✝³ : Algebra R A₃
    inst✝² : Algebra R A₁'
    inst✝¹ : Algebra R A₂'
    inst✝ : Algebra R A₃'
    e₁ : AlgEquiv R A₁ A₁'
    e₂ : AlgEquiv R A₂ A₂'
    e₃ : AlgEquiv R A₃ A₃'
    f : AlgHom R A₁ A₂
    g : AlgHom R A₂ A₃
    x✝ : A₁'
    ⊢ Eq (f (↑e₁.symm x✝)) (↑e₂.symm (↑e₂ (f (↑e₁.symm x✝))))
  -/
  exact (e₂.symm_apply_apply _).symm
  /-
    🎉 no goals
  -/


@[simp]
theorem arrowCongr_refl : arrowCongr AlgEquiv.refl AlgEquiv.refl = Equiv.refl (A₁ →ₐ[R] A₂) :=
  rfl


@[simp]
theorem arrowCongr_trans (e₁ : A₁ ≃ₐ[R] A₂) (e₁' : A₁' ≃ₐ[R] A₂')
    (e₂ : A₂ ≃ₐ[R] A₃) (e₂' : A₂' ≃ₐ[R] A₃') :
    arrowCongr (e₁.trans e₂) (e₁'.trans e₂') = (arrowCongr e₁ e₁').trans (arrowCongr e₂ e₂') :=
  rfl


@[simp]
theorem arrowCongr_symm (e₁ : A₁ ≃ₐ[R] A₁') (e₂ : A₂ ≃ₐ[R] A₂') :
    (arrowCongr e₁ e₂).symm = arrowCongr e₁.symm e₂.symm :=
  rfl


/-- If `A₁` is equivalent to `A₂` and `A₁'` is equivalent to `A₂'`, then the type of maps
`A₁ ≃ₐ[R] A₁'` is equivalent to the type of maps `A₂ ≃ ₐ[R] A₂'`.

This is the `AlgEquiv` version of `AlgEquiv.arrowCongr`. -/
@[simps apply]
def equivCongr (e : A₁ ≃ₐ[R] A₂) (e' : A₁' ≃ₐ[R] A₂') : (A₁ ≃ₐ[R] A₁') ≃ A₂ ≃ₐ[R] A₂' where
  toFun ψ := e.symm.trans (ψ.trans e')
  invFun ψ := e.trans (ψ.trans e'.symm)
  left_inv ψ := by
    /-
      R : Type uR
      A₁ : Type uA₁
      A₂ : Type uA₂
      A₃ : Type uA₃
      A₁' : Type uA₁'
      A₂' : Type uA₂'
      A₃' : Type uA₃'
      inst✝¹² : CommSemiring R
      inst✝¹¹ : Semiring A₁
      inst✝¹⁰ : Semiring A₂
      inst✝⁹ : Semiring A₃
      inst✝⁸ : Semiring A₁'
      inst✝⁷ : Semiring A₂'
      inst✝⁶ : Semiring A₃'
      inst✝⁵ : Algebra R A₁
      inst✝⁴ : Algebra R A₂
      inst✝³ : Algebra R A₃
      inst✝² : Algebra R A₁'
      inst✝¹ : Algebra R A₂'
      inst✝ : Algebra R A₃'
      e✝ e : AlgEquiv R A₁ A₂
      e' : AlgEquiv R A₁' A₂'
      ψ : AlgEquiv R A₁ A₁'
      ⊢ Eq ((fun ψ => e.trans (ψ.trans e'.symm)) ((fun ψ => e.symm.trans (ψ.trans e' …
    -/
    ext
    /-
      case h
      R : Type uR
      A₁ : Type uA₁
      A₂ : Type uA₂
      A₃ : Type uA₃
      A₁' : Type uA₁'
      A₂' : Type uA₂'
      A₃' : Type uA₃'
      inst✝¹² : CommSemiring R
      inst✝¹¹ : Semiring A₁
      inst✝¹⁰ : Semiring A₂
      inst✝⁹ : Semiring A₃
      inst✝⁸ : Semiring A₁'
      inst✝⁷ : Semiring A₂'
      inst✝⁶ : Semiring A₃'
      inst✝⁵ : Algebra R A₁
      inst✝⁴ : Algebra R A₂
      inst✝³ : Algebra R A₃
      inst✝² : Algebra R A₁'
      inst✝¹ : Algebra R A₂'
      inst✝ : Algebra R A₃'
      e✝ e : AlgEquiv R A₁ A₂
      e' : AlgEquiv R A₁' A₂'
      ψ : AlgEquiv R A₁ A₁'
      a✝ : A₁
      ⊢ Eq (((fun ψ => e.trans (ψ.trans e'.symm)) ((fun ψ => e.symm.trans (ψ.trans e …
    -/
    simp_rw [trans_apply, symm_apply_apply]
    /-
      🎉 no goals
    -/
  right_inv ψ := by
    /-
      R : Type uR
      A₁ : Type uA₁
      A₂ : Type uA₂
      A₃ : Type uA₃
      A₁' : Type uA₁'
      A₂' : Type uA₂'
      A₃' : Type uA₃'
      inst✝¹² : CommSemiring R
      inst✝¹¹ : Semiring A₁
      inst✝¹⁰ : Semiring A₂
      inst✝⁹ : Semiring A₃
      inst✝⁸ : Semiring A₁'
      inst✝⁷ : Semiring A₂'
      inst✝⁶ : Semiring A₃'
      inst✝⁵ : Algebra R A₁
      inst✝⁴ : Algebra R A₂
      inst✝³ : Algebra R A₃
      inst✝² : Algebra R A₁'
      inst✝¹ : Algebra R A₂'
      inst✝ : Algebra R A₃'
      e✝ e : AlgEquiv R A₁ A₂
      e' : AlgEquiv R A₁' A₂'
      ψ : AlgEquiv R A₂ A₂'
      ⊢ Eq ((fun ψ => e.symm.trans (ψ.trans e')) ((fun ψ => e.trans (ψ.trans e'.symm …
    -/
    ext
    /-
      case h
      R : Type uR
      A₁ : Type uA₁
      A₂ : Type uA₂
      A₃ : Type uA₃
      A₁' : Type uA₁'
      A₂' : Type uA₂'
      A₃' : Type uA₃'
      inst✝¹² : CommSemiring R
      inst✝¹¹ : Semiring A₁
      inst✝¹⁰ : Semiring A₂
      inst✝⁹ : Semiring A₃
      inst✝⁸ : Semiring A₁'
      inst✝⁷ : Semiring A₂'
      inst✝⁶ : Semiring A₃'
      inst✝⁵ : Algebra R A₁
      inst✝⁴ : Algebra R A₂
      inst✝³ : Algebra R A₃
      inst✝² : Algebra R A₁'
      inst✝¹ : Algebra R A₂'
      inst✝ : Algebra R A₃'
      e✝ e : AlgEquiv R A₁ A₂
      e' : AlgEquiv R A₁' A₂'
      ψ : AlgEquiv R A₂ A₂'
      a✝ : A₂
      ⊢ Eq (((fun ψ => e.symm.trans (ψ.trans e')) ((fun ψ => e.trans (ψ.trans e'.sym …
    -/
    simp_rw [trans_apply, apply_symm_apply]
    /-
      🎉 no goals
    -/


@[simp]
theorem equivCongr_refl : equivCongr AlgEquiv.refl AlgEquiv.refl = Equiv.refl (A₁ ≃ₐ[R] A₁') :=
  rfl


@[simp]
theorem equivCongr_symm (e : A₁ ≃ₐ[R] A₂) (e' : A₁' ≃ₐ[R] A₂') :
    (equivCongr e e').symm = equivCongr e.symm e'.symm :=
  rfl


@[simp]
theorem equivCongr_trans (e₁₂ : A₁ ≃ₐ[R] A₂) (e₁₂' : A₁' ≃ₐ[R] A₂')
    (e₂₃ : A₂ ≃ₐ[R] A₃) (e₂₃' : A₂' ≃ₐ[R] A₃') :
    (equivCongr e₁₂ e₁₂').trans (equivCongr e₂₃ e₂₃') =
      equivCongr (e₁₂.trans e₂₃) (e₁₂'.trans e₂₃') :=
  rfl


/-- If an algebra morphism has an inverse, it is an algebra isomorphism. -/
@[simps]
def ofAlgHom (f : A₁ →ₐ[R] A₂) (g : A₂ →ₐ[R] A₁) (h₁ : f.comp g = AlgHom.id R A₂)
    (h₂ : g.comp f = AlgHom.id R A₁) : A₁ ≃ₐ[R] A₂ :=
  { f with
    toFun := f
    invFun := g
    left_inv := AlgHom.ext_iff.1 h₂
    right_inv := AlgHom.ext_iff.1 h₁ }


theorem coe_algHom_ofAlgHom (f : A₁ →ₐ[R] A₂) (g : A₂ →ₐ[R] A₁) (h₁ h₂) :
    ↑(ofAlgHom f g h₁ h₂) = f :=
  rfl


@[simp]
theorem ofAlgHom_coe_algHom (f : A₁ ≃ₐ[R] A₂) (g : A₂ →ₐ[R] A₁) (h₁ h₂) :
    ofAlgHom (↑f) g h₁ h₂ = f :=
  ext fun _ => rfl


theorem ofAlgHom_symm (f : A₁ →ₐ[R] A₂) (g : A₂ →ₐ[R] A₁) (h₁ h₂) :
    (ofAlgHom f g h₁ h₂).symm = ofAlgHom g f h₂ h₁ :=
  rfl


/-- Promotes a bijective algebra homomorphism to an algebra equivalence. -/
noncomputable def ofBijective (f : A₁ →ₐ[R] A₂) (hf : Function.Bijective f) : A₁ ≃ₐ[R] A₂ :=
  { RingEquiv.ofBijective (f : A₁ →+* A₂) hf, f with }


@[simp]
theorem coe_ofBijective {f : A₁ →ₐ[R] A₂} {hf : Function.Bijective f} :
    (AlgEquiv.ofBijective f hf : A₁ → A₂) = f :=
  rfl


theorem ofBijective_apply {f : A₁ →ₐ[R] A₂} {hf : Function.Bijective f} (a : A₁) :
    (AlgEquiv.ofBijective f hf) a = f a :=
  rfl


/-- Forgetting the multiplicative structures, an equivalence of algebras is a linear equivalence. -/
@[simps apply]
def toLinearEquiv (e : A₁ ≃ₐ[R] A₂) : A₁ ≃ₗ[R] A₂ :=
  { e with
    toFun := e
    map_smul' := map_smul e
    invFun := e.symm }


@[simp]
theorem toLinearEquiv_refl : (AlgEquiv.refl : A₁ ≃ₐ[R] A₁).toLinearEquiv = LinearEquiv.refl R A₁ :=
  rfl


@[simp]
theorem toLinearEquiv_symm (e : A₁ ≃ₐ[R] A₂) : e.toLinearEquiv.symm = e.symm.toLinearEquiv :=
  rfl


@[simp]
theorem toLinearEquiv_trans (e₁ : A₁ ≃ₐ[R] A₂) (e₂ : A₂ ≃ₐ[R] A₃) :
    (e₁.trans e₂).toLinearEquiv = e₁.toLinearEquiv.trans e₂.toLinearEquiv :=
  rfl


theorem toLinearEquiv_injective : Function.Injective (toLinearEquiv : _ → A₁ ≃ₗ[R] A₂) :=
  fun _ _ h => ext <| LinearEquiv.congr_fun h


/-- Interpret an algebra equivalence as a linear map. -/
def toLinearMap : A₁ →ₗ[R] A₂ :=
  e.toAlgHom.toLinearMap


@[simp]
theorem toAlgHom_toLinearMap : (e : A₁ →ₐ[R] A₂).toLinearMap = e.toLinearMap :=
  rfl


theorem toLinearMap_ofAlgHom (f : A₁ →ₐ[R] A₂) (g : A₂ →ₐ[R] A₁) (h₁ h₂) :
    (ofAlgHom f g h₁ h₂).toLinearMap = f.toLinearMap :=
  LinearMap.ext fun _ => rfl


@[simp]
theorem toLinearEquiv_toLinearMap : e.toLinearEquiv.toLinearMap = e.toLinearMap :=
  rfl


@[simp]
theorem toLinearMap_apply (x : A₁) : e.toLinearMap x = e x :=
  rfl


theorem toLinearMap_injective : Function.Injective (toLinearMap : _ → A₁ →ₗ[R] A₂) := fun _ _ h =>
  ext <| LinearMap.congr_fun h


@[simp]
theorem trans_toLinearMap (f : A₁ ≃ₐ[R] A₂) (g : A₂ ≃ₐ[R] A₃) :
    (f.trans g).toLinearMap = g.toLinearMap.comp f.toLinearMap :=
  rfl


/--
Upgrade a linear equivalence to an algebra equivalence,
given that it distributes over multiplication and the identity
-/
@[simps apply]
def ofLinearEquiv : A₁ ≃ₐ[R] A₂ :=
  { l with
    toFun := l
    invFun := l.symm
    map_mul' := map_mul
    commutes' := (AlgHom.ofLinearMap l map_one map_mul : A₁ →ₐ[R] A₂).commutes }


/-- Auxiliary definition to avoid looping in `dsimp` with `AlgEquiv.ofLinearEquiv_symm`. -/
protected def ofLinearEquiv_symm.aux := (ofLinearEquiv l map_one map_mul).symm


@[simp]
theorem ofLinearEquiv_symm :
    (ofLinearEquiv l map_one map_mul).symm =
      ofLinearEquiv l.symm
        (_root_.map_one <| ofLinearEquiv_symm.aux l map_one map_mul)
        (_root_.map_mul <| ofLinearEquiv_symm.aux l map_one map_mul) :=
  rfl


@[simp]
theorem ofLinearEquiv_toLinearEquiv (map_mul) (map_one) :
    ofLinearEquiv e.toLinearEquiv map_mul map_one = e :=
  rfl


@[simp]
theorem toLinearEquiv_ofLinearEquiv : toLinearEquiv (ofLinearEquiv l map_one map_mul) = l :=
  rfl


/-- Promotes a linear `RingEquiv` to an `AlgEquiv`. -/
@[simps apply symm_apply toEquiv] -- Porting note: don't want redundant `toEquiv_symm_apply` simps
def ofRingEquiv {f : A₁ ≃+* A₂} (hf : ∀ x, f (algebraMap R A₁ x) = algebraMap R A₂ x) :
    A₁ ≃ₐ[R] A₂ :=
  { f with
    toFun := f
    invFun := f.symm
    commutes' := hf }


@[stacks 09HR]
instance aut : Group (A₁ ≃ₐ[R] A₁) where
  mul ϕ ψ := ψ.trans ϕ
  mul_assoc _ _ _ := rfl
  one := refl
  one_mul _ := ext fun _ => rfl
  mul_one _ := ext fun _ => rfl
  inv := symm
  inv_mul_cancel ϕ := ext <| symm_apply_apply ϕ


theorem aut_mul (ϕ ψ : A₁ ≃ₐ[R] A₁) : ϕ * ψ = ψ.trans ϕ :=
  rfl


theorem aut_one : 1 = AlgEquiv.refl (R := R) (A₁ := A₁) :=
  rfl


@[simp]
theorem one_apply (x : A₁) : (1 : A₁ ≃ₐ[R] A₁) x = x :=
  rfl


@[simp]
theorem mul_apply (e₁ e₂ : A₁ ≃ₐ[R] A₁) (x : A₁) : (e₁ * e₂) x = e₁ (e₂ x) :=
  rfl


/-- An algebra isomorphism induces a group isomorphism between automorphism groups.

This is a more bundled version of `AlgEquiv.equivCongr`. -/
@[simps apply]
def autCongr (ϕ : A₁ ≃ₐ[R] A₂) : (A₁ ≃ₐ[R] A₁) ≃* A₂ ≃ₐ[R] A₂ where
  __ := equivCongr ϕ ϕ
  toFun ψ := ϕ.symm.trans (ψ.trans ϕ)
  invFun ψ := ϕ.trans (ψ.trans ϕ.symm)
  map_mul' ψ χ := by
    /-
      R : Type uR
      A₁ : Type uA₁
      A₂ : Type uA₂
      A₃ : Type uA₃
      A₁' : Type uA₁'
      A₂' : Type uA₂'
      A₃' : Type uA₃'
      inst✝¹² : CommSemiring R
      inst✝¹¹ : Semiring A₁
      inst✝¹⁰ : Semiring A₂
      inst✝⁹ : Semiring A₃
      inst✝⁸ : Semiring A₁'
      inst✝⁷ : Semiring A₂'
      inst✝⁶ : Semiring A₃'
      inst✝⁵ : Algebra R A₁
      inst✝⁴ : Algebra R A₂
      inst✝³ : Algebra R A₃
      inst✝² : Algebra R A₁'
      inst✝¹ : Algebra R A₂'
      inst✝ : Algebra R A₃'
      e ϕ : AlgEquiv R A₁ A₂
      ψ χ : AlgEquiv R A₁ A₁
      ⊢ Eq ({ toFun := fun ψ => ϕ.symm.trans (ψ.trans ϕ), invFun := fun ψ => ϕ.trans …
    -/
    ext
    /-
      case h
      R : Type uR
      A₁ : Type uA₁
      A₂ : Type uA₂
      A₃ : Type uA₃
      A₁' : Type uA₁'
      A₂' : Type uA₂'
      A₃' : Type uA₃'
      inst✝¹² : CommSemiring R
      inst✝¹¹ : Semiring A₁
      inst✝¹⁰ : Semiring A₂
      inst✝⁹ : Semiring A₃
      inst✝⁸ : Semiring A₁'
      inst✝⁷ : Semiring A₂'
      inst✝⁶ : Semiring A₃'
      inst✝⁵ : Algebra R A₁
      inst✝⁴ : Algebra R A₂
      inst✝³ : Algebra R A₃
      inst✝² : Algebra R A₁'
      inst✝¹ : Algebra R A₂'
      inst✝ : Algebra R A₃'
      e ϕ : AlgEquiv R A₁ A₂
      ψ χ : AlgEquiv R A₁ A₁
      a✝ : A₂
      ⊢ Eq (({ toFun := fun ψ => ϕ.symm.trans (ψ.trans ϕ), invFun := fun ψ => ϕ.tran …
    -/
    simp only [mul_apply, trans_apply, symm_apply_apply]
    /-
      🎉 no goals
    -/


@[simp]
theorem autCongr_refl : autCongr AlgEquiv.refl = MulEquiv.refl (A₁ ≃ₐ[R] A₁) := rfl


@[simp]
theorem autCongr_symm (ϕ : A₁ ≃ₐ[R] A₂) : (autCongr ϕ).symm = autCongr ϕ.symm :=
  rfl


@[simp]
theorem autCongr_trans (ϕ : A₁ ≃ₐ[R] A₂) (ψ : A₂ ≃ₐ[R] A₃) :
    (autCongr ϕ).trans (autCongr ψ) = autCongr (ϕ.trans ψ) :=
  rfl


/-- The tautological action by `A₁ ≃ₐ[R] A₁` on `A₁`.

This generalizes `Function.End.applyMulAction`. -/
instance applyMulSemiringAction : MulSemiringAction (A₁ ≃ₐ[R] A₁) A₁ where
  smul := (· <| ·)
  smul_zero := map_zero
  smul_add := map_add
  smul_one := map_one
  smul_mul := map_mul
  one_smul _ := rfl
  mul_smul _ _ _ := rfl


@[simp]
protected theorem smul_def (f : A₁ ≃ₐ[R] A₁) (a : A₁) : f • a = f a :=
  rfl


instance apply_faithfulSMul : FaithfulSMul (A₁ ≃ₐ[R] A₁) A₁ :=
  ⟨AlgEquiv.ext⟩


instance apply_smulCommClass {S} [SMul S R] [SMul S A₁] [IsScalarTower S R A₁] :
    SMulCommClass S (A₁ ≃ₐ[R] A₁) A₁ where
  smul_comm r e a := (e.toLinearEquiv.map_smul_of_tower r a).symm


instance apply_smulCommClass' {S} [SMul S R] [SMul S A₁] [IsScalarTower S R A₁] :
    SMulCommClass (A₁ ≃ₐ[R] A₁) S A₁ :=
  SMulCommClass.symm _ _ _


instance : MulDistribMulAction (A₁ ≃ₐ[R] A₁) A₁ˣ where
  smul := fun f => Units.map f
                          /-
                            R : Type uR
                            A₁ : Type uA₁
                            A₂ : Type uA₂
                            A₃ : Type uA₃
                            A₁' : Type uA₁'
                            A₂' : Type uA₂'
                            A₃' : Type uA₃'
                            inst✝¹² : CommSemiring R
                            inst✝¹¹ : Semiring A₁
                            inst✝¹⁰ : Semiring A₂
                            inst✝⁹ : Semiring A₃
                            inst✝⁸ : Semiring A₁'
                            inst✝⁷ : Semiring A₂'
                            inst✝⁶ : Semiring A₃'
                            inst✝⁵ : Algebra R A₁
                            inst✝⁴ : Algebra R A₂
                            inst✝³ : Algebra R A₃
                            inst✝² : Algebra R A₁'
                            inst✝¹ : Algebra R A₂'
                            inst✝ : Algebra R A₃'
                            e : AlgEquiv R A₁ A₂
                            x : Units A₁
                            ⊢ Eq (HSMul.hSMul 1 x) x
                          -/
  one_smul := fun x => by ext; rfl
                               /-
                                 🎉 no goals
                               -/
                              /-
                                R : Type uR
                                A₁ : Type uA₁
                                A₂ : Type uA₂
                                A₃ : Type uA₃
                                A₁' : Type uA₁'
                                A₂' : Type uA₂'
                                A₃' : Type uA₃'
                                inst✝¹² : CommSemiring R
                                inst✝¹¹ : Semiring A₁
                                inst✝¹⁰ : Semiring A₂
                                inst✝⁹ : Semiring A₃
                                inst✝⁸ : Semiring A₁'
                                inst✝⁷ : Semiring A₂'
                                inst✝⁶ : Semiring A₃'
                                inst✝⁵ : Algebra R A₁
                                inst✝⁴ : Algebra R A₂
                                inst✝³ : Algebra R A₃
                                inst✝² : Algebra R A₁'
                                inst✝¹ : Algebra R A₂'
                                inst✝ : Algebra R A₃'
                                e : AlgEquiv R A₁ A₂
                                x y : AlgEquiv R A₁ A₁
                                z : Units A₁
                                ⊢ Eq (HSMul.hSMul (HMul.hMul x y) z) (HSMul.hSMul x (HSMul.hSMul y z))
                              -/
  mul_smul := fun x y z => by ext; rfl
                                   /-
                                     🎉 no goals
                                   -/
                              /-
                                R : Type uR
                                A₁ : Type uA₁
                                A₂ : Type uA₂
                                A₃ : Type uA₃
                                A₁' : Type uA₁'
                                A₂' : Type uA₂'
                                A₃' : Type uA₃'
                                inst✝¹² : CommSemiring R
                                inst✝¹¹ : Semiring A₁
                                inst✝¹⁰ : Semiring A₂
                                inst✝⁹ : Semiring A₃
                                inst✝⁸ : Semiring A₁'
                                inst✝⁷ : Semiring A₂'
                                inst✝⁶ : Semiring A₃'
                                inst✝⁵ : Algebra R A₁
                                inst✝⁴ : Algebra R A₂
                                inst✝³ : Algebra R A₃
                                inst✝² : Algebra R A₁'
                                inst✝¹ : Algebra R A₂'
                                inst✝ : Algebra R A₃'
                                e : AlgEquiv R A₁ A₂
                                x : AlgEquiv R A₁ A₁
                                y z : Units A₁
                                ⊢ Eq (HSMul.hSMul x (HMul.hMul y z)) (HMul.hMul (HSMul.hSMul x y) (HSMul.hSMul …
                              -/
  smul_mul := fun x y z => by ext; exact map_mul x _ _
                                   /-
                                     🎉 no goals
                                   -/
                          /-
                            R : Type uR
                            A₁ : Type uA₁
                            A₂ : Type uA₂
                            A₃ : Type uA₃
                            A₁' : Type uA₁'
                            A₂' : Type uA₂'
                            A₃' : Type uA₃'
                            inst✝¹² : CommSemiring R
                            inst✝¹¹ : Semiring A₁
                            inst✝¹⁰ : Semiring A₂
                            inst✝⁹ : Semiring A₃
                            inst✝⁸ : Semiring A₁'
                            inst✝⁷ : Semiring A₂'
                            inst✝⁶ : Semiring A₃'
                            inst✝⁵ : Algebra R A₁
                            inst✝⁴ : Algebra R A₂
                            inst✝³ : Algebra R A₃
                            inst✝² : Algebra R A₁'
                            inst✝¹ : Algebra R A₂'
                            inst✝ : Algebra R A₃'
                            e : AlgEquiv R A₁ A₂
                            x : AlgEquiv R A₁ A₁
                            ⊢ Eq (HSMul.hSMul x 1) 1
                          -/
  smul_one := fun x => by ext; exact map_one x
                               /-
                                 🎉 no goals
                               -/


@[simp]
theorem smul_units_def (f : A₁ ≃ₐ[R] A₁) (x : A₁ˣ) :
    f • x = Units.map f x := rfl


@[simp]
theorem algebraMap_eq_apply (e : A₁ ≃ₐ[R] A₂) {y : R} {x : A₁} :
    algebraMap R A₂ y = e x ↔ algebraMap R A₁ y = x :=
               /-
                 R : Type uR
                 A₁ : Type uA₁
                 A₂ : Type uA₂
                 inst✝⁴ : CommSemiring R
                 inst✝³ : Semiring A₁
                 inst✝² : Semiring A₂
                 inst✝¹ : Algebra R A₁
                 inst✝ : Algebra R A₂
                 e : AlgEquiv R A₁ A₂
                 y : R
                 x : A₁
                 h : Eq ((algebraMap R A₂) y) (e x)
                 ⊢ Eq ((algebraMap R A₁) y) x
               -/
  ⟨fun h => by simpa using e.symm.toAlgHom.algebraMap_eq_apply h, fun h =>
               /-
                 🎉 no goals
               -/
    e.toAlgHom.algebraMap_eq_apply h⟩


/-- `AlgEquiv.toLinearMap` as a `MonoidHom`. -/
@[simps]
def toLinearMapHom (R A) [CommSemiring R] [Semiring A] [Algebra R A] :
    (A ≃ₐ[R] A) →* A →ₗ[R] A where
  toFun := AlgEquiv.toLinearMap
  map_one' := rfl
  map_mul' := fun _ _ ↦ rfl


lemma pow_toLinearMap (σ : A₁ ≃ₐ[R] A₁) (n : ℕ) :
    (σ ^ n).toLinearMap = σ.toLinearMap ^ n :=
  (AlgEquiv.toLinearMapHom R A₁).map_pow σ n


@[simp]
lemma one_toLinearMap :
    (1 : A₁ ≃ₐ[R] A₁).toLinearMap = 1 := rfl


/-- The units group of `S →ₐ[R] S` is `S ≃ₐ[R] S`.
See `LinearMap.GeneralLinearGroup.generalLinearEquiv` for the linear map version. -/
@[simps]
def algHomUnitsEquiv (R S : Type*) [CommSemiring R] [Semiring S] [Algebra R S] :
    (S →ₐ[R] S)ˣ ≃* (S ≃ₐ[R] S) where
  toFun := fun f ↦
    { (f : S →ₐ[R] S) with
      invFun := ↑(f⁻¹)
                                                                  /-
                                                                    R✝ : Type uR
                                                                    A₁ : Type uA₁
                                                                    A₂ : Type uA₂
                                                                    A₃ : Type uA₃
                                                                    A₁' : Type uA₁'
                                                                    A₂' : Type uA₂'
                                                                    A₃' : Type uA₃'
                                                                    inst✝¹⁵ : CommSemiring R✝
                                                                    inst✝¹⁴ : Semiring A₁
                                                                    inst✝¹³ : Semiring A₂
                                                                    inst✝¹² : Semiring A₃
                                                                    inst✝¹¹ : Semiring A₁'
                                                                    inst✝¹⁰ : Semiring A₂'
                                                                    inst✝⁹ : Semiring A₃'
                                                                    inst✝⁸ : Algebra R✝ A₁
                                                                    inst✝⁷ : Algebra R✝ A₂
                                                                    inst✝⁶ : Algebra R✝ A₃
                                                                    inst✝⁵ : Algebra R✝ A₁'
                                                                    inst✝⁴ : Algebra R✝ A₂'
                                                                    inst✝³ : Algebra R✝ A₃'
                                                                    e : AlgEquiv R✝ A₁ A₂
                                                                    R : Type u_1
                                                                    S : Type u_2
                                                                    inst✝² : CommSemiring R
                                                                    inst✝¹ : Semiring S
                                                                    inst✝ : Algebra R S
                                                                    f : Units (AlgHom R S S)
                                                                    x : S
                                                                    ⊢ Eq (↑(HMul.hMul (Inv.inv f) f) x) x
                                                                  -/
      left_inv := (fun x ↦ show (↑(f⁻¹ * f) : S →ₐ[R] S) x = x by rw [inv_mul_cancel]; rfl)
                                                                                       /-
                                                                                         🎉 no goals
                                                                                       -/
                                                                   /-
                                                                     R✝ : Type uR
                                                                     A₁ : Type uA₁
                                                                     A₂ : Type uA₂
                                                                     A₃ : Type uA₃
                                                                     A₁' : Type uA₁'
                                                                     A₂' : Type uA₂'
                                                                     A₃' : Type uA₃'
                                                                     inst✝¹⁵ : CommSemiring R✝
                                                                     inst✝¹⁴ : Semiring A₁
                                                                     inst✝¹³ : Semiring A₂
                                                                     inst✝¹² : Semiring A₃
                                                                     inst✝¹¹ : Semiring A₁'
                                                                     inst✝¹⁰ : Semiring A₂'
                                                                     inst✝⁹ : Semiring A₃'
                                                                     inst✝⁸ : Algebra R✝ A₁
                                                                     inst✝⁷ : Algebra R✝ A₂
                                                                     inst✝⁶ : Algebra R✝ A₃
                                                                     inst✝⁵ : Algebra R✝ A₁'
                                                                     inst✝⁴ : Algebra R✝ A₂'
                                                                     inst✝³ : Algebra R✝ A₃'
                                                                     e : AlgEquiv R✝ A₁ A₂
                                                                     R : Type u_1
                                                                     S : Type u_2
                                                                     inst✝² : CommSemiring R
                                                                     inst✝¹ : Semiring S
                                                                     inst✝ : Algebra R S
                                                                     f : Units (AlgHom R S S)
                                                                     x : S
                                                                     ⊢ Eq (↑(HMul.hMul f (Inv.inv f)) x) x
                                                                   -/
      right_inv := (fun x ↦ show (↑(f * f⁻¹) : S →ₐ[R] S) x = x by rw [mul_inv_cancel]; rfl) }
                                                                                        /-
                                                                                          🎉 no goals
                                                                                        -/
  invFun := fun f ↦ ⟨f, f.symm, f.comp_symm, f.symm_comp⟩
  left_inv := fun _ ↦ rfl
  right_inv := fun _ ↦ rfl
  map_mul' := fun _ _ ↦ rfl


/-- See also `Finite.algHom` -/
instance _root_.Finite.algEquiv [Finite (A₁ →ₐ[R] A₂)] : Finite (A₁ ≃ₐ[R] A₂) :=
  Finite.of_injective _ AlgEquiv.coe_algHom_injective


@[deprecated map_neg (since := "2024-06-20")]
protected theorem map_neg (x) : e (-x) = -e x :=
  map_neg e x


@[deprecated map_sub (since := "2024-06-20")]
protected theorem map_sub (x y) : e (x - y) = e x - e y :=
  map_sub e x y


/-- Each element of the group defines an algebra equivalence.

This is a stronger version of `MulSemiringAction.toRingEquiv` and
`DistribMulAction.toLinearEquiv`. -/
@[simps! apply symm_apply toEquiv] -- Porting note: don't want redundant simps lemma `toEquiv_symm`
def toAlgEquiv (g : G) : A ≃ₐ[R] A :=
  { MulSemiringAction.toRingEquiv _ _ g, MulSemiringAction.toAlgHom R A g with }


theorem toAlgEquiv_injective [FaithfulSMul G A] :
    Function.Injective (MulSemiringAction.toAlgEquiv R A : G → A ≃ₐ[R] A) := fun _ _ h =>
  eq_of_smul_eq_smul fun r => AlgEquiv.ext_iff.1 h r


/-- Each element of the group defines an algebra equivalence.

This is a stronger version of `MulSemiringAction.toRingAut` and
`DistribMulAction.toModuleEnd`. -/
@[simps]
def toAlgAut : G →* A ≃ₐ[R] A where
  toFun := toAlgEquiv R A
  map_one' := AlgEquiv.ext <| one_smul _
  map_mul' g h := AlgEquiv.ext <| mul_smul g h


