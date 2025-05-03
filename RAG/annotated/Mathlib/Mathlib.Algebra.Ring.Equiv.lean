/-- makes a `NonUnitalRingHom` from the bijective inverse of a `NonUnitalRingHom` -/
@[simps] def NonUnitalRingHom.inverse
    [NonUnitalNonAssocSemiring R] [NonUnitalNonAssocSemiring S]
    (f : R →ₙ+* S) (g : S → R)
    (h₁ : Function.LeftInverse g f) (h₂ : Function.RightInverse g f) : S →ₙ+* R :=
  { (f : R →+ S).inverse g h₁ h₂, (f : R →ₙ* S).inverse g h₁ h₂ with toFun := g }


/-- makes a `RingHom` from the bijective inverse of a `RingHom` -/
@[simps] def RingHom.inverse [NonAssocSemiring R] [NonAssocSemiring S]
    (f : RingHom R S) (g : S → R)
    (h₁ : Function.LeftInverse g f) (h₂ : Function.RightInverse g f) : S →+* R :=
  { (f : OneHom R S).inverse g h₁,
    (f : MulHom R S).inverse g h₁ h₂,
    (f : R →+ S).inverse g h₁ h₂ with toFun := g }


/-- An equivalence between two (non-unital non-associative semi)rings that preserves the
algebraic structure. -/
structure RingEquiv (R S : Type*) [Mul R] [Mul S] [Add R] [Add S] extends R ≃ S, R ≃* S, R ≃+ S


/-- Notation for `RingEquiv`. -/
infixl:25 " ≃+* " => RingEquiv


/-- `RingEquivClass F R S` states that `F` is a type of ring structure preserving equivalences.
You should extend this class when you extend `RingEquiv`. -/
class RingEquivClass (F R S : Type*) [Mul R] [Add R] [Mul S] [Add S] [EquivLike F R S]
  extends MulEquivClass F R S : Prop where
  /-- By definition, a ring isomorphism preserves the additive structure. -/
  map_add : ∀ (f : F) (a b), f (a + b) = f a + f b


instance (priority := 100) toAddEquivClass [Mul R] [Add R]
    [Mul S] [Add S] [h : RingEquivClass F R S] : AddEquivClass F R S :=
  { h with }

-- See note [lower instance priority]

instance (priority := 100) toRingHomClass [NonAssocSemiring R] [NonAssocSemiring S]
    [h : RingEquivClass F R S] : RingHomClass F R S :=
  { h with
    map_zero := map_zero
    map_one := map_one }

-- See note [lower instance priority]

instance (priority := 100) toNonUnitalRingHomClass [NonUnitalNonAssocSemiring R]
    [NonUnitalNonAssocSemiring S] [h : RingEquivClass F R S] : NonUnitalRingHomClass F R S :=
  { h with
    map_zero := map_zero }


/-- Turn an element of a type `F` satisfying `RingEquivClass F α β` into an actual
`RingEquiv`. This is declared as the default coercion from `F` to `α ≃+* β`. -/
@[coe]
def toRingEquiv [Mul α] [Add α] [Mul β] [Add β] [EquivLike F α β] [RingEquivClass F α β] (f : F) :
    α ≃+* β :=
  { (f : α ≃* β), (f : α ≃+ β) with }


/-- Any type satisfying `RingEquivClass` can be cast into `RingEquiv` via
`RingEquivClass.toRingEquiv`. -/
instance [Mul α] [Add α] [Mul β] [Add β] [EquivLike F α β] [RingEquivClass F α β] :
    CoeTC F (α ≃+* β) :=
  ⟨RingEquivClass.toRingEquiv⟩


instance : EquivLike (R ≃+* S) R S where
  coe f := f.toFun
  inv f := f.invFun
  coe_injective' e f h₁ h₂ := by
    /-
      F : Type u_1
      α : Type u_2
      β : Type u_3
      R : Type u_4
      S : Type u_5
      S' : Type u_6
      inst✝⁵ : Mul R
      inst✝⁴ : Mul S
      inst✝³ : Add R
      inst✝² : Add S
      inst✝¹ : Mul S'
      inst✝ : Add S'
      e f : RingEquiv R S
      h₁ : Eq ((fun f => f.toFun) e) ((fun f => f.toFun) f)
      h₂ : Eq ((fun f => f.invFun) e) ((fun f => f.invFun) f)
      ⊢ Eq e f
    -/
    cases e
    /-
      case mk
      F : Type u_1
      α : Type u_2
      β : Type u_3
      R : Type u_4
      S : Type u_5
      S' : Type u_6
      inst✝⁵ : Mul R
      inst✝⁴ : Mul S
      inst✝³ : Add R
      inst✝² : Add S
      inst✝¹ : Mul S'
      inst✝ : Add S'
      f : RingEquiv R S
      toEquiv✝ : Equiv R S
      map_mul'✝ : ∀ (x y : R), Eq (toEquiv✝.toFun (HMul.hMul x y)) (HMul.hMul (toEqu …
      map_add'✝ : ∀ (x y : R), Eq (toEquiv✝.toFun (HAdd.hAdd x y)) (HAdd.hAdd (toEqu …
      h₁ : Eq ((fun f => f.toFun) { toEquiv := toEquiv✝, map_mul' := map_mul'✝, map_ …
      h₂ : Eq ((fun f => f.invFun) { toEquiv := toEquiv✝, map_mul' := map_mul'✝, map …
      ⊢ Eq { toEquiv := toEquiv✝, map_mul' := map_mul'✝, map_add' := map_add'✝ } f
    -/
    cases f
    /-
      case mk.mk
      F : Type u_1
      α : Type u_2
      β : Type u_3
      R : Type u_4
      S : Type u_5
      S' : Type u_6
      inst✝⁵ : Mul R
      inst✝⁴ : Mul S
      inst✝³ : Add R
      inst✝² : Add S
      inst✝¹ : Mul S'
      inst✝ : Add S'
      toEquiv✝¹ : Equiv R S
      map_mul'✝¹ : ∀ (x y : R), Eq (toEquiv✝¹.toFun (HMul.hMul x y)) (HMul.hMul (toE …
      map_add'✝¹ : ∀ (x y : R), Eq (toEquiv✝¹.toFun (HAdd.hAdd x y)) (HAdd.hAdd (toE …
      toEquiv✝ : Equiv R S
      map_mul'✝ : ∀ (x y : R), Eq (toEquiv✝.toFun (HMul.hMul x y)) (HMul.hMul (toEqu …
      map_add'✝ : ∀ (x y : R), Eq (toEquiv✝.toFun (HAdd.hAdd x y)) (HAdd.hAdd (toEqu …
      h₁ : Eq ((fun f => f.toFun) { toEquiv := toEquiv✝¹, map_mul' := map_mul'✝¹, ma …
      h₂ : Eq ((fun f => f.invFun) { toEquiv := toEquiv✝¹, map_mul' := map_mul'✝¹, m …
      ⊢ Eq { toEquiv := toEquiv✝¹, map_mul' := map_mul'✝¹, map_add' := map_add'✝¹ }  …
    -/
    congr
    /-
      case mk.mk.e_toEquiv
      F : Type u_1
      α : Type u_2
      β : Type u_3
      R : Type u_4
      S : Type u_5
      S' : Type u_6
      inst✝⁵ : Mul R
      inst✝⁴ : Mul S
      inst✝³ : Add R
      inst✝² : Add S
      inst✝¹ : Mul S'
      inst✝ : Add S'
      toEquiv✝¹ : Equiv R S
      map_mul'✝¹ : ∀ (x y : R), Eq (toEquiv✝¹.toFun (HMul.hMul x y)) (HMul.hMul (toE …
      map_add'✝¹ : ∀ (x y : R), Eq (toEquiv✝¹.toFun (HAdd.hAdd x y)) (HAdd.hAdd (toE …
      toEquiv✝ : Equiv R S
      map_mul'✝ : ∀ (x y : R), Eq (toEquiv✝.toFun (HMul.hMul x y)) (HMul.hMul (toEqu …
      map_add'✝ : ∀ (x y : R), Eq (toEquiv✝.toFun (HAdd.hAdd x y)) (HAdd.hAdd (toEqu …
      h₁ : Eq ((fun f => f.toFun) { toEquiv := toEquiv✝¹, map_mul' := map_mul'✝¹, ma …
      h₂ : Eq ((fun f => f.invFun) { toEquiv := toEquiv✝¹, map_mul' := map_mul'✝¹, m …
      ⊢ Eq toEquiv✝¹ toEquiv✝
    -/
    apply Equiv.coe_fn_injective h₁
    /-
      🎉 no goals
    -/
  left_inv f := f.left_inv
  right_inv f := f.right_inv


instance : RingEquivClass (R ≃+* S) R S where
  map_add f := f.map_add'
  map_mul f := f.map_mul'


/-- Two ring isomorphisms agree if they are defined by the
    same underlying function. -/
@[ext]
theorem ext {f g : R ≃+* S} (h : ∀ x, f x = g x) : f = g :=
  DFunLike.ext f g h


protected theorem congr_arg {f : R ≃+* S} {x x' : R} : x = x' → f x = f x' :=
  DFunLike.congr_arg f


protected theorem congr_fun {f g : R ≃+* S} (h : f = g) (x : R) : f x = g x :=
  DFunLike.congr_fun h x


@[simp]
theorem coe_mk (e h₃ h₄) : ⇑(⟨e, h₃, h₄⟩ : R ≃+* S) = e :=
  rfl

-- Porting note: `toEquiv_mk` no longer needed in Lean4


@[simp]
theorem mk_coe (e : R ≃+* S) (e' h₁ h₂ h₃ h₄) : (⟨⟨e, e', h₁, h₂⟩, h₃, h₄⟩ : R ≃+* S) = e :=
  ext fun _ => rfl


@[simp]
theorem toEquiv_eq_coe (f : R ≃+* S) : f.toEquiv = f :=
  rfl


@[simp]
theorem coe_toEquiv (f : R ≃+* S) : ⇑(f : R ≃ S) = f :=
  rfl


@[simp]
theorem toAddEquiv_eq_coe (f : R ≃+* S) : f.toAddEquiv = ↑f :=
  rfl


@[simp]
theorem toMulEquiv_eq_coe (f : R ≃+* S) : f.toMulEquiv = ↑f :=
  rfl


@[simp, norm_cast]
theorem coe_toMulEquiv (f : R ≃+* S) : ⇑(f : R ≃* S) = f :=
  rfl


@[simp]
theorem coe_toAddEquiv (f : R ≃+* S) : ⇑(f : R ≃+ S) = f :=
  rfl


/-- A ring isomorphism preserves multiplication. -/
protected theorem map_mul (e : R ≃+* S) (x y : R) : e (x * y) = e x * e y :=
  map_mul e x y


/-- A ring isomorphism preserves addition. -/
protected theorem map_add (e : R ≃+* S) (x y : R) : e (x + y) = e x + e y :=
  map_add e x y


protected theorem bijective (e : R ≃+* S) : Function.Bijective e :=
  EquivLike.bijective e


protected theorem injective (e : R ≃+* S) : Function.Injective e :=
  EquivLike.injective e


protected theorem surjective (e : R ≃+* S) : Function.Surjective e :=
  EquivLike.surjective e


/-- The identity map is a ring isomorphism. -/
@[refl]
def refl : R ≃+* R :=
  { MulEquiv.refl R, AddEquiv.refl R with }


instance : Inhabited (R ≃+* R) :=
  ⟨RingEquiv.refl R⟩


@[simp]
theorem refl_apply (x : R) : RingEquiv.refl R x = x :=
  rfl


@[simp]
theorem coe_addEquiv_refl : (RingEquiv.refl R : R ≃+ R) = AddEquiv.refl R :=
  rfl


@[simp]
theorem coe_mulEquiv_refl : (RingEquiv.refl R : R ≃* R) = MulEquiv.refl R :=
  rfl


/-- The inverse of a ring isomorphism is a ring isomorphism. -/
@[symm]
protected def symm (e : R ≃+* S) : S ≃+* R :=
  { e.toMulEquiv.symm, e.toAddEquiv.symm with }


@[simp]
theorem invFun_eq_symm (f : R ≃+* S) : EquivLike.inv f = f.symm :=
  rfl


@[simp]
theorem symm_symm (e : R ≃+* S) : e.symm.symm = e := rfl


theorem symm_bijective : Function.Bijective (RingEquiv.symm : (R ≃+* S) → S ≃+* R) :=
  Function.bijective_iff_has_inverse.mpr ⟨_, symm_symm, symm_symm⟩


@[simp]
theorem mk_coe' (e : R ≃+* S) (f h₁ h₂ h₃ h₄) :
    (⟨⟨f, ⇑e, h₁, h₂⟩, h₃, h₄⟩ : S ≃+* R) = e.symm :=
  symm_bijective.injective <| ext fun _ => rfl


/-- Auxiliary definition to avoid looping in `dsimp` with `RingEquiv.symm_mk`. -/
protected def symm_mk.aux (f : R → S) (g h₁ h₂ h₃ h₄) := (mk ⟨f, g, h₁, h₂⟩ h₃ h₄).symm


@[simp]
theorem symm_mk (f : R → S) (g h₁ h₂ h₃ h₄) :
    (mk ⟨f, g, h₁, h₂⟩ h₃ h₄).symm =
      { symm_mk.aux f g h₁ h₂ h₃ h₄ with
        toFun := g
        invFun := f } :=
  rfl


@[simp]
theorem symm_refl : (RingEquiv.refl R).symm = RingEquiv.refl R :=
  rfl


@[simp]
theorem coe_toEquiv_symm (e : R ≃+* S) : (e.symm : S ≃ R) = (e : R ≃ S).symm :=
  rfl


@[simp]
theorem apply_symm_apply (e : R ≃+* S) : ∀ x, e (e.symm x) = x :=
  e.toEquiv.apply_symm_apply


@[simp]
theorem symm_apply_apply (e : R ≃+* S) : ∀ x, e.symm (e x) = x :=
  e.toEquiv.symm_apply_apply


theorem image_eq_preimage (e : R ≃+* S) (s : Set R) : e '' s = e.symm ⁻¹' s :=
  e.toEquiv.image_eq_preimage s


/-- See Note [custom simps projection] -/
def Simps.symm_apply (e : R ≃+* S) : S → R :=
  e.symm


/-- Transitivity of `RingEquiv`. -/
@[trans]
protected def trans (e₁ : R ≃+* S) (e₂ : S ≃+* S') : R ≃+* S' :=
  { e₁.toMulEquiv.trans e₂.toMulEquiv, e₁.toAddEquiv.trans e₂.toAddEquiv with }


@[simp]
theorem coe_trans (e₁ : R ≃+* S) (e₂ : S ≃+* S') : (e₁.trans e₂ : R → S') = e₂ ∘ e₁ :=
  rfl


theorem trans_apply (e₁ : R ≃+* S) (e₂ : S ≃+* S') (a : R) : e₁.trans e₂ a = e₂ (e₁ a) :=
  rfl


@[simp]
theorem symm_trans_apply (e₁ : R ≃+* S) (e₂ : S ≃+* S') (a : S') :
    (e₁.trans e₂).symm a = e₁.symm (e₂.symm a) :=
  rfl


theorem symm_trans (e₁ : R ≃+* S) (e₂ : S ≃+* S') : (e₁.trans e₂).symm = e₂.symm.trans e₁.symm :=
  rfl


@[simp]
theorem coe_mulEquiv_trans (e₁ : R ≃+* S) (e₂ : S ≃+* S') :
    (e₁.trans e₂ : R ≃* S') = (e₁ : R ≃* S).trans ↑e₂ :=
  rfl


@[simp]
theorem coe_addEquiv_trans (e₁ : R ≃+* S) (e₂ : S ≃+* S') :
    (e₁.trans e₂ : R ≃+ S') = (e₁ : R ≃+ S).trans ↑e₂ :=
  rfl


/-- The `RingEquiv` between two semirings with a unique element. -/
def ofUnique {M N} [Unique M] [Unique N] [Add M] [Mul M] [Add N] [Mul N] : M ≃+* N :=
  { AddEquiv.ofUnique, MulEquiv.ofUnique with }


@[deprecated (since := "2024-12-26")] alias ringEquivOfUnique := ofUnique


instance {M N} [Unique M] [Unique N] [Add M] [Mul M] [Add N] [Mul N] :
    Unique (M ≃+* N) where
  default := .ofUnique
  uniq _ := ext fun _ => Subsingleton.elim _ _


/-- A ring iso `α ≃+* β` can equivalently be viewed as a ring iso `αᵐᵒᵖ ≃+* βᵐᵒᵖ`. -/
@[simps! symm_apply_apply symm_apply_symm_apply apply_apply apply_symm_apply]
protected def op {α β} [Add α] [Mul α] [Add β] [Mul β] :
    α ≃+* β ≃ (αᵐᵒᵖ ≃+* βᵐᵒᵖ) where
  toFun f := { AddEquiv.mulOp f.toAddEquiv, MulEquiv.op f.toMulEquiv with }
  invFun f := { AddEquiv.mulOp.symm f.toAddEquiv, MulEquiv.op.symm f.toMulEquiv with }
  left_inv f := by
    /-
      F : Type u_1
      α✝ : Type u_2
      β✝ : Type u_3
      R : Type u_4
      S : Type u_5
      S' : Type u_6
      α : Type ?u.32125
      β : Type ?u.32131
      inst✝³ : Add α
      inst✝² : Mul α
      inst✝¹ : Add β
      inst✝ : Mul β
      f : RingEquiv α β
      ⊢ Eq
          ((fun f =>
              let __src := AddEquiv.mulOp.symm f.toAddEquiv;
              let __src_1 := MulEquiv.op.symm f.toMulEquiv;
              { toEquiv := __src.toEquiv, map_mul' := ⋯, map_add' := ⋯ })
            ((fun f =>
                let __src := AddEquiv.mulOp f.toAddEquiv;
                let __src_1 := MulEquiv.op f.toMulEquiv;
                { toEquiv := __src.toEquiv, map_mul' := ⋯, map_add' := ⋯ })
              f))
          f
    -/
    ext
    /-
      case h
      F : Type u_1
      α✝ : Type u_2
      β✝ : Type u_3
      R : Type u_4
      S : Type u_5
      S' : Type u_6
      α : Type ?u.32125
      β : Type ?u.32131
      inst✝³ : Add α
      inst✝² : Mul α
      inst✝¹ : Add β
      inst✝ : Mul β
      f : RingEquiv α β
      x✝ : α
      ⊢ Eq
          (((fun f =>
                let __src := AddEquiv.mulOp.symm f.toAddEquiv;
                let __src_1 := MulEquiv.op.symm f.toMulEquiv;
                { toEquiv := __src.toEquiv, map_mul' := ⋯, map_add' := ⋯ })
              ((fun f =>
                  let __src := AddEquiv.mulOp f.toAddEquiv;
                  let __src_1 := MulEquiv.op f.toMulEquiv;
                  { toEquiv := __src.toEquiv, map_mul' := ⋯, map_add' := ⋯ })
                f))
            x✝)
          (f x✝)
    -/
    rfl
    /-
      🎉 no goals
    -/
  right_inv f := by
    /-
      F : Type u_1
      α✝ : Type u_2
      β✝ : Type u_3
      R : Type u_4
      S : Type u_5
      S' : Type u_6
      α : Type ?u.32125
      β : Type ?u.32131
      inst✝³ : Add α
      inst✝² : Mul α
      inst✝¹ : Add β
      inst✝ : Mul β
      f : RingEquiv (MulOpposite α) (MulOpposite β)
      ⊢ Eq
          ((fun f =>
              let __src := AddEquiv.mulOp f.toAddEquiv;
              let __src_1 := MulEquiv.op f.toMulEquiv;
              { toEquiv := __src.toEquiv, map_mul' := ⋯, map_add' := ⋯ })
            ((fun f =>
                let __src := AddEquiv.mulOp.symm f.toAddEquiv;
                let __src_1 := MulEquiv.op.symm f.toMulEquiv;
                { toEquiv := __src.toEquiv, map_mul' := ⋯, map_add' := ⋯ })
              f))
          f
    -/
    ext
    /-
      case h
      F : Type u_1
      α✝ : Type u_2
      β✝ : Type u_3
      R : Type u_4
      S : Type u_5
      S' : Type u_6
      α : Type ?u.32125
      β : Type ?u.32131
      inst✝³ : Add α
      inst✝² : Mul α
      inst✝¹ : Add β
      inst✝ : Mul β
      f : RingEquiv (MulOpposite α) (MulOpposite β)
      x✝ : MulOpposite α
      ⊢ Eq
          (((fun f =>
                let __src := AddEquiv.mulOp f.toAddEquiv;
                let __src_1 := MulEquiv.op f.toMulEquiv;
                { toEquiv := __src.toEquiv, map_mul' := ⋯, map_add' := ⋯ })
              ((fun f =>
                  let __src := AddEquiv.mulOp.symm f.toAddEquiv;
                  let __src_1 := MulEquiv.op.symm f.toMulEquiv;
                  { toEquiv := __src.toEquiv, map_mul' := ⋯, map_add' := ⋯ })
                f))
            x✝)
          (f x✝)
    -/
    rfl
    /-
      🎉 no goals
    -/


/-- The 'unopposite' of a ring iso `αᵐᵒᵖ ≃+* βᵐᵒᵖ`. Inverse to `RingEquiv.op`. -/
@[simp]
protected def unop {α β} [Add α] [Mul α] [Add β] [Mul β] : αᵐᵒᵖ ≃+* βᵐᵒᵖ ≃ (α ≃+* β) :=
  RingEquiv.op.symm


/-- A ring is isomorphic to the opposite of its opposite. -/
@[simps!]
def opOp (R : Type*) [Add R] [Mul R] : R ≃+* Rᵐᵒᵖᵐᵒᵖ where
  __ := MulEquiv.opOp R
  map_add' _ _ := rfl


/-- A non-unital commutative ring is isomorphic to its opposite. -/
def toOpposite : R ≃+* Rᵐᵒᵖ :=
  { MulOpposite.opEquiv with
    map_add' := fun _ _ => rfl
    map_mul' := fun x y => mul_comm (op y) (op x) }


@[simp]
theorem toOpposite_apply (r : R) : toOpposite R r = op r :=
  rfl


@[simp]
theorem toOpposite_symm_apply (r : Rᵐᵒᵖ) : (toOpposite R).symm r = unop r :=
  rfl


/-- A ring isomorphism sends zero to zero. -/
protected theorem map_zero : f 0 = 0 :=
  map_zero f


protected theorem map_eq_zero_iff : f x = 0 ↔ x = 0 :=
  EmbeddingLike.map_eq_zero_iff


theorem map_ne_zero_iff : f x ≠ 0 ↔ x ≠ 0 :=
  EmbeddingLike.map_ne_zero_iff


/-- Produce a ring isomorphism from a bijective ring homomorphism. -/
noncomputable def ofBijective [NonUnitalRingHomClass F R S] (f : F) (hf : Function.Bijective f) :
    R ≃+* S :=
  { Equiv.ofBijective f hf with
    map_mul' := map_mul f
    map_add' := map_add f }


@[simp]
theorem coe_ofBijective [NonUnitalRingHomClass F R S] (f : F) (hf : Function.Bijective f) :
    (ofBijective f hf : R → S) = f :=
  rfl


theorem ofBijective_apply [NonUnitalRingHomClass F R S] (f : F) (hf : Function.Bijective f)
    (x : R) : ofBijective f hf x = f x :=
  rfl


/-- A family of ring isomorphisms `∀ j, (R j ≃+* S j)` generates a
ring isomorphisms between `∀ j, R j` and `∀ j, S j`.

This is the `RingEquiv` version of `Equiv.piCongrRight`, and the dependent version of
`RingEquiv.arrowCongr`.
-/
@[simps apply]
def piCongrRight {ι : Type*} {R S : ι → Type*} [∀ i, NonUnitalNonAssocSemiring (R i)]
    [∀ i, NonUnitalNonAssocSemiring (S i)] (e : ∀ i, R i ≃+* S i) : (∀ i, R i) ≃+* ∀ i, S i :=
  { @MulEquiv.piCongrRight ι R S _ _ fun i => (e i).toMulEquiv,
    @AddEquiv.piCongrRight ι R S _ _ fun i => (e i).toAddEquiv with
    toFun := fun x j => e j (x j)
    invFun := fun x j => (e j).symm (x j) }


@[simp]
theorem piCongrRight_refl {ι : Type*} {R : ι → Type*} [∀ i, NonUnitalNonAssocSemiring (R i)] :
    (piCongrRight fun i => RingEquiv.refl (R i)) = RingEquiv.refl _ :=
  rfl


@[simp]
theorem piCongrRight_symm {ι : Type*} {R S : ι → Type*} [∀ i, NonUnitalNonAssocSemiring (R i)]
    [∀ i, NonUnitalNonAssocSemiring (S i)] (e : ∀ i, R i ≃+* S i) :
    (piCongrRight e).symm = piCongrRight fun i => (e i).symm :=
  rfl


@[simp]
theorem piCongrRight_trans {ι : Type*} {R S T : ι → Type*}
    [∀ i, NonUnitalNonAssocSemiring (R i)] [∀ i, NonUnitalNonAssocSemiring (S i)]
    [∀ i, NonUnitalNonAssocSemiring (T i)] (e : ∀ i, R i ≃+* S i) (f : ∀ i, S i ≃+* T i) :
    (piCongrRight e).trans (piCongrRight f) = piCongrRight fun i => (e i).trans (f i) :=
  rfl


/-- Transport dependent functions through an equivalence of the base space.

This is `Equiv.piCongrLeft'` as a `RingEquiv`. -/
@[simps!]
def piCongrLeft' {ι ι' : Type*} (R : ι → Type*) (e : ι ≃ ι')
    [∀ i, NonUnitalNonAssocSemiring (R i)] :
    ((i : ι) → R i) ≃+* ((i : ι') → R (e.symm i)) where
  toEquiv := Equiv.piCongrLeft' R e
  map_mul' _ _ := rfl
  map_add' _ _ := rfl


@[simp]
theorem piCongrLeft'_symm {R : Type*} [NonUnitalNonAssocSemiring R] (e : α ≃ β) :
    (RingEquiv.piCongrLeft' (fun _ => R) e).symm = RingEquiv.piCongrLeft' _ e.symm := by
  /-
    α : Type u_2
    β : Type u_3
    R : Type u_7
    inst✝ : NonUnitalNonAssocSemiring R
    e : Equiv α β
    ⊢ Eq (RingEquiv.piCongrLeft' (fun x => R) e).symm (RingEquiv.piCongrLeft' (fun …
  -/
  simp only [piCongrLeft', RingEquiv.symm, MulEquiv.symm, Equiv.piCongrLeft'_symm]
  /-
    🎉 no goals
  -/


/-- Transport dependent functions through an equivalence of the base space.

This is `Equiv.piCongrLeft` as a `RingEquiv`. -/
@[simps!]
def piCongrLeft {ι ι' : Type*} (S : ι' → Type*) (e : ι ≃ ι')
    [∀ i, NonUnitalNonAssocSemiring (S i)] :
    ((i : ι) → S (e i)) ≃+* ((i : ι') → S i) :=
  (RingEquiv.piCongrLeft' S e.symm).symm


/-- Splits the indices of ring `∀ (i : ι), Y i` along the predicate `p`. This is
`Equiv.piEquivPiSubtypeProd` as a `RingEquiv`. -/
@[simps!]
def piEquivPiSubtypeProd {ι : Type*} (p : ι → Prop) [DecidablePred p] (Y : ι → Type*)
    [∀ i, NonUnitalNonAssocSemiring (Y i)] :
    ((i : ι) → Y i) ≃+* ((i : { x : ι // p x }) → Y i) × ((i : { x : ι // ¬p x }) → Y i) where
  toEquiv := Equiv.piEquivPiSubtypeProd p Y
  map_mul' _ _ := rfl
  map_add' _ _ := rfl


/-- Product of ring equivalences. This is `Equiv.prodCongr` as a `RingEquiv`. -/
@[simps!]
def prodCongr {R R' S S' : Type*} [NonUnitalNonAssocSemiring R] [NonUnitalNonAssocSemiring R']
    [NonUnitalNonAssocSemiring S] [NonUnitalNonAssocSemiring S']
    (f : R ≃+* R') (g : S ≃+* S') :
    R × S ≃+* R' × S' where
  toEquiv := Equiv.prodCongr f g
  map_mul' _ _ := by
    simp only [Equiv.toFun_as_coe, Equiv.prodCongr_apply, EquivLike.coe_coe,
      Prod.map, Prod.fst_mul, map_mul, Prod.snd_mul, Prod.mk_mul_mk]
  map_add' _ _ := by
    simp only [Equiv.toFun_as_coe, Equiv.prodCongr_apply, EquivLike.coe_coe,
      Prod.map, Prod.fst_add, map_add, Prod.snd_add, Prod.mk_add_mk]


@[simp]
theorem coe_prodCongr {R R' S S' : Type*} [NonUnitalNonAssocSemiring R]
    [NonUnitalNonAssocSemiring R'] [NonUnitalNonAssocSemiring S] [NonUnitalNonAssocSemiring S']
    (f : R ≃+* R') (g : S ≃+* S') :
    ⇑(RingEquiv.prodCongr f g) = Prod.map f g :=
  rfl


/-- A ring isomorphism sends one to one. -/
protected theorem map_one : f 1 = 1 :=
  map_one f


protected theorem map_eq_one_iff : f x = 1 ↔ x = 1 :=
  EmbeddingLike.map_eq_one_iff


theorem map_ne_one_iff : f x ≠ 1 ↔ x ≠ 1 :=
  EmbeddingLike.map_ne_one_iff


theorem coe_monoidHom_refl : (RingEquiv.refl R : R →* R) = MonoidHom.id R :=
  rfl


@[simp]
theorem coe_addMonoidHom_refl : (RingEquiv.refl R : R →+ R) = AddMonoidHom.id R :=
  rfl


@[simp]
theorem coe_ringHom_refl : (RingEquiv.refl R : R →+* R) = RingHom.id R :=
  rfl


@[simp]
theorem coe_monoidHom_trans [NonAssocSemiring S'] (e₁ : R ≃+* S) (e₂ : S ≃+* S') :
    (e₁.trans e₂ : R →* S') = (e₂ : S →* S').comp ↑e₁ :=
  rfl


@[simp]
theorem coe_addMonoidHom_trans [NonAssocSemiring S'] (e₁ : R ≃+* S) (e₂ : S ≃+* S') :
    (e₁.trans e₂ : R →+ S') = (e₂ : S →+ S').comp ↑e₁ :=
  rfl


@[simp]
theorem coe_ringHom_trans [NonAssocSemiring S'] (e₁ : R ≃+* S) (e₂ : S ≃+* S') :
    (e₁.trans e₂ : R →+* S') = (e₂ : S →+* S').comp ↑e₁ :=
  rfl


@[simp]
theorem comp_symm (e : R ≃+* S) : (e : R →+* S).comp (e.symm : S →+* R) = RingHom.id S :=
  RingHom.ext e.apply_symm_apply


@[simp]
theorem symm_comp (e : R ≃+* S) : (e.symm : S →+* R).comp (e : R →+* S) = RingHom.id R :=
  RingHom.ext e.symm_apply_apply


protected theorem map_neg : f (-x) = -f x :=
  map_neg f x


protected theorem map_sub : f (x - y) = f x - f y :=
  map_sub f x y


@[simp]
theorem map_neg_one : f (-1) = -1 :=
  f.map_one ▸ f.map_neg 1


theorem map_eq_neg_one_iff {x : R} : f x = -1 ↔ x = -1 := by
  /-
    R : Type u_4
    S : Type u_5
    inst✝¹ : NonAssocRing R
    inst✝ : NonAssocRing S
    f : RingEquiv R S
    x : R
    ⊢ Iff (Eq (f x) (-1)) (Eq x (-1))
  -/
  rw [← neg_eq_iff_eq_neg, ← neg_eq_iff_eq_neg, ← map_neg, RingEquiv.map_eq_one_iff]
  /-
    🎉 no goals
  -/


/-- Reinterpret a ring equivalence as a non-unital ring homomorphism. -/
def toNonUnitalRingHom (e : R ≃+* S) : R →ₙ+* S :=
  { e.toMulEquiv.toMulHom, e.toAddEquiv.toAddMonoidHom with }


theorem toNonUnitalRingHom_injective :
    Function.Injective (toNonUnitalRingHom : R ≃+* S → R →ₙ+* S) := fun _ _ h =>
  RingEquiv.ext (NonUnitalRingHom.ext_iff.1 h)


theorem toNonUnitalRingHom_eq_coe (f : R ≃+* S) : f.toNonUnitalRingHom = ↑f :=
  rfl


@[simp, norm_cast]
theorem coe_toNonUnitalRingHom (f : R ≃+* S) : ⇑(f : R →ₙ+* S) = f :=
  rfl


theorem coe_nonUnitalRingHom_inj_iff {R S : Type*} [NonUnitalNonAssocSemiring R]
    [NonUnitalNonAssocSemiring S] (f g : R ≃+* S) : f = g ↔ (f : R →ₙ+* S) = g :=
               /-
                 R : Type u_7
                 S : Type u_8
                 inst✝¹ : NonUnitalNonAssocSemiring R
                 inst✝ : NonUnitalNonAssocSemiring S
                 f g : RingEquiv R S
                 h : Eq f g
                 ⊢ Eq ↑f ↑g
               -/
  ⟨fun h => by rw [h], fun h => ext <| NonUnitalRingHom.ext_iff.mp h⟩
               /-
                 🎉 no goals
               -/


@[simp]
theorem toNonUnitalRingHom_refl :
    (RingEquiv.refl R).toNonUnitalRingHom = NonUnitalRingHom.id R :=
  rfl


@[simp]
theorem toNonUnitalRingHom_apply_symm_toNonUnitalRingHom_apply (e : R ≃+* S) :
    ∀ y : S, e.toNonUnitalRingHom (e.symm.toNonUnitalRingHom y) = y :=
  e.toEquiv.apply_symm_apply


@[simp]
theorem symm_toNonUnitalRingHom_apply_toNonUnitalRingHom_apply (e : R ≃+* S) :
    ∀ x : R, e.symm.toNonUnitalRingHom (e.toNonUnitalRingHom x) = x :=
  Equiv.symm_apply_apply e.toEquiv


@[simp]
theorem toNonUnitalRingHom_trans (e₁ : R ≃+* S) (e₂ : S ≃+* S') :
    (e₁.trans e₂).toNonUnitalRingHom = e₂.toNonUnitalRingHom.comp e₁.toNonUnitalRingHom :=
  rfl


@[simp]
theorem toNonUnitalRingHomm_comp_symm_toNonUnitalRingHom (e : R ≃+* S) :
    e.toNonUnitalRingHom.comp e.symm.toNonUnitalRingHom = NonUnitalRingHom.id _ := by
  /-
    R : Type u_4
    S : Type u_5
    inst✝¹ : NonUnitalNonAssocSemiring R
    inst✝ : NonUnitalNonAssocSemiring S
    e : RingEquiv R S
    ⊢ Eq (e.toNonUnitalRingHom.comp e.symm.toNonUnitalRingHom) (NonUnitalRingHom.i …
  -/
  ext
  /-
    case a
    R : Type u_4
    S : Type u_5
    inst✝¹ : NonUnitalNonAssocSemiring R
    inst✝ : NonUnitalNonAssocSemiring S
    e : RingEquiv R S
    x✝ : S
    ⊢ Eq ((e.toNonUnitalRingHom.comp e.symm.toNonUnitalRingHom) x✝) ((NonUnitalRin …
  -/
  simp
  /-
    🎉 no goals
  -/


@[simp]
theorem symm_toNonUnitalRingHom_comp_toNonUnitalRingHom (e : R ≃+* S) :
    e.symm.toNonUnitalRingHom.comp e.toNonUnitalRingHom = NonUnitalRingHom.id _ := by
  /-
    R : Type u_4
    S : Type u_5
    inst✝¹ : NonUnitalNonAssocSemiring R
    inst✝ : NonUnitalNonAssocSemiring S
    e : RingEquiv R S
    ⊢ Eq (e.symm.toNonUnitalRingHom.comp e.toNonUnitalRingHom) (NonUnitalRingHom.i …
  -/
  ext
  /-
    case a
    R : Type u_4
    S : Type u_5
    inst✝¹ : NonUnitalNonAssocSemiring R
    inst✝ : NonUnitalNonAssocSemiring S
    e : RingEquiv R S
    x✝ : R
    ⊢ Eq ((e.symm.toNonUnitalRingHom.comp e.toNonUnitalRingHom) x✝) ((NonUnitalRin …
  -/
  simp
  /-
    🎉 no goals
  -/


/-- Reinterpret a ring equivalence as a ring homomorphism. -/
def toRingHom (e : R ≃+* S) : R →+* S :=
  { e.toMulEquiv.toMonoidHom, e.toAddEquiv.toAddMonoidHom with }


theorem toRingHom_injective : Function.Injective (toRingHom : R ≃+* S → R →+* S) := fun _ _ h =>
  RingEquiv.ext (RingHom.ext_iff.1 h)


@[simp] theorem toRingHom_eq_coe (f : R ≃+* S) : f.toRingHom = ↑f :=
  rfl


@[simp, norm_cast]
theorem coe_toRingHom (f : R ≃+* S) : ⇑(f : R →+* S) = f :=
  rfl


theorem coe_ringHom_inj_iff {R S : Type*} [NonAssocSemiring R] [NonAssocSemiring S]
    (f g : R ≃+* S) : f = g ↔ (f : R →+* S) = g :=
               /-
                 R : Type u_7
                 S : Type u_8
                 inst✝¹ : NonAssocSemiring R
                 inst✝ : NonAssocSemiring S
                 f g : RingEquiv R S
                 h : Eq f g
                 ⊢ Eq ↑f ↑g
               -/
  ⟨fun h => by rw [h], fun h => ext <| RingHom.ext_iff.mp h⟩
               /-
                 🎉 no goals
               -/


/-- The two paths coercion can take to a `NonUnitalRingEquiv` are equivalent -/
@[simp, norm_cast]
theorem toNonUnitalRingHom_commutes (f : R ≃+* S) :
    ((f : R →+* S) : R →ₙ+* S) = (f : R →ₙ+* S) :=
  rfl


/-- Reinterpret a ring equivalence as a monoid homomorphism. -/
abbrev toMonoidHom (e : R ≃+* S) : R →* S :=
  e.toRingHom.toMonoidHom


/-- Reinterpret a ring equivalence as an `AddMonoid` homomorphism. -/
abbrev toAddMonoidHom (e : R ≃+* S) : R →+ S :=
  e.toRingHom.toAddMonoidHom


/-- The two paths coercion can take to an `AddMonoidHom` are equivalent -/
theorem toAddMonoidMom_commutes (f : R ≃+* S) :
    (f : R →+* S).toAddMonoidHom = (f : R ≃+ S).toAddMonoidHom :=
  rfl


/-- The two paths coercion can take to a `MonoidHom` are equivalent -/
theorem toMonoidHom_commutes (f : R ≃+* S) :
    (f : R →+* S).toMonoidHom = (f : R ≃* S).toMonoidHom :=
  rfl


/-- The two paths coercion can take to an `Equiv` are equivalent -/
theorem toEquiv_commutes (f : R ≃+* S) : (f : R ≃+ S).toEquiv = (f : R ≃* S).toEquiv :=
  rfl


@[simp]
theorem toRingHom_refl : (RingEquiv.refl R).toRingHom = RingHom.id R :=
  rfl


@[simp]
theorem toMonoidHom_refl : (RingEquiv.refl R).toMonoidHom = MonoidHom.id R :=
  rfl


@[simp]
theorem toAddMonoidHom_refl : (RingEquiv.refl R).toAddMonoidHom = AddMonoidHom.id R :=
  rfl


theorem toRingHom_apply_symm_toRingHom_apply (e : R ≃+* S) :
    ∀ y : S, e.toRingHom (e.symm.toRingHom y) = y :=
  e.toEquiv.apply_symm_apply


theorem symm_toRingHom_apply_toRingHom_apply (e : R ≃+* S) :
    ∀ x : R, e.symm.toRingHom (e.toRingHom x) = x :=
  Equiv.symm_apply_apply e.toEquiv


@[simp]
theorem toRingHom_trans (e₁ : R ≃+* S) (e₂ : S ≃+* S') :
    (e₁.trans e₂).toRingHom = e₂.toRingHom.comp e₁.toRingHom :=
  rfl


theorem toRingHom_comp_symm_toRingHom (e : R ≃+* S) :
    e.toRingHom.comp e.symm.toRingHom = RingHom.id _ := by
  /-
    R : Type u_4
    S : Type u_5
    inst✝¹ : NonAssocSemiring R
    inst✝ : NonAssocSemiring S
    e : RingEquiv R S
    ⊢ Eq (e.toRingHom.comp e.symm.toRingHom) (RingHom.id S)
  -/
  ext
  /-
    case a
    R : Type u_4
    S : Type u_5
    inst✝¹ : NonAssocSemiring R
    inst✝ : NonAssocSemiring S
    e : RingEquiv R S
    x✝ : S
    ⊢ Eq ((e.toRingHom.comp e.symm.toRingHom) x✝) ((RingHom.id S) x✝)
  -/
  simp
  /-
    🎉 no goals
  -/


theorem symm_toRingHom_comp_toRingHom (e : R ≃+* S) :
    e.symm.toRingHom.comp e.toRingHom = RingHom.id _ := by
  /-
    R : Type u_4
    S : Type u_5
    inst✝¹ : NonAssocSemiring R
    inst✝ : NonAssocSemiring S
    e : RingEquiv R S
    ⊢ Eq (e.symm.toRingHom.comp e.toRingHom) (RingHom.id R)
  -/
  ext
  /-
    case a
    R : Type u_4
    S : Type u_5
    inst✝¹ : NonAssocSemiring R
    inst✝ : NonAssocSemiring S
    e : RingEquiv R S
    x✝ : R
    ⊢ Eq ((e.symm.toRingHom.comp e.toRingHom) x✝) ((RingHom.id R) x✝)
  -/
  simp
  /-
    🎉 no goals
  -/


/-- Construct an equivalence of rings from homomorphisms in both directions, which are inverses.
-/
@[simps]
def ofHomInv' {R S F G : Type*} [NonUnitalNonAssocSemiring R] [NonUnitalNonAssocSemiring S]
    [FunLike F R S] [FunLike G S R]
    [NonUnitalRingHomClass F R S] [NonUnitalRingHomClass G S R] (hom : F) (inv : G)
    (hom_inv_id : (inv : S →ₙ+* R).comp (hom : R →ₙ+* S) = NonUnitalRingHom.id R)
    (inv_hom_id : (hom : R →ₙ+* S).comp (inv : S →ₙ+* R) = NonUnitalRingHom.id S) :
    R ≃+* S where
  toFun := hom
  invFun := inv
  left_inv := DFunLike.congr_fun hom_inv_id
  right_inv := DFunLike.congr_fun inv_hom_id
  map_mul' := map_mul hom
  map_add' := map_add hom


/--
Construct an equivalence of rings from unital homomorphisms in both directions, which are inverses.
-/
@[simps]
def ofHomInv {R S F G : Type*} [NonAssocSemiring R] [NonAssocSemiring S]
    [FunLike F R S] [FunLike G S R] [RingHomClass F R S]
    [RingHomClass G S R] (hom : F) (inv : G)
    (hom_inv_id : (inv : S →+* R).comp (hom : R →+* S) = RingHom.id R)
    (inv_hom_id : (hom : R →+* S).comp (inv : S →+* R) = RingHom.id S) :
    R ≃+* S where
  toFun := hom
  invFun := inv
  left_inv := DFunLike.congr_fun hom_inv_id
  right_inv := DFunLike.congr_fun inv_hom_id
  map_mul' := map_mul hom
  map_add' := map_add hom


protected theorem map_pow (f : R ≃+* S) (a) : ∀ n : ℕ, f (a ^ n) = f a ^ n :=
  map_pow f a


/-- Gives a `RingEquiv` from an element of a `MulEquivClass` preserving addition. -/
def toRingEquiv {R S F : Type*} [Add R] [Add S] [Mul R] [Mul S] [EquivLike F R S]
    [MulEquivClass F R S] (f : F)
    (H : ∀ x y : R, f (x + y) = f x + f y) : R ≃+* S :=
  { (f : R ≃* S).toEquiv, (f : R ≃* S), AddEquiv.mk' (f : R ≃* S).toEquiv H with }


/-- Gives a `RingEquiv` from an element of an `AddEquivClass` preserving addition. -/
def toRingEquiv {R S F : Type*} [Add R] [Add S] [Mul R] [Mul S] [EquivLike F R S]
    [AddEquivClass F R S] (f : F)
    (H : ∀ x y : R, f (x * y) = f x * f y) : R ≃+* S :=
  { (f : R ≃+ S).toEquiv, (f : R ≃+ S), MulEquiv.mk' (f : R ≃+ S).toEquiv H with }


@[simp]
theorem self_trans_symm (e : R ≃+* S) : e.trans e.symm = RingEquiv.refl R :=
  ext e.left_inv


@[simp]
theorem symm_trans_self (e : R ≃+* S) : e.symm.trans e = RingEquiv.refl S :=
  ext e.right_inv


/-- If a ring homomorphism has an inverse, it is a ring isomorphism. -/
@[simps]
def ofRingHom (f : R →+* S) (g : S →+* R) (h₁ : f.comp g = RingHom.id S)
    (h₂ : g.comp f = RingHom.id R) : R ≃+* S :=
  { f with
    toFun := f
    invFun := g
    left_inv := RingHom.ext_iff.1 h₂
    right_inv := RingHom.ext_iff.1 h₁ }


theorem coe_ringHom_ofRingHom (f : R →+* S) (g : S →+* R) (h₁ h₂) : ofRingHom f g h₁ h₂ = f :=
  rfl


@[simp]
theorem ofRingHom_coe_ringHom (f : R ≃+* S) (g : S →+* R) (h₁ h₂) : ofRingHom (↑f) g h₁ h₂ = f :=
  ext fun _ ↦ rfl


theorem ofRingHom_symm (f : R →+* S) (g : S →+* R) (h₁ h₂) :
    (ofRingHom f g h₁ h₂).symm = ofRingHom g f h₂ h₁ :=
  rfl


/-- If two rings are isomorphic, and the second doesn't have zero divisors,
then so does the first. -/
protected theorem noZeroDivisors {A : Type*} (B : Type*) [MulZeroClass A] [MulZeroClass B]
    [NoZeroDivisors B] (e : A ≃* B) : NoZeroDivisors A :=
  e.injective.noZeroDivisors e (map_zero e) (map_mul e)


/-- If two rings are isomorphic, and the second is a domain, then so is the first. -/
protected theorem isDomain {A : Type*} (B : Type*) [Semiring A] [Semiring B] [IsDomain B]
    (e : A ≃* B) : IsDomain A :=
  { e.injective.isLeftCancelMulZero e (map_zero e) (map_mul e),
    e.injective.isRightCancelMulZero e (map_zero e) (map_mul e) with
    exists_pair_ne := ⟨e.symm 0, e.symm 1, e.symm.injective.ne zero_ne_one⟩ }


