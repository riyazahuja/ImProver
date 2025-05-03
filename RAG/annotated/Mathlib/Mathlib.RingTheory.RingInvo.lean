/-- A ring involution -/
structure RingInvo [Semiring R] extends R ≃+* Rᵐᵒᵖ where
  /-- The requirement that the ring homomorphism is its own inverse -/
  involution' : ∀ x, (toFun (toFun x).unop).unop = x


/-- `RingInvoClass F R` states that `F` is a type of ring involutions.
You should extend this class when you extend `RingInvo`. -/
class RingInvoClass (F R : Type*) [Semiring R] [EquivLike F R Rᵐᵒᵖ]
  extends RingEquivClass F R Rᵐᵒᵖ : Prop where
  /-- Every ring involution must be its own inverse -/
  involution : ∀ (f : F) (x), (f (f x).unop).unop = x



/-- Turn an element of a type `F` satisfying `RingInvoClass F R` into an actual
`RingInvo`. This is declared as the default coercion from `F` to `RingInvo R`. -/
@[coe]
def RingInvoClass.toRingInvo {R} [Semiring R] [EquivLike F R Rᵐᵒᵖ] [RingInvoClass F R] (f : F) :
    RingInvo R :=
  { (f : R ≃+* Rᵐᵒᵖ) with involution' := RingInvoClass.involution f }


/-- Any type satisfying `RingInvoClass` can be cast into `RingInvo` via
`RingInvoClass.toRingInvo`. -/
instance [RingInvoClass F R] : CoeTC F (RingInvo R) :=
  ⟨RingInvoClass.toRingInvo⟩


instance : EquivLike (RingInvo R) R Rᵐᵒᵖ where
  coe f := f.toFun
  inv f := f.invFun
  coe_injective' e f h₁ h₂ := by
    /-
      F : Type u_1
      R : Type u_2
      inst✝¹ : Semiring R
      inst✝ : EquivLike F R (MulOpposite R)
      e f : RingInvo R
      h₁ : Eq ((fun f => f.toFun) e) ((fun f => f.toFun) f)
      h₂ : Eq ((fun f => f.invFun) e) ((fun f => f.invFun) f)
      ⊢ Eq e f
    -/
    rcases e with ⟨⟨tE, _⟩, _⟩; rcases f with ⟨⟨tF, _⟩, _⟩
    /-
      case mk.mk.mk.mk
      F : Type u_1
      R : Type u_2
      inst✝¹ : Semiring R
      inst✝ : EquivLike F R (MulOpposite R)
      tE : Equiv R (MulOpposite R)
      map_mul'✝¹ : ∀ (x y : R), Eq (tE.toFun (HMul.hMul x y)) (HMul.hMul (tE.toFun x …
      map_add'✝¹ : ∀ (x y : R), Eq (tE.toFun (HAdd.hAdd x y)) (HAdd.hAdd (tE.toFun x …
      involution'✝¹ : ∀ (x : R), Eq (MulOpposite.unop ({ toEquiv := tE, map_mul' :=  …
      tF : Equiv R (MulOpposite R)
      map_mul'✝ : ∀ (x y : R), Eq (tF.toFun (HMul.hMul x y)) (HMul.hMul (tF.toFun x) …
      map_add'✝ : ∀ (x y : R), Eq (tF.toFun (HAdd.hAdd x y)) (HAdd.hAdd (tF.toFun x) …
      involution'✝ : ∀ (x : R), Eq (MulOpposite.unop ({ toEquiv := tF, map_mul' := m …
      h₁ : Eq ((fun f => f.toFun) { toEquiv := tE, map_mul' := map_mul'✝¹, map_add'  …
      h₂ : Eq ((fun f => f.invFun) { toEquiv := tE, map_mul' := map_mul'✝¹, map_add' …
      ⊢ Eq { toEquiv := tE, map_mul' := map_mul'✝¹, map_add' := map_add'✝¹, involuti …
    -/
    cases tE
    /-
      case mk.mk.mk.mk.mk
      F : Type u_1
      R : Type u_2
      inst✝¹ : Semiring R
      inst✝ : EquivLike F R (MulOpposite R)
      tF : Equiv R (MulOpposite R)
      map_mul'✝¹ : ∀ (x y : R), Eq (tF.toFun (HMul.hMul x y)) (HMul.hMul (tF.toFun x …
      map_add'✝¹ : ∀ (x y : R), Eq (tF.toFun (HAdd.hAdd x y)) (HAdd.hAdd (tF.toFun x …
      involution'✝¹ : ∀ (x : R), Eq (MulOpposite.unop ({ toEquiv := tF, map_mul' :=  …
      toFun✝ : R → MulOpposite R
      invFun✝ : MulOpposite R → R
      left_inv✝ : Function.LeftInverse invFun✝ toFun✝
      right_inv✝ : Function.RightInverse invFun✝ toFun✝
      map_mul'✝ : ∀ (x y : R), Eq ({ toFun := toFun✝, invFun := invFun✝, left_inv := …
      map_add'✝ : ∀ (x y : R), Eq ({ toFun := toFun✝, invFun := invFun✝, left_inv := …
      involution'✝ : ∀ (x : R), Eq (MulOpposite.unop ({ toFun := toFun✝, invFun := i …
      h₁ : Eq ((fun f => f.toFun) { toFun := toFun✝, invFun := invFun✝, left_inv :=  …
      h₂ : Eq ((fun f => f.invFun) { toFun := toFun✝, invFun := invFun✝, left_inv := …
      ⊢ Eq { toFun := toFun✝, invFun := invFun✝, left_inv := left_inv✝, right_inv := …
    -/
    cases tF
    /-
      case mk.mk.mk.mk.mk.mk
      F : Type u_1
      R : Type u_2
      inst✝¹ : Semiring R
      inst✝ : EquivLike F R (MulOpposite R)
      toFun✝¹ : R → MulOpposite R
      invFun✝¹ : MulOpposite R → R
      left_inv✝¹ : Function.LeftInverse invFun✝¹ toFun✝¹
      right_inv✝¹ : Function.RightInverse invFun✝¹ toFun✝¹
      map_mul'✝¹ : ∀ (x y : R), Eq ({ toFun := toFun✝¹, invFun := invFun✝¹, left_inv …
      map_add'✝¹ : ∀ (x y : R), Eq ({ toFun := toFun✝¹, invFun := invFun✝¹, left_inv …
      involution'✝¹ : ∀ (x : R), Eq (MulOpposite.unop ({ toFun := toFun✝¹, invFun := …
      toFun✝ : R → MulOpposite R
      invFun✝ : MulOpposite R → R
      left_inv✝ : Function.LeftInverse invFun✝ toFun✝
      right_inv✝ : Function.RightInverse invFun✝ toFun✝
      map_mul'✝ : ∀ (x y : R), Eq ({ toFun := toFun✝, invFun := invFun✝, left_inv := …
      map_add'✝ : ∀ (x y : R), Eq ({ toFun := toFun✝, invFun := invFun✝, left_inv := …
      involution'✝ : ∀ (x : R), Eq (MulOpposite.unop ({ toFun := toFun✝, invFun := i …
      h₁ : Eq ((fun f => f.toFun) { toFun := toFun✝¹, invFun := invFun✝¹, left_inv : …
      h₂ : Eq ((fun f => f.invFun) { toFun := toFun✝¹, invFun := invFun✝¹, left_inv  …
      ⊢ Eq { toFun := toFun✝¹, invFun := invFun✝¹, left_inv := left_inv✝¹, right_inv …
    -/
    congr
    /-
      🎉 no goals
    -/
  left_inv f := f.left_inv
  right_inv f := f.right_inv


instance : RingInvoClass (RingInvo R) R where
  map_add f := f.map_add'
  map_mul f := f.map_mul'
  involution f := f.involution'


/-- Construct a ring involution from a ring homomorphism. -/
def mk' (f : R →+* Rᵐᵒᵖ) (involution : ∀ r, (f (f r).unop).unop = r) : RingInvo R :=
  { f with
    invFun := fun r => (f r.unop).unop
    left_inv := fun r => involution r
    right_inv := fun _ => MulOpposite.unop_injective <| involution _
    involution' := involution }


@[simp]
theorem involution (f : RingInvo R) (x : R) : (f (f x).unop).unop = x :=
  f.involution' x

-- Porting note: remove Coe instance, not needed
-- instance hasCoeToRingEquiv : Coe (RingInvo R) (R ≃+* Rᵐᵒᵖ) :=
--   ⟨RingInvo.toRingEquiv⟩


@[norm_cast]
theorem coe_ringEquiv (f : RingInvo R) (a : R) : (f : R ≃+* Rᵐᵒᵖ) a = f a :=
  rfl


theorem map_eq_zero_iff (f : RingInvo R) {x : R} : f x = 0 ↔ x = 0 :=
  f.toRingEquiv.map_eq_zero_iff


/-- The identity function of a `CommRing` is a ring involution. -/
protected def RingInvo.id : RingInvo R :=
  { RingEquiv.toOpposite R with involution' := fun _ => rfl }


instance : Inhabited (RingInvo R) :=
  ⟨RingInvo.id _⟩


