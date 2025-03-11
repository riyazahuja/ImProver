/-- A *Shelf* is a structure with a self-distributive binary operation.
The binary operation is regarded as a left action of the type on itself.
-/
class Shelf (α : Type u) where
  /-- The action of the `Shelf` over `α`-/
  act : α → α → α
  /-- A verification that `act` is self-distributive -/
  self_distrib : ∀ {x y z : α}, act x (act y z) = act (act x y) (act x z)


/--
A *unital shelf* is a shelf equipped with an element `1` such that, for all elements `x`,
we have both `x ◃ 1` and `1 ◃ x` equal `x`.
-/
class UnitalShelf (α : Type u) extends Shelf α, One α where
  one_act : ∀ a : α, act 1 a = a
  act_one : ∀ a : α, act a 1 = a


/-- The type of homomorphisms between shelves.
This is also the notion of rack and quandle homomorphisms.
-/
@[ext]
structure ShelfHom (S₁ : Type*) (S₂ : Type*) [Shelf S₁] [Shelf S₂] where
  /-- The function under the Shelf Homomorphism -/
  toFun : S₁ → S₂
  /-- The homomorphism property of a Shelf Homomorphism -/
  map_act' : ∀ {x y : S₁}, toFun (Shelf.act x y) = Shelf.act (toFun x) (toFun y)


/-- A *rack* is an automorphic set (a set with an action on itself by
bijections) that is self-distributive.  It is a shelf such that each
element's action is invertible.

The notations `x ◃ y` and `x ◃⁻¹ y` denote the action and the
inverse action, respectively, and they are right associative.
-/
class Rack (α : Type u) extends Shelf α where
  /-- The inverse actions of the elements -/
  invAct : α → α → α
  /-- Proof of left inverse -/
  left_inv : ∀ x, Function.LeftInverse (invAct x) (act x)
  /-- Proof of right inverse -/
  right_inv : ∀ x, Function.RightInverse (invAct x) (act x)


/-- Action of a Shelf -/
scoped[Quandles] infixr:65 " ◃ " => Shelf.act


/-- Inverse Action of a Rack -/
scoped[Quandles] infixr:65 " ◃⁻¹ " => Rack.invAct


/-- Shelf Homomorphism -/
scoped[Quandles] infixr:25 " →◃ " => ShelfHom


/--
A monoid is *graphic* if, for all `x` and `y`, the *graphic identity*
`(x * y) * x = x * y` holds.  For a unital shelf, this graphic
identity holds.
-/
lemma act_act_self_eq (x y : S) : (x ◃ y) ◃ x = x ◃ y := by
  /-
    S : Type u_1
    inst✝ : UnitalShelf S
    x y : S
    ⊢ Eq (Shelf.act (Shelf.act x y) x) (Shelf.act x y)
  -/
  have h : (x ◃ y) ◃ x = (x ◃ y) ◃ (x ◃ 1) := by rw [act_one]
  /-
    S : Type u_1
    inst✝ : UnitalShelf S
    x y : S
    h : Eq (Shelf.act (Shelf.act x y) x) (Shelf.act (Shelf.act x y) (Shelf.act x 1))
    ⊢ Eq (Shelf.act (Shelf.act x y) x) (Shelf.act x y)
  -/
  rw [h, ← Shelf.self_distrib, act_one]
  /-
    🎉 no goals
  -/


                                           /-
                                             S : Type u_1
                                             inst✝ : UnitalShelf S
                                             x : S
                                             ⊢ Eq (Shelf.act x x) x
                                           -/
lemma act_idem (x : S) : (x ◃ x) = x := by rw [← act_one x, ← Shelf.self_distrib, act_one]
                                           /-
                                             🎉 no goals
                                           -/


lemma act_self_act_eq (x y : S) : x ◃ (x ◃ y) = x ◃ y := by
  /-
    S : Type u_1
    inst✝ : UnitalShelf S
    x y : S
    ⊢ Eq (Shelf.act x (Shelf.act x y)) (Shelf.act x y)
  -/
  have h : x ◃ (x ◃ y) = (x ◃ 1) ◃ (x ◃ y) := by rw [act_one]
  /-
    S : Type u_1
    inst✝ : UnitalShelf S
    x y : S
    h : Eq (Shelf.act x (Shelf.act x y)) (Shelf.act (Shelf.act x 1) (Shelf.act x y))
    ⊢ Eq (Shelf.act x (Shelf.act x y)) (Shelf.act x y)
  -/
  rw [h, ← Shelf.self_distrib, one_act]
  /-
    🎉 no goals
  -/


/--
The associativity of a unital shelf comes for free.
-/
lemma assoc (x y z : S) : (x ◃ y) ◃ z = x ◃ y ◃ z := by
  /-
    S : Type u_1
    inst✝ : UnitalShelf S
    x y z : S
    ⊢ Eq (Shelf.act (Shelf.act x y) z) (Shelf.act x (Shelf.act y z))
  -/
  rw [self_distrib, self_distrib, act_act_self_eq, act_self_act_eq]
  /-
    🎉 no goals
  -/


/-- A rack acts on itself by equivalences.
-/
def act' (x : R) : R ≃ R where
  toFun := Shelf.act x
  invFun := invAct x
  left_inv := left_inv x
  right_inv := right_inv x


@[simp]
theorem act'_apply (x y : R) : act' x y = x ◃ y :=
  rfl


@[simp]
theorem act'_symm_apply (x y : R) : (act' x).symm y = x ◃⁻¹ y :=
  rfl


@[simp]
theorem invAct_apply (x y : R) : (act' x)⁻¹ y = x ◃⁻¹ y :=
  rfl


@[simp]
theorem invAct_act_eq (x y : R) : x ◃⁻¹ x ◃ y = y :=
  left_inv x y


@[simp]
theorem act_invAct_eq (x y : R) : x ◃ x ◃⁻¹ y = y :=
  right_inv x y


theorem left_cancel (x : R) {y y' : R} : x ◃ y = x ◃ y' ↔ y = y' := by
  /-
    R : Type u_1
    inst✝ : Rack R
    x y y' : R
    ⊢ Iff (Eq (Shelf.act x y) (Shelf.act x y')) (Eq y y')
  -/
  constructor
    /-
      case mp
      R : Type u_1
      inst✝ : Rack R
      x y y' : R
      ⊢ Eq (Shelf.act x y) (Shelf.act x y') → Eq y y'
    -/
  · apply (act' x).injective
    /-
      🎉 no goals
    -/
  /-
    case mpr
    R : Type u_1
    inst✝ : Rack R
    x y y' : R
    ⊢ Eq y y' → Eq (Shelf.act x y) (Shelf.act x y')
  -/
  rintro rfl
  /-
    case mpr
    R : Type u_1
    inst✝ : Rack R
    x y : R
    ⊢ Eq (Shelf.act x y) (Shelf.act x y)
  -/
  rfl
  /-
    🎉 no goals
  -/


theorem left_cancel_inv (x : R) {y y' : R} : x ◃⁻¹ y = x ◃⁻¹ y' ↔ y = y' := by
  /-
    R : Type u_1
    inst✝ : Rack R
    x y y' : R
    ⊢ Iff (Eq (Rack.invAct x y) (Rack.invAct x y')) (Eq y y')
  -/
  constructor
    /-
      case mp
      R : Type u_1
      inst✝ : Rack R
      x y y' : R
      ⊢ Eq (Rack.invAct x y) (Rack.invAct x y') → Eq y y'
    -/
  · apply (act' x).symm.injective
    /-
      🎉 no goals
    -/
  /-
    case mpr
    R : Type u_1
    inst✝ : Rack R
    x y y' : R
    ⊢ Eq y y' → Eq (Rack.invAct x y) (Rack.invAct x y')
  -/
  rintro rfl
  /-
    case mpr
    R : Type u_1
    inst✝ : Rack R
    x y : R
    ⊢ Eq (Rack.invAct x y) (Rack.invAct x y)
  -/
  rfl
  /-
    🎉 no goals
  -/


theorem self_distrib_inv {x y z : R} : x ◃⁻¹ y ◃⁻¹ z = (x ◃⁻¹ y) ◃⁻¹ x ◃⁻¹ z := by
  /-
    R : Type u_1
    inst✝ : Rack R
    x y z : R
    ⊢ Eq (Rack.invAct x (Rack.invAct y z)) (Rack.invAct (Rack.invAct x y) (Rack.in …
  -/
  rw [← left_cancel (x ◃⁻¹ y), right_inv, ← left_cancel x, right_inv, self_distrib]
  /-
    R : Type u_1
    inst✝ : Rack R
    x y z : R
    ⊢ Eq (Shelf.act (Shelf.act x (Rack.invAct x y)) (Shelf.act x (Rack.invAct x (R …
  -/
  repeat' rw [right_inv]
  /-
    🎉 no goals
  -/


/-- The *adjoint action* of a rack on itself is `op'`, and the adjoint
action of `x ◃ y` is the conjugate of the action of `y` by the action
of `x`. It is another way to understand the self-distributivity axiom.

This is used in the natural rack homomorphism `toConj` from `R` to
`Conj (R ≃ R)` defined by `op'`.
-/
theorem ad_conj {R : Type*} [Rack R] (x y : R) : act' (x ◃ y) = act' x * act' y * (act' x)⁻¹ := by
  /-
    R : Type u_2
    inst✝ : Rack R
    x y : R
    ⊢ Eq (Rack.act' (Shelf.act x y)) (HMul.hMul (HMul.hMul (Rack.act' x) (Rack.act …
  -/
  rw [eq_mul_inv_iff_mul_eq]; ext z
  /-
    case H
    R : Type u_2
    inst✝ : Rack R
    x y z : R
    ⊢ Eq ((HMul.hMul (Rack.act' (Shelf.act x y)) (Rack.act' x)) z) ((HMul.hMul (Ra …
  -/
  apply self_distrib.symm
  /-
    🎉 no goals
  -/


/-- The opposite rack, swapping the roles of `◃` and `◃⁻¹`.
-/
instance oppositeRack : Rack Rᵐᵒᵖ where
  act x y := op (invAct (unop x) (unop y))
  self_distrib := by
    /-
      R : Type u_1
      inst✝ : Rack R
      ⊢ ∀ {x y z : MulOpposite R}, Eq ((fun x y => MulOpposite.op (Rack.invAct (MulO …
    -/
    intro x y z
    /-
      R : Type u_1
      inst✝ : Rack R
      x y z : MulOpposite R
      ⊢ Eq ((fun x y => MulOpposite.op (Rack.invAct (MulOpposite.unop x) (MulOpposit …
    -/
    induction x
    /-
      case h
      R : Type u_1
      inst✝ : Rack R
      y z : MulOpposite R
      X✝ : R
      ⊢ Eq ((fun x y => MulOpposite.op (Rack.invAct (MulOpposite.unop x) (MulOpposit …
    -/
    induction y
    /-
      case h.h
      R : Type u_1
      inst✝ : Rack R
      z : MulOpposite R
      X✝¹ X✝ : R
      ⊢ Eq ((fun x y => MulOpposite.op (Rack.invAct (MulOpposite.unop x) (MulOpposit …
    -/
    induction z
    /-
      case h.h.h
      R : Type u_1
      inst✝ : Rack R
      X✝² X✝¹ X✝ : R
      ⊢ Eq ((fun x y => MulOpposite.op (Rack.invAct (MulOpposite.unop x) (MulOpposit …
    -/
    simp only [op_inj, unop_op, op_unop]
    /-
      case h.h.h
      R : Type u_1
      inst✝ : Rack R
      X✝² X✝¹ X✝ : R
      ⊢ Eq (Rack.invAct X✝² (Rack.invAct X✝¹ X✝)) (Rack.invAct (Rack.invAct X✝² X✝¹) …
    -/
    rw [self_distrib_inv]
    /-
      🎉 no goals
    -/
  invAct x y := op (Shelf.act (unop x) (unop y))
                                                                     /-
                                                                       R : Type u_1
                                                                       inst✝ : Rack R
                                                                       x y : R
                                                                       ⊢ Eq ((fun x y => MulOpposite.op (Shelf.act (MulOpposite.unop x) (MulOpposite. …
                                                                     -/
  left_inv := MulOpposite.rec' fun x => MulOpposite.rec' fun y => by simp
                                                                     /-
                                                                       🎉 no goals
                                                                     -/
                                                                      /-
                                                                        R : Type u_1
                                                                        inst✝ : Rack R
                                                                        x y : R
                                                                        ⊢ Eq (Shelf.act (MulOpposite.op x) ((fun x y => MulOpposite.op (Shelf.act (Mul …
                                                                      -/
  right_inv := MulOpposite.rec' fun x => MulOpposite.rec' fun y => by simp
                                                                      /-
                                                                        🎉 no goals
                                                                      -/


@[simp]
theorem op_act_op_eq {x y : R} : op x ◃ op y = op (x ◃⁻¹ y) :=
  rfl


@[simp]
theorem op_invAct_op_eq {x y : R} : op x ◃⁻¹ op y = op (x ◃ y) :=
  rfl


@[simp]
                                                              /-
                                                                R : Type u_1
                                                                inst✝ : Rack R
                                                                x y : R
                                                                ⊢ Eq (Shelf.act (Shelf.act x x) y) (Shelf.act x y)
                                                              -/
theorem self_act_act_eq {x y : R} : (x ◃ x) ◃ y = x ◃ y := by rw [← right_inv x y, ← self_distrib]
                                                              /-
                                                                🎉 no goals
                                                              -/


@[simp]
theorem self_invAct_invAct_eq {x y : R} : (x ◃⁻¹ x) ◃⁻¹ y = x ◃⁻¹ y := by
  /-
    R : Type u_1
    inst✝ : Rack R
    x y : R
    ⊢ Eq (Rack.invAct (Rack.invAct x x) y) (Rack.invAct x y)
  -/
  have h := @self_act_act_eq _ _ (op x) (op y)
  /-
    R : Type u_1
    inst✝ : Rack R
    x y : R
    h : Eq (Shelf.act (Shelf.act (MulOpposite.op x) (MulOpposite.op x)) (MulOpposi …
    ⊢ Eq (Rack.invAct (Rack.invAct x x) y) (Rack.invAct x y)
  -/
  simpa using h
  /-
    🎉 no goals
  -/


@[simp]
theorem self_act_invAct_eq {x y : R} : (x ◃ x) ◃⁻¹ y = x ◃⁻¹ y := by
  /-
    R : Type u_1
    inst✝ : Rack R
    x y : R
    ⊢ Eq (Rack.invAct (Shelf.act x x) y) (Rack.invAct x y)
  -/
  rw [← left_cancel (x ◃ x)]
  /-
    R : Type u_1
    inst✝ : Rack R
    x y : R
    ⊢ Eq (Shelf.act (Shelf.act x x) (Rack.invAct (Shelf.act x x) y)) (Shelf.act (S …
  -/
  rw [right_inv]
  /-
    R : Type u_1
    inst✝ : Rack R
    x y : R
    ⊢ Eq y (Shelf.act (Shelf.act x x) (Rack.invAct x y))
  -/
  rw [self_act_act_eq]
  /-
    R : Type u_1
    inst✝ : Rack R
    x y : R
    ⊢ Eq y (Shelf.act x (Rack.invAct x y))
  -/
  rw [right_inv]
  /-
    🎉 no goals
  -/


@[simp]
theorem self_invAct_act_eq {x y : R} : (x ◃⁻¹ x) ◃ y = x ◃ y := by
  /-
    R : Type u_1
    inst✝ : Rack R
    x y : R
    ⊢ Eq (Shelf.act (Rack.invAct x x) y) (Shelf.act x y)
  -/
  have h := @self_act_invAct_eq _ _ (op x) (op y)
  /-
    R : Type u_1
    inst✝ : Rack R
    x y : R
    h : Eq (Rack.invAct (Shelf.act (MulOpposite.op x) (MulOpposite.op x)) (MulOppo …
    ⊢ Eq (Shelf.act (Rack.invAct x x) y) (Shelf.act x y)
  -/
  simpa using h
  /-
    🎉 no goals
  -/


theorem self_act_eq_iff_eq {x y : R} : x ◃ x = y ◃ y ↔ x = y := by
  /-
    R : Type u_1
    inst✝ : Rack R
    x y : R
    ⊢ Iff (Eq (Shelf.act x x) (Shelf.act y y)) (Eq x y)
  -/
  constructor; swap
    /-
      case mpr
      R : Type u_1
      inst✝ : Rack R
      x y : R
      ⊢ Eq x y → Eq (Shelf.act x x) (Shelf.act y y)
    -/
  · rintro rfl; rfl
                /-
                  🎉 no goals
                -/
  /-
    case mp
    R : Type u_1
    inst✝ : Rack R
    x y : R
    ⊢ Eq (Shelf.act x x) (Shelf.act y y) → Eq x y
  -/
  intro h
  /-
    case mp
    R : Type u_1
    inst✝ : Rack R
    x y : R
    h : Eq (Shelf.act x x) (Shelf.act y y)
    ⊢ Eq x y
  -/
  trans (x ◃ x) ◃⁻¹ x ◃ x
    /-
      R : Type u_1
      inst✝ : Rack R
      x y : R
      h : Eq (Shelf.act x x) (Shelf.act y y)
      ⊢ Eq x (Rack.invAct (Shelf.act x x) (Shelf.act x x))
    -/
  · rw [← left_cancel (x ◃ x), right_inv, self_act_act_eq]
    /-
      🎉 no goals
    -/
    /-
      R : Type u_1
      inst✝ : Rack R
      x y : R
      h : Eq (Shelf.act x x) (Shelf.act y y)
      ⊢ Eq (Rack.invAct (Shelf.act x x) (Shelf.act x x)) y
    -/
  · rw [h, ← left_cancel (y ◃ y), right_inv, self_act_act_eq]
    /-
      🎉 no goals
    -/


theorem self_invAct_eq_iff_eq {x y : R} : x ◃⁻¹ x = y ◃⁻¹ y ↔ x = y := by
  /-
    R : Type u_1
    inst✝ : Rack R
    x y : R
    ⊢ Iff (Eq (Rack.invAct x x) (Rack.invAct y y)) (Eq x y)
  -/
  have h := @self_act_eq_iff_eq _ _ (op x) (op y)
  /-
    R : Type u_1
    inst✝ : Rack R
    x y : R
    h : Iff (Eq (Shelf.act (MulOpposite.op x) (MulOpposite.op x)) (Shelf.act (MulO …
    ⊢ Iff (Eq (Rack.invAct x x) (Rack.invAct y y)) (Eq x y)
  -/
  simpa using h
  /-
    🎉 no goals
  -/


/-- The map `x ↦ x ◃ x` is a bijection.  (This has applications for the
regular isotopy version of the Reidemeister I move for knot diagrams.)
-/
def selfApplyEquiv (R : Type*) [Rack R] : R ≃ R where
  toFun x := x ◃ x
  invFun x := x ◃⁻¹ x
                   /-
                     R✝ : Type u_1
                     inst✝¹ : Rack R✝
                     R : Type u_2
                     inst✝ : Rack R
                     x : R
                     ⊢ Eq ((fun x => Rack.invAct x x) ((fun x => Shelf.act x x) x)) x
                   -/
  left_inv x := by simp
                   /-
                     🎉 no goals
                   -/
                    /-
                      R✝ : Type u_1
                      inst✝¹ : Rack R✝
                      R : Type u_2
                      inst✝ : Rack R
                      x : R
                      ⊢ Eq ((fun x => Shelf.act x x) ((fun x => Rack.invAct x x) x)) x
                    -/
  right_inv x := by simp
                    /-
                      🎉 no goals
                    -/


/-- An involutory rack is one for which `Rack.oppositeRack R x` is an involution for every x.
-/
def IsInvolutory (R : Type*) [Rack R] : Prop :=
  ∀ x : R, Function.Involutive (Shelf.act x)


theorem involutory_invAct_eq_act {R : Type*} [Rack R] (h : IsInvolutory R) (x y : R) :
    x ◃⁻¹ y = x ◃ y := by
  /-
    R : Type u_2
    inst✝ : Rack R
    h : Rack.IsInvolutory R
    x y : R
    ⊢ Eq (Rack.invAct x y) (Shelf.act x y)
  -/
  rw [← left_cancel x, right_inv, h x]
  /-
    🎉 no goals
  -/


/-- An abelian rack is one for which the mediality axiom holds.
-/
def IsAbelian (R : Type*) [Rack R] : Prop :=
  ∀ x y z w : R, (x ◃ y) ◃ z ◃ w = (x ◃ z) ◃ y ◃ w


/-- Associative racks are uninteresting.
-/
theorem assoc_iff_id {R : Type*} [Rack R] {x y z : R} : x ◃ y ◃ z = (x ◃ y) ◃ z ↔ x ◃ z = z := by
  /-
    R : Type u_2
    inst✝ : Rack R
    x y z : R
    ⊢ Iff (Eq (Shelf.act x (Shelf.act y z)) (Shelf.act (Shelf.act x y) z)) (Eq (Sh …
  -/
  rw [self_distrib]
  /-
    R : Type u_2
    inst✝ : Rack R
    x y z : R
    ⊢ Iff (Eq (Shelf.act (Shelf.act x y) (Shelf.act x z)) (Shelf.act (Shelf.act x  …
  -/
  rw [left_cancel]
  /-
    🎉 no goals
  -/


instance : FunLike (S₁ →◃ S₂) S₁ S₂ where
  coe := toFun
  coe_injective' | ⟨_, _⟩, ⟨_, _⟩, rfl => rfl


@[simp] theorem toFun_eq_coe (f : S₁ →◃ S₂) : f.toFun = f := rfl


@[simp]
theorem map_act (f : S₁ →◃ S₂) {x y : S₁} : f (x ◃ y) = f x ◃ f y :=
  map_act' f


/-- The identity homomorphism -/
def id (S : Type*) [Shelf S] : S →◃ S where
  toFun := fun x => x
                 /-
                   S₁ : Type u_1
                   S₂ : Type u_2
                   S₃ : Type u_3
                   inst✝³ : Shelf S₁
                   inst✝² : Shelf S₂
                   inst✝¹ : Shelf S₃
                   S : Type u_4
                   inst✝ : Shelf S
                   ⊢ ∀ {x y : S}, Eq ((fun x => x) (Shelf.act x y)) (Shelf.act ((fun x => x) x) ( …
                 -/
  map_act' := by simp
                 /-
                   🎉 no goals
                 -/


instance inhabited (S : Type*) [Shelf S] : Inhabited (S →◃ S) :=
  ⟨id S⟩


/-- The composition of shelf homomorphisms -/
def comp (g : S₂ →◃ S₃) (f : S₁ →◃ S₂) : S₁ →◃ S₃ where
  toFun := g.toFun ∘ f.toFun
                 /-
                   S₁ : Type u_1
                   S₂ : Type u_2
                   S₃ : Type u_3
                   inst✝² : Shelf S₁
                   inst✝¹ : Shelf S₂
                   inst✝ : Shelf S₃
                   g : ShelfHom S₂ S₃
                   f : ShelfHom S₁ S₂
                   ⊢ ∀ {x y : S₁}, Eq (Function.comp g.toFun f.toFun (Shelf.act x y)) (Shelf.act  …
                 -/
  map_act' := by simp
                 /-
                   🎉 no goals
                 -/


@[simp]
theorem comp_apply (g : S₂ →◃ S₃) (f : S₁ →◃ S₂) (x : S₁) : (g.comp f) x = g (f x) :=
  rfl


/-- A quandle is a rack such that each automorphism fixes its corresponding element.
-/
class Quandle (α : Type*) extends Rack α where
  /-- The fixing property of a Quandle -/
  fix : ∀ {x : α}, act x x = x


@[simp]
theorem fix_inv {x : Q} : x ◃⁻¹ x = x := by
  /-
    Q : Type u_1
    inst✝ : Quandle Q
    x : Q
    ⊢ Eq (Rack.invAct x x) x
  -/
  rw [← left_cancel x]
  /-
    Q : Type u_1
    inst✝ : Quandle Q
    x : Q
    ⊢ Eq (Shelf.act x (Rack.invAct x x)) (Shelf.act x x)
  -/
  simp
  /-
    🎉 no goals
  -/


instance oppositeQuandle : Quandle Qᵐᵒᵖ where
  fix := by
    /-
      Q : Type u_1
      inst✝ : Quandle Q
      ⊢ ∀ {x : MulOpposite Q}, Eq (Shelf.act x x) x
    -/
    intro x
    /-
      Q : Type u_1
      inst✝ : Quandle Q
      x : MulOpposite Q
      ⊢ Eq (Shelf.act x x) x
    -/
    induction x
    /-
      case h
      Q : Type u_1
      inst✝ : Quandle Q
      X✝ : Q
      ⊢ Eq (Shelf.act (MulOpposite.op X✝) (MulOpposite.op X✝)) (MulOpposite.op X✝)
    -/
    simp
    /-
      🎉 no goals
    -/


/-- The conjugation quandle of a group.  Each element of the group acts by
the corresponding inner automorphism.
-/
-- Porting note: no need for `nolint` and added `reducible`
abbrev Conj (G : Type*) := G


instance Conj.quandle (G : Type*) [Group G] : Quandle (Conj G) where
  act x := @MulAut.conj G _ x
  self_distrib := by
    /-
      Q : Type u_1
      inst✝¹ : Quandle Q
      G : Type u_2
      inst✝ : Group G
      ⊢ ∀ {x y z : Quandle.Conj G}, Eq ((fun x => ⇑(MulAut.conj x)) x ((fun x => ⇑(M …
    -/
    intro x y z
    /-
      Q : Type u_1
      inst✝¹ : Quandle Q
      G : Type u_2
      inst✝ : Group G
      x y z : Quandle.Conj G
      ⊢ Eq ((fun x => ⇑(MulAut.conj x)) x ((fun x => ⇑(MulAut.conj x)) y z)) ((fun x …
    -/
    dsimp only [MulAut.conj_apply]
    /-
      Q : Type u_1
      inst✝¹ : Quandle Q
      G : Type u_2
      inst✝ : Group G
      x y z : Quandle.Conj G
      ⊢ Eq (HMul.hMul (HMul.hMul x (HMul.hMul (HMul.hMul y z) (Inv.inv y))) (Inv.inv …
    -/
    simp [mul_assoc]
    /-
      🎉 no goals
    -/
  invAct x := (@MulAut.conj G _ x).symm
  left_inv x y := by
    /-
      Q : Type u_1
      inst✝¹ : Quandle Q
      G : Type u_2
      inst✝ : Group G
      x y : Quandle.Conj G
      ⊢ Eq ((fun x => ⇑(MulEquiv.symm (MulAut.conj x))) x (Shelf.act x y)) y
    -/
    simp [act', mul_assoc]
    /-
      🎉 no goals
    -/
  right_inv x y := by
    /-
      Q : Type u_1
      inst✝¹ : Quandle Q
      G : Type u_2
      inst✝ : Group G
      x y : Quandle.Conj G
      ⊢ Eq (Shelf.act x ((fun x => ⇑(MulEquiv.symm (MulAut.conj x))) x y)) y
    -/
    simp [act', mul_assoc]
    /-
      🎉 no goals
    -/
            /-
              Q : Type u_1
              inst✝¹ : Quandle Q
              G : Type u_2
              inst✝ : Group G
              ⊢ ∀ {x : Quandle.Conj G}, Eq (Shelf.act x x) x
            -/
  fix := by simp
            /-
              🎉 no goals
            -/


@[simp]
theorem conj_act_eq_conj {G : Type*} [Group G] (x y : Conj G) :
    x ◃ y = ((x : G) * (y : G) * (x : G)⁻¹ : G) :=
  rfl


theorem conj_swap {G : Type*} [Group G] (x y : Conj G) : x ◃ y = y ↔ y ◃ x = x := by
  /-
    G : Type u_2
    inst✝ : Group G
    x y : Quandle.Conj G
    ⊢ Iff (Eq (Shelf.act x y) y) (Eq (Shelf.act y x) x)
  -/
  dsimp [Conj] at *; constructor
  /-
    case mp
    G : Type u_2
    inst✝ : Group G
    x y : Quandle.Conj G
    ⊢ Eq (HMul.hMul (HMul.hMul x y) (Inv.inv x)) y → Eq (HMul.hMul (HMul.hMul y x) …
  -/
  repeat' intro h; conv_rhs => rw [eq_mul_inv_of_mul_eq (eq_mul_inv_of_mul_eq h)]; simp
  /-
    🎉 no goals
  -/


/-- `Conj` is functorial
-/
def Conj.map {G : Type*} {H : Type*} [Group G] [Group H] (f : G →* H) : Conj G →◃ Conj H where
  toFun := f
                 /-
                   Q : Type u_1
                   inst✝² : Quandle Q
                   G : Type u_2
                   H : Type u_3
                   inst✝¹ : Group G
                   inst✝ : Group H
                   f : MonoidHom G H
                   ⊢ ∀ {x y : Quandle.Conj G}, Eq (f (Shelf.act x y)) (Shelf.act (f x) (f y))
                 -/
  map_act' := by simp
                 /-
                   🎉 no goals
                 -/

-- Porting note: I don't think HasLift exists
-- instance {G : Type*} {H : Type*} [Group G] [Group H] : HasLift (G →* H) (Conj G →◃ Conj H)
--     where lift := Conj.map


/-- The dihedral quandle. This is the conjugation quandle of the dihedral group restrict to flips.

Used for Fox n-colorings of knots.
-/
def Dihedral (n : ℕ) :=
  ZMod n


/-- The operation for the dihedral quandle.  It does not need to be an equivalence
because it is an involution (see `dihedralAct.inv`).
-/
def dihedralAct (n : ℕ) (a : ZMod n) : ZMod n → ZMod n := fun b => 2 * a - b


theorem dihedralAct.inv (n : ℕ) (a : ZMod n) : Function.Involutive (dihedralAct n a) := by
  /-
    n : Nat
    a : ZMod n
    ⊢ Function.Involutive (Quandle.dihedralAct n a)
  -/
  intro b
  /-
    n : Nat
    a b : ZMod n
    ⊢ Eq (Quandle.dihedralAct n a (Quandle.dihedralAct n a b)) b
  -/
  dsimp only [dihedralAct]
  /-
    n : Nat
    a b : ZMod n
    ⊢ Eq (HSub.hSub (HMul.hMul 2 a) (HSub.hSub (HMul.hMul 2 a) b)) b
  -/
  simp
  /-
    🎉 no goals
  -/


instance (n : ℕ) : Quandle (Dihedral n) where
  act := dihedralAct n
  self_distrib := by
    /-
      Q : Type u_1
      inst✝ : Quandle Q
      n : Nat
      ⊢ ∀ {x y z : Quandle.Dihedral n}, Eq (Quandle.dihedralAct n x (Quandle.dihedra …
    -/
    intro x y z
    /-
      Q : Type u_1
      inst✝ : Quandle Q
      n : Nat
      x y z : Quandle.Dihedral n
      ⊢ Eq (Quandle.dihedralAct n x (Quandle.dihedralAct n y z)) (Quandle.dihedralAc …
    -/
    simp only [dihedralAct]
    /-
      Q : Type u_1
      inst✝ : Quandle Q
      n : Nat
      x y z : Quandle.Dihedral n
      ⊢ Eq (HSub.hSub (HMul.hMul 2 x) (HSub.hSub (HMul.hMul 2 y) z)) (HSub.hSub (HMu …
    -/
    ring_nf
    /-
      🎉 no goals
    -/
  invAct := dihedralAct n
  left_inv x := (dihedralAct.inv n x).leftInverse
  right_inv x := (dihedralAct.inv n x).rightInverse
  fix := by
    /-
      Q : Type u_1
      inst✝ : Quandle Q
      n : Nat
      ⊢ ∀ {x : Quandle.Dihedral n}, Eq (Shelf.act x x) x
    -/
    intro x
    /-
      Q : Type u_1
      inst✝ : Quandle Q
      n : Nat
      x : Quandle.Dihedral n
      ⊢ Eq (Shelf.act x x) x
    -/
    simp only [dihedralAct]
    /-
      Q : Type u_1
      inst✝ : Quandle Q
      n : Nat
      x : Quandle.Dihedral n
      ⊢ Eq (HSub.hSub (HMul.hMul 2 x) x) x
    -/
    ring_nf
    /-
      🎉 no goals
    -/


/-- This is the natural rack homomorphism to the conjugation quandle of the group `R ≃ R`
that acts on the rack.
-/
def toConj (R : Type*) [Rack R] : R →◃ Quandle.Conj (R ≃ R) where
  toFun := act'
  map_act' := by
    /-
      R : Type u_1
      inst✝ : Rack R
      ⊢ ∀ {x y : R}, Eq (Rack.act' (Shelf.act x y)) (Shelf.act (Rack.act' x) (Rack.a …
    -/
    intro x y
    /-
      R : Type u_1
      inst✝ : Rack R
      x y : R
      ⊢ Eq (Rack.act' (Shelf.act x y)) (Shelf.act (Rack.act' x) (Rack.act' y))
    -/
    exact ad_conj x y
    /-
      🎉 no goals
    -/


/-- Free generators of the enveloping group.
-/
inductive PreEnvelGroup (R : Type u) : Type u
  | unit : PreEnvelGroup R
  | incl (x : R) : PreEnvelGroup R
  | mul (a b : PreEnvelGroup R) : PreEnvelGroup R
  | inv (a : PreEnvelGroup R) : PreEnvelGroup R


instance PreEnvelGroup.inhabited (R : Type u) : Inhabited (PreEnvelGroup R) :=
  ⟨PreEnvelGroup.unit⟩


/-- Relations for the enveloping group. This is a type-valued relation because
`toEnvelGroup.mapAux.well_def` inducts on it to show `toEnvelGroup.map`
is well-defined.  The relation `PreEnvelGroupRel` is the `Prop`-valued version,
which is used to define `EnvelGroup` itself.
-/
inductive PreEnvelGroupRel' (R : Type u) [Rack R] : PreEnvelGroup R → PreEnvelGroup R → Type u
  | refl {a : PreEnvelGroup R} : PreEnvelGroupRel' R a a
  | symm {a b : PreEnvelGroup R} (hab : PreEnvelGroupRel' R a b) : PreEnvelGroupRel' R b a
  | trans {a b c : PreEnvelGroup R} (hab : PreEnvelGroupRel' R a b)
    (hbc : PreEnvelGroupRel' R b c) : PreEnvelGroupRel' R a c
  | congr_mul {a b a' b' : PreEnvelGroup R} (ha : PreEnvelGroupRel' R a a')
    (hb : PreEnvelGroupRel' R b b') : PreEnvelGroupRel' R (mul a b) (mul a' b')
  | congr_inv {a a' : PreEnvelGroup R} (ha : PreEnvelGroupRel' R a a') :
    PreEnvelGroupRel' R (inv a) (inv a')
  | assoc (a b c : PreEnvelGroup R) : PreEnvelGroupRel' R (mul (mul a b) c) (mul a (mul b c))
  | one_mul (a : PreEnvelGroup R) : PreEnvelGroupRel' R (mul unit a) a
  | mul_one (a : PreEnvelGroup R) : PreEnvelGroupRel' R (mul a unit) a
  | inv_mul_cancel (a : PreEnvelGroup R) : PreEnvelGroupRel' R (mul (inv a) a) unit
  | act_incl (x y : R) :
    PreEnvelGroupRel' R (mul (mul (incl x) (incl y)) (inv (incl x))) (incl (x ◃ y))


instance PreEnvelGroupRel'.inhabited (R : Type u) [Rack R] :
    Inhabited (PreEnvelGroupRel' R unit unit) :=
  ⟨PreEnvelGroupRel'.refl⟩


/--
The `PreEnvelGroupRel` relation as a `Prop`.  Used as the relation for `PreEnvelGroup.setoid`.
-/
inductive PreEnvelGroupRel (R : Type u) [Rack R] : PreEnvelGroup R → PreEnvelGroup R → Prop
  | rel {a b : PreEnvelGroup R} (r : PreEnvelGroupRel' R a b) : PreEnvelGroupRel R a b


/-- A quick way to convert a `PreEnvelGroupRel'` to a `PreEnvelGroupRel`.
-/
theorem PreEnvelGroupRel'.rel {R : Type u} [Rack R] {a b : PreEnvelGroup R} :
    PreEnvelGroupRel' R a b → PreEnvelGroupRel R a b := PreEnvelGroupRel.rel


@[refl]
theorem PreEnvelGroupRel.refl {R : Type u} [Rack R] {a : PreEnvelGroup R} :
    PreEnvelGroupRel R a a :=
  PreEnvelGroupRel.rel PreEnvelGroupRel'.refl


@[symm]
theorem PreEnvelGroupRel.symm {R : Type u} [Rack R] {a b : PreEnvelGroup R} :
    PreEnvelGroupRel R a b → PreEnvelGroupRel R b a
  | ⟨r⟩ => r.symm.rel


@[trans]
theorem PreEnvelGroupRel.trans {R : Type u} [Rack R] {a b c : PreEnvelGroup R} :
    PreEnvelGroupRel R a b → PreEnvelGroupRel R b c → PreEnvelGroupRel R a c
  | ⟨rab⟩, ⟨rbc⟩ => (rab.trans rbc).rel


instance PreEnvelGroup.setoid (R : Type*) [Rack R] : Setoid (PreEnvelGroup R) where
  r := PreEnvelGroupRel R
  iseqv := by
    /-
      R : Type u_1
      inst✝ : Rack R
      ⊢ Equivalence (Rack.PreEnvelGroupRel R)
    -/
    constructor
      /-
        case refl
        R : Type u_1
        inst✝ : Rack R
        ⊢ ∀ (x : Rack.PreEnvelGroup R), Rack.PreEnvelGroupRel R x x
      -/
    · apply PreEnvelGroupRel.refl
      /-
        🎉 no goals
      -/
      /-
        case symm
        R : Type u_1
        inst✝ : Rack R
        ⊢ ∀ {x y : Rack.PreEnvelGroup R}, Rack.PreEnvelGroupRel R x y → Rack.PreEnvelG …
      -/
    · apply PreEnvelGroupRel.symm
      /-
        🎉 no goals
      -/
      /-
        case trans
        R : Type u_1
        inst✝ : Rack R
        ⊢ ∀ {x y z : Rack.PreEnvelGroup R}, Rack.PreEnvelGroupRel R x y → Rack.PreEnve …
      -/
    · apply PreEnvelGroupRel.trans
      /-
        🎉 no goals
      -/

/-- The universal enveloping group for the rack R.
-/
def EnvelGroup (R : Type*) [Rack R] :=
  Quotient (PreEnvelGroup.setoid R)

-- Define the `Group` instances in two steps so `inv` can be inferred correctly.
-- TODO: is there a non-invasive way of defining the instance directly?

instance (R : Type*) [Rack R] : DivInvMonoid (EnvelGroup R) where
  mul a b :=
    Quotient.liftOn₂ a b (fun a b => ⟦PreEnvelGroup.mul a b⟧) fun _ _ _ _ ⟨ha⟩ ⟨hb⟩ =>
      Quotient.sound (PreEnvelGroupRel'.congr_mul ha hb).rel
  one := ⟦unit⟧
  inv a :=
    Quotient.liftOn a (fun a => ⟦PreEnvelGroup.inv a⟧) fun _ _ ⟨ha⟩ =>
      Quotient.sound (PreEnvelGroupRel'.congr_inv ha).rel
  mul_assoc a b c :=
    Quotient.inductionOn₃ a b c fun a b c => Quotient.sound (PreEnvelGroupRel'.assoc a b c).rel
  one_mul a := Quotient.inductionOn a fun a => Quotient.sound (PreEnvelGroupRel'.one_mul a).rel
  mul_one a := Quotient.inductionOn a fun a => Quotient.sound (PreEnvelGroupRel'.mul_one a).rel


instance (R : Type*) [Rack R] : Group (EnvelGroup R) :=
  { inv_mul_cancel := fun a =>
      Quotient.inductionOn a fun a => Quotient.sound (PreEnvelGroupRel'.inv_mul_cancel a).rel }


instance EnvelGroup.inhabited (R : Type*) [Rack R] : Inhabited (EnvelGroup R) :=
  ⟨1⟩


/-- The canonical homomorphism from a rack to its enveloping group.
Satisfies universal properties given by `toEnvelGroup.map` and `toEnvelGroup.univ`.
-/
def toEnvelGroup (R : Type*) [Rack R] : R →◃ Quandle.Conj (EnvelGroup R) where
  toFun x := ⟦incl x⟧
  map_act' := @fun x y => Quotient.sound (PreEnvelGroupRel'.act_incl x y).symm.rel


/-- The preliminary definition of the induced map from the enveloping group.
See `toEnvelGroup.map`.
-/
def toEnvelGroup.mapAux {R : Type*} [Rack R] {G : Type*} [Group G] (f : R →◃ Quandle.Conj G) :
    PreEnvelGroup R → G
  | .unit => 1
  | .incl x => f x
  | .mul a b => toEnvelGroup.mapAux f a * toEnvelGroup.mapAux f b
  | .inv a => (toEnvelGroup.mapAux f a)⁻¹


/-- Show that `toEnvelGroup.mapAux` sends equivalent expressions to equal terms.
-/
theorem well_def {R : Type*} [Rack R] {G : Type*} [Group G] (f : R →◃ Quandle.Conj G) :
    ∀ {a b : PreEnvelGroup R},
      PreEnvelGroupRel' R a b → toEnvelGroup.mapAux f a = toEnvelGroup.mapAux f b
  | _, _, PreEnvelGroupRel'.refl => rfl
  | _, _, PreEnvelGroupRel'.symm h => (well_def f h).symm
  | _, _, PreEnvelGroupRel'.trans hac hcb => Eq.trans (well_def f hac) (well_def f hcb)
  | _, _, PreEnvelGroupRel'.congr_mul ha hb => by
    /-
      R : Type u_1
      inst✝¹ : Rack R
      G : Type u_2
      inst✝ : Group G
      f : ShelfHom R (Quandle.Conj G)
      a✝ b✝ a'✝ b'✝ : Rack.PreEnvelGroup R
      ha : Rack.PreEnvelGroupRel' R a✝ a'✝
      hb : Rack.PreEnvelGroupRel' R b✝ b'✝
      ⊢ Eq (Rack.toEnvelGroup.mapAux f (a✝.mul b✝)) (Rack.toEnvelGroup.mapAux f (a'✝ …
    -/
    simp [toEnvelGroup.mapAux, well_def f ha, well_def f hb]
    /-
      🎉 no goals
    -/
                             /-
                               R : Type u_1
                               inst✝¹ : Rack R
                               G : Type u_2
                               inst✝ : Group G
                               f : ShelfHom R (Quandle.Conj G)
                               a✝ a'✝ : Rack.PreEnvelGroup R
                               ha : Rack.PreEnvelGroupRel' R a✝ a'✝
                               ⊢ Eq (Rack.toEnvelGroup.mapAux f a✝.inv) (Rack.toEnvelGroup.mapAux f a'✝.inv)
                             -/
  | _, _, congr_inv ha => by simp [toEnvelGroup.mapAux, well_def f ha]
                             /-
                               🎉 no goals
                             -/
                            /-
                              R : Type u_1
                              inst✝¹ : Rack R
                              G : Type u_2
                              inst✝ : Group G
                              f : ShelfHom R (Quandle.Conj G)
                              a b c : Rack.PreEnvelGroup R
                              ⊢ Eq (Rack.toEnvelGroup.mapAux f ((a.mul b).mul c)) (Rack.toEnvelGroup.mapAux  …
                            -/
  | _, _, assoc a b c => by apply mul_assoc
                            /-
                              🎉 no goals
                            -/
                                            /-
                                              R : Type u_1
                                              inst✝¹ : Rack R
                                              G : Type u_2
                                              inst✝ : Group G
                                              f : ShelfHom R (Quandle.Conj G)
                                              a : Rack.PreEnvelGroup R
                                              ⊢ Eq (Rack.toEnvelGroup.mapAux f (Rack.PreEnvelGroup.unit.mul a)) (Rack.toEnve …
                                            -/
  | _, _, PreEnvelGroupRel'.one_mul a => by simp [toEnvelGroup.mapAux]
                                            /-
                                              🎉 no goals
                                            -/
                                            /-
                                              R : Type u_1
                                              inst✝¹ : Rack R
                                              G : Type u_2
                                              inst✝ : Group G
                                              f : ShelfHom R (Quandle.Conj G)
                                              a : Rack.PreEnvelGroup R
                                              ⊢ Eq (Rack.toEnvelGroup.mapAux f (a.mul Rack.PreEnvelGroup.unit)) (Rack.toEnve …
                                            -/
  | _, _, PreEnvelGroupRel'.mul_one a => by simp [toEnvelGroup.mapAux]
                                            /-
                                              🎉 no goals
                                            -/
                                                   /-
                                                     R : Type u_1
                                                     inst✝¹ : Rack R
                                                     G : Type u_2
                                                     inst✝ : Group G
                                                     f : ShelfHom R (Quandle.Conj G)
                                                     a : Rack.PreEnvelGroup R
                                                     ⊢ Eq (Rack.toEnvelGroup.mapAux f (a.inv.mul a)) (Rack.toEnvelGroup.mapAux f Ra …
                                                   -/
  | _, _, PreEnvelGroupRel'.inv_mul_cancel a => by simp [toEnvelGroup.mapAux]
                                                   /-
                                                     🎉 no goals
                                                   -/
                             /-
                               R : Type u_1
                               inst✝¹ : Rack R
                               G : Type u_2
                               inst✝ : Group G
                               f : ShelfHom R (Quandle.Conj G)
                               x y : R
                               ⊢ Eq (Rack.toEnvelGroup.mapAux f (((Rack.PreEnvelGroup.incl x).mul (Rack.PreEn …
                             -/
  | _, _, act_incl x y => by simp [toEnvelGroup.mapAux]
                             /-
                               🎉 no goals
                             -/


/-- Given a map from a rack to a group, lift it to being a map from the enveloping group.
More precisely, the `EnvelGroup` functor is left adjoint to `Quandle.Conj`.
-/
def toEnvelGroup.map {R : Type*} [Rack R] {G : Type*} [Group G] :
    (R →◃ Quandle.Conj G) ≃ (EnvelGroup R →* G) where
  toFun f :=
    { toFun := fun x =>
        Quotient.liftOn x (toEnvelGroup.mapAux f) fun _ _ ⟨hab⟩ =>
          toEnvelGroup.mapAux.well_def f hab
      map_one' := by
        /-
          R : Type u_1
          inst✝¹ : Rack R
          G : Type u_2
          inst✝ : Group G
          f : ShelfHom R (Quandle.Conj G)
          ⊢ Eq ((fun x => Quotient.liftOn x (Rack.toEnvelGroup.mapAux f) ⋯) 1) 1
        -/
        change Quotient.liftOn ⟦Rack.PreEnvelGroup.unit⟧ (toEnvelGroup.mapAux f) _ = 1
        /-
          R : Type u_1
          inst✝¹ : Rack R
          G : Type u_2
          inst✝ : Group G
          f : ShelfHom R (Quandle.Conj G)
          ⊢ Eq ((Quotient.mk (Rack.PreEnvelGroup.setoid R) Rack.PreEnvelGroup.unit).lift …
        -/
        simp only [Quotient.lift_mk, mapAux]
        /-
          🎉 no goals
        -/
      map_mul' := fun x y =>
        Quotient.inductionOn₂ x y fun x y => by
          /-
            R : Type u_1
            inst✝¹ : Rack R
            G : Type u_2
            inst✝ : Group G
            f : ShelfHom R (Quandle.Conj G)
            x✝ y✝ : Rack.EnvelGroup R
            x y : Rack.PreEnvelGroup R
            ⊢ Eq ({ toFun := fun x => Quotient.liftOn x (Rack.toEnvelGroup.mapAux f) ⋯, ma …
          -/
          simp only [toEnvelGroup.mapAux]
          /-
            R : Type u_1
            inst✝¹ : Rack R
            G : Type u_2
            inst✝ : Group G
            f : ShelfHom R (Quandle.Conj G)
            x✝ y✝ : Rack.EnvelGroup R
            x y : Rack.PreEnvelGroup R
            ⊢ Eq (Quotient.liftOn (HMul.hMul (Quotient.mk (Rack.PreEnvelGroup.setoid R) x) …
          -/
          change Quotient.liftOn ⟦mul x y⟧ (toEnvelGroup.mapAux f) _ = _
          /-
            R : Type u_1
            inst✝¹ : Rack R
            G : Type u_2
            inst✝ : Group G
            f : ShelfHom R (Quandle.Conj G)
            x✝ y✝ : Rack.EnvelGroup R
            x y : Rack.PreEnvelGroup R
            ⊢ Eq ((Quotient.mk (Rack.PreEnvelGroup.setoid R) (x.mul y)).liftOn (Rack.toEnv …
          -/
          simp [toEnvelGroup.mapAux] }
          /-
            🎉 no goals
          -/
  invFun F := (Quandle.Conj.map F).comp (toEnvelGroup R)
                   /-
                     R : Type u_1
                     inst✝¹ : Rack R
                     G : Type u_2
                     inst✝ : Group G
                     f : ShelfHom R (Quandle.Conj G)
                     ⊢ Eq ((fun F => (Quandle.Conj.map F).comp (Rack.toEnvelGroup R)) ((fun f => {  …
                   -/
  left_inv f := by ext; rfl
                        /-
                          🎉 no goals
                        -/
  right_inv F :=
    MonoidHom.ext fun x =>
      Quotient.inductionOn x fun x => by
        induction x with
        | unit => exact F.map_one.symm
        | incl => rfl
        | mul x y ih_x ih_y =>
          have hm : ⟦x.mul y⟧ = @Mul.mul (EnvelGroup R) _ ⟦x⟧ ⟦y⟧ := rfl
          simp only [MonoidHom.coe_mk, OneHom.coe_mk, Quotient.lift_mk]
          suffices ∀ x y, F (Mul.mul x y) = F (x) * F (y) by
            simp_all only [MonoidHom.coe_mk, OneHom.coe_mk, Quotient.lift_mk, hm]
            rw [← ih_x, ← ih_y, mapAux]
          exact F.map_mul
        | inv x ih_x =>
          have hm : ⟦x.inv⟧ = @Inv.inv (EnvelGroup R) _ ⟦x⟧ := rfl
          rw [hm, F.map_inv, MonoidHom.map_inv, ih_x]


/-- Given a homomorphism from a rack to a group, it factors through the enveloping group.
-/
theorem toEnvelGroup.univ (R : Type*) [Rack R] (G : Type*) [Group G] (f : R →◃ Quandle.Conj G) :
    (Quandle.Conj.map (toEnvelGroup.map f)).comp (toEnvelGroup R) = f :=
  toEnvelGroup.map.symm_apply_apply f


/-- The homomorphism `toEnvelGroup.map f` is the unique map that fits into the commutative
triangle in `toEnvelGroup.univ`.
-/
theorem toEnvelGroup.univ_uniq (R : Type*) [Rack R] (G : Type*) [Group G]
    (f : R →◃ Quandle.Conj G) (g : EnvelGroup R →* G)
    (h : f = (Quandle.Conj.map g).comp (toEnvelGroup R)) : g = toEnvelGroup.map f :=
  h.symm ▸ (toEnvelGroup.map.apply_symm_apply g).symm


/-- The induced group homomorphism from the enveloping group into bijections of the rack,
using `Rack.toConj`. Satisfies the property `envelAction_prop`.

This gives the rack `R` the structure of an augmented rack over `EnvelGroup R`.
-/
def envelAction {R : Type*} [Rack R] : EnvelGroup R →* R ≃ R :=
  toEnvelGroup.map (toConj R)


@[simp]
theorem envelAction_prop {R : Type*} [Rack R] (x y : R) :
    envelAction (toEnvelGroup R x) y = x ◃ y :=
  rfl


