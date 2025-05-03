/-- Bundled non-unital semiring homomorphisms `α →ₙ+* β`; use this for bundled non-unital ring
homomorphisms too.

When possible, instead of parametrizing results over `(f : α →ₙ+* β)`,
you should parametrize over `(F : Type*) [NonUnitalRingHomClass F α β] (f : F)`.

When you extend this structure, make sure to extend `NonUnitalRingHomClass`. -/
structure NonUnitalRingHom (α β : Type*) [NonUnitalNonAssocSemiring α]
  [NonUnitalNonAssocSemiring β] extends α →ₙ* β, α →+ β


/-- `α →ₙ+* β` denotes the type of non-unital ring homomorphisms from `α` to `β`. -/
infixr:25 " →ₙ+* " => NonUnitalRingHom


/-- `NonUnitalRingHomClass F α β` states that `F` is a type of non-unital (semi)ring
homomorphisms. You should extend this class when you extend `NonUnitalRingHom`. -/
class NonUnitalRingHomClass (F : Type*) (α β : outParam Type*) [NonUnitalNonAssocSemiring α]
  [NonUnitalNonAssocSemiring β] [FunLike F α β]
  extends MulHomClass F α β, AddMonoidHomClass F α β : Prop


/-- Turn an element of a type `F` satisfying `NonUnitalRingHomClass F α β` into an actual
`NonUnitalRingHom`. This is declared as the default coercion from `F` to `α →ₙ+* β`. -/
@[coe]
def NonUnitalRingHomClass.toNonUnitalRingHom (f : F) : α →ₙ+* β :=
  { (f : α →ₙ* β), (f : α →+ β) with }


/-- Any type satisfying `NonUnitalRingHomClass` can be cast into `NonUnitalRingHom` via
`NonUnitalRingHomClass.toNonUnitalRingHom`. -/
instance : CoeTC F (α →ₙ+* β) :=
  ⟨NonUnitalRingHomClass.toNonUnitalRingHom⟩


instance : FunLike (α →ₙ+* β) α β where
  coe f := f.toFun
  coe_injective' f g h := by
    /-
      F : Type u_1
      α : Type u_2
      β : Type u_3
      γ : Type u_4
      inst✝¹ : NonUnitalNonAssocSemiring α
      inst✝ : NonUnitalNonAssocSemiring β
      f g : NonUnitalRingHom α β
      h : Eq ((fun f => f.toFun) f) ((fun f => f.toFun) g)
      ⊢ Eq f g
    -/
    cases f
    /-
      case mk
      F : Type u_1
      α : Type u_2
      β : Type u_3
      γ : Type u_4
      inst✝¹ : NonUnitalNonAssocSemiring α
      inst✝ : NonUnitalNonAssocSemiring β
      g : NonUnitalRingHom α β
      toMulHom✝ : MulHom α β
      map_zero'✝ : Eq (toMulHom✝.toFun 0) 0
      map_add'✝ : ∀ (x y : α), Eq (toMulHom✝.toFun (HAdd.hAdd x y)) (HAdd.hAdd (toMu …
      h : Eq ((fun f => f.toFun) { toMulHom := toMulHom✝, map_zero' := map_zero'✝, m …
      ⊢ Eq { toMulHom := toMulHom✝, map_zero' := map_zero'✝, map_add' := map_add'✝ } g
    -/
    cases g
    /-
      case mk.mk
      F : Type u_1
      α : Type u_2
      β : Type u_3
      γ : Type u_4
      inst✝¹ : NonUnitalNonAssocSemiring α
      inst✝ : NonUnitalNonAssocSemiring β
      toMulHom✝¹ : MulHom α β
      map_zero'✝¹ : Eq (toMulHom✝¹.toFun 0) 0
      map_add'✝¹ : ∀ (x y : α), Eq (toMulHom✝¹.toFun (HAdd.hAdd x y)) (HAdd.hAdd (to …
      toMulHom✝ : MulHom α β
      map_zero'✝ : Eq (toMulHom✝.toFun 0) 0
      map_add'✝ : ∀ (x y : α), Eq (toMulHom✝.toFun (HAdd.hAdd x y)) (HAdd.hAdd (toMu …
      h : Eq ((fun f => f.toFun) { toMulHom := toMulHom✝¹, map_zero' := map_zero'✝¹, …
      ⊢ Eq { toMulHom := toMulHom✝¹, map_zero' := map_zero'✝¹, map_add' := map_add'✝ …
    -/
    congr
    /-
      case mk.mk.e_toMulHom
      F : Type u_1
      α : Type u_2
      β : Type u_3
      γ : Type u_4
      inst✝¹ : NonUnitalNonAssocSemiring α
      inst✝ : NonUnitalNonAssocSemiring β
      toMulHom✝¹ : MulHom α β
      map_zero'✝¹ : Eq (toMulHom✝¹.toFun 0) 0
      map_add'✝¹ : ∀ (x y : α), Eq (toMulHom✝¹.toFun (HAdd.hAdd x y)) (HAdd.hAdd (to …
      toMulHom✝ : MulHom α β
      map_zero'✝ : Eq (toMulHom✝.toFun 0) 0
      map_add'✝ : ∀ (x y : α), Eq (toMulHom✝.toFun (HAdd.hAdd x y)) (HAdd.hAdd (toMu …
      h : Eq ((fun f => f.toFun) { toMulHom := toMulHom✝¹, map_zero' := map_zero'✝¹, …
      ⊢ Eq toMulHom✝¹ toMulHom✝
    -/
    apply DFunLike.coe_injective'
    /-
      case mk.mk.e_toMulHom.a
      F : Type u_1
      α : Type u_2
      β : Type u_3
      γ : Type u_4
      inst✝¹ : NonUnitalNonAssocSemiring α
      inst✝ : NonUnitalNonAssocSemiring β
      toMulHom✝¹ : MulHom α β
      map_zero'✝¹ : Eq (toMulHom✝¹.toFun 0) 0
      map_add'✝¹ : ∀ (x y : α), Eq (toMulHom✝¹.toFun (HAdd.hAdd x y)) (HAdd.hAdd (to …
      toMulHom✝ : MulHom α β
      map_zero'✝ : Eq (toMulHom✝.toFun 0) 0
      map_add'✝ : ∀ (x y : α), Eq (toMulHom✝.toFun (HAdd.hAdd x y)) (HAdd.hAdd (toMu …
      h : Eq ((fun f => f.toFun) { toMulHom := toMulHom✝¹, map_zero' := map_zero'✝¹, …
      ⊢ Eq ⇑toMulHom✝¹ ⇑toMulHom✝
    -/
    exact h
    /-
      🎉 no goals
    -/


instance : NonUnitalRingHomClass (α →ₙ+* β) α β where
  map_add := NonUnitalRingHom.map_add'
  map_zero := NonUnitalRingHom.map_zero'
  map_mul f := f.map_mul'

-- Porting note: removed due to new `coe` in Lean4


@[simp]
theorem coe_toMulHom (f : α →ₙ+* β) : ⇑f.toMulHom = f :=
  rfl


@[simp]
theorem coe_mulHom_mk (f : α → β) (h₁ h₂ h₃) :
    ((⟨⟨f, h₁⟩, h₂, h₃⟩ : α →ₙ+* β) : α →ₙ* β) = ⟨f, h₁⟩ :=
  rfl


theorem coe_toAddMonoidHom (f : α →ₙ+* β) : ⇑f.toAddMonoidHom = f := rfl


@[simp]
theorem coe_addMonoidHom_mk (f : α → β) (h₁ h₂ h₃) :
    ((⟨⟨f, h₁⟩, h₂, h₃⟩ : α →ₙ+* β) : α →+ β) = ⟨⟨f, h₂⟩, h₃⟩ :=
  rfl


/-- Copy of a `RingHom` with a new `toFun` equal to the old one. Useful to fix definitional
equalities. -/
protected def copy (f : α →ₙ+* β) (f' : α → β) (h : f' = f) : α →ₙ+* β :=
  { f.toMulHom.copy f' h, f.toAddMonoidHom.copy f' h with }


@[simp]
theorem coe_copy (f : α →ₙ+* β) (f' : α → β) (h : f' = f) : ⇑(f.copy f' h) = f' :=
  rfl


theorem copy_eq (f : α →ₙ+* β) (f' : α → β) (h : f' = f) : f.copy f' h = f :=
  DFunLike.ext' h


@[ext]
theorem ext ⦃f g : α →ₙ+* β⦄ : (∀ x, f x = g x) → f = g :=
  DFunLike.ext _ _


@[simp]
theorem mk_coe (f : α →ₙ+* β) (h₁ h₂ h₃) : NonUnitalRingHom.mk (MulHom.mk f h₁) h₂ h₃ = f :=
  ext fun _ => rfl


theorem coe_addMonoidHom_injective : Injective fun f : α →ₙ+* β => (f : α →+ β) :=
  Injective.of_comp (f := DFunLike.coe) DFunLike.coe_injective


theorem coe_mulHom_injective : Injective fun f : α →ₙ+* β => (f : α →ₙ* β) :=
  Injective.of_comp (f := DFunLike.coe) DFunLike.coe_injective


/-- The identity non-unital ring homomorphism from a non-unital semiring to itself. -/
protected def id (α : Type*) [NonUnitalNonAssocSemiring α] : α →ₙ+* α where
  toFun := id
  map_mul' _ _ := rfl
  map_zero' := rfl
  map_add' _ _ := rfl


instance : Zero (α →ₙ+* β) :=
  ⟨{ toFun := 0, map_mul' := fun _ _ => (mul_zero (0 : β)).symm, map_zero' := rfl,
      map_add' := fun _ _ => (add_zero (0 : β)).symm }⟩


instance : Inhabited (α →ₙ+* β) :=
  ⟨0⟩


@[simp]
theorem coe_zero : ⇑(0 : α →ₙ+* β) = 0 :=
  rfl


@[simp]
theorem zero_apply (x : α) : (0 : α →ₙ+* β) x = 0 :=
  rfl


@[simp]
theorem id_apply (x : α) : NonUnitalRingHom.id α x = x :=
  rfl


@[simp]
theorem coe_addMonoidHom_id : (NonUnitalRingHom.id α : α →+ α) = AddMonoidHom.id α :=
  rfl


@[simp]
theorem coe_mulHom_id : (NonUnitalRingHom.id α : α →ₙ* α) = MulHom.id α :=
  rfl


/-- Composition of non-unital ring homomorphisms is a non-unital ring homomorphism. -/
def comp (g : β →ₙ+* γ) (f : α →ₙ+* β) : α →ₙ+* γ :=
  { g.toMulHom.comp f.toMulHom, g.toAddMonoidHom.comp f.toAddMonoidHom with }


/-- Composition of non-unital ring homomorphisms is associative. -/
theorem comp_assoc {δ} {_ : NonUnitalNonAssocSemiring δ} (f : α →ₙ+* β) (g : β →ₙ+* γ)
    (h : γ →ₙ+* δ) : (h.comp g).comp f = h.comp (g.comp f) :=
  rfl


@[simp]
theorem coe_comp (g : β →ₙ+* γ) (f : α →ₙ+* β) : ⇑(g.comp f) = g ∘ f :=
  rfl


@[simp]
theorem comp_apply (g : β →ₙ+* γ) (f : α →ₙ+* β) (x : α) : g.comp f x = g (f x) :=
  rfl


@[simp]
theorem coe_comp_addMonoidHom (g : β →ₙ+* γ) (f : α →ₙ+* β) :
    AddMonoidHom.mk ⟨g ∘ f, (g.comp f).map_zero'⟩ (g.comp f).map_add' = (g : β →+ γ).comp f :=
  rfl


@[simp]
theorem coe_comp_mulHom (g : β →ₙ+* γ) (f : α →ₙ+* β) :
    MulHom.mk (g ∘ f) (g.comp f).map_mul' = (g : β →ₙ* γ).comp f :=
  rfl


@[simp]
theorem comp_zero (g : β →ₙ+* γ) : g.comp (0 : α →ₙ+* β) = 0 := by
  /-
    α : Type u_2
    β : Type u_3
    γ : Type u_4
    inst✝² : NonUnitalNonAssocSemiring α
    inst✝¹ : NonUnitalNonAssocSemiring β
    inst✝ : NonUnitalNonAssocSemiring γ
    g : NonUnitalRingHom β γ
    ⊢ Eq (g.comp 0) 0
  -/
  ext
  /-
    case a
    α : Type u_2
    β : Type u_3
    γ : Type u_4
    inst✝² : NonUnitalNonAssocSemiring α
    inst✝¹ : NonUnitalNonAssocSemiring β
    inst✝ : NonUnitalNonAssocSemiring γ
    g : NonUnitalRingHom β γ
    x✝ : α
    ⊢ Eq ((g.comp 0) x✝) (0 x✝)
  -/
  simp
  /-
    🎉 no goals
  -/


@[simp]
theorem zero_comp (f : α →ₙ+* β) : (0 : β →ₙ+* γ).comp f = 0 := by
  /-
    α : Type u_2
    β : Type u_3
    γ : Type u_4
    inst✝² : NonUnitalNonAssocSemiring α
    inst✝¹ : NonUnitalNonAssocSemiring β
    inst✝ : NonUnitalNonAssocSemiring γ
    f : NonUnitalRingHom α β
    ⊢ Eq (NonUnitalRingHom.comp 0 f) 0
  -/
  ext
  /-
    case a
    α : Type u_2
    β : Type u_3
    γ : Type u_4
    inst✝² : NonUnitalNonAssocSemiring α
    inst✝¹ : NonUnitalNonAssocSemiring β
    inst✝ : NonUnitalNonAssocSemiring γ
    f : NonUnitalRingHom α β
    x✝ : α
    ⊢ Eq ((NonUnitalRingHom.comp 0 f) x✝) (0 x✝)
  -/
  rfl
  /-
    🎉 no goals
  -/


@[simp]
theorem comp_id (f : α →ₙ+* β) : f.comp (NonUnitalRingHom.id α) = f :=
  ext fun _ => rfl


@[simp]
theorem id_comp (f : α →ₙ+* β) : (NonUnitalRingHom.id β).comp f = f :=
  ext fun _ => rfl


instance : MonoidWithZero (α →ₙ+* α) where
  one := NonUnitalRingHom.id α
  mul := comp
  mul_one := comp_id
  one_mul := id_comp
  mul_assoc _ _ _ := comp_assoc _ _ _
  zero := 0
  mul_zero := comp_zero
  zero_mul := zero_comp


theorem one_def : (1 : α →ₙ+* α) = NonUnitalRingHom.id α :=
  rfl


@[simp]
theorem coe_one : ⇑(1 : α →ₙ+* α) = id :=
  rfl


theorem mul_def (f g : α →ₙ+* α) : f * g = f.comp g :=
  rfl


@[simp]
theorem coe_mul (f g : α →ₙ+* α) : ⇑(f * g) = f ∘ g :=
  rfl


@[simp]
theorem cancel_right {g₁ g₂ : β →ₙ+* γ} {f : α →ₙ+* β} (hf : Surjective f) :
    g₁.comp f = g₂.comp f ↔ g₁ = g₂ :=
  ⟨fun h => ext <| hf.forall.2 (NonUnitalRingHom.ext_iff.1 h), fun h => h ▸ rfl⟩


@[simp]
theorem cancel_left {g : β →ₙ+* γ} {f₁ f₂ : α →ₙ+* β} (hg : Injective g) :
    g.comp f₁ = g.comp f₂ ↔ f₁ = f₂ :=
                                  /-
                                    α : Type u_2
                                    β : Type u_3
                                    γ : Type u_4
                                    inst✝² : NonUnitalNonAssocSemiring α
                                    inst✝¹ : NonUnitalNonAssocSemiring β
                                    inst✝ : NonUnitalNonAssocSemiring γ
                                    g : NonUnitalRingHom β γ
                                    f₁ f₂ : NonUnitalRingHom α β
                                    hg : Function.Injective ⇑g
                                    h : Eq (g.comp f₁) (g.comp f₂)
                                    x : α
                                    ⊢ Eq (g (f₁ x)) (g (f₂ x))
                                  -/
  ⟨fun h => ext fun x => hg <| by rw [← comp_apply, h, comp_apply], fun h => h ▸ rfl⟩
                                  /-
                                    🎉 no goals
                                  -/


/-- Bundled semiring homomorphisms; use this for bundled ring homomorphisms too.

This extends from both `MonoidHom` and `MonoidWithZeroHom` in order to put the fields in a
sensible order, even though `MonoidWithZeroHom` already extends `MonoidHom`. -/
structure RingHom (α : Type*) (β : Type*) [NonAssocSemiring α] [NonAssocSemiring β] extends
  α →* β, α →+ β, α →ₙ+* β, α →*₀ β


/-- `α →+* β` denotes the type of ring homomorphisms from `α` to `β`. -/
infixr:25 " →+* " => RingHom


/-- `RingHomClass F α β` states that `F` is a type of (semi)ring homomorphisms.
You should extend this class when you extend `RingHom`.

This extends from both `MonoidHomClass` and `MonoidWithZeroHomClass` in
order to put the fields in a sensible order, even though
`MonoidWithZeroHomClass` already extends `MonoidHomClass`. -/
class RingHomClass (F : Type*) (α β : outParam Type*)
    [NonAssocSemiring α] [NonAssocSemiring β] [FunLike F α β]
  extends MonoidHomClass F α β, AddMonoidHomClass F α β, MonoidWithZeroHomClass F α β : Prop


/-- Turn an element of a type `F` satisfying `RingHomClass F α β` into an actual
`RingHom`. This is declared as the default coercion from `F` to `α →+* β`. -/
@[coe]
def RingHomClass.toRingHom (f : F) : α →+* β :=
  { (f : α →* β), (f : α →+ β) with }


/-- Any type satisfying `RingHomClass` can be cast into `RingHom` via `RingHomClass.toRingHom`. -/
instance : CoeTC F (α →+* β) :=
  ⟨RingHomClass.toRingHom⟩


instance (priority := 100) RingHomClass.toNonUnitalRingHomClass : NonUnitalRingHomClass F α β :=
  { ‹RingHomClass F α β› with }


instance instFunLike : FunLike (α →+* β) α β where
  coe f := f.toFun
  coe_injective' f g h := by
    /-
      F : Type u_1
      α : Type u_2
      β : Type u_3
      γ : Type u_4
      x✝¹ : NonAssocSemiring α
      x✝ : NonAssocSemiring β
      f g : RingHom α β
      h : Eq ((fun f => (↑f.toMonoidHom).toFun) f) ((fun f => (↑f.toMonoidHom).toFun …
      ⊢ Eq f g
    -/
    cases f
    /-
      case mk
      F : Type u_1
      α : Type u_2
      β : Type u_3
      γ : Type u_4
      x✝¹ : NonAssocSemiring α
      x✝ : NonAssocSemiring β
      g : RingHom α β
      toMonoidHom✝ : MonoidHom α β
      map_zero'✝ : Eq ((↑toMonoidHom✝).toFun 0) 0
      map_add'✝ : ∀ (x y : α), Eq ((↑toMonoidHom✝).toFun (HAdd.hAdd x y)) (HAdd.hAdd …
      h : Eq ((fun f => (↑f.toMonoidHom).toFun) { toMonoidHom := toMonoidHom✝, map_z …
      ⊢ Eq { toMonoidHom := toMonoidHom✝, map_zero' := map_zero'✝, map_add' := map_a …
    -/
    cases g
    /-
      case mk.mk
      F : Type u_1
      α : Type u_2
      β : Type u_3
      γ : Type u_4
      x✝¹ : NonAssocSemiring α
      x✝ : NonAssocSemiring β
      toMonoidHom✝¹ : MonoidHom α β
      map_zero'✝¹ : Eq ((↑toMonoidHom✝¹).toFun 0) 0
      map_add'✝¹ : ∀ (x y : α), Eq ((↑toMonoidHom✝¹).toFun (HAdd.hAdd x y)) (HAdd.hA …
      toMonoidHom✝ : MonoidHom α β
      map_zero'✝ : Eq ((↑toMonoidHom✝).toFun 0) 0
      map_add'✝ : ∀ (x y : α), Eq ((↑toMonoidHom✝).toFun (HAdd.hAdd x y)) (HAdd.hAdd …
      h : Eq ((fun f => (↑f.toMonoidHom).toFun) { toMonoidHom := toMonoidHom✝¹, map_ …
      ⊢ Eq { toMonoidHom := toMonoidHom✝¹, map_zero' := map_zero'✝¹, map_add' := map …
    -/
    congr
    /-
      case mk.mk.e_toMonoidHom
      F : Type u_1
      α : Type u_2
      β : Type u_3
      γ : Type u_4
      x✝¹ : NonAssocSemiring α
      x✝ : NonAssocSemiring β
      toMonoidHom✝¹ : MonoidHom α β
      map_zero'✝¹ : Eq ((↑toMonoidHom✝¹).toFun 0) 0
      map_add'✝¹ : ∀ (x y : α), Eq ((↑toMonoidHom✝¹).toFun (HAdd.hAdd x y)) (HAdd.hA …
      toMonoidHom✝ : MonoidHom α β
      map_zero'✝ : Eq ((↑toMonoidHom✝).toFun 0) 0
      map_add'✝ : ∀ (x y : α), Eq ((↑toMonoidHom✝).toFun (HAdd.hAdd x y)) (HAdd.hAdd …
      h : Eq ((fun f => (↑f.toMonoidHom).toFun) { toMonoidHom := toMonoidHom✝¹, map_ …
      ⊢ Eq toMonoidHom✝¹ toMonoidHom✝
    -/
    apply DFunLike.coe_injective'
    /-
      case mk.mk.e_toMonoidHom.a
      F : Type u_1
      α : Type u_2
      β : Type u_3
      γ : Type u_4
      x✝¹ : NonAssocSemiring α
      x✝ : NonAssocSemiring β
      toMonoidHom✝¹ : MonoidHom α β
      map_zero'✝¹ : Eq ((↑toMonoidHom✝¹).toFun 0) 0
      map_add'✝¹ : ∀ (x y : α), Eq ((↑toMonoidHom✝¹).toFun (HAdd.hAdd x y)) (HAdd.hA …
      toMonoidHom✝ : MonoidHom α β
      map_zero'✝ : Eq ((↑toMonoidHom✝).toFun 0) 0
      map_add'✝ : ∀ (x y : α), Eq ((↑toMonoidHom✝).toFun (HAdd.hAdd x y)) (HAdd.hAdd …
      h : Eq ((fun f => (↑f.toMonoidHom).toFun) { toMonoidHom := toMonoidHom✝¹, map_ …
      ⊢ Eq ⇑toMonoidHom✝¹ ⇑toMonoidHom✝
    -/
    exact h
    /-
      🎉 no goals
    -/


instance instRingHomClass : RingHomClass (α →+* β) α β where
  map_add := RingHom.map_add'
  map_zero := RingHom.map_zero'
  map_mul f := f.map_mul'
  map_one f := f.map_one'


theorem toFun_eq_coe (f : α →+* β) : f.toFun = f :=
  rfl


@[simp]
theorem coe_mk (f : α →* β) (h₁ h₂) : ((⟨f, h₁, h₂⟩ : α →+* β) : α → β) = f :=
  rfl


@[simp]
theorem coe_coe {F : Type*} [FunLike F α β] [RingHomClass F α β] (f : F) :
    ((f : α →+* β) : α → β) = f :=
  rfl


instance coeToMonoidHom : Coe (α →+* β) (α →* β) :=
  ⟨RingHom.toMonoidHom⟩

-- Porting note: `dsimp only` can prove this


@[simp]
theorem toMonoidHom_eq_coe (f : α →+* β) : f.toMonoidHom = f :=
  rfl

-- Porting note: this can't be a simp lemma anymore
-- @[simp]

theorem toMonoidWithZeroHom_eq_coe (f : α →+* β) : (f.toMonoidWithZeroHom : α → β) = f := by
  /-
    α : Type u_2
    β : Type u_3
    x✝¹ : NonAssocSemiring α
    x✝ : NonAssocSemiring β
    f : RingHom α β
    ⊢ Eq ⇑f.toMonoidWithZeroHom ⇑f
  -/
  rfl
  /-
    🎉 no goals
  -/


@[simp]
theorem coe_monoidHom_mk (f : α →* β) (h₁ h₂) : ((⟨f, h₁, h₂⟩ : α →+* β) : α →* β) = f :=
  rfl

-- Porting note: `dsimp only` can prove this


@[simp]
theorem toAddMonoidHom_eq_coe (f : α →+* β) : f.toAddMonoidHom = f :=
  rfl


@[simp]
theorem coe_addMonoidHom_mk (f : α → β) (h₁ h₂ h₃ h₄) :
    ((⟨⟨⟨f, h₁⟩, h₂⟩, h₃, h₄⟩ : α →+* β) : α →+ β) = ⟨⟨f, h₃⟩, h₄⟩ :=
  rfl


/-- Copy of a `RingHom` with a new `toFun` equal to the old one. Useful to fix definitional
equalities. -/
def copy (f : α →+* β) (f' : α → β) (h : f' = f) : α →+* β :=
  { f.toMonoidWithZeroHom.copy f' h, f.toAddMonoidHom.copy f' h with }


@[simp]
theorem coe_copy (f : α →+* β) (f' : α → β) (h : f' = f) : ⇑(f.copy f' h) = f' :=
  rfl


theorem copy_eq (f : α →+* β) (f' : α → β) (h : f' = f) : f.copy f' h = f :=
  DFunLike.ext' h


protected theorem congr_fun {f g : α →+* β} (h : f = g) (x : α) : f x = g x :=
  DFunLike.congr_fun h x


protected theorem congr_arg (f : α →+* β) {x y : α} (h : x = y) : f x = f y :=
  DFunLike.congr_arg f h


theorem coe_inj ⦃f g : α →+* β⦄ (h : (f : α → β) = g) : f = g :=
  DFunLike.coe_injective h


@[ext]
theorem ext ⦃f g : α →+* β⦄ : (∀ x, f x = g x) → f = g :=
  DFunLike.ext _ _


@[simp]
theorem mk_coe (f : α →+* β) (h₁ h₂ h₃ h₄) : RingHom.mk ⟨⟨f, h₁⟩, h₂⟩ h₃ h₄ = f :=
  ext fun _ => rfl


theorem coe_addMonoidHom_injective : Injective (fun f : α →+* β => (f : α →+ β)) := fun _ _ h =>
  ext <| DFunLike.congr_fun (F := α →+ β) h


theorem coe_monoidHom_injective : Injective (fun f : α →+* β => (f : α →* β)) :=
  Injective.of_comp (f := DFunLike.coe) DFunLike.coe_injective


/-- Ring homomorphisms map zero to zero. -/
protected theorem map_zero (f : α →+* β) : f 0 = 0 :=
  map_zero f


/-- Ring homomorphisms map one to one. -/
protected theorem map_one (f : α →+* β) : f 1 = 1 :=
  map_one f


/-- Ring homomorphisms preserve addition. -/
protected theorem map_add (f : α →+* β) : ∀ a b, f (a + b) = f a + f b :=
  map_add f


/-- Ring homomorphisms preserve multiplication. -/
protected theorem map_mul (f : α →+* β) : ∀ a b, f (a * b) = f a * f b :=
  map_mul f


@[simp]
theorem map_ite_zero_one {F : Type*} [FunLike F α β] [RingHomClass F α β] (f : F)
    (p : Prop) [Decidable p] :
    f (ite p 0 1) = ite p 0 1 := by
  /-
    α : Type u_2
    β : Type u_3
    x✝¹ : NonAssocSemiring α
    x✝ : NonAssocSemiring β
    F : Type u_5
    inst✝² : FunLike F α β
    inst✝¹ : RingHomClass F α β
    f : F
    p : Prop
    inst✝ : Decidable p
    ⊢ Eq (f (ite p 0 1)) (ite p 0 1)
  -/
                       /-
                         🎉 no goals
                       -/
  split_ifs with h <;> simp [h]
                       /-
                         🎉 no goals
                       -/


@[simp]
theorem map_ite_one_zero {F : Type*} [FunLike F α β] [RingHomClass F α β] (f : F)
    (p : Prop) [Decidable p] :
    f (ite p 1 0) = ite p 1 0 := by
  /-
    α : Type u_2
    β : Type u_3
    x✝¹ : NonAssocSemiring α
    x✝ : NonAssocSemiring β
    F : Type u_5
    inst✝² : FunLike F α β
    inst✝¹ : RingHomClass F α β
    f : F
    p : Prop
    inst✝ : Decidable p
    ⊢ Eq (f (ite p 1 0)) (ite p 1 0)
  -/
                       /-
                         🎉 no goals
                       -/
  split_ifs with h <;> simp [h]
                       /-
                         🎉 no goals
                       -/


/-- `f : α →+* β` has a trivial codomain iff `f 1 = 0`. -/
                                                                           /-
                                                                             α : Type u_2
                                                                             β : Type u_3
                                                                             x✝¹ : NonAssocSemiring α
                                                                             x✝ : NonAssocSemiring β
                                                                             f : RingHom α β
                                                                             ⊢ Iff (Eq 0 1) (Eq (f 1) 0)
                                                                           -/
theorem codomain_trivial_iff_map_one_eq_zero : (0 : β) = 1 ↔ f 1 = 0 := by rw [map_one, eq_comm]
                                                                           /-
                                                                             🎉 no goals
                                                                           -/


/-- `f : α →+* β` has a trivial codomain iff it has a trivial range. -/
theorem codomain_trivial_iff_range_trivial : (0 : β) = 1 ↔ ∀ x, f x = 0 :=
  f.codomain_trivial_iff_map_one_eq_zero.trans
                   /-
                     α : Type u_2
                     β : Type u_3
                     x✝¹ : NonAssocSemiring α
                     x✝ : NonAssocSemiring β
                     f : RingHom α β
                     h : Eq (f 1) 0
                     x : α
                     ⊢ Eq (f x) 0
                   -/
    ⟨fun h x => by rw [← mul_one x, map_mul, h, mul_zero], fun h => h 1⟩
                   /-
                     🎉 no goals
                   -/


/-- `f : α →+* β` doesn't map `1` to `0` if `β` is nontrivial -/
theorem map_one_ne_zero [Nontrivial β] : f 1 ≠ 0 :=
  mt f.codomain_trivial_iff_map_one_eq_zero.mpr zero_ne_one


include f in
/-- If there is a homomorphism `f : α →+* β` and `β` is nontrivial, then `α` is nontrivial. -/
theorem domain_nontrivial [Nontrivial β] : Nontrivial α :=
                                       /-
                                         α : Type u_2
                                         β : Type u_3
                                         x✝¹ : NonAssocSemiring α
                                         x✝ : NonAssocSemiring β
                                         f : RingHom α β
                                         inst✝ : Nontrivial β
                                         h : Eq 1 0
                                         ⊢ Eq (f 1) 0
                                       -/
  ⟨⟨1, 0, mt (fun h => show f 1 = 0 by rw [h, map_zero]) f.map_one_ne_zero⟩⟩
                                       /-
                                         🎉 no goals
                                       -/


theorem codomain_trivial (f : α →+* β) [h : Subsingleton α] : Subsingleton β :=
  (subsingleton_or_nontrivial β).resolve_right fun _ =>
    not_nontrivial_iff_subsingleton.mpr h f.domain_nontrivial


/-- Ring homomorphisms preserve additive inverse. -/
protected theorem map_neg [NonAssocRing α] [NonAssocRing β] (f : α →+* β) (x : α) : f (-x) = -f x :=
  map_neg f x


/-- Ring homomorphisms preserve subtraction. -/
protected theorem map_sub [NonAssocRing α] [NonAssocRing β] (f : α →+* β) (x y : α) :
    f (x - y) = f x - f y :=
  map_sub f x y


/-- Makes a ring homomorphism from a monoid homomorphism of rings which preserves addition. -/
def mk' [NonAssocSemiring α] [NonAssocRing β] (f : α →* β)
    (map_add : ∀ a b, f (a + b) = f a + f b) : α →+* β :=
  { AddMonoidHom.mk' f map_add, f with }


/-- The identity ring homomorphism from a semiring to itself. -/
def id (α : Type*) [NonAssocSemiring α] : α →+* α where
  toFun := _root_.id
  map_zero' := rfl
  map_one' := rfl
  map_add' _ _ := rfl
  map_mul' _ _ := rfl


instance : Inhabited (α →+* α) :=
  ⟨id α⟩


@[simp]
theorem coe_id : ⇑(RingHom.id α) = _root_.id := rfl


@[simp]
theorem id_apply (x : α) : RingHom.id α x = x :=
  rfl


@[simp]
theorem coe_addMonoidHom_id : (id α : α →+ α) = AddMonoidHom.id α :=
  rfl


@[simp]
theorem coe_monoidHom_id : (id α : α →* α) = MonoidHom.id α :=
  rfl


/-- Composition of ring homomorphisms is a ring homomorphism. -/
def comp (g : β →+* γ) (f : α →+* β) : α →+* γ :=
                                                                                       /-
                                                                                         F : Type u_1
                                                                                         α : Type u_2
                                                                                         β : Type u_3
                                                                                         γ : Type u_4
                                                                                         x✝² : NonAssocSemiring α
                                                                                         x✝¹ : NonAssocSemiring β
                                                                                         x✝ : NonAssocSemiring γ
                                                                                         g : RingHom β γ
                                                                                         f : RingHom α β
                                                                                         ⊢ Eq (Function.comp (⇑g) (⇑f) 1) 1
                                                                                       -/
  { g.toNonUnitalRingHom.comp f.toNonUnitalRingHom with toFun := g ∘ f, map_one' := by simp }
                                                                                       /-
                                                                                         🎉 no goals
                                                                                       -/


/-- Composition of semiring homomorphisms is associative. -/
theorem comp_assoc {δ} {_ : NonAssocSemiring δ} (f : α →+* β) (g : β →+* γ) (h : γ →+* δ) :
    (h.comp g).comp f = h.comp (g.comp f) :=
  rfl


@[simp]
theorem coe_comp (hnp : β →+* γ) (hmn : α →+* β) : (hnp.comp hmn : α → γ) = hnp ∘ hmn :=
  rfl


theorem comp_apply (hnp : β →+* γ) (hmn : α →+* β) (x : α) :
    (hnp.comp hmn : α → γ) x = hnp (hmn x) :=
  rfl


@[simp]
theorem comp_id (f : α →+* β) : f.comp (id α) = f :=
  ext fun _ => rfl


@[simp]
theorem id_comp (f : α →+* β) : (id β).comp f = f :=
  ext fun _ => rfl


instance instOne : One (α →+* α) where one := id _

instance instMul : Mul (α →+* α) where mul := comp


lemma one_def : (1 : α →+* α) = id α := rfl


lemma mul_def (f g : α →+* α) : f * g = f.comp g := rfl


@[simp, norm_cast] lemma coe_one : ⇑(1 : α →+* α) = _root_.id := rfl


@[simp, norm_cast] lemma coe_mul (f g : α →+* α) : ⇑(f * g) = f ∘ g := rfl


instance instMonoid : Monoid (α →+* α) where
  mul_one := comp_id
  one_mul := id_comp
  mul_assoc _ _ _ := comp_assoc _ _ _
                                             /-
                                               F : Type u_1
                                               α : Type u_2
                                               β : Type u_3
                                               γ : Type u_4
                                               x✝² : NonAssocSemiring α
                                               x✝¹ : NonAssocSemiring β
                                               x✝ : NonAssocSemiring γ
                                               n : Nat
                                               f : RingHom α α
                                               ⊢ Eq (Nat.iterate (⇑f) n) ⇑(npowRec n f)
                                             -/
                                                             /-
                                                               🎉 no goals
                                                             -/
  npow n f := (npowRec n f).copy f^[n] <| by induction n <;> simp [npowRec, *]
                                                             /-
                                                               🎉 no goals
                                                             -/
  npow_succ _ _ := DFunLike.coe_injective <| Function.iterate_succ _ _


@[simp, norm_cast] lemma coe_pow (f : α →+* α) (n : ℕ) : ⇑(f ^ n) = f^[n] := rfl


@[simp]
theorem cancel_right {g₁ g₂ : β →+* γ} {f : α →+* β} (hf : Surjective f) :
    g₁.comp f = g₂.comp f ↔ g₁ = g₂ :=
  ⟨fun h => RingHom.ext <| hf.forall.2 (RingHom.ext_iff.1 h), fun h => h ▸ rfl⟩


@[simp]
theorem cancel_left {g : β →+* γ} {f₁ f₂ : α →+* β} (hg : Injective g) :
    g.comp f₁ = g.comp f₂ ↔ f₁ = f₂ :=
                                          /-
                                            α : Type u_2
                                            β : Type u_3
                                            γ : Type u_4
                                            x✝² : NonAssocSemiring α
                                            x✝¹ : NonAssocSemiring β
                                            x✝ : NonAssocSemiring γ
                                            g : RingHom β γ
                                            f₁ f₂ : RingHom α β
                                            hg : Function.Injective ⇑g
                                            h : Eq (g.comp f₁) (g.comp f₂)
                                            x : α
                                            ⊢ Eq (g (f₁ x)) (g (f₂ x))
                                          -/
  ⟨fun h => RingHom.ext fun x => hg <| by rw [← comp_apply, h, comp_apply], fun h => h ▸ rfl⟩
                                          /-
                                            🎉 no goals
                                          -/


protected lemma RingHom.map_pow (f : α →+* β) (a) : ∀ n : ℕ, f (a ^ n) = f a ^ n := map_pow f a


/-- Make a ring homomorphism from an additive group homomorphism from a commutative ring to an
integral domain that commutes with self multiplication, assumes that two is nonzero and `1` is sent
to `1`. -/
def mkRingHomOfMulSelfOfTwoNeZero (h : ∀ x, f (x * x) = f x * f x) (h_two : (2 : α) ≠ 0)
    (h_one : f 1 = 1) : β →+* α :=
  { f with
    map_one' := h_one,
    map_mul' := fun x y => by
      /-
        F : Type u_1
        α : Type u_2
        β : Type u_3
        γ : Type u_4
        inst✝² : CommRing α
        inst✝¹ : IsDomain α
        inst✝ : CommRing β
        f : AddMonoidHom β α
        h : ∀ (x : β), Eq (f (HMul.hMul x x)) (HMul.hMul (f x) (f x))
        h_two : Ne 2 0
        h_one : Eq (f 1) 1
        x y : β
        ⊢ Eq ({ toFun := (↑f).toFun, map_one' := h_one }.toFun (HMul.hMul x y)) (HMul. …
      -/
      have hxy := h (x + y)
      rw [mul_add, add_mul, add_mul, f.map_add, f.map_add, f.map_add, f.map_add, h x, h y, add_mul,
        mul_add, mul_add, ← sub_eq_zero, add_comm (f x * f x + f (y * x)), ← sub_sub, ← sub_sub,
        ← sub_sub, mul_comm y x, mul_comm (f y) (f x)] at hxy
      /-
        F : Type u_1
        α : Type u_2
        β : Type u_3
        γ : Type u_4
        inst✝² : CommRing α
        inst✝¹ : IsDomain α
        inst✝ : CommRing β
        f : AddMonoidHom β α
        h : ∀ (x : β), Eq (f (HMul.hMul x x)) (HMul.hMul (f x) (f x))
        h_two : Ne 2 0
        h_one : Eq (f 1) 1
        x y : β
        hxy : Eq (HSub.hSub (HSub.hSub (HSub.hSub (HSub.hSub (HAdd.hAdd (HAdd.hAdd (f  …
        ⊢ Eq ({ toFun := (↑f).toFun, map_one' := h_one }.toFun (HMul.hMul x y)) (HMul. …
      -/
      simp only [add_assoc, add_sub_assoc, add_sub_cancel] at hxy
      rw [sub_sub, ← two_mul, ← add_sub_assoc, ← two_mul, ← mul_sub, mul_eq_zero (M₀ := α),
        sub_eq_zero, or_iff_not_imp_left] at hxy
      /-
        F : Type u_1
        α : Type u_2
        β : Type u_3
        γ : Type u_4
        inst✝² : CommRing α
        inst✝¹ : IsDomain α
        inst✝ : CommRing β
        f : AddMonoidHom β α
        h : ∀ (x : β), Eq (f (HMul.hMul x x)) (HMul.hMul (f x) (f x))
        h_two : Ne 2 0
        h_one : Eq (f 1) 1
        x y : β
        hxy : Not (Eq 2 0) → Eq (f (HMul.hMul x y)) (HMul.hMul (f x) (f y))
        ⊢ Eq ({ toFun := (↑f).toFun, map_one' := h_one }.toFun (HMul.hMul x y)) (HMul. …
      -/
      exact hxy h_two }
      /-
        🎉 no goals
      -/


@[simp]
theorem coe_fn_mkRingHomOfMulSelfOfTwoNeZero (h h_two h_one) :
    (f.mkRingHomOfMulSelfOfTwoNeZero h h_two h_one : β → α) = f :=
  rfl


@[simp]
theorem coe_addMonoidHom_mkRingHomOfMulSelfOfTwoNeZero (h h_two h_one) :
    (f.mkRingHomOfMulSelfOfTwoNeZero h h_two h_one : β →+ α) = f := by
  /-
    α : Type u_2
    β : Type u_3
    inst✝² : CommRing α
    inst✝¹ : IsDomain α
    inst✝ : CommRing β
    f : AddMonoidHom β α
    h : ∀ (x : β), Eq (f (HMul.hMul x x)) (HMul.hMul (f x) (f x))
    h_two : Ne 2 0
    h_one : Eq (f 1) 1
    ⊢ Eq (↑(f.mkRingHomOfMulSelfOfTwoNeZero h h_two h_one)) f
  -/
  ext
  /-
    case h
    α : Type u_2
    β : Type u_3
    inst✝² : CommRing α
    inst✝¹ : IsDomain α
    inst✝ : CommRing β
    f : AddMonoidHom β α
    h : ∀ (x : β), Eq (f (HMul.hMul x x)) (HMul.hMul (f x) (f x))
    h_two : Ne 2 0
    h_one : Eq (f 1) 1
    x✝ : β
    ⊢ Eq (↑(f.mkRingHomOfMulSelfOfTwoNeZero h h_two h_one) x✝) (f x✝)
  -/
  rfl
  /-
    🎉 no goals
  -/


