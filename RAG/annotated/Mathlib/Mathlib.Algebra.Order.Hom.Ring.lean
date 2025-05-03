/-- `OrderRingHom α β` is the type of monotone semiring homomorphisms from `α` to `β`.

When possible, instead of parametrizing results over `(f : OrderRingHom α β)`,
you should parametrize over `(F : Type*) [OrderRingHomClass F α β] (f : F)`.

When you extend this structure, make sure to extend `OrderRingHomClass`. -/
structure OrderRingHom (α β : Type*) [NonAssocSemiring α] [Preorder α] [NonAssocSemiring β]
  [Preorder β] extends α →+* β where
  /-- The proposition that the function preserves the order. -/
  monotone' : Monotone toFun


@[inherit_doc]
infixl:25 " →+*o " => OrderRingHom

/- Porting note: Needed to reorder instance arguments below:
`[Mul α] [Add α] [LE α] [Mul β] [Add β] [LE β]`
to
`[Mul α] [Mul β] [Add α] [Add β] [LE α] [LE β]`
otherwise the [refl] attribute on `OrderRingIso.refl` complains.
TODO: change back when `refl` attribute is fixed, github issue https://github.com/leanprover-community/mathlib4/issues/2505 -/


/-- `OrderRingHom α β` is the type of order-preserving semiring isomorphisms between `α` and `β`.

When possible, instead of parametrizing results over `(f : OrderRingIso α β)`,
you should parametrize over `(F : Type*) [OrderRingIsoClass F α β] (f : F)`.

When you extend this structure, make sure to extend `OrderRingIsoClass`. -/
structure OrderRingIso (α β : Type*) [Mul α] [Mul β] [Add α] [Add β] [LE α] [LE β] extends
  α ≃+* β where
  /-- The proposition that the function preserves the order bijectively. -/
  map_le_map_iff' {a b : α} : toFun a ≤ toFun b ↔ a ≤ b


@[inherit_doc]
infixl:25 " ≃+*o " => OrderRingIso

-- See module docstring for details


/-- Turn an element of a type `F` satisfying `OrderHomClass F α β` and `RingHomClass F α β`
into an actual `OrderRingHom`.
This is declared as the default coercion from `F` to `α →+*o β`. -/
@[coe]
def OrderRingHomClass.toOrderRingHom [NonAssocSemiring α] [Preorder α] [NonAssocSemiring β]
    [Preorder β] [OrderHomClass F α β] [RingHomClass F α β] (f : F) : α →+*o β :=
{ (f : α →+* β) with monotone' := OrderHomClass.monotone f}


/-- Any type satisfying `OrderRingHomClass` can be cast into `OrderRingHom` via
  `OrderRingHomClass.toOrderRingHom`. -/
instance [NonAssocSemiring α] [Preorder α] [NonAssocSemiring β] [Preorder β]
    [OrderHomClass F α β] [RingHomClass F α β] : CoeTC F (α →+*o β) :=
  ⟨OrderRingHomClass.toOrderRingHom⟩


/-- Turn an element of a type `F` satisfying `OrderIsoClass F α β` and `RingEquivClass F α β`
into an actual `OrderRingIso`.
This is declared as the default coercion from `F` to `α ≃+*o β`. -/
@[coe]
def OrderRingIsoClass.toOrderRingIso [Mul α] [Add α] [LE α] [Mul β] [Add β] [LE β]
    [OrderIsoClass F α β] [RingEquivClass F α β] (f : F) : α ≃+*o β :=
{ (f : α ≃+* β) with map_le_map_iff' := map_le_map_iff f}


/-- Any type satisfying `OrderRingIsoClass` can be cast into `OrderRingIso` via
  `OrderRingIsoClass.toOrderRingIso`. -/
instance [Mul α] [Add α] [LE α] [Mul β] [Add β] [LE β] [OrderIsoClass F α β]
    [RingEquivClass F α β] : CoeTC F (α ≃+*o β) :=
  ⟨OrderRingIsoClass.toOrderRingIso⟩


/-- Reinterpret an ordered ring homomorphism as an ordered additive monoid homomorphism. -/
def toOrderAddMonoidHom (f : α →+*o β) : α →+o β :=
  { f with }


/-- Reinterpret an ordered ring homomorphism as an order homomorphism. -/
def toOrderMonoidWithZeroHom (f : α →+*o β) : α →*₀o β :=
  { f with }


instance : FunLike (α →+*o β) α β where
  coe f := f.toFun
  coe_injective' f g h := by
    /-
      F : Type u_1
      α : Type u_2
      β : Type u_3
      γ : Type u_4
      δ : Type u_5
      inst✝⁷ : NonAssocSemiring α
      inst✝⁶ : Preorder α
      inst✝⁵ : NonAssocSemiring β
      inst✝⁴ : Preorder β
      inst✝³ : NonAssocSemiring γ
      inst✝² : Preorder γ
      inst✝¹ : NonAssocSemiring δ
      inst✝ : Preorder δ
      f g : OrderRingHom α β
      h : Eq ((fun f => (↑↑f.toRingHom).toFun) f) ((fun f => (↑↑f.toRingHom).toFun) g)
      ⊢ Eq f g
    -/
    obtain ⟨⟨_, _⟩, _⟩ := f; obtain ⟨⟨_, _⟩, _⟩ := g; congr
    -- Porting note: needed to add the following line
    /-
      case mk.mk.mk.mk.e_toRingHom.e_toMonoidHom
      F : Type u_1
      α : Type u_2
      β : Type u_3
      γ : Type u_4
      δ : Type u_5
      inst✝⁷ : NonAssocSemiring α
      inst✝⁶ : Preorder α
      inst✝⁵ : NonAssocSemiring β
      inst✝⁴ : Preorder β
      inst✝³ : NonAssocSemiring γ
      inst✝² : Preorder γ
      inst✝¹ : NonAssocSemiring δ
      inst✝ : Preorder δ
      toMonoidHom✝¹ : MonoidHom α β
      map_zero'✝¹ : Eq ((↑toMonoidHom✝¹).toFun 0) 0
      map_add'✝¹ : ∀ (x y : α), Eq ((↑toMonoidHom✝¹).toFun (HAdd.hAdd x y)) (HAdd.hA …
      monotone'✝¹ : Monotone (↑↑{ toMonoidHom := toMonoidHom✝¹, map_zero' := map_zer …
      toMonoidHom✝ : MonoidHom α β
      map_zero'✝ : Eq ((↑toMonoidHom✝).toFun 0) 0
      map_add'✝ : ∀ (x y : α), Eq ((↑toMonoidHom✝).toFun (HAdd.hAdd x y)) (HAdd.hAdd …
      monotone'✝ : Monotone (↑↑{ toMonoidHom := toMonoidHom✝, map_zero' := map_zero' …
      h : Eq ((fun f => (↑↑f.toRingHom).toFun) { toMonoidHom := toMonoidHom✝¹, map_z …
      ⊢ Eq toMonoidHom✝¹ toMonoidHom✝
    -/
    exact DFunLike.coe_injective' h
    /-
      🎉 no goals
    -/


instance : OrderHomClass (α →+*o β) α β where
  map_rel f _ _ h := f.monotone' h


instance : RingHomClass (α →+*o β) α β where
  map_mul f := f.map_mul'
  map_one f := f.map_one'
  map_add f := f.map_add'
  map_zero f := f.map_zero'


theorem toFun_eq_coe (f : α →+*o β) : f.toFun = f :=
  rfl


@[ext]
theorem ext {f g : α →+*o β} (h : ∀ a, f a = g a) : f = g :=
  DFunLike.ext f g h


@[simp]
theorem toRingHom_eq_coe (f : α →+*o β) : f.toRingHom = f :=
  RingHom.ext fun _ => rfl


@[simp]
theorem toOrderAddMonoidHom_eq_coe (f : α →+*o β) : f.toOrderAddMonoidHom = f :=
  rfl


@[simp]
theorem toOrderMonoidWithZeroHom_eq_coe (f : α →+*o β) : f.toOrderMonoidWithZeroHom = f :=
  rfl


@[simp]
theorem coe_coe_ringHom (f : α →+*o β) : ⇑(f : α →+* β) = f :=
  rfl


@[simp]
theorem coe_coe_orderAddMonoidHom (f : α →+*o β) : ⇑(f : α →+o β) = f :=
  rfl


@[simp]
theorem coe_coe_orderMonoidWithZeroHom (f : α →+*o β) : ⇑(f : α →*₀o β) = f :=
  rfl


@[norm_cast]
theorem coe_ringHom_apply (f : α →+*o β) (a : α) : (f : α →+* β) a = f a :=
  rfl


@[norm_cast]
theorem coe_orderAddMonoidHom_apply (f : α →+*o β) (a : α) : (f : α →+o β) a = f a :=
  rfl


@[norm_cast]
theorem coe_orderMonoidWithZeroHom_apply (f : α →+*o β) (a : α) : (f : α →*₀o β) a = f a :=
  rfl


/-- Copy of an `OrderRingHom` with a new `toFun` equal to the old one. Useful to fix definitional
equalities. -/
protected def copy (f : α →+*o β) (f' : α → β) (h : f' = f) : α →+*o β :=
  { f.toRingHom.copy f' h, f.toOrderAddMonoidHom.copy f' h with }


@[simp]
theorem coe_copy (f : α →+*o β) (f' : α → β) (h : f' = f) : ⇑(f.copy f' h) = f' :=
  rfl


theorem copy_eq (f : α →+*o β) (f' : α → β) (h : f' = f) : f.copy f' h = f :=
  DFunLike.ext' h


/-- The identity as an ordered ring homomorphism. -/
protected def id : α →+*o α :=
  { RingHom.id _, OrderHom.id with }


instance : Inhabited (α →+*o α) :=
  ⟨OrderRingHom.id α⟩


@[simp]
theorem coe_id : ⇑(OrderRingHom.id α) = id :=
  rfl


@[simp]
theorem id_apply (a : α) : OrderRingHom.id α a = a :=
  rfl


@[simp]
theorem coe_ringHom_id : (OrderRingHom.id α : α →+* α) = RingHom.id α :=
  rfl


@[simp]
theorem coe_orderAddMonoidHom_id : (OrderRingHom.id α : α →+o α) = OrderAddMonoidHom.id α :=
  rfl


@[simp]
theorem coe_orderMonoidWithZeroHom_id :
    (OrderRingHom.id α : α →*₀o α) = OrderMonoidWithZeroHom.id α :=
  rfl


/-- Composition of two `OrderRingHom`s as an `OrderRingHom`. -/
protected def comp (f : β →+*o γ) (g : α →+*o β) : α →+*o γ :=
  { f.toRingHom.comp g.toRingHom, f.toOrderAddMonoidHom.comp g.toOrderAddMonoidHom with }


@[simp]
theorem coe_comp (f : β →+*o γ) (g : α →+*o β) : ⇑(f.comp g) = f ∘ g :=
  rfl


@[simp]
theorem comp_apply (f : β →+*o γ) (g : α →+*o β) (a : α) : f.comp g a = f (g a) :=
  rfl


theorem comp_assoc (f : γ →+*o δ) (g : β →+*o γ) (h : α →+*o β) :
    (f.comp g).comp h = f.comp (g.comp h) :=
  rfl


@[simp]
theorem comp_id (f : α →+*o β) : f.comp (OrderRingHom.id α) = f :=
  rfl


@[simp]
theorem id_comp (f : α →+*o β) : (OrderRingHom.id β).comp f = f :=
  rfl


@[simp]
theorem cancel_right {f₁ f₂ : β →+*o γ} {g : α →+*o β} (hg : Surjective g) :
    f₁.comp g = f₂.comp g ↔ f₁ = f₂ :=
                                                                    /-
                                                                      α : Type u_2
                                                                      β : Type u_3
                                                                      γ : Type u_4
                                                                      inst✝⁵ : NonAssocSemiring α
                                                                      inst✝⁴ : Preorder α
                                                                      inst✝³ : NonAssocSemiring β
                                                                      inst✝² : Preorder β
                                                                      inst✝¹ : NonAssocSemiring γ
                                                                      inst✝ : Preorder γ
                                                                      f₁ f₂ : OrderRingHom β γ
                                                                      g : OrderRingHom α β
                                                                      hg : Function.Surjective ⇑g
                                                                      h : Eq f₁ f₂
                                                                      ⊢ Eq (f₁.comp g) (f₂.comp g)
                                                                    -/
  ⟨fun h => ext <| hg.forall.2 <| DFunLike.ext_iff.1 h, fun h => by rw [h]⟩
                                                                    /-
                                                                      🎉 no goals
                                                                    -/


@[simp]
theorem cancel_left {f : β →+*o γ} {g₁ g₂ : α →+*o β} (hf : Injective f) :
    f.comp g₁ = f.comp g₂ ↔ g₁ = g₂ :=
                                  /-
                                    α : Type u_2
                                    β : Type u_3
                                    γ : Type u_4
                                    inst✝⁵ : NonAssocSemiring α
                                    inst✝⁴ : Preorder α
                                    inst✝³ : NonAssocSemiring β
                                    inst✝² : Preorder β
                                    inst✝¹ : NonAssocSemiring γ
                                    inst✝ : Preorder γ
                                    f : OrderRingHom β γ
                                    g₁ g₂ : OrderRingHom α β
                                    hf : Function.Injective ⇑f
                                    h : Eq (f.comp g₁) (f.comp g₂)
                                    a : α
                                    ⊢ Eq (f (g₁ a)) (f (g₂ a))
                                  -/
  ⟨fun h => ext fun a => hf <| by rw [← comp_apply, h, comp_apply], congr_arg _⟩
                                  /-
                                    🎉 no goals
                                  -/


instance [Preorder β] : Preorder (OrderRingHom α β) :=
  Preorder.lift ((⇑) : _ → α → β)


instance [PartialOrder β] : PartialOrder (OrderRingHom α β) :=
  PartialOrder.lift _ DFunLike.coe_injective


/-- Reinterpret an ordered ring isomorphism as an order isomorphism. -/
-- Porting note: Added @[coe] attribute
@[coe]
def toOrderIso (f : α ≃+*o β) : α ≃o β :=
  ⟨f.toRingEquiv.toEquiv, f.map_le_map_iff'⟩


instance : EquivLike (α ≃+*o β) α β where
  coe f := f.toFun
  inv f := f.invFun
  coe_injective' f g h₁ h₂ := by
    /-
      F : Type u_1
      α : Type u_2
      β : Type u_3
      γ : Type u_4
      δ : Type u_5
      inst✝⁸ : Mul α
      inst✝⁷ : Add α
      inst✝⁶ : LE α
      inst✝⁵ : Mul β
      inst✝⁴ : Add β
      inst✝³ : LE β
      inst✝² : Mul γ
      inst✝¹ : Add γ
      inst✝ : LE γ
      f g : OrderRingIso α β
      h₁ : Eq ((fun f => f.toFun) f) ((fun f => f.toFun) g)
      h₂ : Eq ((fun f => f.invFun) f) ((fun f => f.invFun) g)
      ⊢ Eq f g
    -/
    obtain ⟨⟨⟨_, _⟩, _⟩, _⟩ := f
    /-
      case mk.mk.mk
      F : Type u_1
      α : Type u_2
      β : Type u_3
      γ : Type u_4
      δ : Type u_5
      inst✝⁸ : Mul α
      inst✝⁷ : Add α
      inst✝⁶ : LE α
      inst✝⁵ : Mul β
      inst✝⁴ : Add β
      inst✝³ : LE β
      inst✝² : Mul γ
      inst✝¹ : Add γ
      inst✝ : LE γ
      g : OrderRingIso α β
      toFun✝ : α → β
      invFun✝ : β → α
      left_inv✝ : Function.LeftInverse invFun✝ toFun✝
      right_inv✝ : Function.RightInverse invFun✝ toFun✝
      map_mul'✝ : ∀ (x y : α), Eq ({ toFun := toFun✝, invFun := invFun✝, left_inv := …
      map_add'✝ : ∀ (x y : α), Eq ({ toFun := toFun✝, invFun := invFun✝, left_inv := …
      map_le_map_iff'✝ : ∀ {a b : α}, Iff (LE.le ({ toFun := toFun✝, invFun := invFu …
      h₁ : Eq ((fun f => f.toFun) { toFun := toFun✝, invFun := invFun✝, left_inv :=  …
      h₂ : Eq ((fun f => f.invFun) { toFun := toFun✝, invFun := invFun✝, left_inv := …
      ⊢ Eq { toFun := toFun✝, invFun := invFun✝, left_inv := left_inv✝, right_inv := …
    -/
    obtain ⟨⟨⟨_, _⟩, _⟩, _⟩ := g
    /-
      case mk.mk.mk.mk.mk.mk
      F : Type u_1
      α : Type u_2
      β : Type u_3
      γ : Type u_4
      δ : Type u_5
      inst✝⁸ : Mul α
      inst✝⁷ : Add α
      inst✝⁶ : LE α
      inst✝⁵ : Mul β
      inst✝⁴ : Add β
      inst✝³ : LE β
      inst✝² : Mul γ
      inst✝¹ : Add γ
      inst✝ : LE γ
      toFun✝¹ : α → β
      invFun✝¹ : β → α
      left_inv✝¹ : Function.LeftInverse invFun✝¹ toFun✝¹
      right_inv✝¹ : Function.RightInverse invFun✝¹ toFun✝¹
      map_mul'✝¹ : ∀ (x y : α), Eq ({ toFun := toFun✝¹, invFun := invFun✝¹, left_inv …
      map_add'✝¹ : ∀ (x y : α), Eq ({ toFun := toFun✝¹, invFun := invFun✝¹, left_inv …
      map_le_map_iff'✝¹ : ∀ {a b : α}, Iff (LE.le ({ toFun := toFun✝¹, invFun := inv …
      toFun✝ : α → β
      invFun✝ : β → α
      left_inv✝ : Function.LeftInverse invFun✝ toFun✝
      right_inv✝ : Function.RightInverse invFun✝ toFun✝
      map_mul'✝ : ∀ (x y : α), Eq ({ toFun := toFun✝, invFun := invFun✝, left_inv := …
      map_add'✝ : ∀ (x y : α), Eq ({ toFun := toFun✝, invFun := invFun✝, left_inv := …
      map_le_map_iff'✝ : ∀ {a b : α}, Iff (LE.le ({ toFun := toFun✝, invFun := invFu …
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


instance : OrderIsoClass (α ≃+*o β) α β where
  map_le_map_iff f _ _ := f.map_le_map_iff'


instance : RingEquivClass (α ≃+*o β) α β where
  map_mul f := f.map_mul'
  map_add f := f.map_add'


theorem toFun_eq_coe (f : α ≃+*o β) : f.toFun = f :=
  rfl


@[ext]
theorem ext {f g : α ≃+*o β} (h : ∀ a, f a = g a) : f = g :=
  DFunLike.ext f g h


@[simp]
theorem coe_mk (e : α ≃+* β) (h) : ⇑(⟨e, h⟩ : α ≃+*o β) = e :=
  rfl


@[simp]
theorem mk_coe (e : α ≃+*o β) (h) : (⟨e, h⟩ : α ≃+*o β) = e :=
  ext fun _ => rfl


@[simp]
theorem toRingEquiv_eq_coe (f : α ≃+*o β) : f.toRingEquiv = f :=
  RingEquiv.ext fun _ => rfl


@[simp]
theorem toOrderIso_eq_coe (f : α ≃+*o β) : f.toOrderIso = f :=
  OrderIso.ext rfl


@[simp, norm_cast]
theorem coe_toRingEquiv (f : α ≃+*o β) : ⇑(f : α ≃+* β) = f :=
  rfl

-- Porting note: needed to add DFunLike.coe on the lhs, bad Equiv coercion otherwise

@[simp, norm_cast]
theorem coe_toOrderIso (f : α ≃+*o β) : DFunLike.coe (f : α ≃o β) = f :=
  rfl


/-- The identity map as an ordered ring isomorphism. -/
@[refl]
protected def refl : α ≃+*o α :=
  ⟨RingEquiv.refl α, Iff.rfl⟩


instance : Inhabited (α ≃+*o α) :=
  ⟨OrderRingIso.refl α⟩


@[simp]
theorem refl_apply (x : α) : OrderRingIso.refl α x = x := by
  /-
    α : Type u_2
    inst✝² : Mul α
    inst✝¹ : Add α
    inst✝ : LE α
    x : α
    ⊢ Eq ((OrderRingIso.refl α) x) x
  -/
  rfl
  /-
    🎉 no goals
  -/


@[simp]
theorem coe_ringEquiv_refl : (OrderRingIso.refl α : α ≃+* α) = RingEquiv.refl α :=
  rfl


@[simp]
theorem coe_orderIso_refl : (OrderRingIso.refl α : α ≃o α) = OrderIso.refl α :=
  rfl


/-- The inverse of an ordered ring isomorphism as an ordered ring isomorphism. -/
@[symm]
protected def symm (e : α ≃+*o β) : β ≃+*o α :=
  ⟨e.toRingEquiv.symm, by
    /-
      F : Type u_1
      α : Type u_2
      β : Type u_3
      γ : Type u_4
      δ : Type u_5
      inst✝⁸ : Mul α
      inst✝⁷ : Add α
      inst✝⁶ : LE α
      inst✝⁵ : Mul β
      inst✝⁴ : Add β
      inst✝³ : LE β
      inst✝² : Mul γ
      inst✝¹ : Add γ
      inst✝ : LE γ
      e : OrderRingIso α β
      ⊢ ∀ {a b : β}, Iff (LE.le (e.symm.toFun a) (e.symm.toFun b)) (LE.le a b)
    -/
    intro a b
    /-
      F : Type u_1
      α : Type u_2
      β : Type u_3
      γ : Type u_4
      δ : Type u_5
      inst✝⁸ : Mul α
      inst✝⁷ : Add α
      inst✝⁶ : LE α
      inst✝⁵ : Mul β
      inst✝⁴ : Add β
      inst✝³ : LE β
      inst✝² : Mul γ
      inst✝¹ : Add γ
      inst✝ : LE γ
      e : OrderRingIso α β
      a b : β
      ⊢ Iff (LE.le (e.symm.toFun a) (e.symm.toFun b)) (LE.le a b)
    -/
    erw [← map_le_map_iff e, e.1.apply_symm_apply, e.1.apply_symm_apply]⟩
    /-
      🎉 no goals
    -/


/-- See Note [custom simps projection] -/
def Simps.symm_apply (e : α ≃+*o β) : β → α :=
  e.symm


@[simp]
theorem symm_symm (e : α ≃+*o β) : e.symm.symm = e := rfl


/-- Composition of `OrderRingIso`s as an `OrderRingIso`. -/
@[trans]
protected def trans (f : α ≃+*o β) (g : β ≃+*o γ) : α ≃+*o γ :=
  ⟨f.toRingEquiv.trans g.toRingEquiv, (map_le_map_iff g).trans (map_le_map_iff f)⟩

/- Porting note: Used to be generated by [simps] on `trans`, but the lhs of this simplifies under
simp, so problem with the simpNF linter. Removed [simps] attribute and added aux version below. -/

theorem trans_toRingEquiv (f : α ≃+*o β) (g : β ≃+*o γ) :
    (OrderRingIso.trans f g).toRingEquiv = RingEquiv.trans f.toRingEquiv g.toRingEquiv :=
  rfl


@[simp]
theorem trans_toRingEquiv_aux (f : α ≃+*o β) (g : β ≃+*o γ) :
    RingEquivClass.toRingEquiv (OrderRingIso.trans f g)
      = RingEquiv.trans f.toRingEquiv g.toRingEquiv :=
  rfl


@[simp]
theorem trans_apply (f : α ≃+*o β) (g : β ≃+*o γ) (a : α) : f.trans g a = g (f a) :=
  rfl


@[simp]
theorem self_trans_symm (e : α ≃+*o β) : e.trans e.symm = OrderRingIso.refl α :=
  ext e.left_inv


@[simp]
theorem symm_trans_self (e : α ≃+*o β) : e.symm.trans e = OrderRingIso.refl β :=
  ext e.right_inv


theorem symm_bijective : Bijective (OrderRingIso.symm : (α ≃+*o β) → β ≃+*o α) :=
  Function.bijective_iff_has_inverse.mpr ⟨_, symm_symm, symm_symm⟩


/-- Reinterpret an ordered ring isomorphism as an ordered ring homomorphism. -/
def toOrderRingHom (f : α ≃+*o β) : α →+*o β :=
  ⟨f.toRingEquiv.toRingHom, fun _ _ => (map_le_map_iff f).2⟩


@[simp]
theorem toOrderRingHom_eq_coe (f : α ≃+*o β) : f.toOrderRingHom = f :=
  rfl


@[simp, norm_cast]
theorem coe_toOrderRingHom (f : α ≃+*o β) : ⇑(f : α →+*o β) = f :=
  rfl


@[simp]
theorem coe_toOrderRingHom_refl : (OrderRingIso.refl α : α →+*o α) = OrderRingHom.id α :=
  rfl


theorem toOrderRingHom_injective : Injective (toOrderRingHom : α ≃+*o β → α →+*o β) :=
                                            /-
                                              α : Type u_2
                                              β : Type u_3
                                              inst✝³ : NonAssocSemiring α
                                              inst✝² : Preorder α
                                              inst✝¹ : NonAssocSemiring β
                                              inst✝ : Preorder β
                                              f g : OrderRingIso α β
                                              h : Eq f.toOrderRingHom g.toOrderRingHom
                                              ⊢ Eq ((fun f => ⇑f) f) ((fun f => ⇑f) g)
                                            -/
  fun f g h => DFunLike.coe_injective <| by convert DFunLike.ext'_iff.1 h using 0
                                            /-
                                              🎉 no goals
                                            -/


