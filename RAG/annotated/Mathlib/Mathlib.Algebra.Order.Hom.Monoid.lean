/-- `α →+o β` is the type of monotone functions `α → β` that preserve the `OrderedAddCommMonoid`
structure.

`OrderAddMonoidHom` is also used for ordered group homomorphisms.

When possible, instead of parametrizing results over `(f : α →+o β)`,
you should parametrize over
`(F : Type*) [FunLike F M N] [MonoidHomClass F M N] [OrderHomClass F M N] (f : F)`. -/
structure OrderAddMonoidHom (α β : Type*) [Preorder α] [Preorder β] [AddZeroClass α]
  [AddZeroClass β] extends α →+ β where
  /-- An `OrderAddMonoidHom` is a monotone function. -/
  monotone' : Monotone toFun


/-- Infix notation for `OrderAddMonoidHom`. -/
infixr:25 " →+o " => OrderAddMonoidHom


/-- `α ≃+o β` is the type of monotone isomorphisms `α ≃ β` that preserve the `OrderedAddCommMonoid`
structure.

`OrderAddMonoidIso` is also used for ordered group isomorphisms.

When possible, instead of parametrizing results over `(f : α ≃+o β)`,
you should parametrize over
`(F : Type*) [FunLike F M N] [AddEquivClass F M N] [OrderIsoClass F M N] (f : F)`. -/
structure OrderAddMonoidIso (α β : Type*) [Preorder α] [Preorder β] [Add α] [Add β]
  extends α ≃+ β where
  /-- An `OrderAddMonoidIso` respects `≤`. -/
  map_le_map_iff' {a b : α} : toFun a ≤ toFun b ↔ a ≤ b


/-- Infix notation for `OrderAddMonoidIso`. -/
infixr:25 " ≃+o " => OrderAddMonoidIso

-- Instances and lemmas are defined below through `@[to_additive]`.

/-- `α →*o β` is the type of functions `α → β` that preserve the `OrderedCommMonoid` structure.

`OrderMonoidHom` is also used for ordered group homomorphisms.

When possible, instead of parametrizing results over `(f : α →*o β)`,
you should parametrize over
`(F : Type*) [FunLike F M N] [MonoidHomClass F M N] [OrderHomClass F M N] (f : F)`. -/
@[to_additive]
structure OrderMonoidHom (α β : Type*) [Preorder α] [Preorder β] [MulOneClass α]
  [MulOneClass β] extends α →* β where
  /-- An `OrderMonoidHom` is a monotone function. -/
  monotone' : Monotone toFun


/-- Infix notation for `OrderMonoidHom`. -/
infixr:25 " →*o " => OrderMonoidHom


/-- Turn an element of a type `F` satisfying `OrderHomClass F α β` and `MonoidHomClass F α β`
into an actual `OrderMonoidHom`. This is declared as the default coercion from `F` to `α →*o β`. -/
@[to_additive (attr := coe)
  "Turn an element of a type `F` satisfying `OrderHomClass F α β` and `AddMonoidHomClass F α β`
  into an actual `OrderAddMonoidHom`.
  This is declared as the default coercion from `F` to `α →+o β`."]
def OrderMonoidHomClass.toOrderMonoidHom [OrderHomClass F α β] [MonoidHomClass F α β] (f : F) :
    α →*o β :=
  { (f : α →* β) with monotone' := OrderHomClass.monotone f }


/-- Any type satisfying `OrderMonoidHomClass` can be cast into `OrderMonoidHom` via
  `OrderMonoidHomClass.toOrderMonoidHom`. -/
@[to_additive "Any type satisfying `OrderAddMonoidHomClass` can be cast into `OrderAddMonoidHom` via
  `OrderAddMonoidHomClass.toOrderAddMonoidHom`"]
instance [OrderHomClass F α β] [MonoidHomClass F α β] : CoeTC F (α →*o β) :=
  ⟨OrderMonoidHomClass.toOrderMonoidHom⟩


/-- `α ≃*o β` is the type of isomorphisms `α ≃ β` that preserve the `OrderedCommMonoid` structure.

`OrderMonoidIso` is also used for ordered group isomorphisms.

When possible, instead of parametrizing results over `(f : α ≃*o β)`,
you should parametrize over
`(F : Type*) [FunLike F M N] [MulEquivClass F M N] [OrderIsoClass F M N] (f : F)`. -/
@[to_additive]
structure OrderMonoidIso (α β : Type*) [Preorder α] [Preorder β] [Mul α] [Mul β]
  extends α ≃* β where
  /-- An `OrderMonoidIso` respects `≤`. -/
  map_le_map_iff' {a b : α} : toFun a ≤ toFun b ↔ a ≤ b


/-- Infix notation for `OrderMonoidIso`. -/
infixr:25 " ≃*o " => OrderMonoidIso


/-- Turn an element of a type `F` satisfying `OrderIsoClass F α β` and `MulEquivClass F α β`
into an actual `OrderMonoidIso`. This is declared as the default coercion from `F` to `α ≃*o β`. -/
@[to_additive (attr := coe)
  "Turn an element of a type `F` satisfying `OrderIsoClass F α β` and `AddEquivClass F α β`
  into an actual `OrderAddMonoidIso`.
  This is declared as the default coercion from `F` to `α ≃+o β`."]
def OrderMonoidIsoClass.toOrderMonoidIso [EquivLike F α β] [OrderIsoClass F α β]
    [MulEquivClass F α β] (f : F) :
    α ≃*o β :=
  { (f : α ≃* β) with map_le_map_iff' := OrderIsoClass.map_le_map_iff f }


/-- Any type satisfying `OrderMonoidIsoClass` can be cast into `OrderMonoidIso` via
  `OrderMonoidIsoClass.toOrderMonoidIso`. -/
@[to_additive "Any type satisfying `OrderAddMonoidIsoClass` can be cast into `OrderAddMonoidIso` via
  `OrderAddMonoidIsoClass.toOrderAddMonoidIso`"]
instance [EquivLike F α β] [OrderIsoClass F α β] [MulEquivClass F α β] : CoeTC F (α ≃*o β) :=
  ⟨OrderMonoidIsoClass.toOrderMonoidIso⟩


/-- `OrderMonoidWithZeroHom α β` is the type of functions `α → β` that preserve
the `MonoidWithZero` structure.

`OrderMonoidWithZeroHom` is also used for group homomorphisms.

When possible, instead of parametrizing results over `(f : α →+ β)`,
you should parameterize over
`(F : Type*) [FunLike F M N] [MonoidWithZeroHomClass F M N] [OrderHomClass F M N] (f : F)`. -/
structure OrderMonoidWithZeroHom (α β : Type*) [Preorder α] [Preorder β] [MulZeroOneClass α]
  [MulZeroOneClass β] extends α →*₀ β where
  /-- An `OrderMonoidWithZeroHom` is a monotone function. -/
  monotone' : Monotone toFun


/-- Infix notation for `OrderMonoidWithZeroHom`. -/
infixr:25 " →*₀o " => OrderMonoidWithZeroHom


/-- Turn an element of a type `F`
satisfying `OrderHomClass F α β` and `MonoidWithZeroHomClass F α β`
into an actual `OrderMonoidWithZeroHom`.
This is declared as the default coercion from `F` to `α →+*₀o β`. -/
@[coe]
def OrderMonoidWithZeroHomClass.toOrderMonoidWithZeroHom [OrderHomClass F α β]
    [MonoidWithZeroHomClass F α β] (f : F) : α →*₀o β :=
{ (f : α →*₀ β) with monotone' := OrderHomClass.monotone f }


instance [OrderHomClass F α β] [MonoidWithZeroHomClass F α β] : CoeTC F (α →*₀o β) :=
  ⟨OrderMonoidWithZeroHomClass.toOrderMonoidWithZeroHom⟩


/-- See also `NonnegHomClass.apply_nonneg`. -/
theorem map_nonneg (ha : 0 ≤ a) : 0 ≤ f a := by
  /-
    F : Type u_1
    α : Type u_2
    β : Type u_3
    inst✝⁶ : FunLike F α β
    inst✝⁵ : Preorder α
    inst✝⁴ : Zero α
    inst✝³ : Preorder β
    inst✝² : Zero β
    inst✝¹ : OrderHomClass F α β
    inst✝ : ZeroHomClass F α β
    f : F
    a : α
    ha : LE.le 0 a
    ⊢ LE.le 0 (f a)
  -/
  rw [← map_zero f]
  /-
    F : Type u_1
    α : Type u_2
    β : Type u_3
    inst✝⁶ : FunLike F α β
    inst✝⁵ : Preorder α
    inst✝⁴ : Zero α
    inst✝³ : Preorder β
    inst✝² : Zero β
    inst✝¹ : OrderHomClass F α β
    inst✝ : ZeroHomClass F α β
    f : F
    a : α
    ha : LE.le 0 a
    ⊢ LE.le (f 0) (f a)
  -/
  exact OrderHomClass.mono _ ha
  /-
    🎉 no goals
  -/


theorem map_nonpos (ha : a ≤ 0) : f a ≤ 0 := by
  /-
    F : Type u_1
    α : Type u_2
    β : Type u_3
    inst✝⁶ : FunLike F α β
    inst✝⁵ : Preorder α
    inst✝⁴ : Zero α
    inst✝³ : Preorder β
    inst✝² : Zero β
    inst✝¹ : OrderHomClass F α β
    inst✝ : ZeroHomClass F α β
    f : F
    a : α
    ha : LE.le a 0
    ⊢ LE.le (f a) 0
  -/
  rw [← map_zero f]
  /-
    F : Type u_1
    α : Type u_2
    β : Type u_3
    inst✝⁶ : FunLike F α β
    inst✝⁵ : Preorder α
    inst✝⁴ : Zero α
    inst✝³ : Preorder β
    inst✝² : Zero β
    inst✝¹ : OrderHomClass F α β
    inst✝ : ZeroHomClass F α β
    f : F
    a : α
    ha : LE.le a 0
    ⊢ LE.le (f a) (f 0)
  -/
  exact OrderHomClass.mono _ ha
  /-
    🎉 no goals
  -/


theorem monotone_iff_map_nonneg [iamhc : AddMonoidHomClass F α β] :
    Monotone (f : α → β) ↔ ∀ a, 0 ≤ a → 0 ≤ f a :=
  ⟨fun h a => by
    /-
      F : Type u_1
      α : Type u_2
      β : Type u_3
      inst✝¹ : OrderedAddCommGroup α
      inst✝ : OrderedAddCommMonoid β
      i : FunLike F α β
      f : F
      iamhc : AddMonoidHomClass F α β
      h : Monotone ⇑f
      a : α
      ⊢ LE.le 0 a → LE.le 0 (f a)
    -/
    rw [← map_zero f]
    /-
      F : Type u_1
      α : Type u_2
      β : Type u_3
      inst✝¹ : OrderedAddCommGroup α
      inst✝ : OrderedAddCommMonoid β
      i : FunLike F α β
      f : F
      iamhc : AddMonoidHomClass F α β
      h : Monotone ⇑f
      a : α
      ⊢ LE.le 0 a → LE.le (f 0) (f a)
    -/
    apply h, fun h a b hl => by
    /-
      🎉 no goals
    -/
    /-
      F : Type u_1
      α : Type u_2
      β : Type u_3
      inst✝¹ : OrderedAddCommGroup α
      inst✝ : OrderedAddCommMonoid β
      i : FunLike F α β
      f : F
      iamhc : AddMonoidHomClass F α β
      h : ∀ (a : α), LE.le 0 a → LE.le 0 (f a)
      a b : α
      hl : LE.le a b
      ⊢ LE.le (f a) (f b)
    -/
    rw [← sub_add_cancel b a, map_add f]
    /-
      F : Type u_1
      α : Type u_2
      β : Type u_3
      inst✝¹ : OrderedAddCommGroup α
      inst✝ : OrderedAddCommMonoid β
      i : FunLike F α β
      f : F
      iamhc : AddMonoidHomClass F α β
      h : ∀ (a : α), LE.le 0 a → LE.le 0 (f a)
      a b : α
      hl : LE.le a b
      ⊢ LE.le (f a) (HAdd.hAdd (f (HSub.hSub b a)) (f a))
    -/
    exact le_add_of_nonneg_left (h _ <| sub_nonneg.2 hl)⟩
    /-
      🎉 no goals
    -/


theorem antitone_iff_map_nonpos : Antitone (f : α → β) ↔ ∀ a, 0 ≤ a → f a ≤ 0 :=
  monotone_toDual_comp_iff.symm.trans <| monotone_iff_map_nonneg (β := βᵒᵈ) (iamhc := iamhc) _


theorem monotone_iff_map_nonpos : Monotone (f : α → β) ↔ ∀ a ≤ 0, f a ≤ 0 :=
  antitone_comp_ofDual_iff.symm.trans <| antitone_iff_map_nonpos (α := αᵒᵈ) (iamhc := iamhc) _


theorem antitone_iff_map_nonneg : Antitone (f : α → β) ↔ ∀ a ≤ 0, 0 ≤ f a :=
  monotone_comp_ofDual_iff.symm.trans <| monotone_iff_map_nonneg (α := αᵒᵈ) (iamhc := iamhc) _


theorem strictMono_iff_map_pos :
    StrictMono (f : α → β) ↔ ∀ a, 0 < a → 0 < f a := by
  /-
    F : Type u_1
    α : Type u_2
    β : Type u_3
    inst✝² : OrderedAddCommGroup α
    inst✝¹ : OrderedAddCommMonoid β
    i : FunLike F α β
    f : F
    iamhc : AddMonoidHomClass F α β
    inst✝ : AddLeftStrictMono β
    ⊢ Iff (StrictMono ⇑f) (∀ (a : α), LT.lt 0 a → LT.lt 0 (f a))
  -/
  refine ⟨fun h a => ?_, fun h a b hl => ?_⟩
    /-
      case refine_1
      F : Type u_1
      α : Type u_2
      β : Type u_3
      inst✝² : OrderedAddCommGroup α
      inst✝¹ : OrderedAddCommMonoid β
      i : FunLike F α β
      f : F
      iamhc : AddMonoidHomClass F α β
      inst✝ : AddLeftStrictMono β
      h : StrictMono ⇑f
      a : α
      ⊢ LT.lt 0 a → LT.lt 0 (f a)
    -/
  · rw [← map_zero f]
    /-
      case refine_1
      F : Type u_1
      α : Type u_2
      β : Type u_3
      inst✝² : OrderedAddCommGroup α
      inst✝¹ : OrderedAddCommMonoid β
      i : FunLike F α β
      f : F
      iamhc : AddMonoidHomClass F α β
      inst✝ : AddLeftStrictMono β
      h : StrictMono ⇑f
      a : α
      ⊢ LT.lt 0 a → LT.lt (f 0) (f a)
    -/
    apply h
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      F : Type u_1
      α : Type u_2
      β : Type u_3
      inst✝² : OrderedAddCommGroup α
      inst✝¹ : OrderedAddCommMonoid β
      i : FunLike F α β
      f : F
      iamhc : AddMonoidHomClass F α β
      inst✝ : AddLeftStrictMono β
      h : ∀ (a : α), LT.lt 0 a → LT.lt 0 (f a)
      a b : α
      hl : LT.lt a b
      ⊢ LT.lt (f a) (f b)
    -/
  · rw [← sub_add_cancel b a, map_add f]
    /-
      case refine_2
      F : Type u_1
      α : Type u_2
      β : Type u_3
      inst✝² : OrderedAddCommGroup α
      inst✝¹ : OrderedAddCommMonoid β
      i : FunLike F α β
      f : F
      iamhc : AddMonoidHomClass F α β
      inst✝ : AddLeftStrictMono β
      h : ∀ (a : α), LT.lt 0 a → LT.lt 0 (f a)
      a b : α
      hl : LT.lt a b
      ⊢ LT.lt (f a) (HAdd.hAdd (f (HSub.hSub b a)) (f a))
    -/
    exact lt_add_of_pos_left _ (h _ <| sub_pos.2 hl)
    /-
      🎉 no goals
    -/


theorem strictAnti_iff_map_neg : StrictAnti (f : α → β) ↔ ∀ a, 0 < a → f a < 0 :=
  strictMono_toDual_comp_iff.symm.trans <| strictMono_iff_map_pos (β := βᵒᵈ) (iamhc := iamhc) _


theorem strictMono_iff_map_neg : StrictMono (f : α → β) ↔ ∀ a < 0, f a < 0 :=
  strictAnti_comp_ofDual_iff.symm.trans <| strictAnti_iff_map_neg (α := αᵒᵈ) (iamhc := iamhc) _


theorem strictAnti_iff_map_pos : StrictAnti (f : α → β) ↔ ∀ a < 0, 0 < f a :=
  strictMono_comp_ofDual_iff.symm.trans <| strictMono_iff_map_pos (α := αᵒᵈ) (iamhc := iamhc) _


@[to_additive]
instance : FunLike (α →*o β) α β where
  coe f := f.toFun
  coe_injective' f g h := by
    /-
      F : Type u_1
      α : Type u_2
      β : Type u_3
      γ : Type u_4
      δ : Type u_5
      inst✝⁷ : Preorder α
      inst✝⁶ : Preorder β
      inst✝⁵ : Preorder γ
      inst✝⁴ : Preorder δ
      inst✝³ : MulOneClass α
      inst✝² : MulOneClass β
      inst✝¹ : MulOneClass γ
      inst✝ : MulOneClass δ
      f✝ g✝ f g : OrderMonoidHom α β
      h : Eq ((fun f => (↑f.toMonoidHom).toFun) f) ((fun f => (↑f.toMonoidHom).toFun …
      ⊢ Eq f g
    -/
    obtain ⟨⟨⟨_, _⟩⟩, _⟩ := f
    /-
      case mk.mk.mk
      F : Type u_1
      α : Type u_2
      β : Type u_3
      γ : Type u_4
      δ : Type u_5
      inst✝⁷ : Preorder α
      inst✝⁶ : Preorder β
      inst✝⁵ : Preorder γ
      inst✝⁴ : Preorder δ
      inst✝³ : MulOneClass α
      inst✝² : MulOneClass β
      inst✝¹ : MulOneClass γ
      inst✝ : MulOneClass δ
      f g✝ g : OrderMonoidHom α β
      toFun✝ : α → β
      map_one'✝ : Eq (toFun✝ 1) 1
      map_mul'✝ : ∀ (x y : α), Eq ({ toFun := toFun✝, map_one' := map_one'✝ }.toFun  …
      monotone'✝ : Monotone (↑{ toFun := toFun✝, map_one' := map_one'✝, map_mul' :=  …
      h : Eq ((fun f => (↑f.toMonoidHom).toFun) { toFun := toFun✝, map_one' := map_o …
      ⊢ Eq { toFun := toFun✝, map_one' := map_one'✝, map_mul' := map_mul'✝, monotone …
    -/
    obtain ⟨⟨⟨_, _⟩⟩, _⟩ := g
    /-
      case mk.mk.mk.mk.mk.mk
      F : Type u_1
      α : Type u_2
      β : Type u_3
      γ : Type u_4
      δ : Type u_5
      inst✝⁷ : Preorder α
      inst✝⁶ : Preorder β
      inst✝⁵ : Preorder γ
      inst✝⁴ : Preorder δ
      inst✝³ : MulOneClass α
      inst✝² : MulOneClass β
      inst✝¹ : MulOneClass γ
      inst✝ : MulOneClass δ
      f g : OrderMonoidHom α β
      toFun✝¹ : α → β
      map_one'✝¹ : Eq (toFun✝¹ 1) 1
      map_mul'✝¹ : ∀ (x y : α), Eq ({ toFun := toFun✝¹, map_one' := map_one'✝¹ }.toF …
      monotone'✝¹ : Monotone (↑{ toFun := toFun✝¹, map_one' := map_one'✝¹, map_mul'  …
      toFun✝ : α → β
      map_one'✝ : Eq (toFun✝ 1) 1
      map_mul'✝ : ∀ (x y : α), Eq ({ toFun := toFun✝, map_one' := map_one'✝ }.toFun  …
      monotone'✝ : Monotone (↑{ toFun := toFun✝, map_one' := map_one'✝, map_mul' :=  …
      h : Eq ((fun f => (↑f.toMonoidHom).toFun) { toFun := toFun✝¹, map_one' := map_ …
      ⊢ Eq { toFun := toFun✝¹, map_one' := map_one'✝¹, map_mul' := map_mul'✝¹, monot …
    -/
    congr
    /-
      🎉 no goals
    -/


@[to_additive]
instance : OrderHomClass (α →*o β) α β where
  map_rel f _ _ h := f.monotone' h


@[to_additive]
instance : MonoidHomClass (α →*o β) α β where
  map_mul f := f.map_mul'
  map_one f := f.map_one'

-- Other lemmas should be accessed through the `FunLike` API

@[to_additive (attr := ext)]
theorem ext (h : ∀ a, f a = g a) : f = g :=
  DFunLike.ext f g h


@[to_additive]
theorem toFun_eq_coe (f : α →*o β) : f.toFun = (f : α → β) :=
  rfl


@[to_additive (attr := simp)]
theorem coe_mk (f : α →* β) (h) : (OrderMonoidHom.mk f h : α → β) = f :=
  rfl


@[to_additive (attr := simp)]
theorem mk_coe (f : α →*o β) (h) : OrderMonoidHom.mk (f : α →* β) h = f := by
  /-
    α : Type u_2
    β : Type u_3
    inst✝³ : Preorder α
    inst✝² : Preorder β
    inst✝¹ : MulOneClass α
    inst✝ : MulOneClass β
    f : OrderMonoidHom α β
    h : Monotone (↑↑f).toFun
    ⊢ Eq { toMonoidHom := ↑f, monotone' := h } f
  -/
  ext
  /-
    case h
    α : Type u_2
    β : Type u_3
    inst✝³ : Preorder α
    inst✝² : Preorder β
    inst✝¹ : MulOneClass α
    inst✝ : MulOneClass β
    f : OrderMonoidHom α β
    h : Monotone (↑↑f).toFun
    a✝ : α
    ⊢ Eq ({ toMonoidHom := ↑f, monotone' := h } a✝) (f a✝)
  -/
  rfl
  /-
    🎉 no goals
  -/


/-- Reinterpret an ordered monoid homomorphism as an order homomorphism. -/
@[to_additive "Reinterpret an ordered additive monoid homomorphism as an order homomorphism."]
def toOrderHom (f : α →*o β) : α →o β :=
  { f with }


@[to_additive (attr := simp)]
theorem coe_monoidHom (f : α →*o β) : ((f : α →* β) : α → β) = f :=
  rfl


@[to_additive (attr := simp)]
theorem coe_orderHom (f : α →*o β) : ((f : α →o β) : α → β) = f :=
  rfl


@[to_additive]
theorem toMonoidHom_injective : Injective (toMonoidHom : _ → α →* β) := fun f g h =>
            /-
              α : Type u_2
              β : Type u_3
              inst✝³ : Preorder α
              inst✝² : Preorder β
              inst✝¹ : MulOneClass α
              inst✝ : MulOneClass β
              f g : OrderMonoidHom α β
              h : Eq f.toMonoidHom g.toMonoidHom
              ⊢ ∀ (a : α), Eq (f a) (g a)
            -/
  ext <| by convert DFunLike.ext_iff.1 h using 0
            /-
              🎉 no goals
            -/


@[to_additive]
theorem toOrderHom_injective : Injective (toOrderHom : _ → α →o β) := fun f g h =>
            /-
              α : Type u_2
              β : Type u_3
              inst✝³ : Preorder α
              inst✝² : Preorder β
              inst✝¹ : MulOneClass α
              inst✝ : MulOneClass β
              f g : OrderMonoidHom α β
              h : Eq f.toOrderHom g.toOrderHom
              ⊢ ∀ (a : α), Eq (f a) (g a)
            -/
  ext <| by convert DFunLike.ext_iff.1 h using 0
            /-
              🎉 no goals
            -/


/-- Copy of an `OrderMonoidHom` with a new `toFun` equal to the old one. Useful to fix
definitional equalities. -/
@[to_additive "Copy of an `OrderAddMonoidHom` with a new `toFun` equal to the old one. Useful to fix
definitional equalities."]
protected def copy (f : α →*o β) (f' : α → β) (h : f' = f) : α →*o β :=
  { f.toMonoidHom.copy f' h with toFun := f', monotone' := h.symm.subst f.monotone' }


@[to_additive (attr := simp)]
theorem coe_copy (f : α →*o β) (f' : α → β) (h : f' = f) : ⇑(f.copy f' h) = f' :=
  rfl


@[to_additive]
theorem copy_eq (f : α →*o β) (f' : α → β) (h : f' = f) : f.copy f' h = f :=
  DFunLike.ext' h


/-- The identity map as an ordered monoid homomorphism. -/
@[to_additive "The identity map as an ordered additive monoid homomorphism."]
protected def id : α →*o α :=
  { MonoidHom.id α, OrderHom.id with }


@[to_additive (attr := simp)]
theorem coe_id : ⇑(OrderMonoidHom.id α) = id :=
  rfl


@[to_additive]
instance : Inhabited (α →*o α) :=
  ⟨OrderMonoidHom.id α⟩


/-- Composition of `OrderMonoidHom`s as an `OrderMonoidHom`. -/
@[to_additive "Composition of `OrderAddMonoidHom`s as an `OrderAddMonoidHom`"]
def comp (f : β →*o γ) (g : α →*o β) : α →*o γ :=
  { f.toMonoidHom.comp (g : α →* β), f.toOrderHom.comp (g : α →o β) with }


@[to_additive (attr := simp)]
theorem coe_comp (f : β →*o γ) (g : α →*o β) : (f.comp g : α → γ) = f ∘ g :=
  rfl


@[to_additive (attr := simp)]
theorem comp_apply (f : β →*o γ) (g : α →*o β) (a : α) : (f.comp g) a = f (g a) :=
  rfl


@[to_additive]
theorem coe_comp_monoidHom (f : β →*o γ) (g : α →*o β) :
    (f.comp g : α →* γ) = (f : β →* γ).comp g :=
  rfl


@[to_additive]
theorem coe_comp_orderHom (f : β →*o γ) (g : α →*o β) :
    (f.comp g : α →o γ) = (f : β →o γ).comp g :=
  rfl


@[to_additive (attr := simp)]
theorem comp_assoc (f : γ →*o δ) (g : β →*o γ) (h : α →*o β) :
    (f.comp g).comp h = f.comp (g.comp h) :=
  rfl


@[to_additive (attr := simp)]
theorem comp_id (f : α →*o β) : f.comp (OrderMonoidHom.id α) = f :=
  rfl


@[to_additive (attr := simp)]
theorem id_comp (f : α →*o β) : (OrderMonoidHom.id β).comp f = f :=
  rfl


@[to_additive (attr := simp)]
theorem cancel_right {g₁ g₂ : β →*o γ} {f : α →*o β} (hf : Function.Surjective f) :
    g₁.comp f = g₂.comp f ↔ g₁ = g₂ :=
                                                                    /-
                                                                      α : Type u_2
                                                                      β : Type u_3
                                                                      γ : Type u_4
                                                                      inst✝⁵ : Preorder α
                                                                      inst✝⁴ : Preorder β
                                                                      inst✝³ : Preorder γ
                                                                      inst✝² : MulOneClass α
                                                                      inst✝¹ : MulOneClass β
                                                                      inst✝ : MulOneClass γ
                                                                      g₁ g₂ : OrderMonoidHom β γ
                                                                      f : OrderMonoidHom α β
                                                                      hf : Function.Surjective ⇑f
                                                                      x✝ : Eq g₁ g₂
                                                                      ⊢ Eq (g₁.comp f) (g₂.comp f)
                                                                    -/
  ⟨fun h => ext <| hf.forall.2 <| DFunLike.ext_iff.1 h, fun _ => by congr⟩
                                                                    /-
                                                                      🎉 no goals
                                                                    -/


@[to_additive (attr := simp)]
theorem cancel_left {g : β →*o γ} {f₁ f₂ : α →*o β} (hg : Function.Injective g) :
    g.comp f₁ = g.comp f₂ ↔ f₁ = f₂ :=
                                  /-
                                    α : Type u_2
                                    β : Type u_3
                                    γ : Type u_4
                                    inst✝⁵ : Preorder α
                                    inst✝⁴ : Preorder β
                                    inst✝³ : Preorder γ
                                    inst✝² : MulOneClass α
                                    inst✝¹ : MulOneClass β
                                    inst✝ : MulOneClass γ
                                    g : OrderMonoidHom β γ
                                    f₁ f₂ : OrderMonoidHom α β
                                    hg : Function.Injective ⇑g
                                    h : Eq (g.comp f₁) (g.comp f₂)
                                    a : α
                                    ⊢ Eq (g (f₁ a)) (g (f₂ a))
                                  -/
  ⟨fun h => ext fun a => hg <| by rw [← comp_apply, h, comp_apply], congr_arg _⟩
                                  /-
                                    🎉 no goals
                                  -/


/-- `1` is the homomorphism sending all elements to `1`. -/
@[to_additive "`0` is the homomorphism sending all elements to `0`."]
instance : One (α →*o β) :=
  ⟨{ (1 : α →* β) with monotone' := monotone_const }⟩


@[to_additive (attr := simp)]
theorem coe_one : ⇑(1 : α →*o β) = 1 :=
  rfl


@[to_additive (attr := simp)]
theorem one_apply (a : α) : (1 : α →*o β) a = 1 :=
  rfl


@[to_additive (attr := simp)]
theorem one_comp (f : α →*o β) : (1 : β →*o γ).comp f = 1 :=
  rfl


@[to_additive (attr := simp)]
theorem comp_one (f : β →*o γ) : f.comp (1 : α →*o β) = 1 :=
  ext fun _ => map_one f


/-- For two ordered monoid morphisms `f` and `g`, their product is the ordered monoid morphism
sending `a` to `f a * g a`. -/
@[to_additive "For two ordered additive monoid morphisms `f` and `g`, their product is the ordered
additive monoid morphism sending `a` to `f a + g a`."]
instance : Mul (α →*o β) :=
  ⟨fun f g => { (f * g : α →* β) with monotone' := f.monotone'.mul' g.monotone' }⟩


@[to_additive (attr := simp)]
theorem coe_mul (f g : α →*o β) : ⇑(f * g) = f * g :=
  rfl


@[to_additive (attr := simp)]
theorem mul_apply (f g : α →*o β) (a : α) : (f * g) a = f a * g a :=
  rfl


@[to_additive]
theorem mul_comp (g₁ g₂ : β →*o γ) (f : α →*o β) : (g₁ * g₂).comp f = g₁.comp f * g₂.comp f :=
  rfl


@[to_additive]
theorem comp_mul (g : β →*o γ) (f₁ f₂ : α →*o β) : g.comp (f₁ * f₂) = g.comp f₁ * g.comp f₂ :=
  ext fun _ => map_mul g _ _


@[to_additive (attr := simp)]
theorem toMonoidHom_eq_coe (f : α →*o β) : f.toMonoidHom = f :=
  rfl


@[to_additive (attr := simp)]
theorem toOrderHom_eq_coe (f : α →*o β) : f.toOrderHom = f :=
  rfl


/-- Makes an ordered group homomorphism from a proof that the map preserves multiplication. -/
@[to_additive
      "Makes an ordered additive group homomorphism from a proof that the map preserves
      addition."]
def mk' (f : α → β) (hf : Monotone f) (map_mul : ∀ a b : α, f (a * b) = f a * f b) : α →*o β :=
  { MonoidHom.mk' f map_mul with monotone' := hf }


@[to_additive]
instance : EquivLike (α ≃*o β) α β where
  coe f := f.toFun
  inv f := f.invFun
  left_inv f := f.left_inv
  right_inv f := f.right_inv
  coe_injective' f g h₁ h₂ := by
    /-
      F : Type u_1
      α : Type u_2
      β : Type u_3
      γ : Type u_4
      δ : Type u_5
      inst✝⁷ : Preorder α
      inst✝⁶ : Preorder β
      inst✝⁵ : Preorder γ
      inst✝⁴ : Preorder δ
      inst✝³ : Mul α
      inst✝² : Mul β
      inst✝¹ : Mul γ
      inst✝ : Mul δ
      f✝ g✝ f g : OrderMonoidIso α β
      h₁ : Eq ((fun f => f.toFun) f) ((fun f => f.toFun) g)
      h₂ : Eq ((fun f => f.invFun) f) ((fun f => f.invFun) g)
      ⊢ Eq f g
    -/
    obtain ⟨⟨⟨_, _⟩⟩, _⟩ := f
    /-
      case mk.mk.mk
      F : Type u_1
      α : Type u_2
      β : Type u_3
      γ : Type u_4
      δ : Type u_5
      inst✝⁷ : Preorder α
      inst✝⁶ : Preorder β
      inst✝⁵ : Preorder γ
      inst✝⁴ : Preorder δ
      inst✝³ : Mul α
      inst✝² : Mul β
      inst✝¹ : Mul γ
      inst✝ : Mul δ
      f g✝ g : OrderMonoidIso α β
      toFun✝ : α → β
      invFun✝ : β → α
      left_inv✝ : Function.LeftInverse invFun✝ toFun✝
      right_inv✝ : Function.RightInverse invFun✝ toFun✝
      map_mul'✝ : ∀ (x y : α), Eq ({ toFun := toFun✝, invFun := invFun✝, left_inv := …
      map_le_map_iff'✝ : ∀ {a b : α}, Iff (LE.le ({ toFun := toFun✝, invFun := invFu …
      h₁ : Eq ((fun f => f.toFun) { toFun := toFun✝, invFun := invFun✝, left_inv :=  …
      h₂ : Eq ((fun f => f.invFun) { toFun := toFun✝, invFun := invFun✝, left_inv := …
      ⊢ Eq { toFun := toFun✝, invFun := invFun✝, left_inv := left_inv✝, right_inv := …
    -/
    obtain ⟨⟨⟨_, _⟩⟩, _⟩ := g
    /-
      case mk.mk.mk.mk.mk.mk
      F : Type u_1
      α : Type u_2
      β : Type u_3
      γ : Type u_4
      δ : Type u_5
      inst✝⁷ : Preorder α
      inst✝⁶ : Preorder β
      inst✝⁵ : Preorder γ
      inst✝⁴ : Preorder δ
      inst✝³ : Mul α
      inst✝² : Mul β
      inst✝¹ : Mul γ
      inst✝ : Mul δ
      f g : OrderMonoidIso α β
      toFun✝¹ : α → β
      invFun✝¹ : β → α
      left_inv✝¹ : Function.LeftInverse invFun✝¹ toFun✝¹
      right_inv✝¹ : Function.RightInverse invFun✝¹ toFun✝¹
      map_mul'✝¹ : ∀ (x y : α), Eq ({ toFun := toFun✝¹, invFun := invFun✝¹, left_inv …
      map_le_map_iff'✝¹ : ∀ {a b : α}, Iff (LE.le ({ toFun := toFun✝¹, invFun := inv …
      toFun✝ : α → β
      invFun✝ : β → α
      left_inv✝ : Function.LeftInverse invFun✝ toFun✝
      right_inv✝ : Function.RightInverse invFun✝ toFun✝
      map_mul'✝ : ∀ (x y : α), Eq ({ toFun := toFun✝, invFun := invFun✝, left_inv := …
      map_le_map_iff'✝ : ∀ {a b : α}, Iff (LE.le ({ toFun := toFun✝, invFun := invFu …
      h₁ : Eq ((fun f => f.toFun) { toFun := toFun✝¹, invFun := invFun✝¹, left_inv : …
      h₂ : Eq ((fun f => f.invFun) { toFun := toFun✝¹, invFun := invFun✝¹, left_inv  …
      ⊢ Eq { toFun := toFun✝¹, invFun := invFun✝¹, left_inv := left_inv✝¹, right_inv …
    -/
    congr
    /-
      🎉 no goals
    -/


@[to_additive]
instance : OrderIsoClass (α ≃*o β) α β where
  map_le_map_iff f := f.map_le_map_iff'


@[to_additive]
instance : MulEquivClass (α ≃*o β) α β where
  map_mul f := map_mul f.toMulEquiv

-- Other lemmas should be accessed through the `FunLike` API

@[to_additive]
theorem toFun_eq_coe (f : α ≃*o β) : f.toFun = (f : α → β) :=
  rfl


@[to_additive (attr := simp)]
theorem coe_mk (f : α ≃* β) (h) : (OrderMonoidIso.mk f h : α → β) = f :=
  rfl


@[to_additive (attr := simp)]
theorem mk_coe (f : α ≃*o β) (h) : OrderMonoidIso.mk (f : α ≃* β) h = f := rfl


/-- Reinterpret an ordered monoid isomorphism as an order isomorphism. -/
@[to_additive "Reinterpret an ordered additive monoid isomomorphism as an order isomomorphism."]
def toOrderIso (f : α ≃*o β) : α ≃o β :=
  { f with
    map_rel_iff' := map_le_map_iff f }


@[to_additive (attr := simp)]
theorem coe_mulEquiv (f : α ≃*o β) : ((f : α ≃* β) : α → β) = f :=
  rfl


@[to_additive (attr := simp)]
theorem coe_orderIso (f : α ≃*o β) : ((f : α →o β) : α → β) = f :=
  rfl


@[to_additive]
theorem toMulEquiv_injective : Injective (toMulEquiv : _ → α ≃* β) := fun f g h =>
            /-
              α : Type u_2
              β : Type u_3
              inst✝³ : Preorder α
              inst✝² : Preorder β
              inst✝¹ : Mul α
              inst✝ : Mul β
              f g : OrderMonoidIso α β
              h : Eq f.toMulEquiv g.toMulEquiv
              ⊢ ∀ (a : α), Eq (f a) (g a)
            -/
  ext <| by convert DFunLike.ext_iff.1 h using 0
            /-
              🎉 no goals
            -/


@[to_additive]
theorem toOrderIso_injective : Injective (toOrderIso : _ → α ≃o β) := fun f g h =>
            /-
              α : Type u_2
              β : Type u_3
              inst✝³ : Preorder α
              inst✝² : Preorder β
              inst✝¹ : Mul α
              inst✝ : Mul β
              f g : OrderMonoidIso α β
              h : Eq f.toOrderIso g.toOrderIso
              ⊢ ∀ (a : α), Eq (f a) (g a)
            -/
  ext <| by convert DFunLike.ext_iff.1 h using 0
            /-
              🎉 no goals
            -/


/-- The identity map as an ordered monoid isomorphism. -/
@[to_additive "The identity map as an ordered additive monoid isomorphism."]
protected def refl : α ≃*o α :=
                                               /-
                                                 F : Type u_1
                                                 α : Type u_2
                                                 β : Type u_3
                                                 γ : Type u_4
                                                 δ : Type u_5
                                                 inst✝⁷ : Preorder α
                                                 inst✝⁶ : Preorder β
                                                 inst✝⁵ : Preorder γ
                                                 inst✝⁴ : Preorder δ
                                                 inst✝³ : Mul α
                                                 inst✝² : Mul β
                                                 inst✝¹ : Mul γ
                                                 inst✝ : Mul δ
                                                 f g : OrderMonoidIso α β
                                                 ⊢ ∀ {a b : α}, Iff (LE.le (__src✝.toFun a) (__src✝.toFun b)) (LE.le a b)
                                               -/
  { MulEquiv.refl α with map_le_map_iff' := by simp }
                                               /-
                                                 🎉 no goals
                                               -/


@[to_additive (attr := simp)]
theorem coe_refl : ⇑(OrderMonoidIso.refl α) = id :=
  rfl


@[to_additive]
instance : Inhabited (α ≃*o α) :=
  ⟨OrderMonoidIso.refl α⟩


/-- Transitivity of multiplication-preserving order isomorphisms -/
@[to_additive (attr := trans) "Transitivity of addition-preserving order isomorphisms"]
def trans (f : α ≃*o β) (g : β ≃*o γ) : α ≃*o γ :=
                                                    /-
                                                      F : Type u_1
                                                      α : Type u_2
                                                      β : Type u_3
                                                      γ : Type u_4
                                                      δ : Type u_5
                                                      inst✝⁷ : Preorder α
                                                      inst✝⁶ : Preorder β
                                                      inst✝⁵ : Preorder γ
                                                      inst✝⁴ : Preorder δ
                                                      inst✝³ : Mul α
                                                      inst✝² : Mul β
                                                      inst✝¹ : Mul γ
                                                      inst✝ : Mul δ
                                                      f✝ g✝ f : OrderMonoidIso α β
                                                      g : OrderMonoidIso β γ
                                                      ⊢ ∀ {a b : α}, Iff (LE.le (__src✝.toFun a) (__src✝.toFun b)) (LE.le a b)
                                                    -/
  { (f : α ≃* β).trans g with map_le_map_iff' := by simp }
                                                    /-
                                                      🎉 no goals
                                                    -/


@[to_additive (attr := simp)]
theorem coe_trans (f : α ≃*o β) (g : β ≃*o γ) : (f.trans g : α → γ) = g ∘ f :=
  rfl


@[to_additive (attr := simp)]
theorem trans_apply (f : α ≃*o β) (g : β ≃*o γ) (a : α) : (f.trans g) a = g (f a) :=
  rfl


@[to_additive]
theorem coe_trans_mulEquiv (f : α ≃*o β) (g : β ≃*o γ) :
    (f.trans g : α ≃* γ) = (f : α ≃* β).trans g :=
  rfl


@[to_additive]
theorem coe_trans_orderIso (f : α ≃*o β) (g : β ≃*o γ) :
    (f.trans g : α ≃o γ) = (f : α ≃o β).trans g :=
  rfl


@[to_additive (attr := simp)]
theorem trans_assoc (f : α ≃*o β) (g : β ≃*o γ) (h : γ ≃*o δ) :
    (f.trans g).trans h = f.trans (g.trans h) :=
  rfl


@[to_additive (attr := simp)]
theorem trans_refl (f : α ≃*o β) : f.trans (OrderMonoidIso.refl β) = f :=
  rfl


@[to_additive (attr := simp)]
theorem refl_trans (f : α ≃*o β) : (OrderMonoidIso.refl α).trans f = f :=
  rfl


@[to_additive (attr := simp)]
theorem cancel_right {g₁ g₂ : α ≃*o β} {f : β ≃*o γ} (hf : Function.Injective f) :
    g₁.trans f = g₂.trans f ↔ g₁ = g₂ :=
                                  /-
                                    α : Type u_2
                                    β : Type u_3
                                    γ : Type u_4
                                    inst✝⁵ : Preorder α
                                    inst✝⁴ : Preorder β
                                    inst✝³ : Preorder γ
                                    inst✝² : Mul α
                                    inst✝¹ : Mul β
                                    inst✝ : Mul γ
                                    g₁ g₂ : OrderMonoidIso α β
                                    f : OrderMonoidIso β γ
                                    hf : Function.Injective ⇑f
                                    h : Eq (g₁.trans f) (g₂.trans f)
                                    a : α
                                    ⊢ Eq (f (g₁ a)) (f (g₂ a))
                                  -/
                                  /-
                                    🎉 no goals
                                  -/
  ⟨fun h => ext fun a => hf <| by rw [← trans_apply, h, trans_apply], by rintro rfl; rfl⟩
                                                                                     /-
                                                                                       🎉 no goals
                                                                                     -/


@[to_additive (attr := simp)]
theorem cancel_left {g : α ≃*o β} {f₁ f₂ : β ≃*o γ} (hg : Function.Surjective g) :
    g.trans f₁ = g.trans f₂ ↔ f₁ = f₂ :=
                                                                    /-
                                                                      α : Type u_2
                                                                      β : Type u_3
                                                                      γ : Type u_4
                                                                      inst✝⁵ : Preorder α
                                                                      inst✝⁴ : Preorder β
                                                                      inst✝³ : Preorder γ
                                                                      inst✝² : Mul α
                                                                      inst✝¹ : Mul β
                                                                      inst✝ : Mul γ
                                                                      g : OrderMonoidIso α β
                                                                      f₁ f₂ : OrderMonoidIso β γ
                                                                      hg : Function.Surjective ⇑g
                                                                      x✝ : Eq f₁ f₂
                                                                      ⊢ Eq (g.trans f₁) (g.trans f₂)
                                                                    -/
  ⟨fun h => ext <| hg.forall.2 <| DFunLike.ext_iff.1 h, fun _ => by congr⟩
                                                                    /-
                                                                      🎉 no goals
                                                                    -/


@[to_additive (attr := simp)]
theorem toMulEquiv_eq_coe (f : α ≃*o β) : f.toMulEquiv = f :=
  rfl


@[to_additive (attr := simp)]
theorem toOrderIso_eq_coe (f : α ≃*o β) : f.toOrderIso = f :=
  rfl


@[to_additive]
protected lemma strictMono : StrictMono f :=
  strictMono_of_le_iff_le fun _ _ ↦ (map_le_map_iff _).symm


@[to_additive]
protected lemma strictMono_symm : StrictMono f.symm :=
  strictMono_of_le_iff_le <| fun a b ↦ by
    /-
      α : Type u_2
      β : Type u_3
      inst✝³ : Preorder α
      inst✝² : Preorder β
      inst✝¹ : Mul α
      inst✝ : Mul β
      f : OrderMonoidIso α β
      a b : β
      ⊢ Iff (LE.le a b) (LE.le (f.symm a) (f.symm b))
    -/
    rw [← map_le_map_iff f]
    /-
      α : Type u_2
      β : Type u_3
      inst✝³ : Preorder α
      inst✝² : Preorder β
      inst✝¹ : Mul α
      inst✝ : Mul β
      f : OrderMonoidIso α β
      a b : β
      ⊢ Iff (LE.le a b) (LE.le (f (f.symm a)) (f (f.symm b)))
    -/
    convert Iff.rfl <;>
    /-
      case h.e'_2.h.e'_3
      α : Type u_2
      β : Type u_3
      inst✝³ : Preorder α
      inst✝² : Preorder β
      inst✝¹ : Mul α
      inst✝ : Mul β
      f : OrderMonoidIso α β
      a b : β
      ⊢ Eq (f (f.symm a)) a
    -/
    /-
      🎉 no goals
    -/
    exact f.toEquiv.apply_symm_apply _
    /-
      🎉 no goals
    -/


/-- Makes an ordered group isomorphism from a proof that the map preserves multiplication. -/
@[to_additive
      "Makes an ordered additive group isomorphism from a proof that the map preserves
      addition."]
def mk' (f : α ≃ β) (hf : ∀ {a b}, f a ≤ f b ↔ a ≤ b) (map_mul : ∀ a b : α, f (a * b) = f a * f b) :
    α ≃*o β :=
  { MulEquiv.mk' f map_mul with map_le_map_iff' := hf }


instance : FunLike (α →*₀o β) α β where
  coe f := f.toFun
  coe_injective' f g h := by
    /-
      F : Type u_1
      α : Type u_2
      β : Type u_3
      γ : Type u_4
      δ : Type u_5
      inst✝⁷ : Preorder α
      inst✝⁶ : Preorder β
      inst✝⁵ : Preorder γ
      inst✝⁴ : Preorder δ
      inst✝³ : MulZeroOneClass α
      inst✝² : MulZeroOneClass β
      inst✝¹ : MulZeroOneClass γ
      inst✝ : MulZeroOneClass δ
      f✝ g✝ f g : OrderMonoidWithZeroHom α β
      h : Eq ((fun f => (↑f.toMonoidWithZeroHom).toFun) f) ((fun f => (↑f.toMonoidWi …
      ⊢ Eq f g
    -/
    obtain ⟨⟨⟨_, _⟩⟩, _⟩ := f
    /-
      case mk.mk.mk
      F : Type u_1
      α : Type u_2
      β : Type u_3
      γ : Type u_4
      δ : Type u_5
      inst✝⁷ : Preorder α
      inst✝⁶ : Preorder β
      inst✝⁵ : Preorder γ
      inst✝⁴ : Preorder δ
      inst✝³ : MulZeroOneClass α
      inst✝² : MulZeroOneClass β
      inst✝¹ : MulZeroOneClass γ
      inst✝ : MulZeroOneClass δ
      f g✝ g : OrderMonoidWithZeroHom α β
      toFun✝ : α → β
      map_zero'✝ : Eq (toFun✝ 0) 0
      map_one'✝ : Eq ({ toFun := toFun✝, map_zero' := map_zero'✝ }.toFun 1) 1
      map_mul'✝ : ∀ (x y : α), Eq ({ toFun := toFun✝, map_zero' := map_zero'✝ }.toFu …
      monotone'✝ : Monotone (↑{ toFun := toFun✝, map_zero' := map_zero'✝, map_one' : …
      h : Eq ((fun f => (↑f.toMonoidWithZeroHom).toFun) { toFun := toFun✝, map_zero' …
      ⊢ Eq { toFun := toFun✝, map_zero' := map_zero'✝, map_one' := map_one'✝, map_mu …
    -/
    obtain ⟨⟨⟨_, _⟩⟩, _⟩ := g
    /-
      case mk.mk.mk.mk.mk.mk
      F : Type u_1
      α : Type u_2
      β : Type u_3
      γ : Type u_4
      δ : Type u_5
      inst✝⁷ : Preorder α
      inst✝⁶ : Preorder β
      inst✝⁵ : Preorder γ
      inst✝⁴ : Preorder δ
      inst✝³ : MulZeroOneClass α
      inst✝² : MulZeroOneClass β
      inst✝¹ : MulZeroOneClass γ
      inst✝ : MulZeroOneClass δ
      f g : OrderMonoidWithZeroHom α β
      toFun✝¹ : α → β
      map_zero'✝¹ : Eq (toFun✝¹ 0) 0
      map_one'✝¹ : Eq ({ toFun := toFun✝¹, map_zero' := map_zero'✝¹ }.toFun 1) 1
      map_mul'✝¹ : ∀ (x y : α), Eq ({ toFun := toFun✝¹, map_zero' := map_zero'✝¹ }.t …
      monotone'✝¹ : Monotone (↑{ toFun := toFun✝¹, map_zero' := map_zero'✝¹, map_one …
      toFun✝ : α → β
      map_zero'✝ : Eq (toFun✝ 0) 0
      map_one'✝ : Eq ({ toFun := toFun✝, map_zero' := map_zero'✝ }.toFun 1) 1
      map_mul'✝ : ∀ (x y : α), Eq ({ toFun := toFun✝, map_zero' := map_zero'✝ }.toFu …
      monotone'✝ : Monotone (↑{ toFun := toFun✝, map_zero' := map_zero'✝, map_one' : …
      h : Eq ((fun f => (↑f.toMonoidWithZeroHom).toFun) { toFun := toFun✝¹, map_zero …
      ⊢ Eq { toFun := toFun✝¹, map_zero' := map_zero'✝¹, map_one' := map_one'✝¹, map …
    -/
    congr
    /-
      🎉 no goals
    -/


instance : MonoidWithZeroHomClass (α →*₀o β) α β where
  map_mul f := f.map_mul'
  map_one f := f.map_one'
  map_zero f := f.map_zero'


instance : OrderHomClass (α →*₀o β) α β where
  map_rel f _ _ h := f.monotone' h

-- Other lemmas should be accessed through the `FunLike` API

@[ext]
theorem ext (h : ∀ a, f a = g a) : f = g :=
  DFunLike.ext f g h


theorem toFun_eq_coe (f : α →*₀o β) : f.toFun = (f : α → β) :=
  rfl


@[simp]
theorem coe_mk (f : α →*₀ β) (h) : (OrderMonoidWithZeroHom.mk f h : α → β) = f :=
  rfl


@[simp]
theorem mk_coe (f : α →*₀o β) (h) : OrderMonoidWithZeroHom.mk (f : α →*₀ β) h = f := rfl


/-- Reinterpret an ordered monoid with zero homomorphism as an order monoid homomorphism. -/
def toOrderMonoidHom (f : α →*₀o β) : α →*o β :=
  { f with }


@[simp]
theorem coe_monoidWithZeroHom (f : α →*₀o β) : ⇑(f : α →*₀ β) = f :=
  rfl


@[simp]
theorem coe_orderMonoidHom (f : α →*₀o β) : ⇑(f : α →*o β) = f :=
  rfl


theorem toOrderMonoidHom_injective : Injective (toOrderMonoidHom : _ → α →*o β) := fun f g h =>
            /-
              α : Type u_2
              β : Type u_3
              inst✝³ : Preorder α
              inst✝² : Preorder β
              inst✝¹ : MulZeroOneClass α
              inst✝ : MulZeroOneClass β
              f g : OrderMonoidWithZeroHom α β
              h : Eq f.toOrderMonoidHom g.toOrderMonoidHom
              ⊢ ∀ (a : α), Eq (f a) (g a)
            -/
  ext <| by convert DFunLike.ext_iff.1 h using 0
            /-
              🎉 no goals
            -/


theorem toMonoidWithZeroHom_injective : Injective (toMonoidWithZeroHom : _ → α →*₀ β) :=
                         /-
                           α : Type u_2
                           β : Type u_3
                           inst✝³ : Preorder α
                           inst✝² : Preorder β
                           inst✝¹ : MulZeroOneClass α
                           inst✝ : MulZeroOneClass β
                           f g : OrderMonoidWithZeroHom α β
                           h : Eq f.toMonoidWithZeroHom g.toMonoidWithZeroHom
                           ⊢ ∀ (a : α), Eq (f a) (g a)
                         -/
  fun f g h => ext <| by convert DFunLike.ext_iff.1 h using 0
                         /-
                           🎉 no goals
                         -/


/-- Copy of an `OrderMonoidWithZeroHom` with a new `toFun` equal to the old one. Useful to fix
definitional equalities. -/
protected def copy (f : α →*₀o β) (f' : α → β) (h : f' = f) : α →*o β :=
  { f.toOrderMonoidHom.copy f' h, f.toMonoidWithZeroHom.copy f' h with toFun := f' }


@[simp]
theorem coe_copy (f : α →*₀o β) (f' : α → β) (h : f' = f) : ⇑(f.copy f' h) = f' :=
  rfl


theorem copy_eq (f : α →*₀o β) (f' : α → β) (h : f' = f) : f.copy f' h = f :=
  DFunLike.ext' h


/-- The identity map as an ordered monoid with zero homomorphism. -/
protected def id : α →*₀o α :=
  { MonoidWithZeroHom.id α, OrderHom.id with }


@[simp]
theorem coe_id : ⇑(OrderMonoidWithZeroHom.id α) = id :=
  rfl


instance : Inhabited (α →*₀o α) :=
  ⟨OrderMonoidWithZeroHom.id α⟩


/-- Composition of `OrderMonoidWithZeroHom`s as an `OrderMonoidWithZeroHom`. -/
def comp (f : β →*₀o γ) (g : α →*₀o β) : α →*₀o γ :=
  { f.toMonoidWithZeroHom.comp (g : α →*₀ β), f.toOrderMonoidHom.comp (g : α →*o β) with }


@[simp]
theorem coe_comp (f : β →*₀o γ) (g : α →*₀o β) : (f.comp g : α → γ) = f ∘ g :=
  rfl


@[simp]
theorem comp_apply (f : β →*₀o γ) (g : α →*₀o β) (a : α) : (f.comp g) a = f (g a) :=
  rfl


theorem coe_comp_monoidWithZeroHom (f : β →*₀o γ) (g : α →*₀o β) :
    (f.comp g : α →*₀ γ) = (f : β →*₀ γ).comp g :=
  rfl


theorem coe_comp_orderMonoidHom (f : β →*₀o γ) (g : α →*₀o β) :
    (f.comp g : α →*o γ) = (f : β →*o γ).comp g :=
  rfl


@[simp]
theorem comp_assoc (f : γ →*₀o δ) (g : β →*₀o γ) (h : α →*₀o β) :
    (f.comp g).comp h = f.comp (g.comp h) :=
  rfl


@[simp]
theorem comp_id (f : α →*₀o β) : f.comp (OrderMonoidWithZeroHom.id α) = f := rfl


@[simp]
theorem id_comp (f : α →*₀o β) : (OrderMonoidWithZeroHom.id β).comp f = f := rfl


@[simp]
theorem cancel_right {g₁ g₂ : β →*₀o γ} {f : α →*₀o β} (hf : Function.Surjective f) :
    g₁.comp f = g₂.comp f ↔ g₁ = g₂ :=
                                                                    /-
                                                                      α : Type u_2
                                                                      β : Type u_3
                                                                      γ : Type u_4
                                                                      inst✝⁵ : Preorder α
                                                                      inst✝⁴ : Preorder β
                                                                      inst✝³ : Preorder γ
                                                                      inst✝² : MulZeroOneClass α
                                                                      inst✝¹ : MulZeroOneClass β
                                                                      inst✝ : MulZeroOneClass γ
                                                                      g₁ g₂ : OrderMonoidWithZeroHom β γ
                                                                      f : OrderMonoidWithZeroHom α β
                                                                      hf : Function.Surjective ⇑f
                                                                      x✝ : Eq g₁ g₂
                                                                      ⊢ Eq (g₁.comp f) (g₂.comp f)
                                                                    -/
  ⟨fun h => ext <| hf.forall.2 <| DFunLike.ext_iff.1 h, fun _ => by congr⟩
                                                                    /-
                                                                      🎉 no goals
                                                                    -/


@[simp]
theorem cancel_left {g : β →*₀o γ} {f₁ f₂ : α →*₀o β} (hg : Function.Injective g) :
    g.comp f₁ = g.comp f₂ ↔ f₁ = f₂ :=
                                  /-
                                    α : Type u_2
                                    β : Type u_3
                                    γ : Type u_4
                                    inst✝⁵ : Preorder α
                                    inst✝⁴ : Preorder β
                                    inst✝³ : Preorder γ
                                    inst✝² : MulZeroOneClass α
                                    inst✝¹ : MulZeroOneClass β
                                    inst✝ : MulZeroOneClass γ
                                    g : OrderMonoidWithZeroHom β γ
                                    f₁ f₂ : OrderMonoidWithZeroHom α β
                                    hg : Function.Injective ⇑g
                                    h : Eq (g.comp f₁) (g.comp f₂)
                                    a : α
                                    ⊢ Eq (g (f₁ a)) (g (f₂ a))
                                  -/
  ⟨fun h => ext fun a => hg <| by rw [← comp_apply, h, comp_apply], congr_arg _⟩
                                  /-
                                    🎉 no goals
                                  -/


/-- For two ordered monoid morphisms `f` and `g`, their product is the ordered monoid morphism
sending `a` to `f a * g a`. -/
instance : Mul (α →*₀o β) :=
  ⟨fun f g => { (f * g : α →*₀ β) with monotone' := f.monotone'.mul' g.monotone' }⟩


@[simp]
theorem coe_mul (f g : α →*₀o β) : ⇑(f * g) = f * g :=
  rfl


@[simp]
theorem mul_apply (f g : α →*₀o β) (a : α) : (f * g) a = f a * g a :=
  rfl


theorem mul_comp (g₁ g₂ : β →*₀o γ) (f : α →*₀o β) : (g₁ * g₂).comp f = g₁.comp f * g₂.comp f :=
  rfl


theorem comp_mul (g : β →*₀o γ) (f₁ f₂ : α →*₀o β) : g.comp (f₁ * f₂) = g.comp f₁ * g.comp f₂ :=
  ext fun _ => map_mul g _ _


@[simp]
theorem toMonoidWithZeroHom_eq_coe (f : α →*₀o β) : f.toMonoidWithZeroHom = f := by
  /-
    α : Type u_2
    β : Type u_3
    hα : Preorder α
    hα' : MulZeroOneClass α
    hβ : Preorder β
    hβ' : MulZeroOneClass β
    f : OrderMonoidWithZeroHom α β
    ⊢ Eq f.toMonoidWithZeroHom ↑f
  -/
  rfl
  /-
    🎉 no goals
  -/


@[simp]
theorem toOrderMonoidHom_eq_coe (f : α →*₀o β) : f.toOrderMonoidHom = f :=
  rfl


/-- Any ordered group is isomorphic to the units of itself adjoined with `0`. -/
@[simps! toFun]
def OrderMonoidIso.unitsWithZero {α : Type*} [Group α] [Preorder α] : (WithZero α)ˣ ≃*o α where
  toMulEquiv := WithZero.unitsWithZeroEquiv
                              /-
                                F : Type u_1
                                α✝ : Type u_2
                                β : Type u_3
                                γ : Type u_4
                                δ : Type u_5
                                α : Type u_6
                                inst✝¹ : Group α
                                inst✝ : Preorder α
                                a b : Units (WithZero α)
                                ⊢ Iff (LE.le (WithZero.unitsWithZeroEquiv.toFun a) (WithZero.unitsWithZeroEqui …
                              -/
  map_le_map_iff' {a b} := by simp [WithZero.unitsWithZeroEquiv]
                              /-
                                🎉 no goals
                              -/

