set_option linter.unusedVariables false in
lemma of_map (f : F) [neZero : NeZero (f a)] : NeZero a :=
                          /-
                            F : Type u_1
                            α : Type u_2
                            β : Type u_3
                            inst✝³ : Zero α
                            inst✝² : Zero β
                            inst✝¹ : FunLike F α β
                            inst✝ : ZeroHomClass F α β
                            a : α
                            f : F
                            neZero : NeZero (f a)
                            h : Eq a 0
                            ⊢ Eq (f a) 0
                          -/
  ⟨fun h ↦ ne (f a) <| by rw [h]; exact ZeroHomClass.map_zero f⟩
                                  /-
                                    🎉 no goals
                                  -/


lemma of_injective {f : F} (hf : Injective f) [NeZero a] : NeZero (f a) :=
      /-
        F : Type u_1
        α : Type u_2
        β : Type u_3
        inst✝⁴ : Zero α
        inst✝³ : Zero β
        inst✝² : FunLike F α β
        inst✝¹ : ZeroHomClass F α β
        a : α
        f : F
        hf : Function.Injective ⇑f
        inst✝ : NeZero a
        ⊢ Ne (f a) 0
      -/
  ⟨by rw [← ZeroHomClass.map_zero f]; exact hf.ne NeZero.out⟩
                                      /-
                                        🎉 no goals
                                      -/


/-- `MonoidWithZeroHomClass F α β` states that `F` is a type of
`MonoidWithZero`-preserving homomorphisms.

You should also extend this typeclass when you extend `MonoidWithZeroHom`. -/
class MonoidWithZeroHomClass (F : Type*) (α β : outParam Type*) [MulZeroOneClass α]
  [MulZeroOneClass β] [FunLike F α β] extends MonoidHomClass F α β, ZeroHomClass F α β : Prop


/-- `α →*₀ β` is the type of functions `α → β` that preserve
the `MonoidWithZero` structure.

`MonoidWithZeroHom` is also used for group homomorphisms.

When possible, instead of parametrizing results over `(f : α →*₀ β)`,
you should parametrize over `(F : Type*) [MonoidWithZeroHomClass F α β] (f : F)`.

When you extend this structure, make sure to extend `MonoidWithZeroHomClass`. -/
structure MonoidWithZeroHom (α β : Type*) [MulZeroOneClass α] [MulZeroOneClass β]
  extends ZeroHom α β, MonoidHom α β


/-- `α →*₀ β` denotes the type of zero-preserving monoid homomorphisms from `α` to `β`. -/
infixr:25 " →*₀ " => MonoidWithZeroHom


/-- Turn an element of a type `F` satisfying `MonoidWithZeroHomClass F α β` into an actual
`MonoidWithZeroHom`. This is declared as the default coercion from `F` to `α →*₀ β`. -/
@[coe]
def MonoidWithZeroHomClass.toMonoidWithZeroHom [FunLike F α β] [MonoidWithZeroHomClass F α β]
    (f : F) : α →*₀ β := { (f : α →* β), (f : ZeroHom α β) with }


/-- Any type satisfying `MonoidWithZeroHomClass` can be cast into `MonoidWithZeroHom` via
`MonoidWithZeroHomClass.toMonoidWithZeroHom`. -/
instance [FunLike F α β] [MonoidWithZeroHomClass F α β] : CoeTC F (α →*₀ β) :=
  ⟨MonoidWithZeroHomClass.toMonoidWithZeroHom⟩


instance funLike : FunLike (α →*₀ β) α β where
  coe f := f.toFun
                             /-
                               F : Type u_1
                               α : Type u_2
                               β : Type u_3
                               γ : Type u_4
                               δ : Type u_5
                               M₀ : Type u_6
                               inst✝³ : MulZeroOneClass α
                               inst✝² : MulZeroOneClass β
                               inst✝¹ : MulZeroOneClass γ
                               inst✝ : MulZeroOneClass δ
                               f g : MonoidWithZeroHom α β
                               h : Eq ((fun f => f.toFun) f) ((fun f => f.toFun) g)
                               ⊢ Eq f g
                             -/
  coe_injective' f g h := by obtain ⟨⟨_, _⟩, _⟩ := f; obtain ⟨⟨_, _⟩, _⟩ := g; congr
                                                                               /-
                                                                                 🎉 no goals
                                                                               -/


instance monoidWithZeroHomClass : MonoidWithZeroHomClass (α →*₀ β) α β where
  map_mul := MonoidWithZeroHom.map_mul'
  map_one := MonoidWithZeroHom.map_one'
  map_zero f := f.map_zero'


instance [Subsingleton α] : Subsingleton (α →*₀ β) := .of_oneHomClass


@[simp] lemma coe_coe [MonoidWithZeroHomClass F α β] (f : F) : ((f : α →*₀ β) : α → β) = f := rfl

-- Completely uninteresting lemmas about coercion to function, that all homs need

/-- `MonoidWithZeroHom` down-cast to a `MonoidHom`, forgetting the 0-preserving property. -/
instance coeToMonoidHom : Coe (α →*₀ β) (α →* β) :=
  ⟨toMonoidHom⟩


/-- `MonoidWithZeroHom` down-cast to a `ZeroHom`, forgetting the monoidal property. -/
instance coeToZeroHom :
  Coe (α →*₀ β) (ZeroHom α β) := ⟨toZeroHom⟩

-- This must come after the coe_toFun definitions

@[simp] lemma coe_mk (f h1 hmul) : (mk f h1 hmul : α → β) = (f : α → β) := rfl


@[simp] lemma toZeroHom_coe (f : α →*₀ β) : (f.toZeroHom : α → β) = f := rfl


lemma toMonoidHom_coe (f : α →*₀ β) : f.toMonoidHom.toFun = f := rfl


@[ext] lemma ext ⦃f g : α →*₀ β⦄ (h : ∀ x, f x = g x) : f = g := DFunLike.ext _ _ h


@[simp] lemma mk_coe (f : α →*₀ β) (h1 hmul) : mk f h1 hmul = f := ext fun _ ↦ rfl


/-- Copy of a `MonoidHom` with a new `toFun` equal to the old one. Useful to fix
definitional equalities. -/
protected def copy (f : α →*₀ β) (f' : α → β) (h : f' = f) : α →* β :=
  { f.toZeroHom.copy f' h, f.toMonoidHom.copy f' h with }


@[simp]
lemma coe_copy (f : α →*₀ β) (f' : α → β) (h) : (f.copy f' h) = f' := rfl


lemma copy_eq (f : α →*₀ β) (f' : α → β) (h) : f.copy f' h = f := DFunLike.ext' h


protected lemma map_one (f : α →*₀ β) : f 1 = 1 := f.map_one'


protected lemma map_zero (f : α →*₀ β) : f 0 = 0 := f.map_zero'


protected lemma map_mul (f : α →*₀ β) (a b : α) : f (a * b) = f a * f b := f.map_mul' a b


/-- The identity map from a `MonoidWithZero` to itself. -/
@[simps]
def id (α : Type*) [MulZeroOneClass α] : α →*₀ α where
  toFun x := x
  map_zero' := rfl
  map_one' := rfl
  map_mul' _ _ := rfl


/-- Composition of `MonoidWithZeroHom`s as a `MonoidWithZeroHom`. -/
def comp (hnp : β →*₀ γ) (hmn : α →*₀ β) : α →*₀ γ where
  toFun := hnp ∘ hmn
                  /-
                    F : Type u_1
                    α : Type u_2
                    β : Type u_3
                    γ : Type u_4
                    δ : Type u_5
                    M₀ : Type u_6
                    inst✝⁴ : MulZeroOneClass α
                    inst✝³ : MulZeroOneClass β
                    inst✝² : MulZeroOneClass γ
                    inst✝¹ : MulZeroOneClass δ
                    inst✝ : FunLike F α β
                    hnp : MonoidWithZeroHom β γ
                    hmn : MonoidWithZeroHom α β
                    ⊢ Eq (Function.comp (⇑hnp) (⇑hmn) 0) 0
                  -/
  map_zero' := by rw [comp_apply, map_zero, map_zero]
                  /-
                    🎉 no goals
                  -/
                 /-
                   F : Type u_1
                   α : Type u_2
                   β : Type u_3
                   γ : Type u_4
                   δ : Type u_5
                   M₀ : Type u_6
                   inst✝⁴ : MulZeroOneClass α
                   inst✝³ : MulZeroOneClass β
                   inst✝² : MulZeroOneClass γ
                   inst✝¹ : MulZeroOneClass δ
                   inst✝ : FunLike F α β
                   hnp : MonoidWithZeroHom β γ
                   hmn : MonoidWithZeroHom α β
                   ⊢ Eq ({ toFun := Function.comp ⇑hnp ⇑hmn, map_zero' := ⋯ }.toFun 1) 1
                 -/
  map_one' := by simp
                 /-
                   🎉 no goals
                 -/
                 /-
                   F : Type u_1
                   α : Type u_2
                   β : Type u_3
                   γ : Type u_4
                   δ : Type u_5
                   M₀ : Type u_6
                   inst✝⁴ : MulZeroOneClass α
                   inst✝³ : MulZeroOneClass β
                   inst✝² : MulZeroOneClass γ
                   inst✝¹ : MulZeroOneClass δ
                   inst✝ : FunLike F α β
                   hnp : MonoidWithZeroHom β γ
                   hmn : MonoidWithZeroHom α β
                   ⊢ ∀ (x y : α), Eq ({ toFun := Function.comp ⇑hnp ⇑hmn, map_zero' := ⋯ }.toFun  …
                 -/
  map_mul' := by simp
                 /-
                   🎉 no goals
                 -/


@[simp] lemma coe_comp (g : β →*₀ γ) (f : α →*₀ β) : ↑(g.comp f) = g ∘ f := rfl


lemma comp_apply (g : β →*₀ γ) (f : α →*₀ β) (x : α) : g.comp f x = g (f x) := rfl


lemma comp_assoc (f : α →*₀ β) (g : β →*₀ γ) (h : γ →*₀ δ) :
    (h.comp g).comp f = h.comp (g.comp f) := rfl


lemma cancel_right {g₁ g₂ : β →*₀ γ} {f : α →*₀ β} (hf : Surjective f) :
    g₁.comp f = g₂.comp f ↔ g₁ = g₂ :=
  ⟨fun h ↦ ext <| hf.forall.2 (DFunLike.ext_iff.1 h), fun h ↦ h ▸ rfl⟩


lemma cancel_left {g : β →*₀ γ} {f₁ f₂ : α →*₀ β} (hg : Injective g) :
    g.comp f₁ = g.comp f₂ ↔ f₁ = f₂ :=
  ⟨fun h ↦ ext fun x ↦ hg <| by rw [← comp_apply, h,
    comp_apply], fun h ↦ h ▸ rfl⟩


lemma toMonoidHom_injective : Injective (toMonoidHom : (α →*₀ β) → α →* β) :=
  Injective.of_comp (f := DFunLike.coe) DFunLike.coe_injective


lemma toZeroHom_injective : Injective (toZeroHom : (α →*₀ β) → ZeroHom α β) :=
  Injective.of_comp (f := DFunLike.coe) DFunLike.coe_injective


@[simp] lemma comp_id (f : α →*₀ β) : f.comp (id α) = f := ext fun _ ↦ rfl


@[simp] lemma id_comp (f : α →*₀ β) : (id β).comp f = f := ext fun _ ↦ rfl

-- Unlike the other homs, `MonoidWithZeroHom` does not have a `1` or `0`

instance : Inhabited (α →*₀ α) := ⟨id α⟩


/-- Given two monoid with zero morphisms `f`, `g` to a commutative monoid with zero, `f * g` is the
monoid with zero morphism sending `x` to `f x * g x`. -/
instance {β} [CommMonoidWithZero β] : Mul (α →*₀ β) where
  mul f g :=
    { (f * g : α →* β) with
                      /-
                        F : Type u_1
                        α : Type u_2
                        β✝ : Type u_3
                        γ : Type u_4
                        δ : Type u_5
                        M₀ : Type u_6
                        inst✝⁵ : MulZeroOneClass α
                        inst✝⁴ : MulZeroOneClass β✝
                        inst✝³ : MulZeroOneClass γ
                        inst✝² : MulZeroOneClass δ
                        inst✝¹ : FunLike F α β✝
                        β : Type ?u.17159
                        inst✝ : CommMonoidWithZero β
                        f g : MonoidWithZeroHom α β
                        ⊢ Eq ((↑__src✝).toFun 0) 0
                      -/
      map_zero' := by dsimp; rw [map_zero, zero_mul] }
                             /-
                               🎉 no goals
                             -/


/-- We define `x ↦ x^n` (for positive `n : ℕ`) as a `MonoidWithZeroHom` -/
def powMonoidWithZeroHom : M₀ →*₀ M₀ :=
  { powMonoidHom n with map_zero' := zero_pow hn }


@[simp] lemma coe_powMonoidWithZeroHom : (powMonoidWithZeroHom hn : M₀ → M₀) = fun x ↦ x ^ n := rfl


@[simp] lemma powMonoidWithZeroHom_apply (a : M₀) : powMonoidWithZeroHom hn a = a ^ n := rfl


instance (priority := 100) toZeroHomClass [MulZeroClass α] [MulZeroClass β] [MulEquivClass F α β] :
    ZeroHomClass F α β where
  map_zero f :=
    calc
                                              /-
                                                F✝ : Type u_1
                                                α✝ : Type u_2
                                                β✝ : Type u_3
                                                γ : Type u_4
                                                δ : Type u_5
                                                M₀ : Type u_6
                                                inst✝⁷ : MulZeroOneClass α✝
                                                inst✝⁶ : MulZeroOneClass β✝
                                                inst✝⁵ : MulZeroOneClass γ
                                                inst✝⁴ : MulZeroOneClass δ
                                                F : Type u_7
                                                α : Type u_8
                                                β : Type u_9
                                                inst✝³ : EquivLike F α β
                                                inst✝² : MulZeroClass α
                                                inst✝¹ : MulZeroClass β
                                                inst✝ : MulEquivClass F α β
                                                f : F
                                                ⊢ Eq (f 0) (HMul.hMul (f 0) (f (EquivLike.inv f 0)))
                                              -/
      f 0 = f 0 * f (EquivLike.inv f 0) := by rw [← map_mul, zero_mul]
                                              /-
                                                🎉 no goals
                                              -/
                    /-
                      F✝ : Type u_1
                      α✝ : Type u_2
                      β✝ : Type u_3
                      γ : Type u_4
                      δ : Type u_5
                      M₀ : Type u_6
                      inst✝⁷ : MulZeroOneClass α✝
                      inst✝⁶ : MulZeroOneClass β✝
                      inst✝⁵ : MulZeroOneClass γ
                      inst✝⁴ : MulZeroOneClass δ
                      F : Type u_7
                      α : Type u_8
                      β : Type u_9
                      inst✝³ : EquivLike F α β
                      inst✝² : MulZeroClass α
                      inst✝¹ : MulZeroClass β
                      inst✝ : MulEquivClass F α β
                      f : F
                      ⊢ Eq (HMul.hMul (f 0) (f (EquivLike.inv f 0))) 0
                    -/
        _ = 0 := by simp
                    /-
                      🎉 no goals
                    -/

-- See note [lower instance priority]

instance (priority := 100) toMonoidWithZeroHomClass
    [MulZeroOneClass α] [MulZeroOneClass β] [MulEquivClass F α β] :
    MonoidWithZeroHomClass F α β :=
  { MulEquivClass.instMonoidHomClass F, MulEquivClass.toZeroHomClass with }


