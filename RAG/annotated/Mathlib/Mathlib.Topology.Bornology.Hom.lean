/-- The type of bounded maps from `α` to `β`, the maps which send a bounded set to a bounded set. -/
structure LocallyBoundedMap (α β : Type*) [Bornology α] [Bornology β] where
  /-- The function underlying a locally bounded map -/
  toFun : α → β
  /-- The pullback of the `Bornology.cobounded` filter under the function is contained in the
  cobounded filter. Equivalently, the function maps bounded sets to bounded sets. -/
  comap_cobounded_le' : (cobounded β).comap toFun ≤ cobounded α


/-- `LocallyBoundedMapClass F α β` states that `F` is a type of bounded maps.

You should extend this class when you extend `LocallyBoundedMap`. -/
class LocallyBoundedMapClass (F : Type*) (α β : outParam Type*) [Bornology α]
    [Bornology β] [FunLike F α β] : Prop where
  /-- The pullback of the `Bornology.cobounded` filter under the function is contained in the
  cobounded filter. Equivalently, the function maps bounded sets to bounded sets. -/
  comap_cobounded_le (f : F) : (cobounded β).comap f ≤ cobounded α


theorem Bornology.IsBounded.image [Bornology α] [Bornology β] [LocallyBoundedMapClass F α β] (f : F)
    {s : Set α} (hs : IsBounded s) : IsBounded (f '' s) :=
  comap_cobounded_le_iff.1 (comap_cobounded_le f) hs


/-- Turn an element of a type `F` satisfying `LocallyBoundedMapClass F α β` into an actual
`LocallyBoundedMap`. This is declared as the default coercion from `F` to
`LocallyBoundedMap α β`. -/
@[coe]
def LocallyBoundedMapClass.toLocallyBoundedMap [Bornology α] [Bornology β]
    [LocallyBoundedMapClass F α β] (f : F) : LocallyBoundedMap α β where
  toFun := f
  comap_cobounded_le' := comap_cobounded_le f


instance [Bornology α] [Bornology β] [LocallyBoundedMapClass F α β] :
    CoeTC F (LocallyBoundedMap α β) :=
  ⟨fun f => ⟨f, comap_cobounded_le f⟩⟩


instance : FunLike (LocallyBoundedMap α β) α β where
  coe f := f.toFun
  coe_injective' f g h := by
    /-
      F : Type u_1
      α : Type u_2
      β : Type u_3
      γ : Type u_4
      δ : Type u_5
      inst✝⁴ : FunLike F α β
      inst✝³ : Bornology α
      inst✝² : Bornology β
      inst✝¹ : Bornology γ
      inst✝ : Bornology δ
      f g : LocallyBoundedMap α β
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
      δ : Type u_5
      inst✝⁴ : FunLike F α β
      inst✝³ : Bornology α
      inst✝² : Bornology β
      inst✝¹ : Bornology γ
      inst✝ : Bornology δ
      g : LocallyBoundedMap α β
      toFun✝ : α → β
      comap_cobounded_le'✝ : LE.le (Filter.comap toFun✝ (Bornology.cobounded β)) (Bo …
      h : Eq ((fun f => f.toFun) { toFun := toFun✝, comap_cobounded_le' := comap_cob …
      ⊢ Eq { toFun := toFun✝, comap_cobounded_le' := comap_cobounded_le'✝ } g
    -/
    cases g
    /-
      case mk.mk
      F : Type u_1
      α : Type u_2
      β : Type u_3
      γ : Type u_4
      δ : Type u_5
      inst✝⁴ : FunLike F α β
      inst✝³ : Bornology α
      inst✝² : Bornology β
      inst✝¹ : Bornology γ
      inst✝ : Bornology δ
      toFun✝¹ : α → β
      comap_cobounded_le'✝¹ : LE.le (Filter.comap toFun✝¹ (Bornology.cobounded β)) ( …
      toFun✝ : α → β
      comap_cobounded_le'✝ : LE.le (Filter.comap toFun✝ (Bornology.cobounded β)) (Bo …
      h : Eq ((fun f => f.toFun) { toFun := toFun✝¹, comap_cobounded_le' := comap_co …
      ⊢ Eq { toFun := toFun✝¹, comap_cobounded_le' := comap_cobounded_le'✝¹ } { toFu …
    -/
    congr
    /-
      🎉 no goals
    -/


instance : LocallyBoundedMapClass (LocallyBoundedMap α β) α β where
  comap_cobounded_le f := f.comap_cobounded_le'


@[ext]
theorem ext {f g : LocallyBoundedMap α β} (h : ∀ a, f a = g a) : f = g :=
  DFunLike.ext f g h


/-- Copy of a `LocallyBoundedMap` with a new `toFun` equal to the old one. Useful to fix
definitional equalities. -/
protected def copy (f : LocallyBoundedMap α β) (f' : α → β) (h : f' = f) : LocallyBoundedMap α β :=
  ⟨f', h.symm ▸ f.comap_cobounded_le'⟩


@[simp]
theorem coe_copy (f : LocallyBoundedMap α β) (f' : α → β) (h : f' = f) : ⇑(f.copy f' h) = f' :=
  rfl


theorem copy_eq (f : LocallyBoundedMap α β) (f' : α → β) (h : f' = f) : f.copy f' h = f :=
  DFunLike.ext' h


/-- Construct a `LocallyBoundedMap` from the fact that the function maps bounded sets to bounded
sets. -/
def ofMapBounded (f : α → β) (h : ∀ ⦃s : Set α⦄, IsBounded s → IsBounded (f '' s)) :
    LocallyBoundedMap α β :=
  ⟨f, comap_cobounded_le_iff.2 h⟩
-- Porting note: I had to provide the type of `h` explicitly.


@[simp]
theorem coe_ofMapBounded (f : α → β) {h} : ⇑(ofMapBounded f h) = f :=
  rfl


@[simp]
theorem ofMapBounded_apply (f : α → β) {h} (a : α) : ofMapBounded f h a = f a :=
  rfl


/-- `id` as a `LocallyBoundedMap`. -/
protected def id : LocallyBoundedMap α α :=
  ⟨id, comap_id.le⟩


instance : Inhabited (LocallyBoundedMap α α) :=
  ⟨LocallyBoundedMap.id α⟩


@[simp]
theorem coe_id : ⇑(LocallyBoundedMap.id α) = id :=
  rfl


@[simp]
theorem id_apply (a : α) : LocallyBoundedMap.id α a = a :=
  rfl


/-- Composition of `LocallyBoundedMap`s as a `LocallyBoundedMap`. -/
def comp (f : LocallyBoundedMap β γ) (g : LocallyBoundedMap α β) : LocallyBoundedMap α γ where
  toFun := f ∘ g
  comap_cobounded_le' :=
    comap_comap.ge.trans <| (comap_mono f.comap_cobounded_le').trans g.comap_cobounded_le'


@[simp]
theorem coe_comp (f : LocallyBoundedMap β γ) (g : LocallyBoundedMap α β) : ⇑(f.comp g) = f ∘ g :=
  rfl


@[simp]
theorem comp_apply (f : LocallyBoundedMap β γ) (g : LocallyBoundedMap α β) (a : α) :
    f.comp g a = f (g a) :=
  rfl


@[simp]
theorem comp_assoc (f : LocallyBoundedMap γ δ) (g : LocallyBoundedMap β γ)
    (h : LocallyBoundedMap α β) : (f.comp g).comp h = f.comp (g.comp h) :=
  rfl


@[simp]
theorem comp_id (f : LocallyBoundedMap α β) : f.comp (LocallyBoundedMap.id α) = f :=
  ext fun _ => rfl


@[simp]
theorem id_comp (f : LocallyBoundedMap α β) : (LocallyBoundedMap.id β).comp f = f :=
  ext fun _ => rfl


@[simp]
theorem cancel_right {g₁ g₂ : LocallyBoundedMap β γ} {f : LocallyBoundedMap α β}
    (hf : Surjective f) : g₁.comp f = g₂.comp f ↔ g₁ = g₂ :=
  ⟨fun h => ext <| hf.forall.2 <| DFunLike.ext_iff.1 h, congrArg (comp · _)⟩


@[simp]
theorem cancel_left {g : LocallyBoundedMap β γ} {f₁ f₂ : LocallyBoundedMap α β} (hg : Injective g) :
    g.comp f₁ = g.comp f₂ ↔ f₁ = f₂ :=
                                  /-
                                    α : Type u_2
                                    β : Type u_3
                                    γ : Type u_4
                                    inst✝² : Bornology α
                                    inst✝¹ : Bornology β
                                    inst✝ : Bornology γ
                                    g : LocallyBoundedMap β γ
                                    f₁ f₂ : LocallyBoundedMap α β
                                    hg : Function.Injective ⇑g
                                    h : Eq (g.comp f₁) (g.comp f₂)
                                    a : α
                                    ⊢ Eq (g (f₁ a)) (g (f₂ a))
                                  -/
  ⟨fun h => ext fun a => hg <| by rw [← comp_apply, h, comp_apply], congr_arg _⟩
                                  /-
                                    🎉 no goals
                                  -/


