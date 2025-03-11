/-- The type of pseudo-epimorphisms, aka p-morphisms, aka bounded maps, from `α` to `β`. -/
structure PseudoEpimorphism (α β : Type*) [Preorder α] [Preorder β] extends α →o β where
  exists_map_eq_of_map_le' ⦃a : α⦄ ⦃b : β⦄ : toFun a ≤ b → ∃ c, a ≤ c ∧ toFun c = b


/-- The type of Esakia morphisms, aka continuous pseudo-epimorphisms, from `α` to `β`. -/
structure EsakiaHom (α β : Type*) [TopologicalSpace α] [Preorder α] [TopologicalSpace β]
  [Preorder β] extends α →Co β where
  exists_map_eq_of_map_le' ⦃a : α⦄ ⦃b : β⦄ : toFun a ≤ b → ∃ c, a ≤ c ∧ toFun c = b


/-- `PseudoEpimorphismClass F α β` states that `F` is a type of `⊔`-preserving morphisms.

You should extend this class when you extend `PseudoEpimorphism`. -/
class PseudoEpimorphismClass (F : Type*) (α β : outParam Type*)
    [Preorder α] [Preorder β] [FunLike F α β]
    extends RelHomClass F ((· ≤ ·) : α → α → Prop) ((· ≤ ·) : β → β → Prop) : Prop where
  exists_map_eq_of_map_le (f : F) ⦃a : α⦄ ⦃b : β⦄ : f a ≤ b → ∃ c, a ≤ c ∧ f c = b


/-- `EsakiaHomClass F α β` states that `F` is a type of lattice morphisms.

You should extend this class when you extend `EsakiaHom`. -/
class EsakiaHomClass (F : Type*) (α β : outParam Type*) [TopologicalSpace α] [Preorder α]
    [TopologicalSpace β] [Preorder β] [FunLike F α β]
    extends ContinuousOrderHomClass F α β : Prop where
  exists_map_eq_of_map_le (f : F) ⦃a : α⦄ ⦃b : β⦄ : f a ≤ b → ∃ c, a ≤ c ∧ f c = b


instance (priority := 100) PseudoEpimorphismClass.toTopHomClass [PartialOrder α] [OrderTop α]
    [Preorder β] [OrderTop β] [PseudoEpimorphismClass F α β] : TopHomClass F α β where
  map_top f := by
    /-
      F : Type u_1
      α : Type u_2
      β : Type u_3
      γ : Type u_4
      δ : Type u_5
      inst✝⁵ : FunLike F α β
      inst✝⁴ : PartialOrder α
      inst✝³ : OrderTop α
      inst✝² : Preorder β
      inst✝¹ : OrderTop β
      inst✝ : PseudoEpimorphismClass F α β
      f : F
      ⊢ Eq (f Top.top) Top.top
    -/
    let ⟨b, h⟩ := exists_map_eq_of_map_le f (@le_top _ _ _ <| f ⊤)
    /-
      F : Type u_1
      α : Type u_2
      β : Type u_3
      γ : Type u_4
      δ : Type u_5
      inst✝⁵ : FunLike F α β
      inst✝⁴ : PartialOrder α
      inst✝³ : OrderTop α
      inst✝² : Preorder β
      inst✝¹ : OrderTop β
      inst✝ : PseudoEpimorphismClass F α β
      f : F
      b : α
      h : And (LE.le Top.top b) (Eq (f b) Top.top)
      ⊢ Eq (f Top.top) Top.top
    -/
    rw [← top_le_iff.1 h.1, h.2]
    /-
      🎉 no goals
    -/

-- See note [lower instance priority]

instance (priority := 100) EsakiaHomClass.toPseudoEpimorphismClass [TopologicalSpace α] [Preorder α]
    [TopologicalSpace β] [Preorder β] [EsakiaHomClass F α β] : PseudoEpimorphismClass F α β :=
  { ‹EsakiaHomClass F α β› with
    map_rel := ContinuousOrderHomClass.map_monotone }


instance [Preorder α] [Preorder β] [PseudoEpimorphismClass F α β] :
    CoeTC F (PseudoEpimorphism α β) :=
  ⟨fun f => ⟨f, exists_map_eq_of_map_le f⟩⟩


instance [TopologicalSpace α] [Preorder α] [TopologicalSpace β] [Preorder β]
    [EsakiaHomClass F α β] : CoeTC F (EsakiaHom α β) :=
  ⟨fun f => ⟨f, exists_map_eq_of_map_le f⟩⟩


instance (priority := 100) OrderIsoClass.toPseudoEpimorphismClass [Preorder α] [Preorder β]
    [EquivLike F α β] [OrderIsoClass F α β] : PseudoEpimorphismClass F α β where
  exists_map_eq_of_map_le f _a b h :=
    ⟨EquivLike.inv f b, (le_map_inv_iff f).2 h, EquivLike.right_inv _ _⟩


instance instFunLike : FunLike (PseudoEpimorphism α β) α β where
  coe f := f.toFun
  coe_injective' f g h := by
    /-
      F : Type u_1
      α : Type u_2
      β : Type u_3
      γ : Type u_4
      δ : Type u_5
      inst✝³ : Preorder α
      inst✝² : Preorder β
      inst✝¹ : Preorder γ
      inst✝ : Preorder δ
      f g : PseudoEpimorphism α β
      h : Eq ((fun f => f.toFun) f) ((fun f => f.toFun) g)
      ⊢ Eq f g
    -/
    obtain ⟨⟨_, _⟩, _⟩ := f
    /-
      case mk.mk
      F : Type u_1
      α : Type u_2
      β : Type u_3
      γ : Type u_4
      δ : Type u_5
      inst✝³ : Preorder α
      inst✝² : Preorder β
      inst✝¹ : Preorder γ
      inst✝ : Preorder δ
      g : PseudoEpimorphism α β
      toFun✝ : α → β
      monotone'✝ : Monotone toFun✝
      exists_map_eq_of_map_le'✝ : ∀ ⦃a : α⦄ ⦃b : β⦄, LE.le ({ toFun := toFun✝, monot …
      h : Eq ((fun f => f.toFun) { toFun := toFun✝, monotone' := monotone'✝, exists_ …
      ⊢ Eq { toFun := toFun✝, monotone' := monotone'✝, exists_map_eq_of_map_le' := e …
    -/
    obtain ⟨⟨_, _⟩, _⟩ := g
    /-
      case mk.mk.mk.mk
      F : Type u_1
      α : Type u_2
      β : Type u_3
      γ : Type u_4
      δ : Type u_5
      inst✝³ : Preorder α
      inst✝² : Preorder β
      inst✝¹ : Preorder γ
      inst✝ : Preorder δ
      toFun✝¹ : α → β
      monotone'✝¹ : Monotone toFun✝¹
      exists_map_eq_of_map_le'✝¹ : ∀ ⦃a : α⦄ ⦃b : β⦄, LE.le ({ toFun := toFun✝¹, mon …
      toFun✝ : α → β
      monotone'✝ : Monotone toFun✝
      exists_map_eq_of_map_le'✝ : ∀ ⦃a : α⦄ ⦃b : β⦄, LE.le ({ toFun := toFun✝, monot …
      h : Eq ((fun f => f.toFun) { toFun := toFun✝¹, monotone' := monotone'✝¹, exist …
      ⊢ Eq { toFun := toFun✝¹, monotone' := monotone'✝¹, exists_map_eq_of_map_le' := …
    -/
    congr
    /-
      🎉 no goals
    -/


instance : PseudoEpimorphismClass (PseudoEpimorphism α β) α β where
  map_rel f _ _ h := f.monotone' h
  exists_map_eq_of_map_le := PseudoEpimorphism.exists_map_eq_of_map_le'


@[simp]
theorem toOrderHom_eq_coe (f : PseudoEpimorphism α β) : ⇑f.toOrderHom = f := rfl


theorem toFun_eq_coe {f : PseudoEpimorphism α β} : f.toFun = (f : α → β) := rfl


@[ext]
theorem ext {f g : PseudoEpimorphism α β} (h : ∀ a, f a = g a) : f = g :=
  DFunLike.ext f g h


/-- Copy of a `PseudoEpimorphism` with a new `toFun` equal to the old one. Useful to fix
definitional equalities. -/
protected def copy (f : PseudoEpimorphism α β) (f' : α → β) (h : f' = f) : PseudoEpimorphism α β :=
                              /-
                                F : Type u_1
                                α : Type u_2
                                β : Type u_3
                                γ : Type u_4
                                δ : Type u_5
                                inst✝³ : Preorder α
                                inst✝² : Preorder β
                                inst✝¹ : Preorder γ
                                inst✝ : Preorder δ
                                f : PseudoEpimorphism α β
                                f' : α → β
                                h : Eq f' ⇑f
                                ⊢ ∀ ⦃a : α⦄ ⦃b : β⦄, LE.le ((f.copy f' h).toFun a) b → Exists fun c => And (LE …
                              -/
  ⟨f.toOrderHom.copy f' h, by simpa only [h.symm, toFun_eq_coe] using f.exists_map_eq_of_map_le'⟩
                              /-
                                🎉 no goals
                              -/


@[simp]
theorem coe_copy (f : PseudoEpimorphism α β) (f' : α → β) (h : f' = f) : ⇑(f.copy f' h) = f' := rfl


theorem copy_eq (f : PseudoEpimorphism α β) (f' : α → β) (h : f' = f) : f.copy f' h = f :=
  DFunLike.ext' h


/-- `id` as a `PseudoEpimorphism`. -/
protected def id : PseudoEpimorphism α α :=
  ⟨OrderHom.id, fun _ b h => ⟨b, h, rfl⟩⟩


instance : Inhabited (PseudoEpimorphism α α) :=
  ⟨PseudoEpimorphism.id α⟩


@[simp]
theorem coe_id : ⇑(PseudoEpimorphism.id α) = id := rfl


@[simp]
theorem coe_id_orderHom : (PseudoEpimorphism.id α : α →o α) = OrderHom.id := rfl


@[simp]
theorem id_apply (a : α) : PseudoEpimorphism.id α a = a := rfl


/-- Composition of `PseudoEpimorphism`s as a `PseudoEpimorphism`. -/
def comp (g : PseudoEpimorphism β γ) (f : PseudoEpimorphism α β) : PseudoEpimorphism α γ :=
  ⟨g.toOrderHom.comp f.toOrderHom, fun a b h₀ => by
    /-
      F : Type u_1
      α : Type u_2
      β : Type u_3
      γ : Type u_4
      δ : Type u_5
      inst✝³ : Preorder α
      inst✝² : Preorder β
      inst✝¹ : Preorder γ
      inst✝ : Preorder δ
      g : PseudoEpimorphism β γ
      f : PseudoEpimorphism α β
      a : α
      b : γ
      h₀ : LE.le ((g.comp f.toOrderHom).toFun a) b
      ⊢ Exists fun c => And (LE.le a c) (Eq ((g.comp f.toOrderHom).toFun c) b)
    -/
    obtain ⟨b, h₁, rfl⟩ := g.exists_map_eq_of_map_le' h₀
    /-
      case intro.intro
      F : Type u_1
      α : Type u_2
      β : Type u_3
      γ : Type u_4
      δ : Type u_5
      inst✝³ : Preorder α
      inst✝² : Preorder β
      inst✝¹ : Preorder γ
      inst✝ : Preorder δ
      g : PseudoEpimorphism β γ
      f : PseudoEpimorphism α β
      a : α
      b : β
      h₁ : LE.le (f.toOrderHom a) b
      h₀ : LE.le ((g.comp f.toOrderHom).toFun a) (g.toFun b)
      ⊢ Exists fun c => And (LE.le a c) (Eq ((g.comp f.toOrderHom).toFun c) (g.toFun …
    -/
    obtain ⟨b, h₂, rfl⟩ := f.exists_map_eq_of_map_le' h₁
    /-
      case intro.intro.intro.intro
      F : Type u_1
      α : Type u_2
      β : Type u_3
      γ : Type u_4
      δ : Type u_5
      inst✝³ : Preorder α
      inst✝² : Preorder β
      inst✝¹ : Preorder γ
      inst✝ : Preorder δ
      g : PseudoEpimorphism β γ
      f : PseudoEpimorphism α β
      a b : α
      h₂ : LE.le a b
      h₁ : LE.le (f.toOrderHom a) (f.toFun b)
      h₀ : LE.le ((g.comp f.toOrderHom).toFun a) (g.toFun (f.toFun b))
      ⊢ Exists fun c => And (LE.le a c) (Eq ((g.comp f.toOrderHom).toFun c) (g.toFun …
    -/
    exact ⟨b, h₂, rfl⟩⟩
    /-
      🎉 no goals
    -/


@[simp]
theorem coe_comp (g : PseudoEpimorphism β γ) (f : PseudoEpimorphism α β) :
    (g.comp f : α → γ) = g ∘ f := rfl


@[simp]
theorem coe_comp_orderHom (g : PseudoEpimorphism β γ) (f : PseudoEpimorphism α β) :
    (g.comp f : α →o γ) = (g : β →o γ).comp f := rfl


@[simp]
theorem comp_apply (g : PseudoEpimorphism β γ) (f : PseudoEpimorphism α β) (a : α) :
    (g.comp f) a = g (f a) := rfl


@[simp]
theorem comp_assoc (h : PseudoEpimorphism γ δ) (g : PseudoEpimorphism β γ)
    (f : PseudoEpimorphism α β) : (h.comp g).comp f = h.comp (g.comp f) := rfl


@[simp]
theorem comp_id (f : PseudoEpimorphism α β) : f.comp (PseudoEpimorphism.id α) = f :=
  ext fun _ => rfl


@[simp]
theorem id_comp (f : PseudoEpimorphism α β) : (PseudoEpimorphism.id β).comp f = f :=
  ext fun _ => rfl


@[simp]
theorem cancel_right {g₁ g₂ : PseudoEpimorphism β γ} {f : PseudoEpimorphism α β}
    (hf : Surjective f) : g₁.comp f = g₂.comp f ↔ g₁ = g₂ :=
  ⟨fun h => ext <| hf.forall.2 <| DFunLike.ext_iff.1 h, congr_arg (comp · f)⟩


@[simp]
theorem cancel_left {g : PseudoEpimorphism β γ} {f₁ f₂ : PseudoEpimorphism α β} (hg : Injective g) :
    g.comp f₁ = g.comp f₂ ↔ f₁ = f₂ :=
                                  /-
                                    α : Type u_2
                                    β : Type u_3
                                    γ : Type u_4
                                    inst✝² : Preorder α
                                    inst✝¹ : Preorder β
                                    inst✝ : Preorder γ
                                    g : PseudoEpimorphism β γ
                                    f₁ f₂ : PseudoEpimorphism α β
                                    hg : Function.Injective ⇑g
                                    h : Eq (g.comp f₁) (g.comp f₂)
                                    a : α
                                    ⊢ Eq (g (f₁ a)) (g (f₂ a))
                                  -/
  ⟨fun h => ext fun a => hg <| by rw [← comp_apply, h, comp_apply], congr_arg _⟩
                                  /-
                                    🎉 no goals
                                  -/


def toPseudoEpimorphism (f : EsakiaHom α β) : PseudoEpimorphism α β :=
  { f with }


instance instFunLike : FunLike (EsakiaHom α β) α β where
  coe f := f.toFun
  coe_injective' f g h := by
    /-
      F : Type u_1
      α : Type u_2
      β : Type u_3
      γ : Type u_4
      δ : Type u_5
      inst✝⁷ : TopologicalSpace α
      inst✝⁶ : Preorder α
      inst✝⁵ : TopologicalSpace β
      inst✝⁴ : Preorder β
      inst✝³ : TopologicalSpace γ
      inst✝² : Preorder γ
      inst✝¹ : TopologicalSpace δ
      inst✝ : Preorder δ
      f g : EsakiaHom α β
      h : Eq ((fun f => f.toFun) f) ((fun f => f.toFun) g)
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
      inst✝⁷ : TopologicalSpace α
      inst✝⁶ : Preorder α
      inst✝⁵ : TopologicalSpace β
      inst✝⁴ : Preorder β
      inst✝³ : TopologicalSpace γ
      inst✝² : Preorder γ
      inst✝¹ : TopologicalSpace δ
      inst✝ : Preorder δ
      g : EsakiaHom α β
      toFun✝ : α → β
      monotone'✝ : Monotone toFun✝
      continuous_toFun✝ : Continuous { toFun := toFun✝, monotone' := monotone'✝ }.to …
      exists_map_eq_of_map_le'✝ : ∀ ⦃a : α⦄ ⦃b : β⦄, LE.le ({ toFun := toFun✝, monot …
      h : Eq ((fun f => f.toFun) { toFun := toFun✝, monotone' := monotone'✝, continu …
      ⊢ Eq { toFun := toFun✝, monotone' := monotone'✝, continuous_toFun := continuou …
    -/
    obtain ⟨⟨⟨_, _⟩, _⟩, _⟩ := g
    /-
      case mk.mk.mk.mk.mk.mk
      F : Type u_1
      α : Type u_2
      β : Type u_3
      γ : Type u_4
      δ : Type u_5
      inst✝⁷ : TopologicalSpace α
      inst✝⁶ : Preorder α
      inst✝⁵ : TopologicalSpace β
      inst✝⁴ : Preorder β
      inst✝³ : TopologicalSpace γ
      inst✝² : Preorder γ
      inst✝¹ : TopologicalSpace δ
      inst✝ : Preorder δ
      toFun✝¹ : α → β
      monotone'✝¹ : Monotone toFun✝¹
      continuous_toFun✝¹ : Continuous { toFun := toFun✝¹, monotone' := monotone'✝¹ } …
      exists_map_eq_of_map_le'✝¹ : ∀ ⦃a : α⦄ ⦃b : β⦄, LE.le ({ toFun := toFun✝¹, mon …
      toFun✝ : α → β
      monotone'✝ : Monotone toFun✝
      continuous_toFun✝ : Continuous { toFun := toFun✝, monotone' := monotone'✝ }.to …
      exists_map_eq_of_map_le'✝ : ∀ ⦃a : α⦄ ⦃b : β⦄, LE.le ({ toFun := toFun✝, monot …
      h : Eq ((fun f => f.toFun) { toFun := toFun✝¹, monotone' := monotone'✝¹, conti …
      ⊢ Eq { toFun := toFun✝¹, monotone' := monotone'✝¹, continuous_toFun := continu …
    -/
    congr
    /-
      🎉 no goals
    -/


instance : EsakiaHomClass (EsakiaHom α β) α β where
  map_monotone f := f.monotone'
  map_continuous f := f.continuous_toFun
  exists_map_eq_of_map_le f := f.exists_map_eq_of_map_le'


@[simp]
theorem toContinuousOrderHom_coe {f : EsakiaHom α β} :
    f.toContinuousOrderHom = (f : α → β) := rfl


theorem toFun_eq_coe {f : EsakiaHom α β} : f.toFun = (f : α → β) := rfl


@[ext]
theorem ext {f g : EsakiaHom α β} (h : ∀ a, f a = g a) : f = g :=
  DFunLike.ext f g h


/-- Copy of an `EsakiaHom` with a new `toFun` equal to the old one. Useful to fix definitional
equalities. -/
protected def copy (f : EsakiaHom α β) (f' : α → β) (h : f' = f) : EsakiaHom α β :=
  ⟨f.toContinuousOrderHom.copy f' h, by
    /-
      F : Type u_1
      α : Type u_2
      β : Type u_3
      γ : Type u_4
      δ : Type u_5
      inst✝⁷ : TopologicalSpace α
      inst✝⁶ : Preorder α
      inst✝⁵ : TopologicalSpace β
      inst✝⁴ : Preorder β
      inst✝³ : TopologicalSpace γ
      inst✝² : Preorder γ
      inst✝¹ : TopologicalSpace δ
      inst✝ : Preorder δ
      f : EsakiaHom α β
      f' : α → β
      h : Eq f' ⇑f
      ⊢ ∀ ⦃a : α⦄ ⦃b : β⦄, LE.le ((f.copy f' h).toFun a) b → Exists fun c => And (LE …
    -/
    simpa only [h.symm, toFun_eq_coe] using f.exists_map_eq_of_map_le'⟩
    /-
      🎉 no goals
    -/


@[simp]
theorem coe_copy (f : EsakiaHom α β) (f' : α → β) (h : f' = f) : ⇑(f.copy f' h) = f' := rfl


theorem copy_eq (f : EsakiaHom α β) (f' : α → β) (h : f' = f) : f.copy f' h = f :=
  DFunLike.ext' h


/-- `id` as an `EsakiaHom`. -/
protected def id : EsakiaHom α α :=
  ⟨ContinuousOrderHom.id α, fun _ b h => ⟨b, h, rfl⟩⟩


instance : Inhabited (EsakiaHom α α) :=
  ⟨EsakiaHom.id α⟩


@[simp]
theorem coe_id : ⇑(EsakiaHom.id α) = id := rfl


@[simp]
theorem coe_id_pseudoEpimorphism :
    (EsakiaHom.id α : PseudoEpimorphism α α) = PseudoEpimorphism.id α := rfl


@[simp]
theorem id_apply (a : α) : EsakiaHom.id α a = a := rfl


@[simp]
theorem coe_id_continuousOrderHom : (EsakiaHom.id α : α →Co α) = ContinuousOrderHom.id α := rfl


/-- Composition of `EsakiaHom`s as an `EsakiaHom`. -/
def comp (g : EsakiaHom β γ) (f : EsakiaHom α β) : EsakiaHom α γ :=
  ⟨g.toContinuousOrderHom.comp f.toContinuousOrderHom, fun a b h₀ => by
    /-
      F : Type u_1
      α : Type u_2
      β : Type u_3
      γ : Type u_4
      δ : Type u_5
      inst✝⁷ : TopologicalSpace α
      inst✝⁶ : Preorder α
      inst✝⁵ : TopologicalSpace β
      inst✝⁴ : Preorder β
      inst✝³ : TopologicalSpace γ
      inst✝² : Preorder γ
      inst✝¹ : TopologicalSpace δ
      inst✝ : Preorder δ
      g : EsakiaHom β γ
      f : EsakiaHom α β
      a : α
      b : γ
      h₀ : LE.le ((g.comp f.toContinuousOrderHom).toFun a) b
      ⊢ Exists fun c => And (LE.le a c) (Eq ((g.comp f.toContinuousOrderHom).toFun c …
    -/
    obtain ⟨b, h₁, rfl⟩ := g.exists_map_eq_of_map_le' h₀
    /-
      case intro.intro
      F : Type u_1
      α : Type u_2
      β : Type u_3
      γ : Type u_4
      δ : Type u_5
      inst✝⁷ : TopologicalSpace α
      inst✝⁶ : Preorder α
      inst✝⁵ : TopologicalSpace β
      inst✝⁴ : Preorder β
      inst✝³ : TopologicalSpace γ
      inst✝² : Preorder γ
      inst✝¹ : TopologicalSpace δ
      inst✝ : Preorder δ
      g : EsakiaHom β γ
      f : EsakiaHom α β
      a : α
      b : β
      h₁ : LE.le (f.toOrderHom a) b
      h₀ : LE.le ((g.comp f.toContinuousOrderHom).toFun a) (g.toFun b)
      ⊢ Exists fun c => And (LE.le a c) (Eq ((g.comp f.toContinuousOrderHom).toFun c …
    -/
    obtain ⟨b, h₂, rfl⟩ := f.exists_map_eq_of_map_le' h₁
    /-
      case intro.intro.intro.intro
      F : Type u_1
      α : Type u_2
      β : Type u_3
      γ : Type u_4
      δ : Type u_5
      inst✝⁷ : TopologicalSpace α
      inst✝⁶ : Preorder α
      inst✝⁵ : TopologicalSpace β
      inst✝⁴ : Preorder β
      inst✝³ : TopologicalSpace γ
      inst✝² : Preorder γ
      inst✝¹ : TopologicalSpace δ
      inst✝ : Preorder δ
      g : EsakiaHom β γ
      f : EsakiaHom α β
      a b : α
      h₂ : LE.le a b
      h₁ : LE.le (f.toOrderHom a) (f.toFun b)
      h₀ : LE.le ((g.comp f.toContinuousOrderHom).toFun a) (g.toFun (f.toFun b))
      ⊢ Exists fun c => And (LE.le a c) (Eq ((g.comp f.toContinuousOrderHom).toFun c …
    -/
    exact ⟨b, h₂, rfl⟩⟩
    /-
      🎉 no goals
    -/


@[simp]
theorem coe_comp_continuousOrderHom (g : EsakiaHom β γ) (f : EsakiaHom α β) :
    (g.comp f : α →Co γ) = (g : β →Co γ).comp f := rfl


@[simp]
theorem coe_comp_pseudoEpimorphism (g : EsakiaHom β γ) (f : EsakiaHom α β) :
    (g.comp f : PseudoEpimorphism α γ) = (g : PseudoEpimorphism β γ).comp f := rfl


@[simp]
theorem coe_comp (g : EsakiaHom β γ) (f : EsakiaHom α β) : (g.comp f : α → γ) = g ∘ f := rfl


@[simp]
theorem comp_apply (g : EsakiaHom β γ) (f : EsakiaHom α β) (a : α) : (g.comp f) a = g (f a) := rfl


@[simp]
theorem comp_assoc (h : EsakiaHom γ δ) (g : EsakiaHom β γ) (f : EsakiaHom α β) :
    (h.comp g).comp f = h.comp (g.comp f) := rfl


@[simp]
theorem comp_id (f : EsakiaHom α β) : f.comp (EsakiaHom.id α) = f :=
  ext fun _ => rfl


@[simp]
theorem id_comp (f : EsakiaHom α β) : (EsakiaHom.id β).comp f = f :=
  ext fun _ => rfl


@[simp]
theorem cancel_right {g₁ g₂ : EsakiaHom β γ} {f : EsakiaHom α β} (hf : Surjective f) :
    g₁.comp f = g₂.comp f ↔ g₁ = g₂ :=
  ⟨fun h => ext <| hf.forall.2 <| DFunLike.ext_iff.1 h, congr_arg (comp · f)⟩


@[simp]
theorem cancel_left {g : EsakiaHom β γ} {f₁ f₂ : EsakiaHom α β} (hg : Injective g) :
    g.comp f₁ = g.comp f₂ ↔ f₁ = f₂ :=
                                  /-
                                    α : Type u_2
                                    β : Type u_3
                                    γ : Type u_4
                                    inst✝⁵ : TopologicalSpace α
                                    inst✝⁴ : Preorder α
                                    inst✝³ : TopologicalSpace β
                                    inst✝² : Preorder β
                                    inst✝¹ : TopologicalSpace γ
                                    inst✝ : Preorder γ
                                    g : EsakiaHom β γ
                                    f₁ f₂ : EsakiaHom α β
                                    hg : Function.Injective ⇑g
                                    h : Eq (g.comp f₁) (g.comp f₂)
                                    a : α
                                    ⊢ Eq (g (f₁ a)) (g (f₂ a))
                                  -/
  ⟨fun h => ext fun a => hg <| by rw [← comp_apply, h, comp_apply], congr_arg _⟩
                                  /-
                                    🎉 no goals
                                  -/


