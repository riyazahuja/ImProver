instance small_subtype (α : Type v) [Small.{w} α] (P : α → Prop) : Small.{w} { x // P x } :=
  small_map (equivShrink α).subtypeEquivOfSubtype'


theorem small_of_injective {α : Type v} {β : Type w} [Small.{u} β] {f : α → β}
    (hf : Function.Injective f) : Small.{u} α :=
  small_map (Equiv.ofInjective f hf)


theorem small_of_surjective {α : Type v} {β : Type w} [Small.{u} α] {f : α → β}
    (hf : Function.Surjective f) : Small.{u} β :=
  small_of_injective (Function.injective_surjInv hf)


instance (priority := 100) small_subsingleton (α : Type v) [Subsingleton α] : Small.{w} α := by
  /-
    α : Type v
    inst✝ : Subsingleton α
    ⊢ Small.{w, v} α
  -/
  rcases isEmpty_or_nonempty α with ⟨⟩
    /-
      case inl
      α : Type v
      inst✝ : Subsingleton α
      h✝ : IsEmpty α
      ⊢ Small.{w, v} α
    -/
  · apply small_map (Equiv.equivPEmpty α)
    /-
      🎉 no goals
    -/
    /-
      case inr
      α : Type v
      inst✝ : Subsingleton α
      h✝ : Nonempty α
      ⊢ Small.{w, v} α
    -/
  · apply small_map Equiv.punitOfNonemptyOfSubsingleton
    /-
      🎉 no goals
    -/


/-- This can be seen as a version of `small_of_surjective` in which the function `f` doesn't
    actually land in `β` but in some larger type `γ` related to `β` via an injective function `g`.
    -/
theorem small_of_injective_of_exists {α : Type v} {β : Type w} {γ : Type v'} [Small.{u} α]
    (f : α → γ) {g : β → γ} (hg : Function.Injective g) (h : ∀ b : β, ∃ a : α, f a = g b) :
    Small.{u} β := by
  /-
    α : Type v
    β : Type w
    γ : Type v'
    inst✝ : Small.{u, v} α
    f : α → γ
    g : β → γ
    hg : Function.Injective g
    h : ∀ (b : β), Exists fun a => Eq (f a) (g b)
    ⊢ Small.{u, w} β
  -/
  by_cases hβ : Nonempty β
    /-
      case pos
      α : Type v
      β : Type w
      γ : Type v'
      inst✝ : Small.{u, v} α
      f : α → γ
      g : β → γ
      hg : Function.Injective g
      h : ∀ (b : β), Exists fun a => Eq (f a) (g b)
      hβ : Nonempty β
      ⊢ Small.{u, w} β
    -/
  · refine small_of_surjective (f := Function.invFun g ∘ f) (fun b => ?_)
    /-
      case pos
      α : Type v
      β : Type w
      γ : Type v'
      inst✝ : Small.{u, v} α
      f : α → γ
      g : β → γ
      hg : Function.Injective g
      h : ∀ (b : β), Exists fun a => Eq (f a) (g b)
      hβ : Nonempty β
      b : β
      ⊢ Exists fun a => Eq (Function.comp (Function.invFun g) f a) b
    -/
    obtain ⟨a, ha⟩ := h b
    /-
      case pos.intro
      α : Type v
      β : Type w
      γ : Type v'
      inst✝ : Small.{u, v} α
      f : α → γ
      g : β → γ
      hg : Function.Injective g
      h : ∀ (b : β), Exists fun a => Eq (f a) (g b)
      hβ : Nonempty β
      b : β
      a : α
      ha : Eq (f a) (g b)
      ⊢ Exists fun a => Eq (Function.comp (Function.invFun g) f a) b
    -/
    exact ⟨a, by rw [Function.comp_apply, ha, Function.leftInverse_invFun hg]⟩
    /-
      🎉 no goals
    -/
    /-
      case neg
      α : Type v
      β : Type w
      γ : Type v'
      inst✝ : Small.{u, v} α
      f : α → γ
      g : β → γ
      hg : Function.Injective g
      h : ∀ (b : β), Exists fun a => Eq (f a) (g b)
      hβ : Not (Nonempty β)
      ⊢ Small.{u, w} β
    -/
  · simp only [not_nonempty_iff] at hβ
    /-
      case neg
      α : Type v
      β : Type w
      γ : Type v'
      inst✝ : Small.{u, v} α
      f : α → γ
      g : β → γ
      hg : Function.Injective g
      h : ∀ (b : β), Exists fun a => Eq (f a) (g b)
      hβ : IsEmpty β
      ⊢ Small.{u, w} β
    -/
    infer_instance
    /-
      🎉 no goals
    -/


instance small_Pi {α} (β : α → Type*) [Small.{w} α] [∀ a, Small.{w} (β a)] :
    Small.{w} (∀ a, β a) :=
  ⟨⟨∀ a' : Shrink α, Shrink (β ((equivShrink α).symm a')),
                                                 /-
                                                   α : Type u_2
                                                   β : α → Type u_1
                                                   inst✝¹ : Small.{w, u_2} α
                                                   inst✝ : ∀ (a : α), Small.{w, u_1} (β a)
                                                   a : α
                                                   ⊢ Equiv (β a) (Shrink.{w, u_1} (β ((equivShrink α).symm ((equivShrink α) a))))
                                                 -/
      ⟨Equiv.piCongr (equivShrink α) fun a => by simpa using equivShrink (β a)⟩⟩⟩
                                                 /-
                                                   🎉 no goals
                                                 -/


instance small_prod {α β} [Small.{w} α] [Small.{w} β] : Small.{w} (α × β) :=
  ⟨⟨Shrink α × Shrink β, ⟨Equiv.prodCongr (equivShrink α) (equivShrink β)⟩⟩⟩


instance small_sum {α β} [Small.{w} α] [Small.{w} β] : Small.{w} (α ⊕ β) :=
  ⟨⟨Shrink α ⊕ Shrink β, ⟨Equiv.sumCongr (equivShrink α) (equivShrink β)⟩⟩⟩


instance small_set {α} [Small.{w} α] : Small.{w} (Set α) :=
  ⟨⟨Set (Shrink α), ⟨Equiv.Set.congr (equivShrink α)⟩⟩⟩

