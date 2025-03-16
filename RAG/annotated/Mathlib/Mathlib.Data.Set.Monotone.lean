theorem _root_.MonotoneOn.congr (h₁ : MonotoneOn f₁ s) (h : s.EqOn f₁ f₂) : MonotoneOn f₂ s := by
  /-
    α : Type u_1
    β : Type u_2
    s : Set α
    f₁ f₂ : α → β
    inst✝¹ : Preorder α
    inst✝ : Preorder β
    h₁ : MonotoneOn f₁ s
    h : Set.EqOn f₁ f₂ s
    ⊢ MonotoneOn f₂ s
  -/
  intro a ha b hb hab
  /-
    α : Type u_1
    β : Type u_2
    s : Set α
    f₁ f₂ : α → β
    inst✝¹ : Preorder α
    inst✝ : Preorder β
    h₁ : MonotoneOn f₁ s
    h : Set.EqOn f₁ f₂ s
    a : α
    ha : Membership.mem s a
    b : α
    hb : Membership.mem s b
    hab : LE.le a b
    ⊢ LE.le (f₂ a) (f₂ b)
  -/
  rw [← h ha, ← h hb]
  /-
    α : Type u_1
    β : Type u_2
    s : Set α
    f₁ f₂ : α → β
    inst✝¹ : Preorder α
    inst✝ : Preorder β
    h₁ : MonotoneOn f₁ s
    h : Set.EqOn f₁ f₂ s
    a : α
    ha : Membership.mem s a
    b : α
    hb : Membership.mem s b
    hab : LE.le a b
    ⊢ LE.le (f₁ a) (f₁ b)
  -/
  exact h₁ ha hb hab
  /-
    🎉 no goals
  -/


theorem _root_.AntitoneOn.congr (h₁ : AntitoneOn f₁ s) (h : s.EqOn f₁ f₂) : AntitoneOn f₂ s :=
  h₁.dual_right.congr h


theorem _root_.StrictMonoOn.congr (h₁ : StrictMonoOn f₁ s) (h : s.EqOn f₁ f₂) :
    StrictMonoOn f₂ s := by
  /-
    α : Type u_1
    β : Type u_2
    s : Set α
    f₁ f₂ : α → β
    inst✝¹ : Preorder α
    inst✝ : Preorder β
    h₁ : StrictMonoOn f₁ s
    h : Set.EqOn f₁ f₂ s
    ⊢ StrictMonoOn f₂ s
  -/
  intro a ha b hb hab
  /-
    α : Type u_1
    β : Type u_2
    s : Set α
    f₁ f₂ : α → β
    inst✝¹ : Preorder α
    inst✝ : Preorder β
    h₁ : StrictMonoOn f₁ s
    h : Set.EqOn f₁ f₂ s
    a : α
    ha : Membership.mem s a
    b : α
    hb : Membership.mem s b
    hab : LT.lt a b
    ⊢ LT.lt (f₂ a) (f₂ b)
  -/
  rw [← h ha, ← h hb]
  /-
    α : Type u_1
    β : Type u_2
    s : Set α
    f₁ f₂ : α → β
    inst✝¹ : Preorder α
    inst✝ : Preorder β
    h₁ : StrictMonoOn f₁ s
    h : Set.EqOn f₁ f₂ s
    a : α
    ha : Membership.mem s a
    b : α
    hb : Membership.mem s b
    hab : LT.lt a b
    ⊢ LT.lt (f₁ a) (f₁ b)
  -/
  exact h₁ ha hb hab
  /-
    🎉 no goals
  -/


theorem _root_.StrictAntiOn.congr (h₁ : StrictAntiOn f₁ s) (h : s.EqOn f₁ f₂) : StrictAntiOn f₂ s :=
  h₁.dual_right.congr h


theorem EqOn.congr_monotoneOn (h : s.EqOn f₁ f₂) : MonotoneOn f₁ s ↔ MonotoneOn f₂ s :=
  ⟨fun h₁ => h₁.congr h, fun h₂ => h₂.congr h.symm⟩


theorem EqOn.congr_antitoneOn (h : s.EqOn f₁ f₂) : AntitoneOn f₁ s ↔ AntitoneOn f₂ s :=
  ⟨fun h₁ => h₁.congr h, fun h₂ => h₂.congr h.symm⟩


theorem EqOn.congr_strictMonoOn (h : s.EqOn f₁ f₂) : StrictMonoOn f₁ s ↔ StrictMonoOn f₂ s :=
  ⟨fun h₁ => h₁.congr h, fun h₂ => h₂.congr h.symm⟩


theorem EqOn.congr_strictAntiOn (h : s.EqOn f₁ f₂) : StrictAntiOn f₁ s ↔ StrictAntiOn f₂ s :=
  ⟨fun h₁ => h₁.congr h, fun h₂ => h₂.congr h.symm⟩


theorem _root_.MonotoneOn.mono (h : MonotoneOn f s) (h' : s₂ ⊆ s) : MonotoneOn f s₂ :=
  fun _ hx _ hy => h (h' hx) (h' hy)


theorem _root_.AntitoneOn.mono (h : AntitoneOn f s) (h' : s₂ ⊆ s) : AntitoneOn f s₂ :=
  fun _ hx _ hy => h (h' hx) (h' hy)


theorem _root_.StrictMonoOn.mono (h : StrictMonoOn f s) (h' : s₂ ⊆ s) : StrictMonoOn f s₂ :=
  fun _ hx _ hy => h (h' hx) (h' hy)


theorem _root_.StrictAntiOn.mono (h : StrictAntiOn f s) (h' : s₂ ⊆ s) : StrictAntiOn f s₂ :=
  fun _ hx _ hy => h (h' hx) (h' hy)


protected theorem _root_.MonotoneOn.monotone (h : MonotoneOn f s) :
    Monotone (f ∘ Subtype.val : s → β) :=
  fun x y hle => h x.coe_prop y.coe_prop hle


protected theorem _root_.AntitoneOn.monotone (h : AntitoneOn f s) :
    Antitone (f ∘ Subtype.val : s → β) :=
  fun x y hle => h x.coe_prop y.coe_prop hle


protected theorem _root_.StrictMonoOn.strictMono (h : StrictMonoOn f s) :
    StrictMono (f ∘ Subtype.val : s → β) :=
  fun x y hlt => h x.coe_prop y.coe_prop hlt


protected theorem _root_.StrictAntiOn.strictAnti (h : StrictAntiOn f s) :
    StrictAnti (f ∘ Subtype.val : s → β) :=
  fun x y hlt => h x.coe_prop y.coe_prop hlt


lemma MonotoneOn_insert_iff {a : α} :
    MonotoneOn f (insert a s) ↔
       (∀ b ∈ s, b ≤ a → f b ≤ f a) ∧ (∀ b ∈ s, a ≤ b → f a ≤ f b) ∧ MonotoneOn f s := by
  /-
    α : Type u_1
    β : Type u_2
    s : Set α
    f : α → β
    inst✝¹ : Preorder α
    inst✝ : Preorder β
    a : α
    ⊢ Iff (MonotoneOn f (Insert.insert a s)) (And (∀ (b : α), Membership.mem s b → …
  -/
  simp [MonotoneOn, forall_and]
  /-
    🎉 no goals
  -/


lemma AntitoneOn_insert_iff {a : α} :
    AntitoneOn f (insert a s) ↔
       (∀ b ∈ s, b ≤ a → f a ≤ f b) ∧ (∀ b ∈ s, a ≤ b → f b ≤ f a) ∧ AntitoneOn f s :=
  @MonotoneOn_insert_iff α βᵒᵈ _ _ _ _ _


protected theorem restrict (h : Monotone f) (s : Set α) : Monotone (s.restrict f) := fun _ _ hxy =>
  h hxy


protected theorem codRestrict (h : Monotone f) {s : Set β} (hs : ∀ x, f x ∈ s) :
    Monotone (s.codRestrict f hs) :=
  h


protected theorem rangeFactorization (h : Monotone f) : Monotone (Set.rangeFactorization f) :=
  h


theorem StrictMonoOn.injOn [LinearOrder α] [Preorder β] {f : α → β} {s : Set α}
    (H : StrictMonoOn f s) : s.InjOn f := fun x hx y hy hxy =>
  show Ordering.eq.Compares x y from (H.compares hx hy).1 hxy


theorem StrictAntiOn.injOn [LinearOrder α] [Preorder β] {f : α → β} {s : Set α}
    (H : StrictAntiOn f s) : s.InjOn f :=
  @StrictMonoOn.injOn α βᵒᵈ _ _ f s H


theorem StrictMonoOn.comp [Preorder α] [Preorder β] [Preorder γ] {g : β → γ} {f : α → β} {s : Set α}
    {t : Set β} (hg : StrictMonoOn g t) (hf : StrictMonoOn f s) (hs : Set.MapsTo f s t) :
    StrictMonoOn (g ∘ f) s := fun _x hx _y hy hxy => hg (hs hx) (hs hy) <| hf hx hy hxy


theorem StrictMonoOn.comp_strictAntiOn [Preorder α] [Preorder β] [Preorder γ] {g : β → γ}
    {f : α → β} {s : Set α} {t : Set β} (hg : StrictMonoOn g t) (hf : StrictAntiOn f s)
    (hs : Set.MapsTo f s t) : StrictAntiOn (g ∘ f) s := fun _x hx _y hy hxy =>
  hg (hs hy) (hs hx) <| hf hx hy hxy


theorem StrictAntiOn.comp [Preorder α] [Preorder β] [Preorder γ] {g : β → γ} {f : α → β} {s : Set α}
    {t : Set β} (hg : StrictAntiOn g t) (hf : StrictAntiOn f s) (hs : Set.MapsTo f s t) :
    StrictMonoOn (g ∘ f) s := fun _x hx _y hy hxy => hg (hs hy) (hs hx) <| hf hx hy hxy


theorem StrictAntiOn.comp_strictMonoOn [Preorder α] [Preorder β] [Preorder γ] {g : β → γ}
    {f : α → β} {s : Set α} {t : Set β} (hg : StrictAntiOn g t) (hf : StrictMonoOn f s)
    (hs : Set.MapsTo f s t) : StrictAntiOn (g ∘ f) s := fun _x hx _y hy hxy =>
  hg (hs hx) (hs hy) <| hf hx hy hxy


@[simp]
theorem strictMono_restrict [Preorder α] [Preorder β] {f : α → β} {s : Set α} :
                                                       /-
                                                         α : Type u_1
                                                         β : Type u_2
                                                         inst✝¹ : Preorder α
                                                         inst✝ : Preorder β
                                                         f : α → β
                                                         s : Set α
                                                         ⊢ Iff (StrictMono (s.restrict f)) (StrictMonoOn f s)
                                                       -/
    StrictMono (s.restrict f) ↔ StrictMonoOn f s := by simp [Set.restrict, StrictMono, StrictMonoOn]
                                                       /-
                                                         🎉 no goals
                                                       -/


alias ⟨_root_.StrictMono.of_restrict, _root_.StrictMonoOn.restrict⟩ := strictMono_restrict


theorem StrictMono.codRestrict [Preorder α] [Preorder β] {f : α → β} (hf : StrictMono f)
    {s : Set β} (hs : ∀ x, f x ∈ s) : StrictMono (Set.codRestrict f s hs) :=
  hf


lemma strictMonoOn_insert_iff [Preorder α] [Preorder β] {f : α → β} {s : Set α} {a : α} :
    StrictMonoOn f (insert a s) ↔
       (∀ b ∈ s, b < a → f b < f a) ∧ (∀ b ∈ s, a < b → f a < f b) ∧ StrictMonoOn f s := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝¹ : Preorder α
    inst✝ : Preorder β
    f : α → β
    s : Set α
    a : α
    ⊢ Iff (StrictMonoOn f (Insert.insert a s)) (And (∀ (b : α), Membership.mem s b …
  -/
  simp [StrictMonoOn, forall_and]
  /-
    🎉 no goals
  -/


lemma strictAntiOn_insert_iff [Preorder α] [Preorder β] {f : α → β} {s : Set α} {a : α} :
    StrictAntiOn f (insert a s) ↔
       (∀ b ∈ s, b < a → f a < f b) ∧ (∀ b ∈ s, a < b → f b < f a) ∧ StrictAntiOn f s :=
  @strictMonoOn_insert_iff α βᵒᵈ _ _ _ _ _


theorem monotoneOn_of_rightInvOn_of_mapsTo {α β : Type*} [PartialOrder α] [LinearOrder β]
    {φ : β → α} {ψ : α → β} {t : Set β} {s : Set α} (hφ : MonotoneOn φ t)
    (φψs : Set.RightInvOn ψ φ s) (ψts : Set.MapsTo ψ s t) : MonotoneOn ψ s := by
  /-
    α : Type u_4
    β : Type u_5
    inst✝¹ : PartialOrder α
    inst✝ : LinearOrder β
    φ : β → α
    ψ : α → β
    t : Set β
    s : Set α
    hφ : MonotoneOn φ t
    φψs : Set.RightInvOn ψ φ s
    ψts : Set.MapsTo ψ s t
    ⊢ MonotoneOn ψ s
  -/
  rintro x xs y ys l
  /-
    α : Type u_4
    β : Type u_5
    inst✝¹ : PartialOrder α
    inst✝ : LinearOrder β
    φ : β → α
    ψ : α → β
    t : Set β
    s : Set α
    hφ : MonotoneOn φ t
    φψs : Set.RightInvOn ψ φ s
    ψts : Set.MapsTo ψ s t
    x : α
    xs : Membership.mem s x
    y : α
    ys : Membership.mem s y
    l : LE.le x y
    ⊢ LE.le (ψ x) (ψ y)
  -/
  rcases le_total (ψ x) (ψ y) with (ψxy|ψyx)
    /-
      case inl
      α : Type u_4
      β : Type u_5
      inst✝¹ : PartialOrder α
      inst✝ : LinearOrder β
      φ : β → α
      ψ : α → β
      t : Set β
      s : Set α
      hφ : MonotoneOn φ t
      φψs : Set.RightInvOn ψ φ s
      ψts : Set.MapsTo ψ s t
      x : α
      xs : Membership.mem s x
      y : α
      ys : Membership.mem s y
      l : LE.le x y
      ψxy : LE.le (ψ x) (ψ y)
      ⊢ LE.le (ψ x) (ψ y)
    -/
  · exact ψxy
    /-
      🎉 no goals
    -/
    /-
      case inr
      α : Type u_4
      β : Type u_5
      inst✝¹ : PartialOrder α
      inst✝ : LinearOrder β
      φ : β → α
      ψ : α → β
      t : Set β
      s : Set α
      hφ : MonotoneOn φ t
      φψs : Set.RightInvOn ψ φ s
      ψts : Set.MapsTo ψ s t
      x : α
      xs : Membership.mem s x
      y : α
      ys : Membership.mem s y
      l : LE.le x y
      ψyx : LE.le (ψ y) (ψ x)
      ⊢ LE.le (ψ x) (ψ y)
    -/
  · have := hφ (ψts ys) (ψts xs) ψyx
    /-
      case inr
      α : Type u_4
      β : Type u_5
      inst✝¹ : PartialOrder α
      inst✝ : LinearOrder β
      φ : β → α
      ψ : α → β
      t : Set β
      s : Set α
      hφ : MonotoneOn φ t
      φψs : Set.RightInvOn ψ φ s
      ψts : Set.MapsTo ψ s t
      x : α
      xs : Membership.mem s x
      y : α
      ys : Membership.mem s y
      l : LE.le x y
      ψyx : LE.le (ψ y) (ψ x)
      this : LE.le (φ (ψ y)) (φ (ψ x))
      ⊢ LE.le (ψ x) (ψ y)
    -/
    rw [φψs.eq ys, φψs.eq xs] at this
    /-
      case inr
      α : Type u_4
      β : Type u_5
      inst✝¹ : PartialOrder α
      inst✝ : LinearOrder β
      φ : β → α
      ψ : α → β
      t : Set β
      s : Set α
      hφ : MonotoneOn φ t
      φψs : Set.RightInvOn ψ φ s
      ψts : Set.MapsTo ψ s t
      x : α
      xs : Membership.mem s x
      y : α
      ys : Membership.mem s y
      l : LE.le x y
      ψyx : LE.le (ψ y) (ψ x)
      this : LE.le y x
      ⊢ LE.le (ψ x) (ψ y)
    -/
    induction le_antisymm l this
    /-
      case inr.refl
      α : Type u_4
      β : Type u_5
      inst✝¹ : PartialOrder α
      inst✝ : LinearOrder β
      φ : β → α
      ψ : α → β
      t : Set β
      s : Set α
      hφ : MonotoneOn φ t
      φψs : Set.RightInvOn ψ φ s
      ψts : Set.MapsTo ψ s t
      x : α
      xs : Membership.mem s x
      y : α
      ys : Membership.mem s x
      l : LE.le x x
      ψyx : LE.le (ψ x) (ψ x)
      this : LE.le x x
      ⊢ LE.le (ψ x) (ψ x)
    -/
    exact le_refl _
    /-
      🎉 no goals
    -/


theorem antitoneOn_of_rightInvOn_of_mapsTo [PartialOrder α] [LinearOrder β]
    {φ : β → α} {ψ : α → β} {t : Set β} {s : Set α} (hφ : AntitoneOn φ t)
    (φψs : Set.RightInvOn ψ φ s) (ψts : Set.MapsTo ψ s t) : AntitoneOn ψ s :=
  (monotoneOn_of_rightInvOn_of_mapsTo hφ.dual_left φψs ψts).dual_right


