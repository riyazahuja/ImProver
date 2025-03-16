/-- Lifts maps `α₁ → β₁ → Finset γ₁` and `α₂ → β₂ → Finset γ₂` to a map
`α₁ ⊕ α₂ → β₁ ⊕ β₂ → Finset (γ₁ ⊕ γ₂)`. Could be generalized to `Alternative` functors if we can
make sure to keep computability and universe polymorphism. -/
@[simp]
def sumLift₂ : ∀ (_ : α₁ ⊕ α₂) (_ : β₁ ⊕ β₂), Finset (γ₁ ⊕ γ₂)
  | inl a, inl b => (f a b).map Embedding.inl
  | inl _, inr _ => ∅
  | inr _, inl _ => ∅
  | inr a, inr b => (g a b).map Embedding.inr


theorem mem_sumLift₂ :
    c ∈ sumLift₂ f g a b ↔
      (∃ a₁ b₁ c₁, a = inl a₁ ∧ b = inl b₁ ∧ c = inl c₁ ∧ c₁ ∈ f a₁ b₁) ∨
        ∃ a₂ b₂ c₂, a = inr a₂ ∧ b = inr b₂ ∧ c = inr c₂ ∧ c₂ ∈ g a₂ b₂ := by
  /-
    α₁ : Type u_1
    α₂ : Type u_2
    β₁ : Type u_3
    β₂ : Type u_4
    γ₁ : Type u_5
    γ₂ : Type u_6
    f : α₁ → β₁ → Finset γ₁
    g : α₂ → β₂ → Finset γ₂
    a : Sum α₁ α₂
    b : Sum β₁ β₂
    c : Sum γ₁ γ₂
    ⊢ Iff (Membership.mem (Finset.sumLift₂ f g a b) c) (Or (Exists fun a₁ => Exist …
  -/
  constructor
    /-
      case mp
      α₁ : Type u_1
      α₂ : Type u_2
      β₁ : Type u_3
      β₂ : Type u_4
      γ₁ : Type u_5
      γ₂ : Type u_6
      f : α₁ → β₁ → Finset γ₁
      g : α₂ → β₂ → Finset γ₂
      a : Sum α₁ α₂
      b : Sum β₁ β₂
      c : Sum γ₁ γ₂
      ⊢ Membership.mem (Finset.sumLift₂ f g a b) c → Or (Exists fun a₁ => Exists fun …
    -/
  · cases' a with a a <;> cases' b with b b
      /-
        case mp.inl.inl
        α₁ : Type u_1
        α₂ : Type u_2
        β₁ : Type u_3
        β₂ : Type u_4
        γ₁ : Type u_5
        γ₂ : Type u_6
        f : α₁ → β₁ → Finset γ₁
        g : α₂ → β₂ → Finset γ₂
        c : Sum γ₁ γ₂
        a : α₁
        b : β₁
        ⊢ Membership.mem (Finset.sumLift₂ f g (Sum.inl a) (Sum.inl b)) c → Or (Exists  …
      -/
    · rw [sumLift₂, mem_map]
      /-
        case mp.inl.inl
        α₁ : Type u_1
        α₂ : Type u_2
        β₁ : Type u_3
        β₂ : Type u_4
        γ₁ : Type u_5
        γ₂ : Type u_6
        f : α₁ → β₁ → Finset γ₁
        g : α₂ → β₂ → Finset γ₂
        c : Sum γ₁ γ₂
        a : α₁
        b : β₁
        ⊢ (Exists fun a_1 => And (Membership.mem (f a b) a_1) (Eq (Function.Embedding. …
      -/
      rintro ⟨c, hc, rfl⟩
      /-
        case mp.inl.inl.intro.intro
        α₁ : Type u_1
        α₂ : Type u_2
        β₁ : Type u_3
        β₂ : Type u_4
        γ₁ : Type u_5
        γ₂ : Type u_6
        f : α₁ → β₁ → Finset γ₁
        g : α₂ → β₂ → Finset γ₂
        a : α₁
        b : β₁
        c : γ₁
        hc : Membership.mem (f a b) c
        ⊢ Or (Exists fun a₁ => Exists fun b₁ => Exists fun c₁ => And (Eq (Sum.inl a) ( …
      -/
      exact Or.inl ⟨a, b, c, rfl, rfl, rfl, hc⟩
      /-
        🎉 no goals
      -/
      /-
        case mp.inl.inr
        α₁ : Type u_1
        α₂ : Type u_2
        β₁ : Type u_3
        β₂ : Type u_4
        γ₁ : Type u_5
        γ₂ : Type u_6
        f : α₁ → β₁ → Finset γ₁
        g : α₂ → β₂ → Finset γ₂
        c : Sum γ₁ γ₂
        a : α₁
        b : β₂
        ⊢ Membership.mem (Finset.sumLift₂ f g (Sum.inl a) (Sum.inr b)) c → Or (Exists  …
      -/
    · refine fun h ↦ (not_mem_empty _ h).elim
      /-
        🎉 no goals
      -/
      /-
        case mp.inr.inl
        α₁ : Type u_1
        α₂ : Type u_2
        β₁ : Type u_3
        β₂ : Type u_4
        γ₁ : Type u_5
        γ₂ : Type u_6
        f : α₁ → β₁ → Finset γ₁
        g : α₂ → β₂ → Finset γ₂
        c : Sum γ₁ γ₂
        a : α₂
        b : β₁
        ⊢ Membership.mem (Finset.sumLift₂ f g (Sum.inr a) (Sum.inl b)) c → Or (Exists  …
      -/
    · refine fun h ↦ (not_mem_empty _ h).elim
      /-
        🎉 no goals
      -/
      /-
        case mp.inr.inr
        α₁ : Type u_1
        α₂ : Type u_2
        β₁ : Type u_3
        β₂ : Type u_4
        γ₁ : Type u_5
        γ₂ : Type u_6
        f : α₁ → β₁ → Finset γ₁
        g : α₂ → β₂ → Finset γ₂
        c : Sum γ₁ γ₂
        a : α₂
        b : β₂
        ⊢ Membership.mem (Finset.sumLift₂ f g (Sum.inr a) (Sum.inr b)) c → Or (Exists  …
      -/
    · rw [sumLift₂, mem_map]
      /-
        case mp.inr.inr
        α₁ : Type u_1
        α₂ : Type u_2
        β₁ : Type u_3
        β₂ : Type u_4
        γ₁ : Type u_5
        γ₂ : Type u_6
        f : α₁ → β₁ → Finset γ₁
        g : α₂ → β₂ → Finset γ₂
        c : Sum γ₁ γ₂
        a : α₂
        b : β₂
        ⊢ (Exists fun a_1 => And (Membership.mem (g a b) a_1) (Eq (Function.Embedding. …
      -/
      rintro ⟨c, hc, rfl⟩
      /-
        case mp.inr.inr.intro.intro
        α₁ : Type u_1
        α₂ : Type u_2
        β₁ : Type u_3
        β₂ : Type u_4
        γ₁ : Type u_5
        γ₂ : Type u_6
        f : α₁ → β₁ → Finset γ₁
        g : α₂ → β₂ → Finset γ₂
        a : α₂
        b : β₂
        c : γ₂
        hc : Membership.mem (g a b) c
        ⊢ Or (Exists fun a₁ => Exists fun b₁ => Exists fun c₁ => And (Eq (Sum.inr a) ( …
      -/
      exact Or.inr ⟨a, b, c, rfl, rfl, rfl, hc⟩
      /-
        🎉 no goals
      -/
    /-
      case mpr
      α₁ : Type u_1
      α₂ : Type u_2
      β₁ : Type u_3
      β₂ : Type u_4
      γ₁ : Type u_5
      γ₂ : Type u_6
      f : α₁ → β₁ → Finset γ₁
      g : α₂ → β₂ → Finset γ₂
      a : Sum α₁ α₂
      b : Sum β₁ β₂
      c : Sum γ₁ γ₂
      ⊢ Or (Exists fun a₁ => Exists fun b₁ => Exists fun c₁ => And (Eq a (Sum.inl a₁ …
    -/
                                                                           /-
                                                                             🎉 no goals
                                                                           -/
  · rintro (⟨a, b, c, rfl, rfl, rfl, h⟩ | ⟨a, b, c, rfl, rfl, rfl, h⟩) <;> exact mem_map_of_mem _ h
                                                                           /-
                                                                             🎉 no goals
                                                                           -/


theorem inl_mem_sumLift₂ {c₁ : γ₁} :
    inl c₁ ∈ sumLift₂ f g a b ↔ ∃ a₁ b₁, a = inl a₁ ∧ b = inl b₁ ∧ c₁ ∈ f a₁ b₁ := by
  /-
    α₁ : Type u_1
    α₂ : Type u_2
    β₁ : Type u_3
    β₂ : Type u_4
    γ₁ : Type u_5
    γ₂ : Type u_6
    f : α₁ → β₁ → Finset γ₁
    g : α₂ → β₂ → Finset γ₂
    a : Sum α₁ α₂
    b : Sum β₁ β₂
    c₁ : γ₁
    ⊢ Iff (Membership.mem (Finset.sumLift₂ f g a b) (Sum.inl c₁)) (Exists fun a₁ = …
  -/
  rw [mem_sumLift₂, or_iff_left]
    /-
      α₁ : Type u_1
      α₂ : Type u_2
      β₁ : Type u_3
      β₂ : Type u_4
      γ₁ : Type u_5
      γ₂ : Type u_6
      f : α₁ → β₁ → Finset γ₁
      g : α₂ → β₂ → Finset γ₂
      a : Sum α₁ α₂
      b : Sum β₁ β₂
      c₁ : γ₁
      ⊢ Iff (Exists fun a₁ => Exists fun b₁ => Exists fun c₁_1 => And (Eq a (Sum.inl …
    -/
  · simp only [inl.injEq, exists_and_left, exists_eq_left']
    /-
      🎉 no goals
    -/
  /-
    α₁ : Type u_1
    α₂ : Type u_2
    β₁ : Type u_3
    β₂ : Type u_4
    γ₁ : Type u_5
    γ₂ : Type u_6
    f : α₁ → β₁ → Finset γ₁
    g : α₂ → β₂ → Finset γ₂
    a : Sum α₁ α₂
    b : Sum β₁ β₂
    c₁ : γ₁
    ⊢ Not (Exists fun a₂ => Exists fun b₂ => Exists fun c₂ => And (Eq a (Sum.inr a …
  -/
  rintro ⟨_, _, c₂, _, _, h, _⟩
  /-
    case intro.intro.intro.intro.intro.intro
    α₁ : Type u_1
    α₂ : Type u_2
    β₁ : Type u_3
    β₂ : Type u_4
    γ₁ : Type u_5
    γ₂ : Type u_6
    f : α₁ → β₁ → Finset γ₁
    g : α₂ → β₂ → Finset γ₂
    a : Sum α₁ α₂
    b : Sum β₁ β₂
    c₁ : γ₁
    w✝¹ : α₂
    w✝ : β₂
    c₂ : γ₂
    left✝¹ : Eq a (Sum.inr w✝¹)
    left✝ : Eq b (Sum.inr w✝)
    h : Eq (Sum.inl c₁) (Sum.inr c₂)
    right✝ : Membership.mem (g w✝¹ w✝) c₂
    ⊢ False
  -/
  exact inl_ne_inr h
  /-
    🎉 no goals
  -/


theorem inr_mem_sumLift₂ {c₂ : γ₂} :
    inr c₂ ∈ sumLift₂ f g a b ↔ ∃ a₂ b₂, a = inr a₂ ∧ b = inr b₂ ∧ c₂ ∈ g a₂ b₂ := by
  /-
    α₁ : Type u_1
    α₂ : Type u_2
    β₁ : Type u_3
    β₂ : Type u_4
    γ₁ : Type u_5
    γ₂ : Type u_6
    f : α₁ → β₁ → Finset γ₁
    g : α₂ → β₂ → Finset γ₂
    a : Sum α₁ α₂
    b : Sum β₁ β₂
    c₂ : γ₂
    ⊢ Iff (Membership.mem (Finset.sumLift₂ f g a b) (Sum.inr c₂)) (Exists fun a₂ = …
  -/
  rw [mem_sumLift₂, or_iff_right]
    /-
      α₁ : Type u_1
      α₂ : Type u_2
      β₁ : Type u_3
      β₂ : Type u_4
      γ₁ : Type u_5
      γ₂ : Type u_6
      f : α₁ → β₁ → Finset γ₁
      g : α₂ → β₂ → Finset γ₂
      a : Sum α₁ α₂
      b : Sum β₁ β₂
      c₂ : γ₂
      ⊢ Iff (Exists fun a₂ => Exists fun b₂ => Exists fun c₂_1 => And (Eq a (Sum.inr …
    -/
  · simp only [inr.injEq, exists_and_left, exists_eq_left']
    /-
      🎉 no goals
    -/
  /-
    α₁ : Type u_1
    α₂ : Type u_2
    β₁ : Type u_3
    β₂ : Type u_4
    γ₁ : Type u_5
    γ₂ : Type u_6
    f : α₁ → β₁ → Finset γ₁
    g : α₂ → β₂ → Finset γ₂
    a : Sum α₁ α₂
    b : Sum β₁ β₂
    c₂ : γ₂
    ⊢ Not (Exists fun a₁ => Exists fun b₁ => Exists fun c₁ => And (Eq a (Sum.inl a …
  -/
  rintro ⟨_, _, c₂, _, _, h, _⟩
  /-
    case intro.intro.intro.intro.intro.intro
    α₁ : Type u_1
    α₂ : Type u_2
    β₁ : Type u_3
    β₂ : Type u_4
    γ₁ : Type u_5
    γ₂ : Type u_6
    f : α₁ → β₁ → Finset γ₁
    g : α₂ → β₂ → Finset γ₂
    a : Sum α₁ α₂
    b : Sum β₁ β₂
    c₂✝ : γ₂
    w✝¹ : α₁
    w✝ : β₁
    c₂ : γ₁
    left✝¹ : Eq a (Sum.inl w✝¹)
    left✝ : Eq b (Sum.inl w✝)
    h : Eq (Sum.inr c₂✝) (Sum.inl c₂)
    right✝ : Membership.mem (f w✝¹ w✝) c₂
    ⊢ False
  -/
  exact inr_ne_inl h
  /-
    🎉 no goals
  -/


theorem sumLift₂_eq_empty :
    sumLift₂ f g a b = ∅ ↔
      (∀ a₁ b₁, a = inl a₁ → b = inl b₁ → f a₁ b₁ = ∅) ∧
        ∀ a₂ b₂, a = inr a₂ → b = inr b₂ → g a₂ b₂ = ∅ := by
  /-
    α₁ : Type u_1
    α₂ : Type u_2
    β₁ : Type u_3
    β₂ : Type u_4
    γ₁ : Type u_5
    γ₂ : Type u_6
    f : α₁ → β₁ → Finset γ₁
    g : α₂ → β₂ → Finset γ₂
    a : Sum α₁ α₂
    b : Sum β₁ β₂
    ⊢ Iff (Eq (Finset.sumLift₂ f g a b) EmptyCollection.emptyCollection) (And (∀ ( …
  -/
  refine ⟨fun h ↦ ?_, fun h ↦ ?_⟩
    /-
      case refine_1
      α₁ : Type u_1
      α₂ : Type u_2
      β₁ : Type u_3
      β₂ : Type u_4
      γ₁ : Type u_5
      γ₂ : Type u_6
      f : α₁ → β₁ → Finset γ₁
      g : α₂ → β₂ → Finset γ₂
      a : Sum α₁ α₂
      b : Sum β₁ β₂
      h : Eq (Finset.sumLift₂ f g a b) EmptyCollection.emptyCollection
      ⊢ And (∀ (a₁ : α₁) (b₁ : β₁), Eq a (Sum.inl a₁) → Eq b (Sum.inl b₁) → Eq (f a₁ …
    -/
  · constructor <;>
      /-
        case refine_1.left
        α₁ : Type u_1
        α₂ : Type u_2
        β₁ : Type u_3
        β₂ : Type u_4
        γ₁ : Type u_5
        γ₂ : Type u_6
        f : α₁ → β₁ → Finset γ₁
        g : α₂ → β₂ → Finset γ₂
        a : Sum α₁ α₂
        b : Sum β₁ β₂
        h : Eq (Finset.sumLift₂ f g a b) EmptyCollection.emptyCollection
        ⊢ ∀ (a₁ : α₁) (b₁ : β₁), Eq a (Sum.inl a₁) → Eq b (Sum.inl b₁) → Eq (f a₁ b₁)  …
      -/
      /-
        case refine_1.left
        α₁ : Type u_1
        α₂ : Type u_2
        β₁ : Type u_3
        β₂ : Type u_4
        γ₁ : Type u_5
        γ₂ : Type u_6
        f : α₁ → β₁ → Finset γ₁
        g : α₂ → β₂ → Finset γ₂
        a : α₁
        b : β₁
        h : Eq (Finset.sumLift₂ f g (Sum.inl a) (Sum.inl b)) EmptyCollection.emptyColl …
        ⊢ Eq (f a b) EmptyCollection.emptyCollection
      -/
      /-
        🎉 no goals
      -/
      /-
        case refine_1.right
        α₁ : Type u_1
        α₂ : Type u_2
        β₁ : Type u_3
        β₂ : Type u_4
        γ₁ : Type u_5
        γ₂ : Type u_6
        f : α₁ → β₁ → Finset γ₁
        g : α₂ → β₂ → Finset γ₂
        a : α₂
        b : β₂
        h : Eq (Finset.sumLift₂ f g (Sum.inr a) (Sum.inr b)) EmptyCollection.emptyColl …
        ⊢ Eq (g a b) EmptyCollection.emptyCollection
      -/
      exact map_eq_empty.1 h
      /-
        🎉 no goals
      -/
  /-
    case refine_2
    α₁ : Type u_1
    α₂ : Type u_2
    β₁ : Type u_3
    β₂ : Type u_4
    γ₁ : Type u_5
    γ₂ : Type u_6
    f : α₁ → β₁ → Finset γ₁
    g : α₂ → β₂ → Finset γ₂
    a : Sum α₁ α₂
    b : Sum β₁ β₂
    h : And (∀ (a₁ : α₁) (b₁ : β₁), Eq a (Sum.inl a₁) → Eq b (Sum.inl b₁) → Eq (f  …
    ⊢ Eq (Finset.sumLift₂ f g a b) EmptyCollection.emptyCollection
  -/
  cases a <;> cases b
    /-
      case refine_2.inl.inl
      α₁ : Type u_1
      α₂ : Type u_2
      β₁ : Type u_3
      β₂ : Type u_4
      γ₁ : Type u_5
      γ₂ : Type u_6
      f : α₁ → β₁ → Finset γ₁
      g : α₂ → β₂ → Finset γ₂
      val✝¹ : α₁
      val✝ : β₁
      h : And (∀ (a₁ : α₁) (b₁ : β₁), Eq (Sum.inl val✝¹) (Sum.inl a₁) → Eq (Sum.inl  …
      ⊢ Eq (Finset.sumLift₂ f g (Sum.inl val✝¹) (Sum.inl val✝)) EmptyCollection.empt …
    -/
  · exact map_eq_empty.2 (h.1 _ _ rfl rfl)
    /-
      🎉 no goals
    -/
    /-
      case refine_2.inl.inr
      α₁ : Type u_1
      α₂ : Type u_2
      β₁ : Type u_3
      β₂ : Type u_4
      γ₁ : Type u_5
      γ₂ : Type u_6
      f : α₁ → β₁ → Finset γ₁
      g : α₂ → β₂ → Finset γ₂
      val✝¹ : α₁
      val✝ : β₂
      h : And (∀ (a₁ : α₁) (b₁ : β₁), Eq (Sum.inl val✝¹) (Sum.inl a₁) → Eq (Sum.inr  …
      ⊢ Eq (Finset.sumLift₂ f g (Sum.inl val✝¹) (Sum.inr val✝)) EmptyCollection.empt …
    -/
  · rfl
    /-
      🎉 no goals
    -/
    /-
      case refine_2.inr.inl
      α₁ : Type u_1
      α₂ : Type u_2
      β₁ : Type u_3
      β₂ : Type u_4
      γ₁ : Type u_5
      γ₂ : Type u_6
      f : α₁ → β₁ → Finset γ₁
      g : α₂ → β₂ → Finset γ₂
      val✝¹ : α₂
      val✝ : β₁
      h : And (∀ (a₁ : α₁) (b₁ : β₁), Eq (Sum.inr val✝¹) (Sum.inl a₁) → Eq (Sum.inl  …
      ⊢ Eq (Finset.sumLift₂ f g (Sum.inr val✝¹) (Sum.inl val✝)) EmptyCollection.empt …
    -/
  · rfl
    /-
      🎉 no goals
    -/
    /-
      case refine_2.inr.inr
      α₁ : Type u_1
      α₂ : Type u_2
      β₁ : Type u_3
      β₂ : Type u_4
      γ₁ : Type u_5
      γ₂ : Type u_6
      f : α₁ → β₁ → Finset γ₁
      g : α₂ → β₂ → Finset γ₂
      val✝¹ : α₂
      val✝ : β₂
      h : And (∀ (a₁ : α₁) (b₁ : β₁), Eq (Sum.inr val✝¹) (Sum.inl a₁) → Eq (Sum.inr  …
      ⊢ Eq (Finset.sumLift₂ f g (Sum.inr val✝¹) (Sum.inr val✝)) EmptyCollection.empt …
    -/
  · exact map_eq_empty.2 (h.2 _ _ rfl rfl)
    /-
      🎉 no goals
    -/


theorem sumLift₂_nonempty :
    (sumLift₂ f g a b).Nonempty ↔
      (∃ a₁ b₁, a = inl a₁ ∧ b = inl b₁ ∧ (f a₁ b₁).Nonempty) ∨
        ∃ a₂ b₂, a = inr a₂ ∧ b = inr b₂ ∧ (g a₂ b₂).Nonempty := by
  /-
    α₁ : Type u_1
    α₂ : Type u_2
    β₁ : Type u_3
    β₂ : Type u_4
    γ₁ : Type u_5
    γ₂ : Type u_6
    f : α₁ → β₁ → Finset γ₁
    g : α₂ → β₂ → Finset γ₂
    a : Sum α₁ α₂
    b : Sum β₁ β₂
    ⊢ Iff (Finset.sumLift₂ f g a b).Nonempty (Or (Exists fun a₁ => Exists fun b₁ = …
  -/
  simp only [nonempty_iff_ne_empty, Ne, sumLift₂_eq_empty, not_and_or, not_forall, exists_prop]
  /-
    🎉 no goals
  -/


theorem sumLift₂_mono (h₁ : ∀ a b, f₁ a b ⊆ g₁ a b) (h₂ : ∀ a b, f₂ a b ⊆ g₂ a b) :
    ∀ a b, sumLift₂ f₁ f₂ a b ⊆ sumLift₂ g₁ g₂ a b
  | inl _, inl _ => map_subset_map.2 (h₁ _ _)
  | inl _, inr _ => Subset.rfl
  | inr _, inl _ => Subset.rfl
  | inr _, inr _ => map_subset_map.2 (h₂ _ _)


/-- Lifts maps `α₁ → β₁ → Finset γ₁`, `α₂ → β₂ → Finset γ₂`, `α₁ → β₂ → Finset γ₁`,
`α₂ → β₂ → Finset γ₂`  to a map `α₁ ⊕ α₂ → β₁ ⊕ β₂ → Finset (γ₁ ⊕ γ₂)`. Could be generalized to
alternative monads if we can make sure to keep computability and universe polymorphism. -/
def sumLexLift : α₁ ⊕ α₂ → β₁ ⊕ β₂ → Finset (γ₁ ⊕ γ₂)
  | inl a, inl b => (f₁ a b).map Embedding.inl
  | inl a, inr b => (g₁ a b).disjSum (g₂ a b)
  | inr _, inl _ => ∅
  | inr a, inr b => (f₂ a b).map ⟨_, inr_injective⟩


@[simp]
lemma sumLexLift_inl_inl (a : α₁) (b : β₁) :
    sumLexLift f₁ f₂ g₁ g₂ (inl a) (inl b) = (f₁ a b).map Embedding.inl := rfl


@[simp]
lemma sumLexLift_inl_inr (a : α₁) (b : β₂) :
    sumLexLift f₁ f₂ g₁ g₂ (inl a) (inr b) = (g₁ a b).disjSum (g₂ a b) := rfl


@[simp]
lemma sumLexLift_inr_inl (a : α₂) (b : β₁) : sumLexLift f₁ f₂ g₁ g₂ (inr a) (inl b) = ∅ := rfl


@[simp]
lemma sumLexLift_inr_inr (a : α₂) (b : β₂) :
    sumLexLift f₁ f₂ g₁ g₂ (inr a) (inr b) = (f₂ a b).map ⟨_, inr_injective⟩ := rfl


lemma mem_sumLexLift :
    c ∈ sumLexLift f₁ f₂ g₁ g₂ a b ↔
      (∃ a₁ b₁ c₁, a = inl a₁ ∧ b = inl b₁ ∧ c = inl c₁ ∧ c₁ ∈ f₁ a₁ b₁) ∨
        (∃ a₁ b₂ c₁, a = inl a₁ ∧ b = inr b₂ ∧ c = inl c₁ ∧ c₁ ∈ g₁ a₁ b₂) ∨
          (∃ a₁ b₂ c₂, a = inl a₁ ∧ b = inr b₂ ∧ c = inr c₂ ∧ c₂ ∈ g₂ a₁ b₂) ∨
            ∃ a₂ b₂ c₂, a = inr a₂ ∧ b = inr b₂ ∧ c = inr c₂ ∧ c₂ ∈ f₂ a₂ b₂ := by
  /-
    α₁ : Type u_1
    α₂ : Type u_2
    β₁ : Type u_3
    β₂ : Type u_4
    γ₁ : Type u_5
    γ₂ : Type u_6
    f₁ : α₁ → β₁ → Finset γ₁
    f₂ : α₂ → β₂ → Finset γ₂
    g₁ : α₁ → β₂ → Finset γ₁
    g₂ : α₁ → β₂ → Finset γ₂
    a : Sum α₁ α₂
    b : Sum β₁ β₂
    c : Sum γ₁ γ₂
    ⊢ Iff (Membership.mem (Finset.sumLexLift f₁ f₂ g₁ g₂ a b) c) (Or (Exists fun a …
  -/
  constructor
    /-
      case mp
      α₁ : Type u_1
      α₂ : Type u_2
      β₁ : Type u_3
      β₂ : Type u_4
      γ₁ : Type u_5
      γ₂ : Type u_6
      f₁ : α₁ → β₁ → Finset γ₁
      f₂ : α₂ → β₂ → Finset γ₂
      g₁ : α₁ → β₂ → Finset γ₁
      g₂ : α₁ → β₂ → Finset γ₂
      a : Sum α₁ α₂
      b : Sum β₁ β₂
      c : Sum γ₁ γ₂
      ⊢ Membership.mem (Finset.sumLexLift f₁ f₂ g₁ g₂ a b) c → Or (Exists fun a₁ =>  …
    -/
  · obtain a | a := a <;> obtain b | b := b
      /-
        case mp.inl.inl
        α₁ : Type u_1
        α₂ : Type u_2
        β₁ : Type u_3
        β₂ : Type u_4
        γ₁ : Type u_5
        γ₂ : Type u_6
        f₁ : α₁ → β₁ → Finset γ₁
        f₂ : α₂ → β₂ → Finset γ₂
        g₁ : α₁ → β₂ → Finset γ₁
        g₂ : α₁ → β₂ → Finset γ₂
        c : Sum γ₁ γ₂
        a : α₁
        b : β₁
        ⊢ Membership.mem (Finset.sumLexLift f₁ f₂ g₁ g₂ (Sum.inl a) (Sum.inl b)) c → O …
      -/
    · rw [sumLexLift, mem_map]
      /-
        case mp.inl.inl
        α₁ : Type u_1
        α₂ : Type u_2
        β₁ : Type u_3
        β₂ : Type u_4
        γ₁ : Type u_5
        γ₂ : Type u_6
        f₁ : α₁ → β₁ → Finset γ₁
        f₂ : α₂ → β₂ → Finset γ₂
        g₁ : α₁ → β₂ → Finset γ₁
        g₂ : α₁ → β₂ → Finset γ₂
        c : Sum γ₁ γ₂
        a : α₁
        b : β₁
        ⊢ (Exists fun a_1 => And (Membership.mem (f₁ a b) a_1) (Eq (Function.Embedding …
      -/
      rintro ⟨c, hc, rfl⟩
      /-
        case mp.inl.inl.intro.intro
        α₁ : Type u_1
        α₂ : Type u_2
        β₁ : Type u_3
        β₂ : Type u_4
        γ₁ : Type u_5
        γ₂ : Type u_6
        f₁ : α₁ → β₁ → Finset γ₁
        f₂ : α₂ → β₂ → Finset γ₂
        g₁ : α₁ → β₂ → Finset γ₁
        g₂ : α₁ → β₂ → Finset γ₂
        a : α₁
        b : β₁
        c : γ₁
        hc : Membership.mem (f₁ a b) c
        ⊢ Or (Exists fun a₁ => Exists fun b₁ => Exists fun c₁ => And (Eq (Sum.inl a) ( …
      -/
      exact Or.inl ⟨a, b, c, rfl, rfl, rfl, hc⟩
      /-
        🎉 no goals
      -/
      /-
        case mp.inl.inr
        α₁ : Type u_1
        α₂ : Type u_2
        β₁ : Type u_3
        β₂ : Type u_4
        γ₁ : Type u_5
        γ₂ : Type u_6
        f₁ : α₁ → β₁ → Finset γ₁
        f₂ : α₂ → β₂ → Finset γ₂
        g₁ : α₁ → β₂ → Finset γ₁
        g₂ : α₁ → β₂ → Finset γ₂
        c : Sum γ₁ γ₂
        a : α₁
        b : β₂
        ⊢ Membership.mem (Finset.sumLexLift f₁ f₂ g₁ g₂ (Sum.inl a) (Sum.inr b)) c → O …
      -/
    · refine fun h ↦ (mem_disjSum.1 h).elim ?_ ?_
        /-
          case mp.inl.inr.refine_1
          α₁ : Type u_1
          α₂ : Type u_2
          β₁ : Type u_3
          β₂ : Type u_4
          γ₁ : Type u_5
          γ₂ : Type u_6
          f₁ : α₁ → β₁ → Finset γ₁
          f₂ : α₂ → β₂ → Finset γ₂
          g₁ : α₁ → β₂ → Finset γ₁
          g₂ : α₁ → β₂ → Finset γ₂
          c : Sum γ₁ γ₂
          a : α₁
          b : β₂
          h : Membership.mem (Finset.sumLexLift f₁ f₂ g₁ g₂ (Sum.inl a) (Sum.inr b)) c
          ⊢ (Exists fun a_1 => And (Membership.mem (g₁ a b) a_1) (Eq (Sum.inl a_1) c)) → …
        -/
      · rintro ⟨c, hc, rfl⟩
        /-
          case mp.inl.inr.refine_1.intro.intro
          α₁ : Type u_1
          α₂ : Type u_2
          β₁ : Type u_3
          β₂ : Type u_4
          γ₁ : Type u_5
          γ₂ : Type u_6
          f₁ : α₁ → β₁ → Finset γ₁
          f₂ : α₂ → β₂ → Finset γ₂
          g₁ : α₁ → β₂ → Finset γ₁
          g₂ : α₁ → β₂ → Finset γ₂
          a : α₁
          b : β₂
          c : γ₁
          hc : Membership.mem (g₁ a b) c
          h : Membership.mem (Finset.sumLexLift f₁ f₂ g₁ g₂ (Sum.inl a) (Sum.inr b)) (Su …
          ⊢ Or (Exists fun a₁ => Exists fun b₁ => Exists fun c₁ => And (Eq (Sum.inl a) ( …
        -/
        exact Or.inr (Or.inl ⟨a, b, c, rfl, rfl, rfl, hc⟩)
        /-
          🎉 no goals
        -/
        /-
          case mp.inl.inr.refine_2
          α₁ : Type u_1
          α₂ : Type u_2
          β₁ : Type u_3
          β₂ : Type u_4
          γ₁ : Type u_5
          γ₂ : Type u_6
          f₁ : α₁ → β₁ → Finset γ₁
          f₂ : α₂ → β₂ → Finset γ₂
          g₁ : α₁ → β₂ → Finset γ₁
          g₂ : α₁ → β₂ → Finset γ₂
          c : Sum γ₁ γ₂
          a : α₁
          b : β₂
          h : Membership.mem (Finset.sumLexLift f₁ f₂ g₁ g₂ (Sum.inl a) (Sum.inr b)) c
          ⊢ (Exists fun b_1 => And (Membership.mem (g₂ a b) b_1) (Eq (Sum.inr b_1) c)) → …
        -/
      · rintro ⟨c, hc, rfl⟩
        /-
          case mp.inl.inr.refine_2.intro.intro
          α₁ : Type u_1
          α₂ : Type u_2
          β₁ : Type u_3
          β₂ : Type u_4
          γ₁ : Type u_5
          γ₂ : Type u_6
          f₁ : α₁ → β₁ → Finset γ₁
          f₂ : α₂ → β₂ → Finset γ₂
          g₁ : α₁ → β₂ → Finset γ₁
          g₂ : α₁ → β₂ → Finset γ₂
          a : α₁
          b : β₂
          c : γ₂
          hc : Membership.mem (g₂ a b) c
          h : Membership.mem (Finset.sumLexLift f₁ f₂ g₁ g₂ (Sum.inl a) (Sum.inr b)) (Su …
          ⊢ Or (Exists fun a₁ => Exists fun b₁ => Exists fun c₁ => And (Eq (Sum.inl a) ( …
        -/
        exact Or.inr (Or.inr <| Or.inl ⟨a, b, c, rfl, rfl, rfl, hc⟩)
        /-
          🎉 no goals
        -/
      /-
        case mp.inr.inl
        α₁ : Type u_1
        α₂ : Type u_2
        β₁ : Type u_3
        β₂ : Type u_4
        γ₁ : Type u_5
        γ₂ : Type u_6
        f₁ : α₁ → β₁ → Finset γ₁
        f₂ : α₂ → β₂ → Finset γ₂
        g₁ : α₁ → β₂ → Finset γ₁
        g₂ : α₁ → β₂ → Finset γ₂
        c : Sum γ₁ γ₂
        a : α₂
        b : β₁
        ⊢ Membership.mem (Finset.sumLexLift f₁ f₂ g₁ g₂ (Sum.inr a) (Sum.inl b)) c → O …
      -/
    · exact fun h ↦ (not_mem_empty _ h).elim
      /-
        🎉 no goals
      -/
      /-
        case mp.inr.inr
        α₁ : Type u_1
        α₂ : Type u_2
        β₁ : Type u_3
        β₂ : Type u_4
        γ₁ : Type u_5
        γ₂ : Type u_6
        f₁ : α₁ → β₁ → Finset γ₁
        f₂ : α₂ → β₂ → Finset γ₂
        g₁ : α₁ → β₂ → Finset γ₁
        g₂ : α₁ → β₂ → Finset γ₂
        c : Sum γ₁ γ₂
        a : α₂
        b : β₂
        ⊢ Membership.mem (Finset.sumLexLift f₁ f₂ g₁ g₂ (Sum.inr a) (Sum.inr b)) c → O …
      -/
    · rw [sumLexLift, mem_map]
      /-
        case mp.inr.inr
        α₁ : Type u_1
        α₂ : Type u_2
        β₁ : Type u_3
        β₂ : Type u_4
        γ₁ : Type u_5
        γ₂ : Type u_6
        f₁ : α₁ → β₁ → Finset γ₁
        f₂ : α₂ → β₂ → Finset γ₂
        g₁ : α₁ → β₂ → Finset γ₁
        g₂ : α₁ → β₂ → Finset γ₂
        c : Sum γ₁ γ₂
        a : α₂
        b : β₂
        ⊢ (Exists fun a_1 => And (Membership.mem (f₂ a b) a_1) (Eq ({ toFun := Sum.inr …
      -/
      rintro ⟨c, hc, rfl⟩
      /-
        case mp.inr.inr.intro.intro
        α₁ : Type u_1
        α₂ : Type u_2
        β₁ : Type u_3
        β₂ : Type u_4
        γ₁ : Type u_5
        γ₂ : Type u_6
        f₁ : α₁ → β₁ → Finset γ₁
        f₂ : α₂ → β₂ → Finset γ₂
        g₁ : α₁ → β₂ → Finset γ₁
        g₂ : α₁ → β₂ → Finset γ₂
        a : α₂
        b : β₂
        c : γ₂
        hc : Membership.mem (f₂ a b) c
        ⊢ Or (Exists fun a₁ => Exists fun b₁ => Exists fun c₁ => And (Eq (Sum.inr a) ( …
      -/
      exact Or.inr (Or.inr <| Or.inr <| ⟨a, b, c, rfl, rfl, rfl, hc⟩)
      /-
        🎉 no goals
      -/
  · rintro (⟨a, b, c, rfl, rfl, rfl, hc⟩ | ⟨a, b, c, rfl, rfl, rfl, hc⟩ |
      ⟨a, b, c, rfl, rfl, rfl, hc⟩ | ⟨a, b, c, rfl, rfl, rfl, hc⟩)
      /-
        case mpr.inl.intro.intro.intro.intro.intro.intro
        α₁ : Type u_1
        α₂ : Type u_2
        β₁ : Type u_3
        β₂ : Type u_4
        γ₁ : Type u_5
        γ₂ : Type u_6
        f₁ : α₁ → β₁ → Finset γ₁
        f₂ : α₂ → β₂ → Finset γ₂
        g₁ : α₁ → β₂ → Finset γ₁
        g₂ : α₁ → β₂ → Finset γ₂
        a : α₁
        b : β₁
        c : γ₁
        hc : Membership.mem (f₁ a b) c
        ⊢ Membership.mem (Finset.sumLexLift f₁ f₂ g₁ g₂ (Sum.inl a) (Sum.inl b)) (Sum. …
      -/
    · exact mem_map_of_mem _ hc
      /-
        🎉 no goals
      -/
      /-
        case mpr.inr.inl.intro.intro.intro.intro.intro.intro
        α₁ : Type u_1
        α₂ : Type u_2
        β₁ : Type u_3
        β₂ : Type u_4
        γ₁ : Type u_5
        γ₂ : Type u_6
        f₁ : α₁ → β₁ → Finset γ₁
        f₂ : α₂ → β₂ → Finset γ₂
        g₁ : α₁ → β₂ → Finset γ₁
        g₂ : α₁ → β₂ → Finset γ₂
        a : α₁
        b : β₂
        c : γ₁
        hc : Membership.mem (g₁ a b) c
        ⊢ Membership.mem (Finset.sumLexLift f₁ f₂ g₁ g₂ (Sum.inl a) (Sum.inr b)) (Sum. …
      -/
    · exact inl_mem_disjSum.2 hc
      /-
        🎉 no goals
      -/
      /-
        case mpr.inr.inr.inl.intro.intro.intro.intro.intro.intro
        α₁ : Type u_1
        α₂ : Type u_2
        β₁ : Type u_3
        β₂ : Type u_4
        γ₁ : Type u_5
        γ₂ : Type u_6
        f₁ : α₁ → β₁ → Finset γ₁
        f₂ : α₂ → β₂ → Finset γ₂
        g₁ : α₁ → β₂ → Finset γ₁
        g₂ : α₁ → β₂ → Finset γ₂
        a : α₁
        b : β₂
        c : γ₂
        hc : Membership.mem (g₂ a b) c
        ⊢ Membership.mem (Finset.sumLexLift f₁ f₂ g₁ g₂ (Sum.inl a) (Sum.inr b)) (Sum. …
      -/
    · exact inr_mem_disjSum.2 hc
      /-
        🎉 no goals
      -/
      /-
        case mpr.inr.inr.inr.intro.intro.intro.intro.intro.intro
        α₁ : Type u_1
        α₂ : Type u_2
        β₁ : Type u_3
        β₂ : Type u_4
        γ₁ : Type u_5
        γ₂ : Type u_6
        f₁ : α₁ → β₁ → Finset γ₁
        f₂ : α₂ → β₂ → Finset γ₂
        g₁ : α₁ → β₂ → Finset γ₁
        g₂ : α₁ → β₂ → Finset γ₂
        a : α₂
        b : β₂
        c : γ₂
        hc : Membership.mem (f₂ a b) c
        ⊢ Membership.mem (Finset.sumLexLift f₁ f₂ g₁ g₂ (Sum.inr a) (Sum.inr b)) (Sum. …
      -/
    · exact mem_map_of_mem _ hc
      /-
        🎉 no goals
      -/


lemma inl_mem_sumLexLift {c₁ : γ₁} :
    inl c₁ ∈ sumLexLift f₁ f₂ g₁ g₂ a b ↔
      (∃ a₁ b₁, a = inl a₁ ∧ b = inl b₁ ∧ c₁ ∈ f₁ a₁ b₁) ∨
        ∃ a₁ b₂, a = inl a₁ ∧ b = inr b₂ ∧ c₁ ∈ g₁ a₁ b₂ := by
  /-
    α₁ : Type u_1
    α₂ : Type u_2
    β₁ : Type u_3
    β₂ : Type u_4
    γ₁ : Type u_5
    γ₂ : Type u_6
    f₁ : α₁ → β₁ → Finset γ₁
    f₂ : α₂ → β₂ → Finset γ₂
    g₁ : α₁ → β₂ → Finset γ₁
    g₂ : α₁ → β₂ → Finset γ₂
    a : Sum α₁ α₂
    b : Sum β₁ β₂
    c₁ : γ₁
    ⊢ Iff (Membership.mem (Finset.sumLexLift f₁ f₂ g₁ g₂ a b) (Sum.inl c₁)) (Or (E …
  -/
  simp [mem_sumLexLift]
  /-
    🎉 no goals
  -/


lemma inr_mem_sumLexLift {c₂ : γ₂} :
    inr c₂ ∈ sumLexLift f₁ f₂ g₁ g₂ a b ↔
      (∃ a₁ b₂, a = inl a₁ ∧ b = inr b₂ ∧ c₂ ∈ g₂ a₁ b₂) ∨
        ∃ a₂ b₂, a = inr a₂ ∧ b = inr b₂ ∧ c₂ ∈ f₂ a₂ b₂ := by
  /-
    α₁ : Type u_1
    α₂ : Type u_2
    β₁ : Type u_3
    β₂ : Type u_4
    γ₁ : Type u_5
    γ₂ : Type u_6
    f₁ : α₁ → β₁ → Finset γ₁
    f₂ : α₂ → β₂ → Finset γ₂
    g₁ : α₁ → β₂ → Finset γ₁
    g₂ : α₁ → β₂ → Finset γ₂
    a : Sum α₁ α₂
    b : Sum β₁ β₂
    c₂ : γ₂
    ⊢ Iff (Membership.mem (Finset.sumLexLift f₁ f₂ g₁ g₂ a b) (Sum.inr c₂)) (Or (E …
  -/
  simp [mem_sumLexLift]
  /-
    🎉 no goals
  -/


lemma sumLexLift_mono (hf₁ : ∀ a b, f₁ a b ⊆ f₁' a b) (hf₂ : ∀ a b, f₂ a b ⊆ f₂' a b)
    (hg₁ : ∀ a b, g₁ a b ⊆ g₁' a b) (hg₂ : ∀ a b, g₂ a b ⊆ g₂' a b) (a : α₁ ⊕ α₂)
    (b : β₁ ⊕ β₂) : sumLexLift f₁ f₂ g₁ g₂ a b ⊆ sumLexLift f₁' f₂' g₁' g₂' a b := by
  /-
    α₁ : Type u_1
    α₂ : Type u_2
    β₁ : Type u_3
    β₂ : Type u_4
    γ₁ : Type u_5
    γ₂ : Type u_6
    f₁ f₁' : α₁ → β₁ → Finset γ₁
    f₂ f₂' : α₂ → β₂ → Finset γ₂
    g₁ g₁' : α₁ → β₂ → Finset γ₁
    g₂ g₂' : α₁ → β₂ → Finset γ₂
    hf₁ : ∀ (a : α₁) (b : β₁), HasSubset.Subset (f₁ a b) (f₁' a b)
    hf₂ : ∀ (a : α₂) (b : β₂), HasSubset.Subset (f₂ a b) (f₂' a b)
    hg₁ : ∀ (a : α₁) (b : β₂), HasSubset.Subset (g₁ a b) (g₁' a b)
    hg₂ : ∀ (a : α₁) (b : β₂), HasSubset.Subset (g₂ a b) (g₂' a b)
    a : Sum α₁ α₂
    b : Sum β₁ β₂
    ⊢ HasSubset.Subset (Finset.sumLexLift f₁ f₂ g₁ g₂ a b) (Finset.sumLexLift f₁'  …
  -/
  cases a <;> cases b
  exacts [map_subset_map.2 (hf₁ _ _), disjSum_mono (hg₁ _ _) (hg₂ _ _), Subset.rfl,
    map_subset_map.2 (hf₂ _ _)]


lemma sumLexLift_eq_empty :
    sumLexLift f₁ f₂ g₁ g₂ a b = ∅ ↔
      (∀ a₁ b₁, a = inl a₁ → b = inl b₁ → f₁ a₁ b₁ = ∅) ∧
        (∀ a₁ b₂, a = inl a₁ → b = inr b₂ → g₁ a₁ b₂ = ∅ ∧ g₂ a₁ b₂ = ∅) ∧
          ∀ a₂ b₂, a = inr a₂ → b = inr b₂ → f₂ a₂ b₂ = ∅ := by
  /-
    α₁ : Type u_1
    α₂ : Type u_2
    β₁ : Type u_3
    β₂ : Type u_4
    γ₁ : Type u_5
    γ₂ : Type u_6
    f₁ : α₁ → β₁ → Finset γ₁
    f₂ : α₂ → β₂ → Finset γ₂
    g₁ : α₁ → β₂ → Finset γ₁
    g₂ : α₁ → β₂ → Finset γ₂
    a : Sum α₁ α₂
    b : Sum β₁ β₂
    ⊢ Iff (Eq (Finset.sumLexLift f₁ f₂ g₁ g₂ a b) EmptyCollection.emptyCollection) …
  -/
  refine ⟨fun h ↦ ⟨?_, ?_, ?_⟩, fun h ↦ ?_⟩
  /-
    case refine_1
    α₁ : Type u_1
    α₂ : Type u_2
    β₁ : Type u_3
    β₂ : Type u_4
    γ₁ : Type u_5
    γ₂ : Type u_6
    f₁ : α₁ → β₁ → Finset γ₁
    f₂ : α₂ → β₂ → Finset γ₂
    g₁ : α₁ → β₂ → Finset γ₁
    g₂ : α₁ → β₂ → Finset γ₂
    a : Sum α₁ α₂
    b : Sum β₁ β₂
    h : Eq (Finset.sumLexLift f₁ f₂ g₁ g₂ a b) EmptyCollection.emptyCollection
    ⊢ ∀ (a₁ : α₁) (b₁ : β₁), Eq a (Sum.inl a₁) → Eq b (Sum.inl b₁) → Eq (f₁ a₁ b₁) …
  -/
  any_goals rintro a b rfl rfl; exact map_eq_empty.1 h
    /-
      case refine_2
      α₁ : Type u_1
      α₂ : Type u_2
      β₁ : Type u_3
      β₂ : Type u_4
      γ₁ : Type u_5
      γ₂ : Type u_6
      f₁ : α₁ → β₁ → Finset γ₁
      f₂ : α₂ → β₂ → Finset γ₂
      g₁ : α₁ → β₂ → Finset γ₁
      g₂ : α₁ → β₂ → Finset γ₂
      a : Sum α₁ α₂
      b : Sum β₁ β₂
      h : Eq (Finset.sumLexLift f₁ f₂ g₁ g₂ a b) EmptyCollection.emptyCollection
      ⊢ ∀ (a₁ : α₁) (b₂ : β₂), Eq a (Sum.inl a₁) → Eq b (Sum.inr b₂) → And (Eq (g₁ a …
    -/
  · rintro a b rfl rfl; exact disjSum_eq_empty.1 h
                        /-
                          🎉 no goals
                        -/
  /-
    case refine_4
    α₁ : Type u_1
    α₂ : Type u_2
    β₁ : Type u_3
    β₂ : Type u_4
    γ₁ : Type u_5
    γ₂ : Type u_6
    f₁ : α₁ → β₁ → Finset γ₁
    f₂ : α₂ → β₂ → Finset γ₂
    g₁ : α₁ → β₂ → Finset γ₁
    g₂ : α₁ → β₂ → Finset γ₂
    a : Sum α₁ α₂
    b : Sum β₁ β₂
    h : And (∀ (a₁ : α₁) (b₁ : β₁), Eq a (Sum.inl a₁) → Eq b (Sum.inl b₁) → Eq (f₁ …
    ⊢ Eq (Finset.sumLexLift f₁ f₂ g₁ g₂ a b) EmptyCollection.emptyCollection
  -/
  cases a <;> cases b
    /-
      case refine_4.inl.inl
      α₁ : Type u_1
      α₂ : Type u_2
      β₁ : Type u_3
      β₂ : Type u_4
      γ₁ : Type u_5
      γ₂ : Type u_6
      f₁ : α₁ → β₁ → Finset γ₁
      f₂ : α₂ → β₂ → Finset γ₂
      g₁ : α₁ → β₂ → Finset γ₁
      g₂ : α₁ → β₂ → Finset γ₂
      val✝¹ : α₁
      val✝ : β₁
      h : And (∀ (a₁ : α₁) (b₁ : β₁), Eq (Sum.inl val✝¹) (Sum.inl a₁) → Eq (Sum.inl  …
      ⊢ Eq (Finset.sumLexLift f₁ f₂ g₁ g₂ (Sum.inl val✝¹) (Sum.inl val✝)) EmptyColle …
    -/
  · exact map_eq_empty.2 (h.1 _ _ rfl rfl)
    /-
      🎉 no goals
    -/
    /-
      case refine_4.inl.inr
      α₁ : Type u_1
      α₂ : Type u_2
      β₁ : Type u_3
      β₂ : Type u_4
      γ₁ : Type u_5
      γ₂ : Type u_6
      f₁ : α₁ → β₁ → Finset γ₁
      f₂ : α₂ → β₂ → Finset γ₂
      g₁ : α₁ → β₂ → Finset γ₁
      g₂ : α₁ → β₂ → Finset γ₂
      val✝¹ : α₁
      val✝ : β₂
      h : And (∀ (a₁ : α₁) (b₁ : β₁), Eq (Sum.inl val✝¹) (Sum.inl a₁) → Eq (Sum.inr  …
      ⊢ Eq (Finset.sumLexLift f₁ f₂ g₁ g₂ (Sum.inl val✝¹) (Sum.inr val✝)) EmptyColle …
    -/
  · simp [h.2.1 _ _ rfl rfl]
    /-
      🎉 no goals
    -/
    /-
      case refine_4.inr.inl
      α₁ : Type u_1
      α₂ : Type u_2
      β₁ : Type u_3
      β₂ : Type u_4
      γ₁ : Type u_5
      γ₂ : Type u_6
      f₁ : α₁ → β₁ → Finset γ₁
      f₂ : α₂ → β₂ → Finset γ₂
      g₁ : α₁ → β₂ → Finset γ₁
      g₂ : α₁ → β₂ → Finset γ₂
      val✝¹ : α₂
      val✝ : β₁
      h : And (∀ (a₁ : α₁) (b₁ : β₁), Eq (Sum.inr val✝¹) (Sum.inl a₁) → Eq (Sum.inl  …
      ⊢ Eq (Finset.sumLexLift f₁ f₂ g₁ g₂ (Sum.inr val✝¹) (Sum.inl val✝)) EmptyColle …
    -/
  · rfl
    /-
      🎉 no goals
    -/
    /-
      case refine_4.inr.inr
      α₁ : Type u_1
      α₂ : Type u_2
      β₁ : Type u_3
      β₂ : Type u_4
      γ₁ : Type u_5
      γ₂ : Type u_6
      f₁ : α₁ → β₁ → Finset γ₁
      f₂ : α₂ → β₂ → Finset γ₂
      g₁ : α₁ → β₂ → Finset γ₁
      g₂ : α₁ → β₂ → Finset γ₂
      val✝¹ : α₂
      val✝ : β₂
      h : And (∀ (a₁ : α₁) (b₁ : β₁), Eq (Sum.inr val✝¹) (Sum.inl a₁) → Eq (Sum.inr  …
      ⊢ Eq (Finset.sumLexLift f₁ f₂ g₁ g₂ (Sum.inr val✝¹) (Sum.inr val✝)) EmptyColle …
    -/
  · exact map_eq_empty.2 (h.2.2 _ _ rfl rfl)
    /-
      🎉 no goals
    -/


lemma sumLexLift_nonempty :
    (sumLexLift f₁ f₂ g₁ g₂ a b).Nonempty ↔
      (∃ a₁ b₁, a = inl a₁ ∧ b = inl b₁ ∧ (f₁ a₁ b₁).Nonempty) ∨
        (∃ a₁ b₂, a = inl a₁ ∧ b = inr b₂ ∧ ((g₁ a₁ b₂).Nonempty ∨ (g₂ a₁ b₂).Nonempty)) ∨
          ∃ a₂ b₂, a = inr a₂ ∧ b = inr b₂ ∧ (f₂ a₂ b₂).Nonempty := by
  -- Porting note (https://github.com/leanprover-community/mathlib4/issues/10745): was `simp [nonempty_iff_ne_empty, sumLexLift_eq_empty, not_and_or]`.
  -- Could add `-exists_and_left, -not_and, -exists_and_right` but easier to squeeze.
  simp only [nonempty_iff_ne_empty, Ne, sumLexLift_eq_empty, not_and_or, exists_prop,
    not_forall]


instance instLocallyFiniteOrder : LocallyFiniteOrder (α ⊕ β) where
  finsetIcc := sumLift₂ Icc Icc
  finsetIco := sumLift₂ Ico Ico
  finsetIoc := sumLift₂ Ioc Ioc
  finsetIoo := sumLift₂ Ioo Ioo
                       /-
                         α : Type u_1
                         β : Type u_2
                         inst✝³ : Preorder α
                         inst✝² : Preorder β
                         inst✝¹ : LocallyFiniteOrder α
                         inst✝ : LocallyFiniteOrder β
                         ⊢ ∀ (a b x : Sum α β), Iff (Membership.mem (Finset.sumLift₂ Finset.Icc Finset. …
                       -/
                                                          /-
                                                            🎉 no goals
                                                          -/
                                                          /-
                                                            🎉 no goals
                                                          -/
                                                          /-
                                                            🎉 no goals
                                                          -/
                                                          /-
                                                            🎉 no goals
                                                          -/
                                                          /-
                                                            🎉 no goals
                                                          -/
                                                          /-
                                                            🎉 no goals
                                                          -/
                                                          /-
                                                            🎉 no goals
                                                          -/
  finset_mem_Icc := by rintro (a | a) (b | b) (x | x) <;> simp
                                                          /-
                                                            🎉 no goals
                                                          -/
                       /-
                         α : Type u_1
                         β : Type u_2
                         inst✝³ : Preorder α
                         inst✝² : Preorder β
                         inst✝¹ : LocallyFiniteOrder α
                         inst✝ : LocallyFiniteOrder β
                         ⊢ ∀ (a b x : Sum α β), Iff (Membership.mem (Finset.sumLift₂ Finset.Ico Finset. …
                       -/
                                                          /-
                                                            🎉 no goals
                                                          -/
                                                          /-
                                                            🎉 no goals
                                                          -/
                                                          /-
                                                            🎉 no goals
                                                          -/
                                                          /-
                                                            🎉 no goals
                                                          -/
                                                          /-
                                                            🎉 no goals
                                                          -/
                                                          /-
                                                            🎉 no goals
                                                          -/
                                                          /-
                                                            🎉 no goals
                                                          -/
  finset_mem_Ico := by rintro (a | a) (b | b) (x | x) <;> simp
                                                          /-
                                                            🎉 no goals
                                                          -/
                       /-
                         α : Type u_1
                         β : Type u_2
                         inst✝³ : Preorder α
                         inst✝² : Preorder β
                         inst✝¹ : LocallyFiniteOrder α
                         inst✝ : LocallyFiniteOrder β
                         ⊢ ∀ (a b x : Sum α β), Iff (Membership.mem (Finset.sumLift₂ Finset.Ioc Finset. …
                       -/
                                                          /-
                                                            🎉 no goals
                                                          -/
                                                          /-
                                                            🎉 no goals
                                                          -/
                                                          /-
                                                            🎉 no goals
                                                          -/
                                                          /-
                                                            🎉 no goals
                                                          -/
                                                          /-
                                                            🎉 no goals
                                                          -/
                                                          /-
                                                            🎉 no goals
                                                          -/
                                                          /-
                                                            🎉 no goals
                                                          -/
  finset_mem_Ioc := by rintro (a | a) (b | b) (x | x) <;> simp
                                                          /-
                                                            🎉 no goals
                                                          -/
                       /-
                         α : Type u_1
                         β : Type u_2
                         inst✝³ : Preorder α
                         inst✝² : Preorder β
                         inst✝¹ : LocallyFiniteOrder α
                         inst✝ : LocallyFiniteOrder β
                         ⊢ ∀ (a b x : Sum α β), Iff (Membership.mem (Finset.sumLift₂ Finset.Ioo Finset. …
                       -/
                                                          /-
                                                            🎉 no goals
                                                          -/
                                                          /-
                                                            🎉 no goals
                                                          -/
                                                          /-
                                                            🎉 no goals
                                                          -/
                                                          /-
                                                            🎉 no goals
                                                          -/
                                                          /-
                                                            🎉 no goals
                                                          -/
                                                          /-
                                                            🎉 no goals
                                                          -/
                                                          /-
                                                            🎉 no goals
                                                          -/
  finset_mem_Ioo := by rintro (a | a) (b | b) (x | x) <;> simp
                                                          /-
                                                            🎉 no goals
                                                          -/


theorem Icc_inl_inl : Icc (inl a₁ : α ⊕ β) (inl a₂) = (Icc a₁ a₂).map Embedding.inl :=
  rfl


theorem Ico_inl_inl : Ico (inl a₁ : α ⊕ β) (inl a₂) = (Ico a₁ a₂).map Embedding.inl :=
  rfl


theorem Ioc_inl_inl : Ioc (inl a₁ : α ⊕ β) (inl a₂) = (Ioc a₁ a₂).map Embedding.inl :=
  rfl


theorem Ioo_inl_inl : Ioo (inl a₁ : α ⊕ β) (inl a₂) = (Ioo a₁ a₂).map Embedding.inl :=
  rfl


@[simp]
theorem Icc_inl_inr : Icc (inl a₁) (inr b₂) = ∅ :=
  rfl


@[simp]
theorem Ico_inl_inr : Ico (inl a₁) (inr b₂) = ∅ :=
  rfl


@[simp]
theorem Ioc_inl_inr : Ioc (inl a₁) (inr b₂) = ∅ :=
  rfl


@[simp]
theorem Ioo_inl_inr : Ioo (inl a₁) (inr b₂) = ∅ := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝³ : Preorder α
    inst✝² : Preorder β
    inst✝¹ : LocallyFiniteOrder α
    inst✝ : LocallyFiniteOrder β
    a₁ : α
    b₂ : β
    ⊢ Eq (Finset.Ioo (Sum.inl a₁) (Sum.inr b₂)) EmptyCollection.emptyCollection
  -/
  rfl
  /-
    🎉 no goals
  -/


@[simp]
theorem Icc_inr_inl : Icc (inr b₁) (inl a₂) = ∅ :=
  rfl


@[simp]
theorem Ico_inr_inl : Ico (inr b₁) (inl a₂) = ∅ :=
  rfl


@[simp]
theorem Ioc_inr_inl : Ioc (inr b₁) (inl a₂) = ∅ :=
  rfl


@[simp]
theorem Ioo_inr_inl : Ioo (inr b₁) (inl a₂) = ∅ := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝³ : Preorder α
    inst✝² : Preorder β
    inst✝¹ : LocallyFiniteOrder α
    inst✝ : LocallyFiniteOrder β
    a₂ : α
    b₁ : β
    ⊢ Eq (Finset.Ioo (Sum.inr b₁) (Sum.inl a₂)) EmptyCollection.emptyCollection
  -/
  rfl
  /-
    🎉 no goals
  -/


theorem Icc_inr_inr : Icc (inr b₁ : α ⊕ β) (inr b₂) = (Icc b₁ b₂).map Embedding.inr :=
  rfl


theorem Ico_inr_inr : Ico (inr b₁ : α ⊕ β) (inr b₂) = (Ico b₁ b₂).map Embedding.inr :=
  rfl


theorem Ioc_inr_inr : Ioc (inr b₁ : α ⊕ β) (inr b₂) = (Ioc b₁ b₂).map Embedding.inr :=
  rfl


theorem Ioo_inr_inr : Ioo (inr b₁ : α ⊕ β) (inr b₂) = (Ioo b₁ b₂).map Embedding.inr :=
  rfl


/-- Throwaway tactic. -/
local elab "simp_lex" : tactic => do
  Lean.Elab.Tactic.evalTactic <| ← `(tactic|
    refine toLex.surjective.forall₃.2 ?_;
    rintro (a | a) (b | b) (c | c) <;> simp only
      [sumLexLift_inl_inl, sumLexLift_inl_inr, sumLexLift_inr_inl, sumLexLift_inr_inr,
        inl_le_inl_iff, inl_le_inr, not_inr_le_inl, inr_le_inr_iff, inl_lt_inl_iff, inl_lt_inr,
        not_inr_lt_inl, inr_lt_inr_iff, mem_Icc, mem_Ico, mem_Ioc, mem_Ioo, mem_Ici, mem_Ioi,
        mem_Iic, mem_Iio, Equiv.coe_toEmbedding, toLex_inj, exists_false, and_false, false_and,
        map_empty, not_mem_empty, true_and, inl_mem_disjSum, inr_mem_disjSum, and_true, ofLex_toLex,
        mem_map, Embedding.coeFn_mk, exists_prop, exists_eq_right, Embedding.inl_apply,
        -- Porting note: added
        inl.injEq, inr.injEq, reduceCtorEq]
  )


instance locallyFiniteOrder : LocallyFiniteOrder (α ⊕ₗ β) where
  finsetIcc a b :=
    (sumLexLift Icc Icc (fun a _ => Ici a) (fun _ => Iic) (ofLex a) (ofLex b)).map toLex.toEmbedding
  finsetIco a b :=
    (sumLexLift Ico Ico (fun a _ => Ici a) (fun _ => Iio) (ofLex a) (ofLex b)).map toLex.toEmbedding
  finsetIoc a b :=
    (sumLexLift Ioc Ioc (fun a _ => Ioi a) (fun _ => Iic) (ofLex a) (ofLex b)).map toLex.toEmbedding
  finsetIoo a b :=
    (sumLexLift Ioo Ioo (fun a _ => Ioi a) (fun _ => Iio) (ofLex a) (ofLex b)).map toLex.toEmbedding
                       /-
                         α : Type u_1
                         β : Type u_2
                         inst✝⁵ : Preorder α
                         inst✝⁴ : Preorder β
                         inst✝³ : OrderTop α
                         inst✝² : OrderBot β
                         inst✝¹ : LocallyFiniteOrder α
                         inst✝ : LocallyFiniteOrder β
                         ⊢ ∀ (a b x : _root_.Lex (Sum α β)), Iff (Membership.mem ((fun a b => Finset.ma …
                       -/
  finset_mem_Icc := by simp_lex
                       /-
                         🎉 no goals
                       -/
                       /-
                         α : Type u_1
                         β : Type u_2
                         inst✝⁵ : Preorder α
                         inst✝⁴ : Preorder β
                         inst✝³ : OrderTop α
                         inst✝² : OrderBot β
                         inst✝¹ : LocallyFiniteOrder α
                         inst✝ : LocallyFiniteOrder β
                         ⊢ ∀ (a b x : _root_.Lex (Sum α β)), Iff (Membership.mem ((fun a b => Finset.ma …
                       -/
  finset_mem_Ico := by simp_lex
                       /-
                         🎉 no goals
                       -/
                       /-
                         α : Type u_1
                         β : Type u_2
                         inst✝⁵ : Preorder α
                         inst✝⁴ : Preorder β
                         inst✝³ : OrderTop α
                         inst✝² : OrderBot β
                         inst✝¹ : LocallyFiniteOrder α
                         inst✝ : LocallyFiniteOrder β
                         ⊢ ∀ (a b x : _root_.Lex (Sum α β)), Iff (Membership.mem ((fun a b => Finset.ma …
                       -/
  finset_mem_Ioc := by simp_lex
                       /-
                         🎉 no goals
                       -/
                       /-
                         α : Type u_1
                         β : Type u_2
                         inst✝⁵ : Preorder α
                         inst✝⁴ : Preorder β
                         inst✝³ : OrderTop α
                         inst✝² : OrderBot β
                         inst✝¹ : LocallyFiniteOrder α
                         inst✝ : LocallyFiniteOrder β
                         ⊢ ∀ (a b x : _root_.Lex (Sum α β)), Iff (Membership.mem ((fun a b => Finset.ma …
                       -/
  finset_mem_Ioo := by simp_lex
                       /-
                         🎉 no goals
                       -/


lemma Icc_inl_inl :
    Icc (inlₗ a₁ : α ⊕ₗ β) (inlₗ a₂) = (Icc a₁ a₂).map (Embedding.inl.trans toLex.toEmbedding) := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝⁵ : Preorder α
    inst✝⁴ : Preorder β
    inst✝³ : OrderTop α
    inst✝² : OrderBot β
    inst✝¹ : LocallyFiniteOrder α
    inst✝ : LocallyFiniteOrder β
    a₁ a₂ : α
    ⊢ Eq (Finset.Icc (Sum.inlₗ a₁) (Sum.inlₗ a₂)) (Finset.map (Function.Embedding. …
  -/
  rw [← Finset.map_map]; rfl
                         /-
                           🎉 no goals
                         -/


lemma Ico_inl_inl :
    Ico (inlₗ a₁ : α ⊕ₗ β) (inlₗ a₂) = (Ico a₁ a₂).map (Embedding.inl.trans toLex.toEmbedding) := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝⁵ : Preorder α
    inst✝⁴ : Preorder β
    inst✝³ : OrderTop α
    inst✝² : OrderBot β
    inst✝¹ : LocallyFiniteOrder α
    inst✝ : LocallyFiniteOrder β
    a₁ a₂ : α
    ⊢ Eq (Finset.Ico (Sum.inlₗ a₁) (Sum.inlₗ a₂)) (Finset.map (Function.Embedding. …
  -/
  rw [← Finset.map_map]; rfl
                         /-
                           🎉 no goals
                         -/


lemma Ioc_inl_inl :
    Ioc (inlₗ a₁ : α ⊕ₗ β) (inlₗ a₂) = (Ioc a₁ a₂).map (Embedding.inl.trans toLex.toEmbedding) := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝⁵ : Preorder α
    inst✝⁴ : Preorder β
    inst✝³ : OrderTop α
    inst✝² : OrderBot β
    inst✝¹ : LocallyFiniteOrder α
    inst✝ : LocallyFiniteOrder β
    a₁ a₂ : α
    ⊢ Eq (Finset.Ioc (Sum.inlₗ a₁) (Sum.inlₗ a₂)) (Finset.map (Function.Embedding. …
  -/
  rw [← Finset.map_map]; rfl
                         /-
                           🎉 no goals
                         -/


lemma Ioo_inl_inl :
    Ioo (inlₗ a₁ : α ⊕ₗ β) (inlₗ a₂) = (Ioo a₁ a₂).map (Embedding.inl.trans toLex.toEmbedding) := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝⁵ : Preorder α
    inst✝⁴ : Preorder β
    inst✝³ : OrderTop α
    inst✝² : OrderBot β
    inst✝¹ : LocallyFiniteOrder α
    inst✝ : LocallyFiniteOrder β
    a₁ a₂ : α
    ⊢ Eq (Finset.Ioo (Sum.inlₗ a₁) (Sum.inlₗ a₂)) (Finset.map (Function.Embedding. …
  -/
  rw [← Finset.map_map]; rfl
                         /-
                           🎉 no goals
                         -/


@[simp]
lemma Icc_inl_inr : Icc (inlₗ a) (inrₗ b) = ((Ici a).disjSum (Iic b)).map toLex.toEmbedding := rfl


@[simp]
lemma Ico_inl_inr : Ico (inlₗ a) (inrₗ b) = ((Ici a).disjSum (Iio b)).map toLex.toEmbedding := rfl


@[simp]
lemma Ioc_inl_inr : Ioc (inlₗ a) (inrₗ b) = ((Ioi a).disjSum (Iic b)).map toLex.toEmbedding := rfl


@[simp]
lemma Ioo_inl_inr : Ioo (inlₗ a) (inrₗ b) = ((Ioi a).disjSum (Iio b)).map toLex.toEmbedding := rfl


@[simp]
lemma Icc_inr_inl : Icc (inrₗ b) (inlₗ a) = ∅ := rfl


@[simp]
lemma Ico_inr_inl : Ico (inrₗ b) (inlₗ a) = ∅ := rfl


@[simp]
lemma Ioc_inr_inl : Ioc (inrₗ b) (inlₗ a) = ∅ := rfl


@[simp]
lemma Ioo_inr_inl : Ioo (inrₗ b) (inlₗ a) = ∅ := rfl


lemma Icc_inr_inr :
    Icc (inrₗ b₁ : α ⊕ₗ β) (inrₗ b₂) = (Icc b₁ b₂).map (Embedding.inr.trans toLex.toEmbedding) := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝⁵ : Preorder α
    inst✝⁴ : Preorder β
    inst✝³ : OrderTop α
    inst✝² : OrderBot β
    inst✝¹ : LocallyFiniteOrder α
    inst✝ : LocallyFiniteOrder β
    b₁ b₂ : β
    ⊢ Eq (Finset.Icc (Sum.inrₗ b₁) (Sum.inrₗ b₂)) (Finset.map (Function.Embedding. …
  -/
  rw [← Finset.map_map]; rfl
                         /-
                           🎉 no goals
                         -/


lemma Ico_inr_inr :
    Ico (inrₗ b₁ : α ⊕ₗ β) (inrₗ b₂) = (Ico b₁ b₂).map (Embedding.inr.trans toLex.toEmbedding) := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝⁵ : Preorder α
    inst✝⁴ : Preorder β
    inst✝³ : OrderTop α
    inst✝² : OrderBot β
    inst✝¹ : LocallyFiniteOrder α
    inst✝ : LocallyFiniteOrder β
    b₁ b₂ : β
    ⊢ Eq (Finset.Ico (Sum.inrₗ b₁) (Sum.inrₗ b₂)) (Finset.map (Function.Embedding. …
  -/
  rw [← Finset.map_map]; rfl
                         /-
                           🎉 no goals
                         -/


lemma Ioc_inr_inr :
    Ioc (inrₗ b₁ : α ⊕ₗ β) (inrₗ b₂) = (Ioc b₁ b₂).map (Embedding.inr.trans toLex.toEmbedding) := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝⁵ : Preorder α
    inst✝⁴ : Preorder β
    inst✝³ : OrderTop α
    inst✝² : OrderBot β
    inst✝¹ : LocallyFiniteOrder α
    inst✝ : LocallyFiniteOrder β
    b₁ b₂ : β
    ⊢ Eq (Finset.Ioc (Sum.inrₗ b₁) (Sum.inrₗ b₂)) (Finset.map (Function.Embedding. …
  -/
  rw [← Finset.map_map]; rfl
                         /-
                           🎉 no goals
                         -/


lemma Ioo_inr_inr :
    Ioo (inrₗ b₁ : α ⊕ₗ β) (inrₗ b₂) = (Ioo b₁ b₂).map (Embedding.inr.trans toLex.toEmbedding) := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝⁵ : Preorder α
    inst✝⁴ : Preorder β
    inst✝³ : OrderTop α
    inst✝² : OrderBot β
    inst✝¹ : LocallyFiniteOrder α
    inst✝ : LocallyFiniteOrder β
    b₁ b₂ : β
    ⊢ Eq (Finset.Ioo (Sum.inrₗ b₁) (Sum.inrₗ b₂)) (Finset.map (Function.Embedding. …
  -/
  rw [← Finset.map_map]; rfl
                         /-
                           🎉 no goals
                         -/


