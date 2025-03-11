theorem pairwise_on_bool (hr : Symmetric r) {a b : α} :
                                                      /-
                                                        α : Type u_1
                                                        r : α → α → Prop
                                                        hr : Symmetric r
                                                        a b : α
                                                        ⊢ Iff (Pairwise (Function.onFun r fun c => cond c a b)) (r a b)
                                                      -/
    Pairwise (r on fun c => cond c a b) ↔ r a b := by simpa [Pairwise, Function.onFun] using @hr a b
                                                      /-
                                                        🎉 no goals
                                                      -/


theorem pairwise_disjoint_on_bool [PartialOrder α] [OrderBot α] {a b : α} :
    Pairwise (Disjoint on fun c => cond c a b) ↔ Disjoint a b :=
  pairwise_on_bool Disjoint.symm


theorem Symmetric.pairwise_on [LinearOrder ι] (hr : Symmetric r) (f : ι → α) :
    Pairwise (r on f) ↔ ∀ ⦃m n⦄, m < n → r (f m) (f n) :=
  ⟨fun h _m _n hmn => h hmn.ne, fun h _m _n hmn => hmn.lt_or_lt.elim (@h _ _) fun h' => hr (h h')⟩


theorem pairwise_disjoint_on [PartialOrder α] [OrderBot α] [LinearOrder ι] (f : ι → α) :
    Pairwise (Disjoint on f) ↔ ∀ ⦃m n⦄, m < n → Disjoint (f m) (f n) :=
  Symmetric.pairwise_on Disjoint.symm f


theorem pairwise_disjoint_mono [PartialOrder α] [OrderBot α] (hs : Pairwise (Disjoint on f))
    (h : g ≤ f) : Pairwise (Disjoint on g) :=
  hs.mono fun i j hij => Disjoint.mono (h i) (h j) hij


theorem Pairwise.disjoint_extend_bot [PartialOrder γ] [OrderBot γ]
    {e : α → β} {f : α → γ} (hf : Pairwise (Disjoint on f)) (he : FactorsThrough f e) :
    Pairwise (Disjoint on extend e f ⊥) := by
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    inst✝¹ : PartialOrder γ
    inst✝ : OrderBot γ
    e : α → β
    f : α → γ
    hf : Pairwise (Function.onFun Disjoint f)
    he : Function.FactorsThrough f e
    ⊢ Pairwise (Function.onFun Disjoint (Function.extend e f Bot.bot))
  -/
  intro b₁ b₂ hne
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    inst✝¹ : PartialOrder γ
    inst✝ : OrderBot γ
    e : α → β
    f : α → γ
    hf : Pairwise (Function.onFun Disjoint f)
    he : Function.FactorsThrough f e
    b₁ b₂ : β
    hne : Ne b₁ b₂
    ⊢ Function.onFun Disjoint (Function.extend e f Bot.bot) b₁ b₂
  -/
  rcases em (∃ a₁, e a₁ = b₁) with ⟨a₁, rfl⟩ | hb₁
    /-
      case inl.intro
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      inst✝¹ : PartialOrder γ
      inst✝ : OrderBot γ
      e : α → β
      f : α → γ
      hf : Pairwise (Function.onFun Disjoint f)
      he : Function.FactorsThrough f e
      b₂ : β
      a₁ : α
      hne : Ne (e a₁) b₂
      ⊢ Function.onFun Disjoint (Function.extend e f Bot.bot) (e a₁) b₂
    -/
  · rcases em (∃ a₂, e a₂ = b₂) with ⟨a₂, rfl⟩ | hb₂
      /-
        case inl.intro.inl.intro
        α : Type u_1
        β : Type u_2
        γ : Type u_3
        inst✝¹ : PartialOrder γ
        inst✝ : OrderBot γ
        e : α → β
        f : α → γ
        hf : Pairwise (Function.onFun Disjoint f)
        he : Function.FactorsThrough f e
        a₁ a₂ : α
        hne : Ne (e a₁) (e a₂)
        ⊢ Function.onFun Disjoint (Function.extend e f Bot.bot) (e a₁) (e a₂)
      -/
    · simpa only [onFun, he.extend_apply] using hf (ne_of_apply_ne e hne)
      /-
        🎉 no goals
      -/
      /-
        case inl.intro.inr
        α : Type u_1
        β : Type u_2
        γ : Type u_3
        inst✝¹ : PartialOrder γ
        inst✝ : OrderBot γ
        e : α → β
        f : α → γ
        hf : Pairwise (Function.onFun Disjoint f)
        he : Function.FactorsThrough f e
        b₂ : β
        a₁ : α
        hne : Ne (e a₁) b₂
        hb₂ : Not (Exists fun a₂ => Eq (e a₂) b₂)
        ⊢ Function.onFun Disjoint (Function.extend e f Bot.bot) (e a₁) b₂
      -/
    · simpa only [onFun, extend_apply' _ _ _ hb₂] using disjoint_bot_right
      /-
        🎉 no goals
      -/
    /-
      case inr
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      inst✝¹ : PartialOrder γ
      inst✝ : OrderBot γ
      e : α → β
      f : α → γ
      hf : Pairwise (Function.onFun Disjoint f)
      he : Function.FactorsThrough f e
      b₁ b₂ : β
      hne : Ne b₁ b₂
      hb₁ : Not (Exists fun a₁ => Eq (e a₁) b₁)
      ⊢ Function.onFun Disjoint (Function.extend e f Bot.bot) b₁ b₂
    -/
  · simpa only [onFun, extend_apply' _ _ _ hb₁] using disjoint_bot_left
    /-
      🎉 no goals
    -/


theorem Pairwise.mono (h : t ⊆ s) (hs : s.Pairwise r) : t.Pairwise r :=
  fun _x xt _y yt => hs (h xt) (h yt)


theorem Pairwise.mono' (H : r ≤ p) (hr : s.Pairwise r) : s.Pairwise p :=
  hr.imp H


theorem pairwise_top (s : Set α) : s.Pairwise ⊤ :=
  pairwise_of_forall s _ fun _ _ => trivial


protected theorem Subsingleton.pairwise (h : s.Subsingleton) (r : α → α → Prop) : s.Pairwise r :=
  fun _x hx _y hy hne => (hne (h hx hy)).elim


@[simp]
theorem pairwise_empty (r : α → α → Prop) : (∅ : Set α).Pairwise r :=
  subsingleton_empty.pairwise r


@[simp]
theorem pairwise_singleton (a : α) (r : α → α → Prop) : Set.Pairwise {a} r :=
  subsingleton_singleton.pairwise r


theorem pairwise_iff_of_refl [IsRefl α r] : s.Pairwise r ↔ ∀ ⦃a⦄, a ∈ s → ∀ ⦃b⦄, b ∈ s → r a b :=
  forall₄_congr fun _ _ _ _ => or_iff_not_imp_left.symm.trans <| or_iff_right_of_imp of_eq


alias ⟨Pairwise.of_refl, _⟩ := pairwise_iff_of_refl


theorem Nonempty.pairwise_iff_exists_forall [IsEquiv α r] {s : Set ι} (hs : s.Nonempty) :
    s.Pairwise (r on f) ↔ ∃ z, ∀ x ∈ s, r (f x) z := by
  /-
    α : Type u_1
    ι : Type u_4
    r : α → α → Prop
    f : ι → α
    inst✝ : IsEquiv α r
    s : Set ι
    hs : s.Nonempty
    ⊢ Iff (s.Pairwise (Function.onFun r f)) (Exists fun z => ∀ (x : ι), Membership …
  -/
  constructor
    /-
      case mp
      α : Type u_1
      ι : Type u_4
      r : α → α → Prop
      f : ι → α
      inst✝ : IsEquiv α r
      s : Set ι
      hs : s.Nonempty
      ⊢ s.Pairwise (Function.onFun r f) → Exists fun z => ∀ (x : ι), Membership.mem  …
    -/
  · rcases hs with ⟨y, hy⟩
    /-
      case mp.intro
      α : Type u_1
      ι : Type u_4
      r : α → α → Prop
      f : ι → α
      inst✝ : IsEquiv α r
      s : Set ι
      y : ι
      hy : Membership.mem s y
      ⊢ s.Pairwise (Function.onFun r f) → Exists fun z => ∀ (x : ι), Membership.mem  …
    -/
    refine fun H => ⟨f y, fun x hx => ?_⟩
    /-
      case mp.intro
      α : Type u_1
      ι : Type u_4
      r : α → α → Prop
      f : ι → α
      inst✝ : IsEquiv α r
      s : Set ι
      y : ι
      hy : Membership.mem s y
      H : s.Pairwise (Function.onFun r f)
      x : ι
      hx : Membership.mem s x
      ⊢ r (f x) (f y)
    -/
    rcases eq_or_ne x y with (rfl | hne)
      /-
        case mp.intro.inl
        α : Type u_1
        ι : Type u_4
        r : α → α → Prop
        f : ι → α
        inst✝ : IsEquiv α r
        s : Set ι
        H : s.Pairwise (Function.onFun r f)
        x : ι
        hx hy : Membership.mem s x
        ⊢ r (f x) (f x)
      -/
    · apply IsRefl.refl
      /-
        🎉 no goals
      -/
      /-
        case mp.intro.inr
        α : Type u_1
        ι : Type u_4
        r : α → α → Prop
        f : ι → α
        inst✝ : IsEquiv α r
        s : Set ι
        y : ι
        hy : Membership.mem s y
        H : s.Pairwise (Function.onFun r f)
        x : ι
        hx : Membership.mem s x
        hne : Ne x y
        ⊢ r (f x) (f y)
      -/
    · exact H hx hy hne
      /-
        🎉 no goals
      -/
    /-
      case mpr
      α : Type u_1
      ι : Type u_4
      r : α → α → Prop
      f : ι → α
      inst✝ : IsEquiv α r
      s : Set ι
      hs : s.Nonempty
      ⊢ (Exists fun z => ∀ (x : ι), Membership.mem s x → r (f x) z) → s.Pairwise (Fu …
    -/
  · rintro ⟨z, hz⟩ x hx y hy _
    /-
      case mpr.intro
      α : Type u_1
      ι : Type u_4
      r : α → α → Prop
      f : ι → α
      inst✝ : IsEquiv α r
      s : Set ι
      hs : s.Nonempty
      z : α
      hz : ∀ (x : ι), Membership.mem s x → r (f x) z
      x : ι
      hx : Membership.mem s x
      y : ι
      hy : Membership.mem s y
      a✝ : Ne x y
      ⊢ Function.onFun r f x y
    -/
    exact @IsTrans.trans α r _ (f x) z (f y) (hz _ hx) (IsSymm.symm _ _ <| hz _ hy)
    /-
      🎉 no goals
    -/


/-- For a nonempty set `s`, a function `f` takes pairwise equal values on `s` if and only if
for some `z` in the codomain, `f` takes value `z` on all `x ∈ s`. See also
`Set.pairwise_eq_iff_exists_eq` for a version that assumes `[Nonempty ι]` instead of
`Set.Nonempty s`. -/
theorem Nonempty.pairwise_eq_iff_exists_eq {s : Set α} (hs : s.Nonempty) {f : α → ι} :
    (s.Pairwise fun x y => f x = f y) ↔ ∃ z, ∀ x ∈ s, f x = z :=
  hs.pairwise_iff_exists_forall


theorem pairwise_iff_exists_forall [Nonempty ι] (s : Set α) (f : α → ι) {r : ι → ι → Prop}
    [IsEquiv ι r] : s.Pairwise (r on f) ↔ ∃ z, ∀ x ∈ s, r (f x) z := by
  /-
    α : Type u_1
    ι : Type u_4
    inst✝¹ : Nonempty ι
    s : Set α
    f : α → ι
    r : ι → ι → Prop
    inst✝ : IsEquiv ι r
    ⊢ Iff (s.Pairwise (Function.onFun r f)) (Exists fun z => ∀ (x : α), Membership …
  -/
  rcases s.eq_empty_or_nonempty with (rfl | hne)
    /-
      case inl
      α : Type u_1
      ι : Type u_4
      inst✝¹ : Nonempty ι
      f : α → ι
      r : ι → ι → Prop
      inst✝ : IsEquiv ι r
      ⊢ Iff (EmptyCollection.emptyCollection.Pairwise (Function.onFun r f)) (Exists  …
    -/
  · simp
    /-
      🎉 no goals
    -/
    /-
      case inr
      α : Type u_1
      ι : Type u_4
      inst✝¹ : Nonempty ι
      s : Set α
      f : α → ι
      r : ι → ι → Prop
      inst✝ : IsEquiv ι r
      hne : s.Nonempty
      ⊢ Iff (s.Pairwise (Function.onFun r f)) (Exists fun z => ∀ (x : α), Membership …
    -/
  · exact hne.pairwise_iff_exists_forall
    /-
      🎉 no goals
    -/


/-- A function `f : α → ι` with nonempty codomain takes pairwise equal values on a set `s` if and
only if for some `z` in the codomain, `f` takes value `z` on all `x ∈ s`. See also
`Set.Nonempty.pairwise_eq_iff_exists_eq` for a version that assumes `Set.Nonempty s` instead of
`[Nonempty ι]`. -/
theorem pairwise_eq_iff_exists_eq [Nonempty ι] (s : Set α) (f : α → ι) :
    (s.Pairwise fun x y => f x = f y) ↔ ∃ z, ∀ x ∈ s, f x = z :=
  pairwise_iff_exists_forall s f


theorem pairwise_union :
    (s ∪ t).Pairwise r ↔
    s.Pairwise r ∧ t.Pairwise r ∧ ∀ a ∈ s, ∀ b ∈ t, a ≠ b → r a b ∧ r b a := by
  /-
    α : Type u_1
    r : α → α → Prop
    s t : Set α
    ⊢ Iff ((Union.union s t).Pairwise r) (And (s.Pairwise r) (And (t.Pairwise r) ( …
  -/
  simp only [Set.Pairwise, mem_union, or_imp, forall_and]
  /-
    α : Type u_1
    r : α → α → Prop
    s t : Set α
    ⊢ Iff (And (And (∀ (x : α), Membership.mem s x → ∀ (x_1 : α), Membership.mem s …
  -/
  aesop
  /-
    🎉 no goals
  -/


theorem pairwise_union_of_symmetric (hr : Symmetric r) :
    (s ∪ t).Pairwise r ↔ s.Pairwise r ∧ t.Pairwise r ∧ ∀ a ∈ s, ∀ b ∈ t, a ≠ b → r a b :=
                             /-
                               α : Type u_1
                               r : α → α → Prop
                               s t : Set α
                               hr : Symmetric r
                               ⊢ Iff (And (s.Pairwise r) (And (t.Pairwise r) (∀ (a : α), Membership.mem s a → …
                             -/
  pairwise_union.trans <| by simp only [hr.iff, and_self_iff]
                             /-
                               🎉 no goals
                             -/


theorem pairwise_insert :
    (insert a s).Pairwise r ↔ s.Pairwise r ∧ ∀ b ∈ s, a ≠ b → r a b ∧ r b a := by
  /-
    α : Type u_1
    r : α → α → Prop
    s : Set α
    a : α
    ⊢ Iff ((Insert.insert a s).Pairwise r) (And (s.Pairwise r) (∀ (b : α), Members …
  -/
  simp only [insert_eq, pairwise_union, pairwise_singleton, true_and, mem_singleton_iff, forall_eq]
  /-
    🎉 no goals
  -/


theorem pairwise_insert_of_not_mem (ha : a ∉ s) :
    (insert a s).Pairwise r ↔ s.Pairwise r ∧ ∀ b ∈ s, r a b ∧ r b a :=
  pairwise_insert.trans <|
                                                     /-
                                                       α : Type u_1
                                                       r : α → α → Prop
                                                       s : Set α
                                                       a : α
                                                       ha : Not (Membership.mem s a)
                                                       b : α
                                                       hb : Membership.mem s b
                                                       ⊢ Iff (Ne a b → And (r a b) (r b a)) (And (r a b) (r b a))
                                                     -/
    and_congr_right' <| forall₂_congr fun b hb => by simp [(ne_of_mem_of_not_mem hb ha).symm]
                                                     /-
                                                       🎉 no goals
                                                     -/


protected theorem Pairwise.insert (hs : s.Pairwise r) (h : ∀ b ∈ s, a ≠ b → r a b ∧ r b a) :
    (insert a s).Pairwise r :=
  pairwise_insert.2 ⟨hs, h⟩


theorem Pairwise.insert_of_not_mem (ha : a ∉ s) (hs : s.Pairwise r) (h : ∀ b ∈ s, r a b ∧ r b a) :
    (insert a s).Pairwise r :=
  (pairwise_insert_of_not_mem ha).2 ⟨hs, h⟩


theorem pairwise_insert_of_symmetric (hr : Symmetric r) :
    (insert a s).Pairwise r ↔ s.Pairwise r ∧ ∀ b ∈ s, a ≠ b → r a b := by
  /-
    α : Type u_1
    r : α → α → Prop
    s : Set α
    a : α
    hr : Symmetric r
    ⊢ Iff ((Insert.insert a s).Pairwise r) (And (s.Pairwise r) (∀ (b : α), Members …
  -/
  simp only [pairwise_insert, hr.iff a, and_self_iff]
  /-
    🎉 no goals
  -/


theorem pairwise_insert_of_symmetric_of_not_mem (hr : Symmetric r) (ha : a ∉ s) :
    (insert a s).Pairwise r ↔ s.Pairwise r ∧ ∀ b ∈ s, r a b := by
  /-
    α : Type u_1
    r : α → α → Prop
    s : Set α
    a : α
    hr : Symmetric r
    ha : Not (Membership.mem s a)
    ⊢ Iff ((Insert.insert a s).Pairwise r) (And (s.Pairwise r) (∀ (b : α), Members …
  -/
  simp only [pairwise_insert_of_not_mem ha, hr.iff a, and_self_iff]
  /-
    🎉 no goals
  -/


theorem Pairwise.insert_of_symmetric (hs : s.Pairwise r) (hr : Symmetric r)
    (h : ∀ b ∈ s, a ≠ b → r a b) : (insert a s).Pairwise r :=
  (pairwise_insert_of_symmetric hr).2 ⟨hs, h⟩


theorem Pairwise.insert_of_symmetric_of_not_mem (hs : s.Pairwise r) (hr : Symmetric r) (ha : a ∉ s)
    (h : ∀ b ∈ s, r a b) : (insert a s).Pairwise r :=
  (pairwise_insert_of_symmetric_of_not_mem hr ha).2 ⟨hs, h⟩


                                                                            /-
                                                                              α : Type u_1
                                                                              r : α → α → Prop
                                                                              a b : α
                                                                              ⊢ Iff ((Insert.insert a (Singleton.singleton b)).Pairwise r) (Ne a b → And (r  …
                                                                            -/
theorem pairwise_pair : Set.Pairwise {a, b} r ↔ a ≠ b → r a b ∧ r b a := by simp [pairwise_insert]
                                                                            /-
                                                                              🎉 no goals
                                                                            -/


theorem pairwise_pair_of_symmetric (hr : Symmetric r) : Set.Pairwise {a, b} r ↔ a ≠ b → r a b := by
  /-
    α : Type u_1
    r : α → α → Prop
    a b : α
    hr : Symmetric r
    ⊢ Iff ((Insert.insert a (Singleton.singleton b)).Pairwise r) (Ne a b → r a b)
  -/
  simp [pairwise_insert_of_symmetric hr]
  /-
    🎉 no goals
  -/


theorem pairwise_univ : (univ : Set α).Pairwise r ↔ Pairwise r := by
  /-
    α : Type u_1
    r : α → α → Prop
    ⊢ Iff (Set.univ.Pairwise r) (Pairwise r)
  -/
  simp only [Set.Pairwise, Pairwise, mem_univ, forall_const]
  /-
    🎉 no goals
  -/


@[simp]
theorem pairwise_bot_iff : s.Pairwise (⊥ : α → α → Prop) ↔ (s : Set α).Subsingleton :=
  ⟨fun h _a ha _b hb => h.eq ha hb id, fun h => h.pairwise _⟩


alias ⟨Pairwise.subsingleton, _⟩ := pairwise_bot_iff


/-- See also `Function.injective_iff_pairwise_ne` -/
lemma injOn_iff_pairwise_ne {s : Set ι} : InjOn f s ↔ s.Pairwise (f · ≠ f ·) := by
  /-
    α : Type u_1
    ι : Type u_4
    f : ι → α
    s : Set ι
    ⊢ Iff (Set.InjOn f s) (s.Pairwise fun x1 x2 => Ne (f x1) (f x2))
  -/
  simp only [InjOn, Set.Pairwise, not_imp_not]
  /-
    🎉 no goals
  -/


alias ⟨InjOn.pairwise_ne, _⟩ := injOn_iff_pairwise_ne


protected theorem Pairwise.image {s : Set ι} (h : s.Pairwise (r on f)) : (f '' s).Pairwise r :=
  forall_mem_image.2 fun _x hx ↦ forall_mem_image.2 fun _y hy hne ↦ h hx hy <| ne_of_apply_ne _ hne


/-- See also `Set.Pairwise.image`. -/
theorem InjOn.pairwise_image {s : Set ι} (h : s.InjOn f) :
    (f '' s).Pairwise r ↔ s.Pairwise (r on f) := by
  /-
    α : Type u_1
    ι : Type u_4
    r : α → α → Prop
    f : ι → α
    s : Set ι
    h : Set.InjOn f s
    ⊢ Iff ((Set.image f s).Pairwise r) (s.Pairwise (Function.onFun r f))
  -/
  simp +contextual [h.eq_iff, Set.Pairwise]
  /-
    🎉 no goals
  -/


lemma _root_.Pairwise.range_pairwise (hr : Pairwise (r on f)) : (Set.range f).Pairwise r :=
  image_univ ▸ (pairwise_univ.mpr hr).image


theorem pairwise_subtype_iff_pairwise_set (s : Set α) (r : α → α → Prop) :
    (Pairwise fun (x : s) (y : s) => r x y) ↔ s.Pairwise r := by
  /-
    α : Type u_1
    s : Set α
    r : α → α → Prop
    ⊢ Iff (Pairwise fun x y => r ↑x ↑y) (s.Pairwise r)
  -/
  simp only [Pairwise, Set.Pairwise, SetCoe.forall, Ne, Subtype.ext_iff, Subtype.coe_mk]
  /-
    🎉 no goals
  -/


alias ⟨Pairwise.set_of_subtype, Set.Pairwise.subtype⟩ := pairwise_subtype_iff_pairwise_set


/-- A set is `PairwiseDisjoint` under `f`, if the images of any distinct two elements under `f`
are disjoint.

`s.Pairwise Disjoint` is (definitionally) the same as `s.PairwiseDisjoint id`. We prefer the latter
in order to allow dot notation on `Set.PairwiseDisjoint`, even though the former unfolds more
nicely. -/
def PairwiseDisjoint (s : Set ι) (f : ι → α) : Prop :=
  s.Pairwise (Disjoint on f)


theorem PairwiseDisjoint.subset (ht : t.PairwiseDisjoint f) (h : s ⊆ t) : s.PairwiseDisjoint f :=
  Pairwise.mono h ht


theorem PairwiseDisjoint.mono_on (hs : s.PairwiseDisjoint f) (h : ∀ ⦃i⦄, i ∈ s → g i ≤ f i) :
    s.PairwiseDisjoint g := fun _a ha _b hb hab => (hs ha hb hab).mono (h ha) (h hb)


theorem PairwiseDisjoint.mono (hs : s.PairwiseDisjoint f) (h : g ≤ f) : s.PairwiseDisjoint g :=
  hs.mono_on fun i _ => h i


@[simp]
theorem pairwiseDisjoint_empty : (∅ : Set ι).PairwiseDisjoint f :=
  pairwise_empty _


@[simp]
theorem pairwiseDisjoint_singleton (i : ι) (f : ι → α) : PairwiseDisjoint {i} f :=
  pairwise_singleton i _


theorem pairwiseDisjoint_insert {i : ι} :
    (insert i s).PairwiseDisjoint f ↔
      s.PairwiseDisjoint f ∧ ∀ j ∈ s, i ≠ j → Disjoint (f i) (f j) :=
  pairwise_insert_of_symmetric <| symmetric_disjoint.comap f


theorem pairwiseDisjoint_insert_of_not_mem {i : ι} (hi : i ∉ s) :
    (insert i s).PairwiseDisjoint f ↔ s.PairwiseDisjoint f ∧ ∀ j ∈ s, Disjoint (f i) (f j) :=
  pairwise_insert_of_symmetric_of_not_mem (symmetric_disjoint.comap f) hi


protected theorem PairwiseDisjoint.insert (hs : s.PairwiseDisjoint f) {i : ι}
    (h : ∀ j ∈ s, i ≠ j → Disjoint (f i) (f j)) : (insert i s).PairwiseDisjoint f :=
  pairwiseDisjoint_insert.2 ⟨hs, h⟩


theorem PairwiseDisjoint.insert_of_not_mem (hs : s.PairwiseDisjoint f) {i : ι} (hi : i ∉ s)
    (h : ∀ j ∈ s, Disjoint (f i) (f j)) : (insert i s).PairwiseDisjoint f :=
  (pairwiseDisjoint_insert_of_not_mem hi).2 ⟨hs, h⟩


theorem PairwiseDisjoint.image_of_le (hs : s.PairwiseDisjoint f) {g : ι → ι} (hg : f ∘ g ≤ f) :
    (g '' s).PairwiseDisjoint f := by
  /-
    α : Type u_1
    ι : Type u_4
    inst✝¹ : PartialOrder α
    inst✝ : OrderBot α
    s : Set ι
    f : ι → α
    hs : s.PairwiseDisjoint f
    g : ι → ι
    hg : LE.le (Function.comp f g) f
    ⊢ (Set.image g s).PairwiseDisjoint f
  -/
  rintro _ ⟨a, ha, rfl⟩ _ ⟨b, hb, rfl⟩ h
  /-
    case intro.intro.intro.intro
    α : Type u_1
    ι : Type u_4
    inst✝¹ : PartialOrder α
    inst✝ : OrderBot α
    s : Set ι
    f : ι → α
    hs : s.PairwiseDisjoint f
    g : ι → ι
    hg : LE.le (Function.comp f g) f
    a : ι
    ha : Membership.mem s a
    b : ι
    hb : Membership.mem s b
    h : Ne (g a) (g b)
    ⊢ Function.onFun Disjoint f (g a) (g b)
  -/
  exact (hs ha hb <| ne_of_apply_ne _ h).mono (hg a) (hg b)
  /-
    🎉 no goals
  -/


theorem InjOn.pairwiseDisjoint_image {g : ι' → ι} {s : Set ι'} (h : s.InjOn g) :
    (g '' s).PairwiseDisjoint f ↔ s.PairwiseDisjoint (f ∘ g) :=
  h.pairwise_image


theorem PairwiseDisjoint.range (g : s → ι) (hg : ∀ i : s, f (g i) ≤ f i)
    (ht : s.PairwiseDisjoint f) : (range g).PairwiseDisjoint f := by
  /-
    α : Type u_1
    ι : Type u_4
    inst✝¹ : PartialOrder α
    inst✝ : OrderBot α
    s : Set ι
    f : ι → α
    g : ↑s → ι
    hg : ∀ (i : ↑s), LE.le (f (g i)) (f ↑i)
    ht : s.PairwiseDisjoint f
    ⊢ (Set.range g).PairwiseDisjoint f
  -/
  rintro _ ⟨x, rfl⟩ _ ⟨y, rfl⟩ hxy
  /-
    case intro.intro
    α : Type u_1
    ι : Type u_4
    inst✝¹ : PartialOrder α
    inst✝ : OrderBot α
    s : Set ι
    f : ι → α
    g : ↑s → ι
    hg : ∀ (i : ↑s), LE.le (f (g i)) (f ↑i)
    ht : s.PairwiseDisjoint f
    x y : ↑s
    hxy : Ne (g x) (g y)
    ⊢ Function.onFun Disjoint f (g x) (g y)
  -/
  exact ((ht x.2 y.2) fun h => hxy <| congr_arg g <| Subtype.ext h).mono (hg x) (hg y)
  /-
    🎉 no goals
  -/


theorem pairwiseDisjoint_union :
    (s ∪ t).PairwiseDisjoint f ↔
      s.PairwiseDisjoint f ∧
        t.PairwiseDisjoint f ∧ ∀ ⦃i⦄, i ∈ s → ∀ ⦃j⦄, j ∈ t → i ≠ j → Disjoint (f i) (f j) :=
  pairwise_union_of_symmetric <| symmetric_disjoint.comap f


theorem PairwiseDisjoint.union (hs : s.PairwiseDisjoint f) (ht : t.PairwiseDisjoint f)
    (h : ∀ ⦃i⦄, i ∈ s → ∀ ⦃j⦄, j ∈ t → i ≠ j → Disjoint (f i) (f j)) : (s ∪ t).PairwiseDisjoint f :=
  pairwiseDisjoint_union.2 ⟨hs, ht, h⟩

-- classical

theorem PairwiseDisjoint.elim (hs : s.PairwiseDisjoint f) {i j : ι} (hi : i ∈ s) (hj : j ∈ s)
    (h : ¬Disjoint (f i) (f j)) : i = j :=
  hs.eq hi hj h


lemma PairwiseDisjoint.eq_or_disjoint
    (h : s.PairwiseDisjoint f) {i j : ι} (hi : i ∈ s) (hj : j ∈ s) :
    i = j ∨ Disjoint (f i) (f j) := by
  /-
    α : Type u_1
    ι : Type u_4
    inst✝¹ : PartialOrder α
    inst✝ : OrderBot α
    s : Set ι
    f : ι → α
    h : s.PairwiseDisjoint f
    i j : ι
    hi : Membership.mem s i
    hj : Membership.mem s j
    ⊢ Or (Eq i j) (Disjoint (f i) (f j))
  -/
  rw [or_iff_not_imp_right]
  /-
    α : Type u_1
    ι : Type u_4
    inst✝¹ : PartialOrder α
    inst✝ : OrderBot α
    s : Set ι
    f : ι → α
    h : s.PairwiseDisjoint f
    i j : ι
    hi : Membership.mem s i
    hj : Membership.mem s j
    ⊢ Not (Disjoint (f i) (f j)) → Eq i j
  -/
  exact h.elim hi hj
  /-
    🎉 no goals
  -/


lemma pairwiseDisjoint_range_iff {α β : Type*} {f : α → (Set β)} :
    (range f).PairwiseDisjoint id ↔ ∀ x y, f x ≠ f y → Disjoint (f x) (f y) := by
  /-
    α : Type u_6
    β : Type u_7
    f : α → Set β
    ⊢ Iff ((Set.range f).PairwiseDisjoint id) (∀ (x y : α), Ne (f x) (f y) → Disjo …
  -/
  aesop (add simp [PairwiseDisjoint, Set.Pairwise])
  /-
    🎉 no goals
  -/


/-- If the range of `f` is pairwise disjoint, then the image of any set `s` under `f` is as well. -/
lemma _root_.Pairwise.pairwiseDisjoint (h : Pairwise (Disjoint on f)) (s : Set ι) :
    s.PairwiseDisjoint f := h.set_pairwise s


theorem PairwiseDisjoint.elim' (hs : s.PairwiseDisjoint f) {i j : ι} (hi : i ∈ s) (hj : j ∈ s)
    (h : f i ⊓ f j ≠ ⊥) : i = j :=
  (hs.elim hi hj) fun hij => h hij.eq_bot


theorem PairwiseDisjoint.eq_of_le (hs : s.PairwiseDisjoint f) {i j : ι} (hi : i ∈ s) (hj : j ∈ s)
    (hf : f i ≠ ⊥) (hij : f i ≤ f j) : i = j :=
  (hs.elim' hi hj) fun h => hf <| (inf_of_le_left hij).symm.trans h


theorem pairwiseDisjoint_range_singleton :
    (range (singleton : ι → Set ι)).PairwiseDisjoint id :=
  Pairwise.range_pairwise fun _ _ => disjoint_singleton.2


theorem pairwiseDisjoint_fiber (f : ι → α) (s : Set α) : s.PairwiseDisjoint fun a => f ⁻¹' {a} :=
  fun _a _ _b _ h => disjoint_iff_inf_le.mpr fun _i ⟨hia, hib⟩ => h <| (Eq.symm hia).trans hib

-- classical

theorem PairwiseDisjoint.elim_set {s : Set ι} {f : ι → Set α} (hs : s.PairwiseDisjoint f) {i j : ι}
    (hi : i ∈ s) (hj : j ∈ s) (a : α) (hai : a ∈ f i) (haj : a ∈ f j) : i = j :=
  hs.elim hi hj <| not_disjoint_iff.2 ⟨a, hai, haj⟩


theorem PairwiseDisjoint.prod {f : ι → Set α} {g : ι' → Set β} (hs : s.PairwiseDisjoint f)
    (ht : t.PairwiseDisjoint g) :
    (s ×ˢ t : Set (ι × ι')).PairwiseDisjoint fun i => f i.1 ×ˢ g i.2 :=
  fun ⟨_, _⟩ ⟨hi, hi'⟩ ⟨_, _⟩ ⟨hj, hj'⟩ hij =>
  disjoint_left.2 fun ⟨_, _⟩ ⟨hai, hbi⟩ ⟨haj, hbj⟩ =>
    hij <| Prod.ext (hs.elim_set hi hj _ hai haj) <| ht.elim_set hi' hj' _ hbi hbj


theorem pairwiseDisjoint_pi {ι' α : ι → Type*} {s : ∀ i, Set (ι' i)} {f : ∀ i, ι' i → Set (α i)}
    (hs : ∀ i, (s i).PairwiseDisjoint (f i)) :
    ((univ : Set ι).pi s).PairwiseDisjoint fun I => (univ : Set ι).pi fun i => f _ (I i) :=
  fun _ hI _ hJ hIJ =>
  disjoint_left.2 fun a haI haJ =>
    hIJ <|
      funext fun i =>
        (hs i).elim_set (hI i trivial) (hJ i trivial) (a i) (haI i trivial) (haJ i trivial)


/-- The partial images of a binary function `f` whose partial evaluations are injective are pairwise
disjoint iff `f` is injective . -/
theorem pairwiseDisjoint_image_right_iff {f : α → β → γ} {s : Set α} {t : Set β}
    (hf : ∀ a ∈ s, Injective (f a)) :
    (s.PairwiseDisjoint fun a => f a '' t) ↔ (s ×ˢ t).InjOn fun p => f p.1 p.2 := by
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    f : α → β → γ
    s : Set α
    t : Set β
    hf : ∀ (a : α), Membership.mem s a → Function.Injective (f a)
    ⊢ Iff (s.PairwiseDisjoint fun a => Set.image (f a) t) (Set.InjOn (fun p => f p …
  -/
  refine ⟨fun hs x hx y hy (h : f _ _ = _) => ?_, fun hs x hx y hy h => ?_⟩
    /-
      case refine_1
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      f : α → β → γ
      s : Set α
      t : Set β
      hf : ∀ (a : α), Membership.mem s a → Function.Injective (f a)
      hs : s.PairwiseDisjoint fun a => Set.image (f a) t
      x : Prod α β
      hx : Membership.mem (SProd.sprod s t) x
      y : Prod α β
      hy : Membership.mem (SProd.sprod s t) y
      h : Eq (f x.1 x.2) ((fun p => f p.1 p.2) y)
      ⊢ Eq x y
    -/
  · suffices x.1 = y.1 by exact Prod.ext this (hf _ hx.1 <| h.trans <| by rw [this])
    /-
      case refine_1
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      f : α → β → γ
      s : Set α
      t : Set β
      hf : ∀ (a : α), Membership.mem s a → Function.Injective (f a)
      hs : s.PairwiseDisjoint fun a => Set.image (f a) t
      x : Prod α β
      hx : Membership.mem (SProd.sprod s t) x
      y : Prod α β
      hy : Membership.mem (SProd.sprod s t) y
      h : Eq (f x.1 x.2) ((fun p => f p.1 p.2) y)
      ⊢ Eq x.1 y.1
    -/
    refine hs.elim hx.1 hy.1 (not_disjoint_iff.2 ⟨_, mem_image_of_mem _ hx.2, ?_⟩)
    /-
      case refine_1
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      f : α → β → γ
      s : Set α
      t : Set β
      hf : ∀ (a : α), Membership.mem s a → Function.Injective (f a)
      hs : s.PairwiseDisjoint fun a => Set.image (f a) t
      x : Prod α β
      hx : Membership.mem (SProd.sprod s t) x
      y : Prod α β
      hy : Membership.mem (SProd.sprod s t) y
      h : Eq (f x.1 x.2) ((fun p => f p.1 p.2) y)
      ⊢ Membership.mem (Set.image (f y.1) t) (f x.1 x.2)
    -/
    rw [h]
    /-
      case refine_1
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      f : α → β → γ
      s : Set α
      t : Set β
      hf : ∀ (a : α), Membership.mem s a → Function.Injective (f a)
      hs : s.PairwiseDisjoint fun a => Set.image (f a) t
      x : Prod α β
      hx : Membership.mem (SProd.sprod s t) x
      y : Prod α β
      hy : Membership.mem (SProd.sprod s t) y
      h : Eq (f x.1 x.2) ((fun p => f p.1 p.2) y)
      ⊢ Membership.mem (Set.image (f y.1) t) ((fun p => f p.1 p.2) y)
    -/
    exact mem_image_of_mem _ hy.2
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      f : α → β → γ
      s : Set α
      t : Set β
      hf : ∀ (a : α), Membership.mem s a → Function.Injective (f a)
      hs : Set.InjOn (fun p => f p.1 p.2) (SProd.sprod s t)
      x : α
      hx : Membership.mem s x
      y : α
      hy : Membership.mem s y
      h : Ne x y
      ⊢ Function.onFun Disjoint (fun a => Set.image (f a) t) x y
    -/
  · refine disjoint_iff_inf_le.mpr ?_
    /-
      case refine_2
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      f : α → β → γ
      s : Set α
      t : Set β
      hf : ∀ (a : α), Membership.mem s a → Function.Injective (f a)
      hs : Set.InjOn (fun p => f p.1 p.2) (SProd.sprod s t)
      x : α
      hx : Membership.mem s x
      y : α
      hy : Membership.mem s y
      h : Ne x y
      ⊢ LE.le (Min.min ((fun a => Set.image (f a) t) x) ((fun a => Set.image (f a) t …
    -/
    rintro _ ⟨⟨a, ha, hab⟩, b, hb, rfl⟩
    /-
      case refine_2.intro.intro.intro.intro.intro
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      f : α → β → γ
      s : Set α
      t : Set β
      hf : ∀ (a : α), Membership.mem s a → Function.Injective (f a)
      hs : Set.InjOn (fun p => f p.1 p.2) (SProd.sprod s t)
      x : α
      hx : Membership.mem s x
      y : α
      hy : Membership.mem s y
      h : Ne x y
      a : β
      ha : Membership.mem t a
      b : β
      hb : Membership.mem t b
      hab : Eq (f x a) (f y b)
      ⊢ Membership.mem Bot.bot (f y b)
    -/
    exact h (congr_arg Prod.fst <| hs (mk_mem_prod hx ha) (mk_mem_prod hy hb) hab)
    /-
      🎉 no goals
    -/


/-- The partial images of a binary function `f` whose partial evaluations are injective are pairwise
disjoint iff `f` is injective . -/
theorem pairwiseDisjoint_image_left_iff {f : α → β → γ} {s : Set α} {t : Set β}
    (hf : ∀ b ∈ t, Injective fun a => f a b) :
    (t.PairwiseDisjoint fun b => (fun a => f a b) '' s) ↔ (s ×ˢ t).InjOn fun p => f p.1 p.2 := by
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    f : α → β → γ
    s : Set α
    t : Set β
    hf : ∀ (b : β), Membership.mem t b → Function.Injective fun a => f a b
    ⊢ Iff (t.PairwiseDisjoint fun b => Set.image (fun a => f a b) s) (Set.InjOn (f …
  -/
  refine ⟨fun ht x hx y hy (h : f _ _ = _) => ?_, fun ht x hx y hy h => ?_⟩
    /-
      case refine_1
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      f : α → β → γ
      s : Set α
      t : Set β
      hf : ∀ (b : β), Membership.mem t b → Function.Injective fun a => f a b
      ht : t.PairwiseDisjoint fun b => Set.image (fun a => f a b) s
      x : Prod α β
      hx : Membership.mem (SProd.sprod s t) x
      y : Prod α β
      hy : Membership.mem (SProd.sprod s t) y
      h : Eq (f x.1 x.2) ((fun p => f p.1 p.2) y)
      ⊢ Eq x y
    -/
  · suffices x.2 = y.2 by exact Prod.ext (hf _ hx.2 <| h.trans <| by rw [this]) this
    /-
      case refine_1
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      f : α → β → γ
      s : Set α
      t : Set β
      hf : ∀ (b : β), Membership.mem t b → Function.Injective fun a => f a b
      ht : t.PairwiseDisjoint fun b => Set.image (fun a => f a b) s
      x : Prod α β
      hx : Membership.mem (SProd.sprod s t) x
      y : Prod α β
      hy : Membership.mem (SProd.sprod s t) y
      h : Eq (f x.1 x.2) ((fun p => f p.1 p.2) y)
      ⊢ Eq x.2 y.2
    -/
    refine ht.elim hx.2 hy.2 (not_disjoint_iff.2 ⟨_, mem_image_of_mem _ hx.1, ?_⟩)
    /-
      case refine_1
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      f : α → β → γ
      s : Set α
      t : Set β
      hf : ∀ (b : β), Membership.mem t b → Function.Injective fun a => f a b
      ht : t.PairwiseDisjoint fun b => Set.image (fun a => f a b) s
      x : Prod α β
      hx : Membership.mem (SProd.sprod s t) x
      y : Prod α β
      hy : Membership.mem (SProd.sprod s t) y
      h : Eq (f x.1 x.2) ((fun p => f p.1 p.2) y)
      ⊢ Membership.mem (Set.image (fun a => f a y.2) s) (f x.1 x.2)
    -/
    rw [h]
    /-
      case refine_1
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      f : α → β → γ
      s : Set α
      t : Set β
      hf : ∀ (b : β), Membership.mem t b → Function.Injective fun a => f a b
      ht : t.PairwiseDisjoint fun b => Set.image (fun a => f a b) s
      x : Prod α β
      hx : Membership.mem (SProd.sprod s t) x
      y : Prod α β
      hy : Membership.mem (SProd.sprod s t) y
      h : Eq (f x.1 x.2) ((fun p => f p.1 p.2) y)
      ⊢ Membership.mem (Set.image (fun a => f a y.2) s) ((fun p => f p.1 p.2) y)
    -/
    exact mem_image_of_mem _ hy.1
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      f : α → β → γ
      s : Set α
      t : Set β
      hf : ∀ (b : β), Membership.mem t b → Function.Injective fun a => f a b
      ht : Set.InjOn (fun p => f p.1 p.2) (SProd.sprod s t)
      x : β
      hx : Membership.mem t x
      y : β
      hy : Membership.mem t y
      h : Ne x y
      ⊢ Function.onFun Disjoint (fun b => Set.image (fun a => f a b) s) x y
    -/
  · refine disjoint_iff_inf_le.mpr ?_
    /-
      case refine_2
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      f : α → β → γ
      s : Set α
      t : Set β
      hf : ∀ (b : β), Membership.mem t b → Function.Injective fun a => f a b
      ht : Set.InjOn (fun p => f p.1 p.2) (SProd.sprod s t)
      x : β
      hx : Membership.mem t x
      y : β
      hy : Membership.mem t y
      h : Ne x y
      ⊢ LE.le (Min.min ((fun b => Set.image (fun a => f a b) s) x) ((fun b => Set.im …
    -/
    rintro _ ⟨⟨a, ha, hab⟩, b, hb, rfl⟩
    /-
      case refine_2.intro.intro.intro.intro.intro
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      f : α → β → γ
      s : Set α
      t : Set β
      hf : ∀ (b : β), Membership.mem t b → Function.Injective fun a => f a b
      ht : Set.InjOn (fun p => f p.1 p.2) (SProd.sprod s t)
      x : β
      hx : Membership.mem t x
      y : β
      hy : Membership.mem t y
      h : Ne x y
      a : α
      ha : Membership.mem s a
      b : α
      hb : Membership.mem s b
      hab : Eq ((fun a => f a x) a) ((fun a => f a y) b)
      ⊢ Membership.mem Bot.bot ((fun a => f a y) b)
    -/
    exact h (congr_arg Prod.snd <| ht (mk_mem_prod ha hx) (mk_mem_prod hb hy) hab)
    /-
      🎉 no goals
    -/


lemma exists_ne_mem_inter_of_not_pairwiseDisjoint
    {f : ι → Set α} (h : ¬ s.PairwiseDisjoint f) :
    ∃ i ∈ s, ∃ j ∈ s, i ≠ j ∧ ∃ x : α, x ∈ f i ∩ f j := by
  /-
    α : Type u_1
    ι : Type u_4
    s : Set ι
    f : ι → Set α
    h : Not (s.PairwiseDisjoint f)
    ⊢ Exists fun i => And (Membership.mem s i) (Exists fun j => And (Membership.me …
  -/
  change ¬ ∀ i, i ∈ s → ∀ j, j ∈ s → i ≠ j → ∀ t, t ≤ f i → t ≤ f j → t ≤ ⊥ at h
  /-
    α : Type u_1
    ι : Type u_4
    s : Set ι
    f : ι → Set α
    h : Not (∀ (i : ι), Membership.mem s i → ∀ (j : ι), Membership.mem s j → Ne i  …
    ⊢ Exists fun i => And (Membership.mem s i) (Exists fun j => And (Membership.me …
  -/
  simp only [not_forall] at h
  /-
    α : Type u_1
    ι : Type u_4
    s : Set ι
    f : ι → Set α
    h : Exists fun x => Exists fun h => Exists fun x_1 => Exists fun h => Exists f …
    ⊢ Exists fun i => And (Membership.mem s i) (Exists fun j => And (Membership.me …
  -/
  obtain ⟨i, hi, j, hj, h_ne, t, hfi, hfj, ht⟩ := h
  replace ht : t.Nonempty := by
    rwa [le_bot_iff, bot_eq_empty, ← Ne, ← nonempty_iff_ne_empty] at ht
  /-
    case intro.intro.intro.intro.intro.intro.intro.intro
    α : Type u_1
    ι : Type u_4
    s : Set ι
    f : ι → Set α
    i : ι
    hi : Membership.mem s i
    j : ι
    hj : Membership.mem s j
    h_ne : Ne i j
    t : Set α
    hfi : LE.le t (f i)
    hfj : LE.le t (f j)
    ht : t.Nonempty
    ⊢ Exists fun i => And (Membership.mem s i) (Exists fun j => And (Membership.me …
  -/
  obtain ⟨x, hx⟩ := ht
  /-
    case intro.intro.intro.intro.intro.intro.intro.intro.intro
    α : Type u_1
    ι : Type u_4
    s : Set ι
    f : ι → Set α
    i : ι
    hi : Membership.mem s i
    j : ι
    hj : Membership.mem s j
    h_ne : Ne i j
    t : Set α
    hfi : LE.le t (f i)
    hfj : LE.le t (f j)
    x : α
    hx : Membership.mem t x
    ⊢ Exists fun i => And (Membership.mem s i) (Exists fun j => And (Membership.me …
  -/
  exact ⟨i, hi, j, hj, h_ne, x, hfi hx, hfj hx⟩
  /-
    🎉 no goals
  -/


lemma exists_lt_mem_inter_of_not_pairwiseDisjoint [LinearOrder ι]
    {f : ι → Set α} (h : ¬ s.PairwiseDisjoint f) :
    ∃ i ∈ s, ∃ j ∈ s, i < j ∧ ∃ x, x ∈ f i ∩ f j := by
  /-
    α : Type u_1
    ι : Type u_4
    s : Set ι
    inst✝ : LinearOrder ι
    f : ι → Set α
    h : Not (s.PairwiseDisjoint f)
    ⊢ Exists fun i => And (Membership.mem s i) (Exists fun j => And (Membership.me …
  -/
  obtain ⟨i, hi, j, hj, hne, x, hx₁, hx₂⟩ := exists_ne_mem_inter_of_not_pairwiseDisjoint h
  /-
    case intro.intro.intro.intro.intro.intro.intro
    α : Type u_1
    ι : Type u_4
    s : Set ι
    inst✝ : LinearOrder ι
    f : ι → Set α
    h : Not (s.PairwiseDisjoint f)
    i : ι
    hi : Membership.mem s i
    j : ι
    hj : Membership.mem s j
    hne : Ne i j
    x : α
    hx₁ : Membership.mem (f i) x
    hx₂ : Membership.mem (f j) x
    ⊢ Exists fun i => And (Membership.mem s i) (Exists fun j => And (Membership.me …
  -/
  cases' lt_or_lt_iff_ne.mpr hne with h_lt h_lt
    /-
      case intro.intro.intro.intro.intro.intro.intro.inl
      α : Type u_1
      ι : Type u_4
      s : Set ι
      inst✝ : LinearOrder ι
      f : ι → Set α
      h : Not (s.PairwiseDisjoint f)
      i : ι
      hi : Membership.mem s i
      j : ι
      hj : Membership.mem s j
      hne : Ne i j
      x : α
      hx₁ : Membership.mem (f i) x
      hx₂ : Membership.mem (f j) x
      h_lt : LT.lt i j
      ⊢ Exists fun i => And (Membership.mem s i) (Exists fun j => And (Membership.me …
    -/
  · exact ⟨i, hi, j, hj, h_lt, x, hx₁, hx₂⟩
    /-
      🎉 no goals
    -/
    /-
      case intro.intro.intro.intro.intro.intro.intro.inr
      α : Type u_1
      ι : Type u_4
      s : Set ι
      inst✝ : LinearOrder ι
      f : ι → Set α
      h : Not (s.PairwiseDisjoint f)
      i : ι
      hi : Membership.mem s i
      j : ι
      hj : Membership.mem s j
      hne : Ne i j
      x : α
      hx₁ : Membership.mem (f i) x
      hx₂ : Membership.mem (f j) x
      h_lt : LT.lt j i
      ⊢ Exists fun i => And (Membership.mem s i) (Exists fun j => And (Membership.me …
    -/
  · exact ⟨j, hj, i, hi, h_lt, x, hx₂, hx₁⟩
    /-
      🎉 no goals
    -/


lemma exists_ne_mem_inter_of_not_pairwise_disjoint
    {f : ι → Set α} (h : ¬ Pairwise (Disjoint on f)) :
    ∃ i j : ι, i ≠ j ∧ ∃ x, x ∈ f i ∩ f j := by
  /-
    α : Type u_1
    ι : Type u_4
    f : ι → Set α
    h : Not (Pairwise (Function.onFun Disjoint f))
    ⊢ Exists fun i => Exists fun j => And (Ne i j) (Exists fun x => Membership.mem …
  -/
  rw [← pairwise_univ] at h
  /-
    α : Type u_1
    ι : Type u_4
    f : ι → Set α
    h : Not (Set.univ.Pairwise (Function.onFun Disjoint f))
    ⊢ Exists fun i => Exists fun j => And (Ne i j) (Exists fun x => Membership.mem …
  -/
  obtain ⟨i, _hi, j, _hj, h⟩ := exists_ne_mem_inter_of_not_pairwiseDisjoint h
  /-
    case intro.intro.intro.intro
    α : Type u_1
    ι : Type u_4
    f : ι → Set α
    h✝ : Not (Set.univ.Pairwise (Function.onFun Disjoint f))
    i : ι
    _hi : Membership.mem Set.univ i
    j : ι
    _hj : Membership.mem Set.univ j
    h : And (Ne i j) (Exists fun x => Membership.mem (Inter.inter (f i) (f j)) x)
    ⊢ Exists fun i => Exists fun j => And (Ne i j) (Exists fun x => Membership.mem …
  -/
  exact ⟨i, j, h⟩
  /-
    🎉 no goals
  -/


lemma exists_lt_mem_inter_of_not_pairwise_disjoint [LinearOrder ι]
    {f : ι → Set α} (h : ¬ Pairwise (Disjoint on f)) :
    ∃ i j : ι, i < j ∧ ∃ x, x ∈ f i ∩ f j := by
  /-
    α : Type u_1
    ι : Type u_4
    inst✝ : LinearOrder ι
    f : ι → Set α
    h : Not (Pairwise (Function.onFun Disjoint f))
    ⊢ Exists fun i => Exists fun j => And (LT.lt i j) (Exists fun x => Membership. …
  -/
  rw [← pairwise_univ] at h
  /-
    α : Type u_1
    ι : Type u_4
    inst✝ : LinearOrder ι
    f : ι → Set α
    h : Not (Set.univ.Pairwise (Function.onFun Disjoint f))
    ⊢ Exists fun i => Exists fun j => And (LT.lt i j) (Exists fun x => Membership. …
  -/
  obtain ⟨i, _hi, j, _hj, h⟩ := exists_lt_mem_inter_of_not_pairwiseDisjoint h
  /-
    case intro.intro.intro.intro
    α : Type u_1
    ι : Type u_4
    inst✝ : LinearOrder ι
    f : ι → Set α
    h✝ : Not (Set.univ.Pairwise (Function.onFun Disjoint f))
    i : ι
    _hi : Membership.mem Set.univ i
    j : ι
    _hj : Membership.mem Set.univ j
    h : And (LT.lt i j) (Exists fun x => Membership.mem (Inter.inter (f i) (f j)) x)
    ⊢ Exists fun i => Exists fun j => And (LT.lt i j) (Exists fun x => Membership. …
  -/
  exact ⟨i, j, h⟩
  /-
    🎉 no goals
  -/


theorem pairwise_disjoint_fiber (f : ι → α) : Pairwise (Disjoint on fun a : α => f ⁻¹' {a}) :=
  pairwise_univ.1 <| Set.pairwiseDisjoint_fiber f univ


lemma subsingleton_setOf_mem_iff_pairwise_disjoint {f : ι → Set α} :
    (∀ a, {i | a ∈ f i}.Subsingleton) ↔ Pairwise (Disjoint on f) :=
  ⟨fun h _ _ hij ↦ disjoint_left.2 fun a hi hj ↦ hij (h a hi hj),
   fun h _ _ hx _ hy ↦ by_contra fun hne ↦ disjoint_left.1 (h hne) hx hy⟩

