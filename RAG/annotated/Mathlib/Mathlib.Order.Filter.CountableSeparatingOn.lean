/-- We say that a type `α` has a *countable separating family of sets* satisfying a predicate
`p : Set α → Prop` on a set `t` if there exists a countable family of sets `S : Set (Set α)` such
that all sets `s ∈ S` satisfy `p` and any two distinct points `x y ∈ t`, `x ≠ y`, can be separated
by `s ∈ S`: there exists `s ∈ S` such that exactly one of `x` and `y` belongs to `s`.

E.g., if `α` is a `T₀` topological space with second countable topology, then it has a countable
separating family of open sets and a countable separating family of closed sets.
-/
class HasCountableSeparatingOn (α : Type*) (p : Set α → Prop) (t : Set α) : Prop where
  exists_countable_separating : ∃ S : Set (Set α), S.Countable ∧ (∀ s ∈ S, p s) ∧
    ∀ x ∈ t, ∀ y ∈ t, (∀ s ∈ S, x ∈ s ↔ y ∈ s) → x = y


theorem exists_countable_separating (α : Type*) (p : Set α → Prop) (t : Set α)
    [h : HasCountableSeparatingOn α p t] :
    ∃ S : Set (Set α), S.Countable ∧ (∀ s ∈ S, p s) ∧
      ∀ x ∈ t, ∀ y ∈ t, (∀ s ∈ S, x ∈ s ↔ y ∈ s) → x = y :=
  h.1


theorem exists_nonempty_countable_separating (α : Type*) {p : Set α → Prop} {s₀} (hp : p s₀)
    (t : Set α) [HasCountableSeparatingOn α p t] :
    ∃ S : Set (Set α), S.Nonempty ∧ S.Countable ∧ (∀ s ∈ S, p s) ∧
      ∀ x ∈ t, ∀ y ∈ t, (∀ s ∈ S, x ∈ s ↔ y ∈ s) → x = y :=
  let ⟨S, hSc, hSp, hSt⟩ := exists_countable_separating α p t
  ⟨insert s₀ S, insert_nonempty _ _, hSc.insert _, forall_insert_of_forall hSp hp,
    fun x hx y hy hxy ↦ hSt x hx y hy <| forall_of_forall_insert hxy⟩


theorem exists_seq_separating (α : Type*) {p : Set α → Prop} {s₀} (hp : p s₀) (t : Set α)
    [HasCountableSeparatingOn α p t] :
    ∃ S : ℕ → Set α, (∀ n, p (S n)) ∧ ∀ x ∈ t, ∀ y ∈ t, (∀ n, x ∈ S n ↔ y ∈ S n) → x = y := by
  /-
    α : Type u_1
    p : Set α → Prop
    s₀ : Set α
    hp : p s₀
    t : Set α
    inst✝ : HasCountableSeparatingOn α p t
    ⊢ Exists fun S => And (∀ (n : Nat), p (S n)) (∀ (x : α), Membership.mem t x →  …
  -/
  rcases exists_nonempty_countable_separating α hp t with ⟨S, hSne, hSc, hS⟩
  /-
    case intro.intro.intro
    α : Type u_1
    p : Set α → Prop
    s₀ : Set α
    hp : p s₀
    t : Set α
    inst✝ : HasCountableSeparatingOn α p t
    S : Set (Set α)
    hSne : S.Nonempty
    hSc : S.Countable
    hS : And (∀ (s : Set α), Membership.mem S s → p s) (∀ (x : α), Membership.mem  …
    ⊢ Exists fun S => And (∀ (n : Nat), p (S n)) (∀ (x : α), Membership.mem t x →  …
  -/
  rcases hSc.exists_eq_range hSne with ⟨S, rfl⟩
  /-
    case intro.intro.intro.intro
    α : Type u_1
    p : Set α → Prop
    s₀ : Set α
    hp : p s₀
    t : Set α
    inst✝ : HasCountableSeparatingOn α p t
    S : Nat → Set α
    hSne : (Set.range S).Nonempty
    hSc : (Set.range S).Countable
    hS : And (∀ (s : Set α), Membership.mem (Set.range S) s → p s) (∀ (x : α), Mem …
    ⊢ Exists fun S => And (∀ (n : Nat), p (S n)) (∀ (x : α), Membership.mem t x →  …
  -/
  use S
  /-
    case h
    α : Type u_1
    p : Set α → Prop
    s₀ : Set α
    hp : p s₀
    t : Set α
    inst✝ : HasCountableSeparatingOn α p t
    S : Nat → Set α
    hSne : (Set.range S).Nonempty
    hSc : (Set.range S).Countable
    hS : And (∀ (s : Set α), Membership.mem (Set.range S) s → p s) (∀ (x : α), Mem …
    ⊢ And (∀ (n : Nat), p (S n)) (∀ (x : α), Membership.mem t x → ∀ (y : α), Membe …
  -/
  simpa only [forall_mem_range] using hS
  /-
    🎉 no goals
  -/


theorem HasCountableSeparatingOn.mono {α} {p₁ p₂ : Set α → Prop} {t₁ t₂ : Set α}
    [h : HasCountableSeparatingOn α p₁ t₁] (hp : ∀ s, p₁ s → p₂ s) (ht : t₂ ⊆ t₁) :
    HasCountableSeparatingOn α p₂ t₂ where
  exists_countable_separating :=
    let ⟨S, hSc, hSp, hSt⟩ := h.1
    ⟨S, hSc, fun s hs ↦ hp s (hSp s hs), fun x hx y hy ↦ hSt x (ht hx) y (ht hy)⟩


theorem HasCountableSeparatingOn.of_subtype {α : Type*} {p : Set α → Prop} {t : Set α}
    {q : Set t → Prop} [h : HasCountableSeparatingOn t q univ]
    (hpq : ∀ U, q U → ∃ V, p V ∧ (↑) ⁻¹' V = U) : HasCountableSeparatingOn α p t := by
  /-
    α : Type u_1
    p : Set α → Prop
    t : Set α
    q : Set ↑t → Prop
    h : HasCountableSeparatingOn (↑t) q Set.univ
    hpq : ∀ (U : Set ↑t), q U → Exists fun V => And (p V) (Eq (Set.preimage Subtyp …
    ⊢ HasCountableSeparatingOn α p t
  -/
  rcases h.1 with ⟨S, hSc, hSq, hS⟩
  /-
    case intro.intro.intro
    α : Type u_1
    p : Set α → Prop
    t : Set α
    q : Set ↑t → Prop
    h : HasCountableSeparatingOn (↑t) q Set.univ
    hpq : ∀ (U : Set ↑t), q U → Exists fun V => And (p V) (Eq (Set.preimage Subtyp …
    S : Set (Set ↑t)
    hSc : S.Countable
    hSq : ∀ (s : Set ↑t), Membership.mem S s → q s
    hS : ∀ (x : ↑t), Membership.mem Set.univ x → ∀ (y : ↑t), Membership.mem Set.un …
    ⊢ HasCountableSeparatingOn α p t
  -/
  choose! V hpV hV using fun s hs ↦ hpq s (hSq s hs)
  /-
    case intro.intro.intro
    α : Type u_1
    p : Set α → Prop
    t : Set α
    q : Set ↑t → Prop
    h : HasCountableSeparatingOn (↑t) q Set.univ
    hpq : ∀ (U : Set ↑t), q U → Exists fun V => And (p V) (Eq (Set.preimage Subtyp …
    S : Set (Set ↑t)
    hSc : S.Countable
    hSq : ∀ (s : Set ↑t), Membership.mem S s → q s
    hS : ∀ (x : ↑t), Membership.mem Set.univ x → ∀ (y : ↑t), Membership.mem Set.un …
    V : Set ↑t → Set α
    hpV : ∀ (s : Set ↑t), Membership.mem S s → p (V s)
    hV : ∀ (s : Set ↑t), Membership.mem S s → Eq (Set.preimage Subtype.val (V s)) s
    ⊢ HasCountableSeparatingOn α p t
  -/
  refine ⟨⟨V '' S, hSc.image _, forall_mem_image.2 hpV, fun x hx y hy h ↦ ?_⟩⟩
  /-
    case intro.intro.intro
    α : Type u_1
    p : Set α → Prop
    t : Set α
    q : Set ↑t → Prop
    h✝ : HasCountableSeparatingOn (↑t) q Set.univ
    hpq : ∀ (U : Set ↑t), q U → Exists fun V => And (p V) (Eq (Set.preimage Subtyp …
    S : Set (Set ↑t)
    hSc : S.Countable
    hSq : ∀ (s : Set ↑t), Membership.mem S s → q s
    hS : ∀ (x : ↑t), Membership.mem Set.univ x → ∀ (y : ↑t), Membership.mem Set.un …
    V : Set ↑t → Set α
    hpV : ∀ (s : Set ↑t), Membership.mem S s → p (V s)
    hV : ∀ (s : Set ↑t), Membership.mem S s → Eq (Set.preimage Subtype.val (V s)) s
    x : α
    hx : Membership.mem t x
    y : α
    hy : Membership.mem t y
    h : ∀ (s : Set α), Membership.mem (Set.image V S) s → Iff (Membership.mem s x) …
    ⊢ Eq x y
  -/
  refine congr_arg Subtype.val (hS ⟨x, hx⟩ trivial ⟨y, hy⟩ trivial fun U hU ↦ ?_)
  /-
    case intro.intro.intro
    α : Type u_1
    p : Set α → Prop
    t : Set α
    q : Set ↑t → Prop
    h✝ : HasCountableSeparatingOn (↑t) q Set.univ
    hpq : ∀ (U : Set ↑t), q U → Exists fun V => And (p V) (Eq (Set.preimage Subtyp …
    S : Set (Set ↑t)
    hSc : S.Countable
    hSq : ∀ (s : Set ↑t), Membership.mem S s → q s
    hS : ∀ (x : ↑t), Membership.mem Set.univ x → ∀ (y : ↑t), Membership.mem Set.un …
    V : Set ↑t → Set α
    hpV : ∀ (s : Set ↑t), Membership.mem S s → p (V s)
    hV : ∀ (s : Set ↑t), Membership.mem S s → Eq (Set.preimage Subtype.val (V s)) s
    x : α
    hx : Membership.mem t x
    y : α
    hy : Membership.mem t y
    h : ∀ (s : Set α), Membership.mem (Set.image V S) s → Iff (Membership.mem s x) …
    U : Set ↑t
    hU : Membership.mem S U
    ⊢ Iff (Membership.mem U ⟨x, hx⟩) (Membership.mem U ⟨y, hy⟩)
  -/
  rw [← hV U hU]
  /-
    case intro.intro.intro
    α : Type u_1
    p : Set α → Prop
    t : Set α
    q : Set ↑t → Prop
    h✝ : HasCountableSeparatingOn (↑t) q Set.univ
    hpq : ∀ (U : Set ↑t), q U → Exists fun V => And (p V) (Eq (Set.preimage Subtyp …
    S : Set (Set ↑t)
    hSc : S.Countable
    hSq : ∀ (s : Set ↑t), Membership.mem S s → q s
    hS : ∀ (x : ↑t), Membership.mem Set.univ x → ∀ (y : ↑t), Membership.mem Set.un …
    V : Set ↑t → Set α
    hpV : ∀ (s : Set ↑t), Membership.mem S s → p (V s)
    hV : ∀ (s : Set ↑t), Membership.mem S s → Eq (Set.preimage Subtype.val (V s)) s
    x : α
    hx : Membership.mem t x
    y : α
    hy : Membership.mem t y
    h : ∀ (s : Set α), Membership.mem (Set.image V S) s → Iff (Membership.mem s x) …
    U : Set ↑t
    hU : Membership.mem S U
    ⊢ Iff (Membership.mem (Set.preimage Subtype.val (V U)) ⟨x, hx⟩) (Membership.me …
  -/
  exact h _ (mem_image_of_mem _ hU)
  /-
    🎉 no goals
  -/


theorem HasCountableSeparatingOn.subtype_iff {α : Type*} {p : Set α → Prop} {t : Set α} :
    HasCountableSeparatingOn t (fun u ↦ ∃ v, p v ∧ (↑) ⁻¹' v = u) univ ↔
    HasCountableSeparatingOn α p t := by
  /-
    α : Type u_1
    p : Set α → Prop
    t : Set α
    ⊢ Iff (HasCountableSeparatingOn (↑t) (fun u => Exists fun v => And (p v) (Eq ( …
  -/
  constructor <;> intro h
    /-
      case mp
      α : Type u_1
      p : Set α → Prop
      t : Set α
      h : HasCountableSeparatingOn (↑t) (fun u => Exists fun v => And (p v) (Eq (Set …
      ⊢ HasCountableSeparatingOn α p t
    -/
  · exact h.of_subtype <| fun s ↦ id
    /-
      🎉 no goals
    -/
  /-
    case mpr
    α : Type u_1
    p : Set α → Prop
    t : Set α
    h : HasCountableSeparatingOn α p t
    ⊢ HasCountableSeparatingOn (↑t) (fun u => Exists fun v => And (p v) (Eq (Set.p …
  -/
  rcases h with ⟨S, Sct, Sp, hS⟩
  /-
    case mpr.mk.intro.intro.intro
    α : Type u_1
    p : Set α → Prop
    t : Set α
    S : Set (Set α)
    Sct : S.Countable
    Sp : ∀ (s : Set α), Membership.mem S s → p s
    hS : ∀ (x : α), Membership.mem t x → ∀ (y : α), Membership.mem t y → (∀ (s : S …
    ⊢ HasCountableSeparatingOn (↑t) (fun u => Exists fun v => And (p v) (Eq (Set.p …
  -/
  use {Subtype.val ⁻¹' s | s ∈ S}, Sct.image _, ?_, ?_
    /-
      case left
      α : Type u_1
      p : Set α → Prop
      t : Set α
      S : Set (Set α)
      Sct : S.Countable
      Sp : ∀ (s : Set α), Membership.mem S s → p s
      hS : ∀ (x : α), Membership.mem t x → ∀ (y : α), Membership.mem t y → (∀ (s : S …
      ⊢ ∀ (s : Set ↑t), Membership.mem (setOf fun x => Exists fun s => And (Membersh …
    -/
  · rintro u ⟨t, tS, rfl⟩
    /-
      case left.intro.intro
      α : Type u_1
      p : Set α → Prop
      t✝ : Set α
      S : Set (Set α)
      Sct : S.Countable
      Sp : ∀ (s : Set α), Membership.mem S s → p s
      hS : ∀ (x : α), Membership.mem t✝ x → ∀ (y : α), Membership.mem t✝ y → (∀ (s : …
      t : Set α
      tS : Membership.mem S t
      ⊢ Exists fun v => And (p v) (Eq (Set.preimage Subtype.val v) (Set.preimage Sub …
    -/
    exact ⟨t, Sp _ tS, rfl⟩
    /-
      🎉 no goals
    -/
  /-
    case right
    α : Type u_1
    p : Set α → Prop
    t : Set α
    S : Set (Set α)
    Sct : S.Countable
    Sp : ∀ (s : Set α), Membership.mem S s → p s
    hS : ∀ (x : α), Membership.mem t x → ∀ (y : α), Membership.mem t y → (∀ (s : S …
    ⊢ ∀ (x : ↑t), Membership.mem Set.univ x → ∀ (y : ↑t), Membership.mem Set.univ  …
  -/
  rintro x - y - hxy
  exact Subtype.val_injective <| hS _ (Subtype.coe_prop _) _ (Subtype.coe_prop _)
    fun s hs ↦ hxy (Subtype.val ⁻¹' s) ⟨s, hs, rfl⟩


theorem exists_subset_subsingleton_mem_of_forall_separating (p : Set α → Prop)
    {s : Set α} [h : HasCountableSeparatingOn α p s] (hs : s ∈ l)
    (hl : ∀ U, p U → U ∈ l ∨ Uᶜ ∈ l) : ∃ t, t ⊆ s ∧ t.Subsingleton ∧ t ∈ l := by
  /-
    α : Type u_1
    l : Filter α
    inst✝ : CountableInterFilter l
    p : Set α → Prop
    s : Set α
    h : HasCountableSeparatingOn α p s
    hs : Membership.mem l s
    hl : ∀ (U : Set α), p U → Or (Membership.mem l U) (Membership.mem l (HasCompl. …
    ⊢ Exists fun t => And (HasSubset.Subset t s) (And t.Subsingleton (Membership.m …
  -/
  rcases h.1 with ⟨S, hSc, hSp, hS⟩
  /-
    case intro.intro.intro
    α : Type u_1
    l : Filter α
    inst✝ : CountableInterFilter l
    p : Set α → Prop
    s : Set α
    h : HasCountableSeparatingOn α p s
    hs : Membership.mem l s
    hl : ∀ (U : Set α), p U → Or (Membership.mem l U) (Membership.mem l (HasCompl. …
    S : Set (Set α)
    hSc : S.Countable
    hSp : ∀ (s : Set α), Membership.mem S s → p s
    hS : ∀ (x : α), Membership.mem s x → ∀ (y : α), Membership.mem s y → (∀ (s : S …
    ⊢ Exists fun t => And (HasSubset.Subset t s) (And t.Subsingleton (Membership.m …
  -/
  refine ⟨s ∩ ⋂₀ (S ∩ l.sets) ∩ ⋂ (U ∈ S) (_ : Uᶜ ∈ l), Uᶜ, ?_, ?_, ?_⟩
    /-
      case intro.intro.intro.refine_1
      α : Type u_1
      l : Filter α
      inst✝ : CountableInterFilter l
      p : Set α → Prop
      s : Set α
      h : HasCountableSeparatingOn α p s
      hs : Membership.mem l s
      hl : ∀ (U : Set α), p U → Or (Membership.mem l U) (Membership.mem l (HasCompl. …
      S : Set (Set α)
      hSc : S.Countable
      hSp : ∀ (s : Set α), Membership.mem S s → p s
      hS : ∀ (x : α), Membership.mem s x → ∀ (y : α), Membership.mem s y → (∀ (s : S …
      ⊢ HasSubset.Subset (Inter.inter (Inter.inter s (Inter.inter S l.sets).sInter)  …
    -/
  · exact fun _ h ↦ h.1.1
    /-
      🎉 no goals
    -/
    /-
      case intro.intro.intro.refine_2
      α : Type u_1
      l : Filter α
      inst✝ : CountableInterFilter l
      p : Set α → Prop
      s : Set α
      h : HasCountableSeparatingOn α p s
      hs : Membership.mem l s
      hl : ∀ (U : Set α), p U → Or (Membership.mem l U) (Membership.mem l (HasCompl. …
      S : Set (Set α)
      hSc : S.Countable
      hSp : ∀ (s : Set α), Membership.mem S s → p s
      hS : ∀ (x : α), Membership.mem s x → ∀ (y : α), Membership.mem s y → (∀ (s : S …
      ⊢ (Inter.inter (Inter.inter s (Inter.inter S l.sets).sInter) (Set.iInter fun U …
    -/
  · intro x hx y hy
    /-
      case intro.intro.intro.refine_2
      α : Type u_1
      l : Filter α
      inst✝ : CountableInterFilter l
      p : Set α → Prop
      s : Set α
      h : HasCountableSeparatingOn α p s
      hs : Membership.mem l s
      hl : ∀ (U : Set α), p U → Or (Membership.mem l U) (Membership.mem l (HasCompl. …
      S : Set (Set α)
      hSc : S.Countable
      hSp : ∀ (s : Set α), Membership.mem S s → p s
      hS : ∀ (x : α), Membership.mem s x → ∀ (y : α), Membership.mem s y → (∀ (s : S …
      x : α
      hx : Membership.mem (Inter.inter (Inter.inter s (Inter.inter S l.sets).sInter) …
      y : α
      hy : Membership.mem (Inter.inter (Inter.inter s (Inter.inter S l.sets).sInter) …
      ⊢ Eq x y
    -/
    simp only [mem_sInter, mem_inter_iff, mem_iInter, mem_compl_iff] at hx hy
    /-
      case intro.intro.intro.refine_2
      α : Type u_1
      l : Filter α
      inst✝ : CountableInterFilter l
      p : Set α → Prop
      s : Set α
      h : HasCountableSeparatingOn α p s
      hs : Membership.mem l s
      hl : ∀ (U : Set α), p U → Or (Membership.mem l U) (Membership.mem l (HasCompl. …
      S : Set (Set α)
      hSc : S.Countable
      hSp : ∀ (s : Set α), Membership.mem S s → p s
      hS : ∀ (x : α), Membership.mem s x → ∀ (y : α), Membership.mem s y → (∀ (s : S …
      x y : α
      hx : And (And (Membership.mem s x) (∀ (t : Set α), And (Membership.mem S t) (M …
      hy : And (And (Membership.mem s y) (∀ (t : Set α), And (Membership.mem S t) (M …
      ⊢ Eq x y
    -/
    refine hS x hx.1.1 y hy.1.1 (fun s hsS ↦ ?_)
    cases hl s (hSp s hsS) with
    | inl hsl => simp only [hx.1.2 s ⟨hsS, hsl⟩, hy.1.2 s ⟨hsS, hsl⟩]
    | inr hsl => simp only [hx.2 s hsS hsl, hy.2 s hsS hsl]
  · exact inter_mem
      (inter_mem hs ((countable_sInter_mem (hSc.mono inter_subset_left)).2 fun _ h ↦ h.2))
      ((countable_bInter_mem hSc).2 fun U hU ↦ iInter_mem'.2 id)


theorem exists_mem_singleton_mem_of_mem_of_nonempty_of_forall_separating (p : Set α → Prop)
    {s : Set α} [HasCountableSeparatingOn α p s] (hs : s ∈ l) (hne : s.Nonempty)
    (hl : ∀ U, p U → U ∈ l ∨ Uᶜ ∈ l) : ∃ a ∈ s, {a} ∈ l := by
  /-
    α : Type u_1
    l : Filter α
    inst✝¹ : CountableInterFilter l
    p : Set α → Prop
    s : Set α
    inst✝ : HasCountableSeparatingOn α p s
    hs : Membership.mem l s
    hne : s.Nonempty
    hl : ∀ (U : Set α), p U → Or (Membership.mem l U) (Membership.mem l (HasCompl. …
    ⊢ Exists fun a => And (Membership.mem s a) (Membership.mem l (Singleton.single …
  -/
  rcases exists_subset_subsingleton_mem_of_forall_separating p hs hl with ⟨t, hts, ht, htl⟩
  /-
    case intro.intro.intro
    α : Type u_1
    l : Filter α
    inst✝¹ : CountableInterFilter l
    p : Set α → Prop
    s : Set α
    inst✝ : HasCountableSeparatingOn α p s
    hs : Membership.mem l s
    hne : s.Nonempty
    hl : ∀ (U : Set α), p U → Or (Membership.mem l U) (Membership.mem l (HasCompl. …
    t : Set α
    hts : HasSubset.Subset t s
    ht : t.Subsingleton
    htl : Membership.mem l t
    ⊢ Exists fun a => And (Membership.mem s a) (Membership.mem l (Singleton.single …
  -/
  rcases ht.eq_empty_or_singleton with rfl | ⟨x, rfl⟩
    /-
      case intro.intro.intro.inl
      α : Type u_1
      l : Filter α
      inst✝¹ : CountableInterFilter l
      p : Set α → Prop
      s : Set α
      inst✝ : HasCountableSeparatingOn α p s
      hs : Membership.mem l s
      hne : s.Nonempty
      hl : ∀ (U : Set α), p U → Or (Membership.mem l U) (Membership.mem l (HasCompl. …
      hts : HasSubset.Subset EmptyCollection.emptyCollection s
      ht : EmptyCollection.emptyCollection.Subsingleton
      htl : Membership.mem l EmptyCollection.emptyCollection
      ⊢ Exists fun a => And (Membership.mem s a) (Membership.mem l (Singleton.single …
    -/
  · exact hne.imp fun a ha ↦ ⟨ha, mem_of_superset htl (empty_subset _)⟩
    /-
      🎉 no goals
    -/
    /-
      case intro.intro.intro.inr.intro
      α : Type u_1
      l : Filter α
      inst✝¹ : CountableInterFilter l
      p : Set α → Prop
      s : Set α
      inst✝ : HasCountableSeparatingOn α p s
      hs : Membership.mem l s
      hne : s.Nonempty
      hl : ∀ (U : Set α), p U → Or (Membership.mem l U) (Membership.mem l (HasCompl. …
      x : α
      hts : HasSubset.Subset (Singleton.singleton x) s
      ht : (Singleton.singleton x).Subsingleton
      htl : Membership.mem l (Singleton.singleton x)
      ⊢ Exists fun a => And (Membership.mem s a) (Membership.mem l (Singleton.single …
    -/
  · exact ⟨x, hts rfl, htl⟩
    /-
      🎉 no goals
    -/


theorem exists_singleton_mem_of_mem_of_forall_separating [Nonempty α] (p : Set α → Prop)
    {s : Set α} [HasCountableSeparatingOn α p s] (hs : s ∈ l) (hl : ∀ U, p U → U ∈ l ∨ Uᶜ ∈ l) :
    ∃ a, {a} ∈ l := by
  /-
    α : Type u_1
    l : Filter α
    inst✝² : CountableInterFilter l
    inst✝¹ : Nonempty α
    p : Set α → Prop
    s : Set α
    inst✝ : HasCountableSeparatingOn α p s
    hs : Membership.mem l s
    hl : ∀ (U : Set α), p U → Or (Membership.mem l U) (Membership.mem l (HasCompl. …
    ⊢ Exists fun a => Membership.mem l (Singleton.singleton a)
  -/
  rcases s.eq_empty_or_nonempty with rfl | hne
    /-
      case inl
      α : Type u_1
      l : Filter α
      inst✝² : CountableInterFilter l
      inst✝¹ : Nonempty α
      p : Set α → Prop
      hl : ∀ (U : Set α), p U → Or (Membership.mem l U) (Membership.mem l (HasCompl. …
      inst✝ : HasCountableSeparatingOn α p EmptyCollection.emptyCollection
      hs : Membership.mem l EmptyCollection.emptyCollection
      ⊢ Exists fun a => Membership.mem l (Singleton.singleton a)
    -/
  · exact ‹Nonempty α›.elim fun a ↦ ⟨a, mem_of_superset hs (empty_subset _)⟩
    /-
      🎉 no goals
    -/
  · exact (exists_mem_singleton_mem_of_mem_of_nonempty_of_forall_separating p hs hne hl).imp fun _ ↦
      And.right


theorem exists_subsingleton_mem_of_forall_separating (p : Set α → Prop)
    [HasCountableSeparatingOn α p univ] (hl : ∀ U, p U → U ∈ l ∨ Uᶜ ∈ l) :
    ∃ s : Set α, s.Subsingleton ∧ s ∈ l :=
  let ⟨t, _, hts, htl⟩ := exists_subset_subsingleton_mem_of_forall_separating p univ_mem hl
  ⟨t, hts, htl⟩


theorem exists_singleton_mem_of_forall_separating [Nonempty α] (p : Set α → Prop)
    [HasCountableSeparatingOn α p univ] (hl : ∀ U, p U → U ∈ l ∨ Uᶜ ∈ l) :
    ∃ x : α, {x} ∈ l :=
  exists_singleton_mem_of_mem_of_forall_separating p univ_mem hl


theorem exists_mem_eventuallyEq_const_of_eventually_mem_of_forall_separating (p : Set β → Prop)
    {s : Set β} [HasCountableSeparatingOn β p s] (hs : ∀ᶠ x in l, f x ∈ s) (hne : s.Nonempty)
    (h : ∀ U, p U → (∀ᶠ x in l, f x ∈ U) ∨ (∀ᶠ x in l, f x ∉ U)) :
    ∃ a ∈ s, f =ᶠ[l] const α a :=
  exists_mem_singleton_mem_of_mem_of_nonempty_of_forall_separating p (l := map f l) hs hne h


theorem exists_eventuallyEq_const_of_eventually_mem_of_forall_separating [Nonempty β]
    (p : Set β → Prop) {s : Set β} [HasCountableSeparatingOn β p s] (hs : ∀ᶠ x in l, f x ∈ s)
    (h : ∀ U, p U → (∀ᶠ x in l, f x ∈ U) ∨ (∀ᶠ x in l, f x ∉ U)) :
    ∃ a, f =ᶠ[l] const α a :=
  exists_singleton_mem_of_mem_of_forall_separating (l := map f l) p hs h


theorem exists_eventuallyEq_const_of_forall_separating [Nonempty β] (p : Set β → Prop)
    [HasCountableSeparatingOn β p univ]
    (h : ∀ U, p U → (∀ᶠ x in l, f x ∈ U) ∨ (∀ᶠ x in l, f x ∉ U)) :
    ∃ a, f =ᶠ[l] const α a :=
  exists_singleton_mem_of_forall_separating (l := map f l) p h


theorem of_eventually_mem_of_forall_separating_mem_iff (p : Set β → Prop) {s : Set β}
    [h' : HasCountableSeparatingOn β p s] (hf : ∀ᶠ x in l, f x ∈ s) (hg : ∀ᶠ x in l, g x ∈ s)
    (h : ∀ U : Set β, p U → ∀ᶠ x in l, f x ∈ U ↔ g x ∈ U) : f =ᶠ[l] g := by
  /-
    α : Type u_1
    β : Type u_2
    l : Filter α
    inst✝ : CountableInterFilter l
    f g : α → β
    p : Set β → Prop
    s : Set β
    h' : HasCountableSeparatingOn β p s
    hf : Filter.Eventually (fun x => Membership.mem s (f x)) l
    hg : Filter.Eventually (fun x => Membership.mem s (g x)) l
    h : ∀ (U : Set β), p U → Filter.Eventually (fun x => Iff (Membership.mem U (f  …
    ⊢ l.EventuallyEq f g
  -/
  rcases h'.1 with ⟨S, hSc, hSp, hS⟩
  have H : ∀ᶠ x in l, ∀ s ∈ S, f x ∈ s ↔ g x ∈ s :=
    (eventually_countable_ball hSc).2 fun s hs ↦ (h _ (hSp _ hs))
  /-
    case intro.intro.intro
    α : Type u_1
    β : Type u_2
    l : Filter α
    inst✝ : CountableInterFilter l
    f g : α → β
    p : Set β → Prop
    s : Set β
    h' : HasCountableSeparatingOn β p s
    hf : Filter.Eventually (fun x => Membership.mem s (f x)) l
    hg : Filter.Eventually (fun x => Membership.mem s (g x)) l
    h : ∀ (U : Set β), p U → Filter.Eventually (fun x => Iff (Membership.mem U (f  …
    S : Set (Set β)
    hSc : S.Countable
    hSp : ∀ (s : Set β), Membership.mem S s → p s
    hS : ∀ (x : β), Membership.mem s x → ∀ (y : β), Membership.mem s y → (∀ (s : S …
    H : Filter.Eventually (fun x => ∀ (s : Set β), Membership.mem S s → Iff (Membe …
    ⊢ l.EventuallyEq f g
  -/
  filter_upwards [H, hf, hg] with x hx hxf hxg using hS _ hxf _ hxg hx
  /-
    🎉 no goals
  -/


theorem of_forall_separating_mem_iff (p : Set β → Prop)
    [HasCountableSeparatingOn β p univ] (h : ∀ U : Set β, p U → ∀ᶠ x in l, f x ∈ U ↔ g x ∈ U) :
    f =ᶠ[l] g :=
  of_eventually_mem_of_forall_separating_mem_iff p (s := univ) univ_mem univ_mem h


theorem of_eventually_mem_of_forall_separating_preimage (p : Set β → Prop) {s : Set β}
    [HasCountableSeparatingOn β p s] (hf : ∀ᶠ x in l, f x ∈ s) (hg : ∀ᶠ x in l, g x ∈ s)
    (h : ∀ U : Set β, p U → f ⁻¹' U =ᶠ[l] g ⁻¹' U) : f =ᶠ[l] g :=
  of_eventually_mem_of_forall_separating_mem_iff p hf hg fun U hU ↦ (h U hU).mem_iff


theorem of_forall_separating_preimage (p : Set β → Prop) [HasCountableSeparatingOn β p univ]
    (h : ∀ U : Set β, p U → f ⁻¹' U =ᶠ[l] g ⁻¹' U) : f =ᶠ[l] g :=
  of_eventually_mem_of_forall_separating_preimage p (s := univ) univ_mem univ_mem h


