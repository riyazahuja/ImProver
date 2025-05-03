/-- Given `δ : α → Sort*`, `Pi.empty δ` is the trivial dependent function out of the empty
multiset. -/
def empty (δ : α → Sort*) : ∀ a ∈ (0 : Multiset α), δ a :=
  nofun


/-- Given `δ : α → Sort*`, a multiset `m` and a term `a`, as well as a term `b : δ a` and a
function `f` such that `f a' : δ a'` for all `a'` in `m`, `Pi.cons m a b f` is a function `g` such
that `g a'' : δ a''` for all `a''` in `a ::ₘ m`. -/
def cons (b : δ a) (f : ∀ a ∈ m, δ a) : ∀ a' ∈ a ::ₘ m, δ a' :=
  fun a' ha' => if h : a' = a then Eq.ndrec b h.symm else f a' <| (mem_cons.1 ha').resolve_left h


theorem cons_same {b : δ a} {f : ∀ a ∈ m, δ a} (h : a ∈ a ::ₘ m) :
    cons m a b f a h = b :=
  dif_pos rfl


theorem cons_ne {a a' : α} {b : δ a} {f : ∀ a ∈ m, δ a} (h' : a' ∈ a ::ₘ m)
    (h : a' ≠ a) : Pi.cons m a b f a' h' = f a' ((mem_cons.1 h').resolve_left h) :=
  dif_neg h


theorem cons_swap {a a' : α} {b : δ a} {b' : δ a'} {m : Multiset α} {f : ∀ a ∈ m, δ a}
    (h : a ≠ a') : HEq (Pi.cons (a' ::ₘ m) a b (Pi.cons m a' b' f))
      (Pi.cons (a ::ₘ m) a' b' (Pi.cons m a b f)) := by
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    δ : α → Sort u_2
    a a' : α
    b : δ a
    b' : δ a'
    m : Multiset α
    f : (a : α) → Membership.mem m a → δ a
    h : Ne a a'
    ⊢ HEq (Multiset.Pi.cons (Multiset.cons a' m) a b (Multiset.Pi.cons m a' b' f)) …
  -/
  apply hfunext rfl
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    δ : α → Sort u_2
    a a' : α
    b : δ a
    b' : δ a'
    m : Multiset α
    f : (a : α) → Membership.mem m a → δ a
    h : Ne a a'
    ⊢ ∀ (a_1 a'_1 : α), HEq a_1 a'_1 → HEq (Multiset.Pi.cons (Multiset.cons a' m)  …
  -/
  simp only [heq_iff_eq]
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    δ : α → Sort u_2
    a a' : α
    b : δ a
    b' : δ a'
    m : Multiset α
    f : (a : α) → Membership.mem m a → δ a
    h : Ne a a'
    ⊢ ∀ (a_1 a'_1 : α), Eq a_1 a'_1 → HEq (Multiset.Pi.cons (Multiset.cons a' m) a …
  -/
  rintro a'' _ rfl
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    δ : α → Sort u_2
    a a' : α
    b : δ a
    b' : δ a'
    m : Multiset α
    f : (a : α) → Membership.mem m a → δ a
    h : Ne a a'
    a'' : α
    ⊢ HEq (Multiset.Pi.cons (Multiset.cons a' m) a b (Multiset.Pi.cons m a' b' f)  …
  -/
  refine hfunext (by rw [Multiset.cons_swap]) fun ha₁ ha₂ _ => ?_
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    δ : α → Sort u_2
    a a' : α
    b : δ a
    b' : δ a'
    m : Multiset α
    f : (a : α) → Membership.mem m a → δ a
    h : Ne a a'
    a'' : α
    ha₁ : Membership.mem (Multiset.cons a (Multiset.cons a' m)) a''
    ha₂ : Membership.mem (Multiset.cons a' (Multiset.cons a m)) a''
    x✝ : HEq ha₁ ha₂
    ⊢ HEq (Multiset.Pi.cons (Multiset.cons a' m) a b (Multiset.Pi.cons m a' b' f)  …
  -/
  rcases Decidable.ne_or_eq a'' a with (h₁ | rfl)
  /-
    case inl
    α : Type u_1
    inst✝ : DecidableEq α
    δ : α → Sort u_2
    a a' : α
    b : δ a
    b' : δ a'
    m : Multiset α
    f : (a : α) → Membership.mem m a → δ a
    h : Ne a a'
    a'' : α
    ha₁ : Membership.mem (Multiset.cons a (Multiset.cons a' m)) a''
    ha₂ : Membership.mem (Multiset.cons a' (Multiset.cons a m)) a''
    x✝ : HEq ha₁ ha₂
    h₁ : Ne a'' a
    ⊢ HEq (Multiset.Pi.cons (Multiset.cons a' m) a b (Multiset.Pi.cons m a' b' f)  …
  -/
  on_goal 1 => rcases Decidable.eq_or_ne a'' a' with (rfl | h₂)
  /-
    case inl.inl
    α : Type u_1
    inst✝ : DecidableEq α
    δ : α → Sort u_2
    a : α
    b : δ a
    m : Multiset α
    f : (a : α) → Membership.mem m a → δ a
    a'' : α
    h₁ : Ne a'' a
    b' : δ a''
    h : Ne a a''
    ha₁ : Membership.mem (Multiset.cons a (Multiset.cons a'' m)) a''
    ha₂ : Membership.mem (Multiset.cons a'' (Multiset.cons a m)) a''
    x✝ : HEq ha₁ ha₂
    ⊢ HEq (Multiset.Pi.cons (Multiset.cons a'' m) a b (Multiset.Pi.cons m a'' b' f …
  -/
  all_goals simp [*, Pi.cons_same, Pi.cons_ne]
  /-
    🎉 no goals
  -/


@[simp, nolint simpNF] -- Porting note: false positive, this lemma can prove itself
theorem cons_eta {m : Multiset α} {a : α} (f : ∀ a' ∈ a ::ₘ m, δ a') :
    (cons m a (f _ (mem_cons_self _ _)) fun a' ha' => f a' (mem_cons_of_mem ha')) = f := by
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    δ : α → Sort u_2
    m : Multiset α
    a : α
    f : (a' : α) → Membership.mem (Multiset.cons a m) a' → δ a'
    ⊢ Eq (Multiset.Pi.cons m a (f a ⋯) fun a' ha' => f a' ⋯) f
  -/
  ext a' h'
  /-
    case h.h
    α : Type u_1
    inst✝ : DecidableEq α
    δ : α → Sort u_2
    m : Multiset α
    a : α
    f : (a' : α) → Membership.mem (Multiset.cons a m) a' → δ a'
    a' : α
    h' : Membership.mem (Multiset.cons a m) a'
    ⊢ Eq (Multiset.Pi.cons m a (f a ⋯) (fun a' ha' => f a' ⋯) a' h') (f a' h')
  -/
  by_cases h : a' = a
    /-
      case pos
      α : Type u_1
      inst✝ : DecidableEq α
      δ : α → Sort u_2
      m : Multiset α
      a : α
      f : (a' : α) → Membership.mem (Multiset.cons a m) a' → δ a'
      a' : α
      h' : Membership.mem (Multiset.cons a m) a'
      h : Eq a' a
      ⊢ Eq (Multiset.Pi.cons m a (f a ⋯) (fun a' ha' => f a' ⋯) a' h') (f a' h')
    -/
  · subst h
    /-
      case pos
      α : Type u_1
      inst✝ : DecidableEq α
      δ : α → Sort u_2
      m : Multiset α
      a' : α
      f : (a'_1 : α) → Membership.mem (Multiset.cons a' m) a'_1 → δ a'_1
      h' : Membership.mem (Multiset.cons a' m) a'
      ⊢ Eq (Multiset.Pi.cons m a' (f a' ⋯) (fun a'_1 ha' => f a'_1 ⋯) a' h') (f a' h')
    -/
    rw [Pi.cons_same]
    /-
      🎉 no goals
    -/
    /-
      case neg
      α : Type u_1
      inst✝ : DecidableEq α
      δ : α → Sort u_2
      m : Multiset α
      a : α
      f : (a' : α) → Membership.mem (Multiset.cons a m) a' → δ a'
      a' : α
      h' : Membership.mem (Multiset.cons a m) a'
      h : Not (Eq a' a)
      ⊢ Eq (Multiset.Pi.cons m a (f a ⋯) (fun a' ha' => f a' ⋯) a' h') (f a' h')
    -/
  · rw [Pi.cons_ne _ h]
    /-
      🎉 no goals
    -/


theorem cons_map (b : δ a) (f : ∀ a' ∈ m, δ a')
    {δ' : α → Sort*} (φ : ∀ ⦃a'⦄, δ a' → δ' a') :
    Pi.cons _ _ (φ b) (fun a' ha' ↦ φ (f a' ha')) = (fun a' ha' ↦ φ ((cons _ _ b f) a' ha')) := by
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    δ : α → Sort u_2
    m : Multiset α
    a : α
    b : δ a
    f : (a' : α) → Membership.mem m a' → δ a'
    δ' : α → Sort u_3
    φ : ⦃a' : α⦄ → δ a' → δ' a'
    ⊢ Eq (Multiset.Pi.cons m a (φ b) fun a' ha' => φ (f a' ha')) fun a' ha' => φ ( …
  -/
  ext a' ha'
  /-
    case h.h
    α : Type u_1
    inst✝ : DecidableEq α
    δ : α → Sort u_2
    m : Multiset α
    a : α
    b : δ a
    f : (a' : α) → Membership.mem m a' → δ a'
    δ' : α → Sort u_3
    φ : ⦃a' : α⦄ → δ a' → δ' a'
    a' : α
    ha' : Membership.mem (Multiset.cons a m) a'
    ⊢ Eq (Multiset.Pi.cons m a (φ b) (fun a' ha' => φ (f a' ha')) a' ha') (φ (Mult …
  -/
  refine (congrArg₂ _ ?_ rfl).trans (apply_dite (@φ _) (a' = a) _ _).symm
  /-
    case h.h
    α : Type u_1
    inst✝ : DecidableEq α
    δ : α → Sort u_2
    m : Multiset α
    a : α
    b : δ a
    f : (a' : α) → Membership.mem m a' → δ a'
    δ' : α → Sort u_3
    φ : ⦃a' : α⦄ → δ a' → δ' a'
    a' : α
    ha' : Membership.mem (Multiset.cons a m) a'
    ⊢ Eq (fun h => Eq.ndrec (φ b) ⋯) fun h => φ (Eq.ndrec b ⋯)
  -/
  ext rfl
  /-
    case h.h.h
    α : Type u_1
    inst✝ : DecidableEq α
    δ : α → Sort u_2
    m : Multiset α
    f : (a' : α) → Membership.mem m a' → δ a'
    δ' : α → Sort u_3
    φ : ⦃a' : α⦄ → δ a' → δ' a'
    a' : α
    b : δ a'
    ha' : Membership.mem (Multiset.cons a' m) a'
    ⊢ Eq (Eq.ndrec (φ b) ⋯) (φ (Eq.ndrec b ⋯))
  -/
  rfl
  /-
    🎉 no goals
  -/


theorem forall_rel_cons_ext {r : ∀ ⦃a⦄, δ a → δ a → Prop} {b₁ b₂ : δ a} {f₁ f₂ : ∀ a' ∈ m, δ a'}
    (hb : r b₁ b₂) (hf : ∀ (a : α) (ha : a ∈ m), r (f₁ a ha) (f₂ a ha)) :
    ∀ a ha, r (cons _ _ b₁ f₁ a ha) (cons _ _ b₂ f₂ a ha) := by
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    δ : α → Sort u_2
    m : Multiset α
    a : α
    r : ⦃a : α⦄ → δ a → δ a → Prop
    b₁ b₂ : δ a
    f₁ f₂ : (a' : α) → Membership.mem m a' → δ a'
    hb : r b₁ b₂
    hf : ∀ (a : α) (ha : Membership.mem m a), r (f₁ a ha) (f₂ a ha)
    ⊢ ∀ (a_1 : α) (ha : Membership.mem (Multiset.cons a m) a_1), r (Multiset.Pi.co …
  -/
  intro a ha
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    δ : α → Sort u_2
    m : Multiset α
    a✝ : α
    r : ⦃a : α⦄ → δ a → δ a → Prop
    b₁ b₂ : δ a✝
    f₁ f₂ : (a' : α) → Membership.mem m a' → δ a'
    hb : r b₁ b₂
    hf : ∀ (a : α) (ha : Membership.mem m a), r (f₁ a ha) (f₂ a ha)
    a : α
    ha : Membership.mem (Multiset.cons a✝ m) a
    ⊢ r (Multiset.Pi.cons m a✝ b₁ f₁ a ha) (Multiset.Pi.cons m a✝ b₂ f₂ a ha)
  -/
  dsimp [cons]
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    δ : α → Sort u_2
    m : Multiset α
    a✝ : α
    r : ⦃a : α⦄ → δ a → δ a → Prop
    b₁ b₂ : δ a✝
    f₁ f₂ : (a' : α) → Membership.mem m a' → δ a'
    hb : r b₁ b₂
    hf : ∀ (a : α) (ha : Membership.mem m a), r (f₁ a ha) (f₂ a ha)
    a : α
    ha : Membership.mem (Multiset.cons a✝ m) a
    ⊢ r (dite (Eq a a✝) (fun h => Eq.rec b₁ ⋯) fun h => f₁ a ⋯) (dite (Eq a a✝) (f …
  -/
  split_ifs with H
    /-
      case pos
      α : Type u_1
      inst✝ : DecidableEq α
      δ : α → Sort u_2
      m : Multiset α
      a✝ : α
      r : ⦃a : α⦄ → δ a → δ a → Prop
      b₁ b₂ : δ a✝
      f₁ f₂ : (a' : α) → Membership.mem m a' → δ a'
      hb : r b₁ b₂
      hf : ∀ (a : α) (ha : Membership.mem m a), r (f₁ a ha) (f₂ a ha)
      a : α
      ha : Membership.mem (Multiset.cons a✝ m) a
      H : Eq a a✝
      ⊢ r (Eq.rec b₁ ⋯) (Eq.rec b₂ ⋯)
    -/
  · cases H
    /-
      case pos.refl
      α : Type u_1
      inst✝ : DecidableEq α
      δ : α → Sort u_2
      m : Multiset α
      a : α
      r : ⦃a : α⦄ → δ a → δ a → Prop
      b₁ b₂ : δ a
      f₁ f₂ : (a' : α) → Membership.mem m a' → δ a'
      hb : r b₁ b₂
      hf : ∀ (a : α) (ha : Membership.mem m a), r (f₁ a ha) (f₂ a ha)
      ha : Membership.mem (Multiset.cons a m) a
      ⊢ r (Eq.rec b₁ ⋯) (Eq.rec b₂ ⋯)
    -/
    exact hb
    /-
      🎉 no goals
    -/
    /-
      case neg
      α : Type u_1
      inst✝ : DecidableEq α
      δ : α → Sort u_2
      m : Multiset α
      a✝ : α
      r : ⦃a : α⦄ → δ a → δ a → Prop
      b₁ b₂ : δ a✝
      f₁ f₂ : (a' : α) → Membership.mem m a' → δ a'
      hb : r b₁ b₂
      hf : ∀ (a : α) (ha : Membership.mem m a), r (f₁ a ha) (f₂ a ha)
      a : α
      ha : Membership.mem (Multiset.cons a✝ m) a
      H : Not (Eq a a✝)
      ⊢ r (f₁ a ⋯) (f₂ a ⋯)
    -/
  · exact hf _ _
    /-
      🎉 no goals
    -/


theorem cons_injective {a : α} {b : δ a} {s : Multiset α} (hs : a ∉ s) :
    Function.Injective (Pi.cons s a b) := fun f₁ f₂ eq =>
  funext fun a' =>
    funext fun h' =>
      have ne : a ≠ a' := fun h => hs <| h.symm ▸ h'
      have : a' ∈ a ::ₘ s := mem_cons_of_mem h'
      calc
                                                  /-
                                                    α : Type u_1
                                                    inst✝ : DecidableEq α
                                                    δ : α → Sort u_2
                                                    a : α
                                                    b : δ a
                                                    s : Multiset α
                                                    hs : Not (Membership.mem s a)
                                                    f₁ f₂ : (a : α) → Membership.mem s a → δ a
                                                    eq : Eq (Multiset.Pi.cons s a b f₁) (Multiset.Pi.cons s a b f₂)
                                                    a' : α
                                                    h' : Membership.mem s a'
                                                    ne : Ne a a'
                                                    this : Membership.mem (Multiset.cons a s) a'
                                                    ⊢ Eq (f₁ a' h') (Multiset.Pi.cons s a b f₁ a' this)
                                                  -/
        f₁ a' h' = Pi.cons s a b f₁ a' this := by rw [Pi.cons_ne this ne.symm]
                                                  /-
                                                    🎉 no goals
                                                  -/
                                                  /-
                                                    α : Type u_1
                                                    inst✝ : DecidableEq α
                                                    δ : α → Sort u_2
                                                    a : α
                                                    b : δ a
                                                    s : Multiset α
                                                    hs : Not (Membership.mem s a)
                                                    f₁ f₂ : (a : α) → Membership.mem s a → δ a
                                                    eq : Eq (Multiset.Pi.cons s a b f₁) (Multiset.Pi.cons s a b f₂)
                                                    a' : α
                                                    h' : Membership.mem s a'
                                                    ne : Ne a a'
                                                    this : Membership.mem (Multiset.cons a s) a'
                                                    ⊢ Eq (Multiset.Pi.cons s a b f₁ a' this) (Multiset.Pi.cons s a b f₂ a' this)
                                                  -/
               _ = Pi.cons s a b f₂ a' this := by rw [eq]
                                                  /-
                                                    🎉 no goals
                                                  -/
                                  /-
                                    α : Type u_1
                                    inst✝ : DecidableEq α
                                    δ : α → Sort u_2
                                    a : α
                                    b : δ a
                                    s : Multiset α
                                    hs : Not (Membership.mem s a)
                                    f₁ f₂ : (a : α) → Membership.mem s a → δ a
                                    eq : Eq (Multiset.Pi.cons s a b f₁) (Multiset.Pi.cons s a b f₂)
                                    a' : α
                                    h' : Membership.mem s a'
                                    ne : Ne a a'
                                    this : Membership.mem (Multiset.cons a s) a'
                                    ⊢ Eq (Multiset.Pi.cons s a b f₂ a' this) (f₂ a' h')
                                  -/
               _ = f₂ a' h' := by rw [Pi.cons_ne this ne.symm]
                                  /-
                                    🎉 no goals
                                  -/


/-- `pi m t` constructs the Cartesian product over `t` indexed by `m`. -/
def pi (m : Multiset α) (t : ∀ a, Multiset (β a)) : Multiset (∀ a ∈ m, β a) :=
  m.recOn {Pi.empty β}
    (fun a m (p : Multiset (∀ a ∈ m, β a)) => (t a).bind fun b => p.map <| Pi.cons m a b)
    (by
      /-
        α : Type u_1
        inst✝ : DecidableEq α
        β : α → Type u_2
        m : Multiset α
        t : (a : α) → Multiset (β a)
        ⊢ ∀ (a a' : α) (m : Multiset α) (b : Multiset ((a : α) → Membership.mem m a →  …
      -/
      intro a a' m n
      /-
        α : Type u_1
        inst✝ : DecidableEq α
        β : α → Type u_2
        m✝ : Multiset α
        t : (a : α) → Multiset (β a)
        a a' : α
        m : Multiset α
        n : Multiset ((a : α) → Membership.mem m a → β a)
        ⊢ HEq ((t a).bind fun b => Multiset.map (Multiset.Pi.cons (Multiset.cons a' m) …
      -/
      by_cases eq : a = a'
        /-
          case pos
          α : Type u_1
          inst✝ : DecidableEq α
          β : α → Type u_2
          m✝ : Multiset α
          t : (a : α) → Multiset (β a)
          a a' : α
          m : Multiset α
          n : Multiset ((a : α) → Membership.mem m a → β a)
          eq : Eq a a'
          ⊢ HEq ((t a).bind fun b => Multiset.map (Multiset.Pi.cons (Multiset.cons a' m) …
        -/
      · subst eq; rfl
                  /-
                    🎉 no goals
                  -/
        /-
          case neg
          α : Type u_1
          inst✝ : DecidableEq α
          β : α → Type u_2
          m✝ : Multiset α
          t : (a : α) → Multiset (β a)
          a a' : α
          m : Multiset α
          n : Multiset ((a : α) → Membership.mem m a → β a)
          eq : Not (Eq a a')
          ⊢ HEq ((t a).bind fun b => Multiset.map (Multiset.Pi.cons (Multiset.cons a' m) …
        -/
      · simp only [map_bind, map_map, comp_apply, bind_bind (t a') (t a)]
        /-
          case neg
          α : Type u_1
          inst✝ : DecidableEq α
          β : α → Type u_2
          m✝ : Multiset α
          t : (a : α) → Multiset (β a)
          a a' : α
          m : Multiset α
          n : Multiset ((a : α) → Membership.mem m a → β a)
          eq : Not (Eq a a')
          ⊢ HEq ((t a).bind fun b => (t a').bind fun a_1 => Multiset.map (fun x => Multi …
        -/
        apply bind_hcongr
          /-
            case neg.h
            α : Type u_1
            inst✝ : DecidableEq α
            β : α → Type u_2
            m✝ : Multiset α
            t : (a : α) → Multiset (β a)
            a a' : α
            m : Multiset α
            n : Multiset ((a : α) → Membership.mem m a → β a)
            eq : Not (Eq a a')
            ⊢ Eq ((a_1 : α) → Membership.mem (Multiset.cons a (Multiset.cons a' m)) a_1 →  …
          -/
        · rw [cons_swap a a']
          /-
            🎉 no goals
          -/
        /-
          case neg.hf
          α : Type u_1
          inst✝ : DecidableEq α
          β : α → Type u_2
          m✝ : Multiset α
          t : (a : α) → Multiset (β a)
          a a' : α
          m : Multiset α
          n : Multiset ((a : α) → Membership.mem m a → β a)
          eq : Not (Eq a a')
          ⊢ ∀ (a_1 : β a), Membership.mem (t a) a_1 → HEq ((t a').bind fun a_3 => Multis …
        -/
        intro b _
        /-
          case neg.hf
          α : Type u_1
          inst✝ : DecidableEq α
          β : α → Type u_2
          m✝ : Multiset α
          t : (a : α) → Multiset (β a)
          a a' : α
          m : Multiset α
          n : Multiset ((a : α) → Membership.mem m a → β a)
          eq : Not (Eq a a')
          b : β a
          a✝ : Membership.mem (t a) b
          ⊢ HEq ((t a').bind fun a_1 => Multiset.map (fun x => Multiset.Pi.cons (Multise …
        -/
        apply bind_hcongr
          /-
            case neg.hf.h
            α : Type u_1
            inst✝ : DecidableEq α
            β : α → Type u_2
            m✝ : Multiset α
            t : (a : α) → Multiset (β a)
            a a' : α
            m : Multiset α
            n : Multiset ((a : α) → Membership.mem m a → β a)
            eq : Not (Eq a a')
            b : β a
            a✝ : Membership.mem (t a) b
            ⊢ Eq ((a_1 : α) → Membership.mem (Multiset.cons a (Multiset.cons a' m)) a_1 →  …
          -/
        · rw [cons_swap a a']
          /-
            🎉 no goals
          -/
        /-
          case neg.hf.hf
          α : Type u_1
          inst✝ : DecidableEq α
          β : α → Type u_2
          m✝ : Multiset α
          t : (a : α) → Multiset (β a)
          a a' : α
          m : Multiset α
          n : Multiset ((a : α) → Membership.mem m a → β a)
          eq : Not (Eq a a')
          b : β a
          a✝ : Membership.mem (t a) b
          ⊢ ∀ (a_1 : β a'), Membership.mem (t a') a_1 → HEq (Multiset.map (fun x => Mult …
        -/
        intro b' _
        /-
          case neg.hf.hf
          α : Type u_1
          inst✝ : DecidableEq α
          β : α → Type u_2
          m✝ : Multiset α
          t : (a : α) → Multiset (β a)
          a a' : α
          m : Multiset α
          n : Multiset ((a : α) → Membership.mem m a → β a)
          eq : Not (Eq a a')
          b : β a
          a✝¹ : Membership.mem (t a) b
          b' : β a'
          a✝ : Membership.mem (t a') b'
          ⊢ HEq (Multiset.map (fun x => Multiset.Pi.cons (Multiset.cons a' m) a b (Multi …
        -/
        apply map_hcongr
          /-
            case neg.hf.hf.h
            α : Type u_1
            inst✝ : DecidableEq α
            β : α → Type u_2
            m✝ : Multiset α
            t : (a : α) → Multiset (β a)
            a a' : α
            m : Multiset α
            n : Multiset ((a : α) → Membership.mem m a → β a)
            eq : Not (Eq a a')
            b : β a
            a✝¹ : Membership.mem (t a) b
            b' : β a'
            a✝ : Membership.mem (t a') b'
            ⊢ Eq ((a_1 : α) → Membership.mem (Multiset.cons a (Multiset.cons a' m)) a_1 →  …
          -/
        · rw [cons_swap a a']
          /-
            🎉 no goals
          -/
        /-
          case neg.hf.hf.hf
          α : Type u_1
          inst✝ : DecidableEq α
          β : α → Type u_2
          m✝ : Multiset α
          t : (a : α) → Multiset (β a)
          a a' : α
          m : Multiset α
          n : Multiset ((a : α) → Membership.mem m a → β a)
          eq : Not (Eq a a')
          b : β a
          a✝¹ : Membership.mem (t a) b
          b' : β a'
          a✝ : Membership.mem (t a') b'
          ⊢ ∀ (a_1 : (a : α) → Membership.mem m a → β a), Membership.mem n a_1 → HEq (Mu …
        -/
        intro f _
        /-
          case neg.hf.hf.hf
          α : Type u_1
          inst✝ : DecidableEq α
          β : α → Type u_2
          m✝ : Multiset α
          t : (a : α) → Multiset (β a)
          a a' : α
          m : Multiset α
          n : Multiset ((a : α) → Membership.mem m a → β a)
          eq : Not (Eq a a')
          b : β a
          a✝² : Membership.mem (t a) b
          b' : β a'
          a✝¹ : Membership.mem (t a') b'
          f : (a : α) → Membership.mem m a → β a
          a✝ : Membership.mem n f
          ⊢ HEq (Multiset.Pi.cons (Multiset.cons a' m) a b (Multiset.Pi.cons m a' b' f)) …
        -/
        exact Pi.cons_swap eq)
        /-
          🎉 no goals
        -/


@[simp]
theorem pi_zero (t : ∀ a, Multiset (β a)) : pi 0 t = {Pi.empty β} :=
  rfl


@[simp]
theorem pi_cons (m : Multiset α) (t : ∀ a, Multiset (β a)) (a : α) :
    pi (a ::ₘ m) t = (t a).bind fun b => (pi m t).map <| Pi.cons m a b :=
  recOn_cons a m


theorem card_pi (m : Multiset α) (t : ∀ a, Multiset (β a)) :
    card (pi m t) = prod (m.map fun a => card (t a)) :=
                              /-
                                α : Type u_1
                                inst✝ : DecidableEq α
                                β : α → Type u_2
                                m : Multiset α
                                t : (a : α) → Multiset (β a)
                                ⊢ Eq (Multiset.pi 0 t).card (Multiset.map (fun a => (t a).card) 0).prod
                              -/
                              /-
                                🎉 no goals
                              -/
  Multiset.induction_on m (by simp) (by simp +contextual [mul_comm])
                                        /-
                                          🎉 no goals
                                        -/


protected theorem Nodup.pi {s : Multiset α} {t : ∀ a, Multiset (β a)} :
    Nodup s → (∀ a ∈ s, Nodup (t a)) → Nodup (pi s t) :=
  Multiset.induction_on s (fun _ _ => nodup_singleton _)
    (by
      /-
        α : Type u_1
        inst✝ : DecidableEq α
        β : α → Type u_2
        s : Multiset α
        t : (a : α) → Multiset (β a)
        ⊢ ∀ (a : α) (s : Multiset α), (s.Nodup → (∀ (a : α), Membership.mem s a → (t a …
      -/
      intro a s ih hs ht
      /-
        α : Type u_1
        inst✝ : DecidableEq α
        β : α → Type u_2
        s✝ : Multiset α
        t : (a : α) → Multiset (β a)
        a : α
        s : Multiset α
        ih : s.Nodup → (∀ (a : α), Membership.mem s a → (t a).Nodup) → (s.pi t).Nodup
        hs : (Multiset.cons a s).Nodup
        ht : ∀ (a_1 : α), Membership.mem (Multiset.cons a s) a_1 → (t a_1).Nodup
        ⊢ ((Multiset.cons a s).pi t).Nodup
      -/
      have has : a ∉ s := by simp only [nodup_cons] at hs; exact hs.1
      /-
        α : Type u_1
        inst✝ : DecidableEq α
        β : α → Type u_2
        s✝ : Multiset α
        t : (a : α) → Multiset (β a)
        a : α
        s : Multiset α
        ih : s.Nodup → (∀ (a : α), Membership.mem s a → (t a).Nodup) → (s.pi t).Nodup
        hs : (Multiset.cons a s).Nodup
        ht : ∀ (a_1 : α), Membership.mem (Multiset.cons a s) a_1 → (t a_1).Nodup
        has : Not (Membership.mem s a)
        ⊢ ((Multiset.cons a s).pi t).Nodup
      -/
      have hs : Nodup s := by simp only [nodup_cons] at hs; exact hs.2
      /-
        α : Type u_1
        inst✝ : DecidableEq α
        β : α → Type u_2
        s✝ : Multiset α
        t : (a : α) → Multiset (β a)
        a : α
        s : Multiset α
        ih : s.Nodup → (∀ (a : α), Membership.mem s a → (t a).Nodup) → (s.pi t).Nodup
        hs✝ : (Multiset.cons a s).Nodup
        ht : ∀ (a_1 : α), Membership.mem (Multiset.cons a s) a_1 → (t a_1).Nodup
        has : Not (Membership.mem s a)
        hs : s.Nodup
        ⊢ ((Multiset.cons a s).pi t).Nodup
      -/
      simp only [pi_cons, nodup_bind]
      refine
        ⟨fun b _ => ((ih hs) fun a' h' => ht a' <| mem_cons_of_mem h').map (Pi.cons_injective has),
          ?_⟩
      /-
        α : Type u_1
        inst✝ : DecidableEq α
        β : α → Type u_2
        s✝ : Multiset α
        t : (a : α) → Multiset (β a)
        a : α
        s : Multiset α
        ih : s.Nodup → (∀ (a : α), Membership.mem s a → (t a).Nodup) → (s.pi t).Nodup
        hs✝ : (Multiset.cons a s).Nodup
        ht : ∀ (a_1 : α), Membership.mem (Multiset.cons a s) a_1 → (t a_1).Nodup
        has : Not (Membership.mem s a)
        hs : s.Nodup
        ⊢ Multiset.Pairwise (Function.onFun Disjoint fun b => Multiset.map (Multiset.P …
      -/
      refine (ht a <| mem_cons_self _ _).pairwise ?_
      exact fun b₁ _ b₂ _ neb =>
        disjoint_map_map.2 fun f _ g _ eq =>
          have : Pi.cons s a b₁ f a (mem_cons_self _ _) = Pi.cons s a b₂ g a (mem_cons_self _ _) :=
            by rw [eq]
          neb <| show b₁ = b₂ by rwa [Pi.cons_same, Pi.cons_same] at this)


theorem mem_pi (m : Multiset α) (t : ∀ a, Multiset (β a)) :
    ∀ f : ∀ a ∈ m, β a, f ∈ pi m t ↔ ∀ (a) (h : a ∈ m), f a h ∈ t a := by
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    β : α → Type u_2
    m : Multiset α
    t : (a : α) → Multiset (β a)
    ⊢ ∀ (f : (a : α) → Membership.mem m a → β a), Iff (Membership.mem (m.pi t) f)  …
  -/
  intro f
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    β : α → Type u_2
    m : Multiset α
    t : (a : α) → Multiset (β a)
    f : (a : α) → Membership.mem m a → β a
    ⊢ Iff (Membership.mem (m.pi t) f) (∀ (a : α) (h : Membership.mem m a), Members …
  -/
  induction' m using Multiset.induction_on with a m ih
    /-
      case empty
      α : Type u_1
      inst✝ : DecidableEq α
      β : α → Type u_2
      t : (a : α) → Multiset (β a)
      f : (a : α) → Membership.mem 0 a → β a
      ⊢ Iff (Membership.mem (Multiset.pi 0 t) f) (∀ (a : α) (h : Membership.mem 0 a) …
    -/
  · have : f = Pi.empty β := funext (fun _ => funext fun h => (not_mem_zero _ h).elim)
    /-
      case empty
      α : Type u_1
      inst✝ : DecidableEq α
      β : α → Type u_2
      t : (a : α) → Multiset (β a)
      f : (a : α) → Membership.mem 0 a → β a
      this : Eq f (Multiset.Pi.empty β)
      ⊢ Iff (Membership.mem (Multiset.pi 0 t) f) (∀ (a : α) (h : Membership.mem 0 a) …
    -/
    simp only [this, pi_zero, mem_singleton, true_iff]
    /-
      case empty
      α : Type u_1
      inst✝ : DecidableEq α
      β : α → Type u_2
      t : (a : α) → Multiset (β a)
      f : (a : α) → Membership.mem 0 a → β a
      this : Eq f (Multiset.Pi.empty β)
      ⊢ ∀ (a : α) (h : Membership.mem 0 a), Membership.mem (t a) (Multiset.Pi.empty  …
    -/
    intro _ h; exact (not_mem_zero _ h).elim
               /-
                 🎉 no goals
               -/
  /-
    case cons
    α : Type u_1
    inst✝ : DecidableEq α
    β : α → Type u_2
    t : (a : α) → Multiset (β a)
    a : α
    m : Multiset α
    ih : ∀ (f : (a : α) → Membership.mem m a → β a), Iff (Membership.mem (m.pi t)  …
    f : (a_1 : α) → Membership.mem (Multiset.cons a m) a_1 → β a_1
    ⊢ Iff (Membership.mem ((Multiset.cons a m).pi t) f) (∀ (a_1 : α) (h : Membersh …
  -/
  simp_rw [pi_cons, mem_bind, mem_map, ih]
  /-
    case cons
    α : Type u_1
    inst✝ : DecidableEq α
    β : α → Type u_2
    t : (a : α) → Multiset (β a)
    a : α
    m : Multiset α
    ih : ∀ (f : (a : α) → Membership.mem m a → β a), Iff (Membership.mem (m.pi t)  …
    f : (a_1 : α) → Membership.mem (Multiset.cons a m) a_1 → β a_1
    ⊢ Iff (Exists fun a_1 => And (Membership.mem (t a) a_1) (Exists fun a_2 => And …
  -/
  constructor
    /-
      case cons.mp
      α : Type u_1
      inst✝ : DecidableEq α
      β : α → Type u_2
      t : (a : α) → Multiset (β a)
      a : α
      m : Multiset α
      ih : ∀ (f : (a : α) → Membership.mem m a → β a), Iff (Membership.mem (m.pi t)  …
      f : (a_1 : α) → Membership.mem (Multiset.cons a m) a_1 → β a_1
      ⊢ (Exists fun a_1 => And (Membership.mem (t a) a_1) (Exists fun a_2 => And (∀  …
    -/
  · rintro ⟨b, hb, f', hf', rfl⟩ a' ha'
    /-
      case cons.mp.intro.intro.intro.intro
      α : Type u_1
      inst✝ : DecidableEq α
      β : α → Type u_2
      t : (a : α) → Multiset (β a)
      a : α
      m : Multiset α
      ih : ∀ (f : (a : α) → Membership.mem m a → β a), Iff (Membership.mem (m.pi t)  …
      b : β a
      hb : Membership.mem (t a) b
      f' : (a : α) → Membership.mem m a → β a
      hf' : ∀ (a : α) (h : Membership.mem m a), Membership.mem (t a) (f' a h)
      a' : α
      ha' : Membership.mem (Multiset.cons a m) a'
      ⊢ Membership.mem (t a') (Multiset.Pi.cons m a b f' a' ha')
    -/
    by_cases h : a' = a
      /-
        case pos
        α : Type u_1
        inst✝ : DecidableEq α
        β : α → Type u_2
        t : (a : α) → Multiset (β a)
        a : α
        m : Multiset α
        ih : ∀ (f : (a : α) → Membership.mem m a → β a), Iff (Membership.mem (m.pi t)  …
        b : β a
        hb : Membership.mem (t a) b
        f' : (a : α) → Membership.mem m a → β a
        hf' : ∀ (a : α) (h : Membership.mem m a), Membership.mem (t a) (f' a h)
        a' : α
        ha' : Membership.mem (Multiset.cons a m) a'
        h : Eq a' a
        ⊢ Membership.mem (t a') (Multiset.Pi.cons m a b f' a' ha')
      -/
    · subst h
      /-
        case pos
        α : Type u_1
        inst✝ : DecidableEq α
        β : α → Type u_2
        t : (a : α) → Multiset (β a)
        m : Multiset α
        ih : ∀ (f : (a : α) → Membership.mem m a → β a), Iff (Membership.mem (m.pi t)  …
        f' : (a : α) → Membership.mem m a → β a
        hf' : ∀ (a : α) (h : Membership.mem m a), Membership.mem (t a) (f' a h)
        a' : α
        b : β a'
        hb : Membership.mem (t a') b
        ha' : Membership.mem (Multiset.cons a' m) a'
        ⊢ Membership.mem (t a') (Multiset.Pi.cons m a' b f' a' ha')
      -/
      rwa [Pi.cons_same]
      /-
        🎉 no goals
      -/
      /-
        case neg
        α : Type u_1
        inst✝ : DecidableEq α
        β : α → Type u_2
        t : (a : α) → Multiset (β a)
        a : α
        m : Multiset α
        ih : ∀ (f : (a : α) → Membership.mem m a → β a), Iff (Membership.mem (m.pi t)  …
        b : β a
        hb : Membership.mem (t a) b
        f' : (a : α) → Membership.mem m a → β a
        hf' : ∀ (a : α) (h : Membership.mem m a), Membership.mem (t a) (f' a h)
        a' : α
        ha' : Membership.mem (Multiset.cons a m) a'
        h : Not (Eq a' a)
        ⊢ Membership.mem (t a') (Multiset.Pi.cons m a b f' a' ha')
      -/
    · rw [Pi.cons_ne _ h]
      /-
        case neg
        α : Type u_1
        inst✝ : DecidableEq α
        β : α → Type u_2
        t : (a : α) → Multiset (β a)
        a : α
        m : Multiset α
        ih : ∀ (f : (a : α) → Membership.mem m a → β a), Iff (Membership.mem (m.pi t)  …
        b : β a
        hb : Membership.mem (t a) b
        f' : (a : α) → Membership.mem m a → β a
        hf' : ∀ (a : α) (h : Membership.mem m a), Membership.mem (t a) (f' a h)
        a' : α
        ha' : Membership.mem (Multiset.cons a m) a'
        h : Not (Eq a' a)
        ⊢ Membership.mem (t a') (f' a' ⋯)
      -/
      apply hf'
      /-
        🎉 no goals
      -/
    /-
      case cons.mpr
      α : Type u_1
      inst✝ : DecidableEq α
      β : α → Type u_2
      t : (a : α) → Multiset (β a)
      a : α
      m : Multiset α
      ih : ∀ (f : (a : α) → Membership.mem m a → β a), Iff (Membership.mem (m.pi t)  …
      f : (a_1 : α) → Membership.mem (Multiset.cons a m) a_1 → β a_1
      ⊢ (∀ (a_1 : α) (h : Membership.mem (Multiset.cons a m) a_1), Membership.mem (t …
    -/
  · intro hf
    /-
      case cons.mpr
      α : Type u_1
      inst✝ : DecidableEq α
      β : α → Type u_2
      t : (a : α) → Multiset (β a)
      a : α
      m : Multiset α
      ih : ∀ (f : (a : α) → Membership.mem m a → β a), Iff (Membership.mem (m.pi t)  …
      f : (a_1 : α) → Membership.mem (Multiset.cons a m) a_1 → β a_1
      hf : ∀ (a_1 : α) (h : Membership.mem (Multiset.cons a m) a_1), Membership.mem  …
      ⊢ Exists fun a_1 => And (Membership.mem (t a) a_1) (Exists fun a_2 => And (∀ ( …
    -/
    refine ⟨_, hf a (mem_cons_self _ _), _, fun a ha => hf a (mem_cons_of_mem ha), ?_⟩
    /-
      case cons.mpr
      α : Type u_1
      inst✝ : DecidableEq α
      β : α → Type u_2
      t : (a : α) → Multiset (β a)
      a : α
      m : Multiset α
      ih : ∀ (f : (a : α) → Membership.mem m a → β a), Iff (Membership.mem (m.pi t)  …
      f : (a_1 : α) → Membership.mem (Multiset.cons a m) a_1 → β a_1
      hf : ∀ (a_1 : α) (h : Membership.mem (Multiset.cons a m) a_1), Membership.mem  …
      ⊢ Eq (Multiset.Pi.cons m a (f a ⋯) fun a_1 ha => f a_1 ⋯) f
    -/
    rw [Pi.cons_eta]
    /-
      🎉 no goals
    -/


