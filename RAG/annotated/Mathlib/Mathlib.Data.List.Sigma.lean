/-- List of keys from a list of key-value pairs -/
def keys : List (Sigma β) → List α :=
  map Sigma.fst


@[simp]
theorem keys_nil : @keys α β [] = [] :=
  rfl


@[simp]
theorem keys_cons {s} {l : List (Sigma β)} : (s :: l).keys = s.1 :: l.keys :=
  rfl


theorem mem_keys_of_mem {s : Sigma β} {l : List (Sigma β)} : s ∈ l → s.1 ∈ l.keys :=
  mem_map_of_mem Sigma.fst


theorem exists_of_mem_keys {a} {l : List (Sigma β)} (h : a ∈ l.keys) :
    ∃ b : β a, Sigma.mk a b ∈ l :=
  let ⟨⟨_, b'⟩, m, e⟩ := exists_of_mem_map h
  Eq.recOn e (Exists.intro b' m)


theorem mem_keys {a} {l : List (Sigma β)} : a ∈ l.keys ↔ ∃ b : β a, Sigma.mk a b ∈ l :=
  ⟨exists_of_mem_keys, fun ⟨_, h⟩ => mem_keys_of_mem h⟩


theorem not_mem_keys {a} {l : List (Sigma β)} : a ∉ l.keys ↔ ∀ b : β a, Sigma.mk a b ∉ l :=
  (not_congr mem_keys).trans not_exists


theorem not_eq_key {a} {l : List (Sigma β)} : a ∉ l.keys ↔ ∀ s : Sigma β, s ∈ l → a ≠ s.1 :=
                                                              /-
                                                                α : Type u
                                                                β : α → Type v
                                                                a : α
                                                                l : List (Sigma β)
                                                                h₁ : Not (Membership.mem l.keys a)
                                                                s : Sigma β
                                                                h₂ : Membership.mem l s
                                                                e : Eq a s.fst
                                                                ⊢ Not (Membership.mem l.keys s.fst)
                                                              -/
  Iff.intro (fun h₁ s h₂ e => absurd (mem_keys_of_mem h₂) (by rwa [e] at h₁)) fun f h₁ =>
                                                              /-
                                                                🎉 no goals
                                                              -/
    let ⟨_, h₂⟩ := exists_of_mem_keys h₁
    f _ h₂ rfl


/-- Determines whether the store uses a key several times. -/
def NodupKeys (l : List (Sigma β)) : Prop :=
  l.keys.Nodup


theorem nodupKeys_iff_pairwise {l} : NodupKeys l ↔ Pairwise (fun s s' : Sigma β => s.1 ≠ s'.1) l :=
  pairwise_map


theorem NodupKeys.pairwise_ne {l} (h : NodupKeys l) :
    Pairwise (fun s s' : Sigma β => s.1 ≠ s'.1) l :=
  nodupKeys_iff_pairwise.1 h


@[simp]
theorem nodupKeys_nil : @NodupKeys α β [] :=
  Pairwise.nil


@[simp]
theorem nodupKeys_cons {s : Sigma β} {l : List (Sigma β)} :
                                                          /-
                                                            α : Type u
                                                            β : α → Type v
                                                            s : Sigma β
                                                            l : List (Sigma β)
                                                            ⊢ Iff (List.cons s l).NodupKeys (And (Not (Membership.mem l.keys s.fst)) l.Nod …
                                                          -/
    NodupKeys (s :: l) ↔ s.1 ∉ l.keys ∧ NodupKeys l := by simp [keys, NodupKeys]
                                                          /-
                                                            🎉 no goals
                                                          -/


theorem not_mem_keys_of_nodupKeys_cons {s : Sigma β} {l : List (Sigma β)} (h : NodupKeys (s :: l)) :
    s.1 ∉ l.keys :=
  (nodupKeys_cons.1 h).1


theorem nodupKeys_of_nodupKeys_cons {s : Sigma β} {l : List (Sigma β)} (h : NodupKeys (s :: l)) :
    NodupKeys l :=
  (nodupKeys_cons.1 h).2


theorem NodupKeys.eq_of_fst_eq {l : List (Sigma β)} (nd : NodupKeys l) {s s' : Sigma β} (h : s ∈ l)
    (h' : s' ∈ l) : s.1 = s'.1 → s = s' :=
  @Pairwise.forall_of_forall _ (fun s s' : Sigma β => s.1 = s'.1 → s = s') _
    (fun _ _ H h => (H h.symm).symm) (fun _ _ _ => rfl)
    ((nodupKeys_iff_pairwise.1 nd).imp fun h h' => (h h').elim) _ h _ h'


theorem NodupKeys.eq_of_mk_mem {a : α} {b b' : β a} {l : List (Sigma β)} (nd : NodupKeys l)
    (h : Sigma.mk a b ∈ l) (h' : Sigma.mk a b' ∈ l) : b = b' := by
  /-
    α : Type u
    β : α → Type v
    a : α
    b b' : β a
    l : List (Sigma β)
    nd : l.NodupKeys
    h : Membership.mem l ⟨a, b⟩
    h' : Membership.mem l ⟨a, b'⟩
    ⊢ Eq b b'
  -/
  cases nd.eq_of_fst_eq h h' rfl; rfl
                                  /-
                                    🎉 no goals
                                  -/


theorem nodupKeys_singleton (s : Sigma β) : NodupKeys [s] :=
  nodup_singleton _


theorem NodupKeys.sublist {l₁ l₂ : List (Sigma β)} (h : l₁ <+ l₂) : NodupKeys l₂ → NodupKeys l₁ :=
  Nodup.sublist <| h.map _


protected theorem NodupKeys.nodup {l : List (Sigma β)} : NodupKeys l → Nodup l :=
  Nodup.of_map _


theorem perm_nodupKeys {l₁ l₂ : List (Sigma β)} (h : l₁ ~ l₂) : NodupKeys l₁ ↔ NodupKeys l₂ :=
  (h.map _).nodup_iff


theorem nodupKeys_flatten {L : List (List (Sigma β))} :
    NodupKeys (flatten L) ↔ (∀ l ∈ L, NodupKeys l) ∧ Pairwise Disjoint (L.map keys) := by
  /-
    α : Type u
    β : α → Type v
    L : List (List (Sigma β))
    ⊢ Iff L.flatten.NodupKeys (And (∀ (l : List (Sigma β)), Membership.mem L l → l …
  -/
  rw [nodupKeys_iff_pairwise, pairwise_flatten, pairwise_map]
  /-
    α : Type u
    β : α → Type v
    L : List (List (Sigma β))
    ⊢ Iff (And (∀ (l : List (Sigma β)), Membership.mem L l → List.Pairwise (fun s  …
  -/
  refine and_congr (forall₂_congr fun l _ => by simp [nodupKeys_iff_pairwise]) ?_
  /-
    α : Type u
    β : α → Type v
    L : List (List (Sigma β))
    ⊢ Iff (List.Pairwise (fun l₁ l₂ => ∀ (x : Sigma β), Membership.mem l₁ x → ∀ (y …
  -/
  apply iff_of_eq; congr with (l₁ l₂)
  /-
    case a.e_R.h.h.a
    α : Type u
    β : α → Type v
    L : List (List (Sigma β))
    l₁ l₂ : List (Sigma β)
    ⊢ Iff (∀ (x : Sigma β), Membership.mem l₁ x → ∀ (y : Sigma β), Membership.mem  …
  -/
  simp [keys, disjoint_iff_ne, Sigma.forall]
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-10-15")] alias nodupKeys_join := nodupKeys_flatten


                                                                            /-
                                                                              α : Type u
                                                                              l : List α
                                                                              ⊢ (List.map Prod.fst l.enum).Nodup
                                                                            -/
theorem nodup_enum_map_fst (l : List α) : (l.enum.map Prod.fst).Nodup := by simp [List.nodup_range]
                                                                            /-
                                                                              🎉 no goals
                                                                            -/


theorem mem_ext {l₀ l₁ : List (Sigma β)} (nd₀ : l₀.Nodup) (nd₁ : l₁.Nodup)
    (h : ∀ x, x ∈ l₀ ↔ x ∈ l₁) : l₀ ~ l₁ :=
  (perm_ext_iff_of_nodup nd₀ nd₁).2 h


/-- `dlookup a l` is the first value in `l` corresponding to the key `a`,
  or `none` if no such element exists. -/
def dlookup (a : α) : List (Sigma β) → Option (β a)
  | [] => none
  | ⟨a', b⟩ :: l => if h : a' = a then some (Eq.recOn h b) else dlookup a l


@[simp]
theorem dlookup_nil (a : α) : dlookup a [] = @none (β a) :=
  rfl


@[simp]
theorem dlookup_cons_eq (l) (a : α) (b : β a) : dlookup a (⟨a, b⟩ :: l) = some b :=
  dif_pos rfl


@[simp]
theorem dlookup_cons_ne (l) {a} : ∀ s : Sigma β, a ≠ s.1 → dlookup a (s :: l) = dlookup a l
  | ⟨_, _⟩, h => dif_neg h.symm


theorem dlookup_isSome {a : α} : ∀ {l : List (Sigma β)}, (dlookup a l).isSome ↔ a ∈ l.keys
             /-
               α : Type u
               β : α → Type v
               inst✝ : DecidableEq α
               a : α
               ⊢ Iff (Eq (List.dlookup a List.nil).isSome Bool.true) (Membership.mem List.nil …
             -/
  | [] => by simp
             /-
               🎉 no goals
             -/
  | ⟨a', b⟩ :: l => by
    /-
      α : Type u
      β : α → Type v
      inst✝ : DecidableEq α
      a a' : α
      b : β a'
      l : List (Sigma β)
      ⊢ Iff (Eq (List.dlookup a (List.cons ⟨a', b⟩ l)).isSome Bool.true) (Membership …
    -/
    by_cases h : a = a'
      /-
        case pos
        α : Type u
        β : α → Type v
        inst✝ : DecidableEq α
        a a' : α
        b : β a'
        l : List (Sigma β)
        h : Eq a a'
        ⊢ Iff (Eq (List.dlookup a (List.cons ⟨a', b⟩ l)).isSome Bool.true) (Membership …
      -/
    · subst a'
      /-
        case pos
        α : Type u
        β : α → Type v
        inst✝ : DecidableEq α
        a : α
        l : List (Sigma β)
        b : β a
        ⊢ Iff (Eq (List.dlookup a (List.cons ⟨a, b⟩ l)).isSome Bool.true) (Membership. …
      -/
      simp
      /-
        🎉 no goals
      -/
      /-
        case neg
        α : Type u
        β : α → Type v
        inst✝ : DecidableEq α
        a a' : α
        b : β a'
        l : List (Sigma β)
        h : Not (Eq a a')
        ⊢ Iff (Eq (List.dlookup a (List.cons ⟨a', b⟩ l)).isSome Bool.true) (Membership …
      -/
    · simp [h, dlookup_isSome]
      /-
        🎉 no goals
      -/


theorem dlookup_eq_none {a : α} {l : List (Sigma β)} : dlookup a l = none ↔ a ∉ l.keys := by
  /-
    α : Type u
    β : α → Type v
    inst✝ : DecidableEq α
    a : α
    l : List (Sigma β)
    ⊢ Iff (Eq (List.dlookup a l) Option.none) (Not (Membership.mem l.keys a))
  -/
  simp [← dlookup_isSome, Option.isNone_iff_eq_none]
  /-
    🎉 no goals
  -/


theorem of_mem_dlookup {a : α} {b : β a} :
    ∀ {l : List (Sigma β)}, b ∈ dlookup a l → Sigma.mk a b ∈ l
  | ⟨a', b'⟩ :: l, H => by
    /-
      α : Type u
      β : α → Type v
      inst✝ : DecidableEq α
      a : α
      b : β a
      a' : α
      b' : β a'
      l : List (Sigma β)
      H : Membership.mem (List.dlookup a (List.cons ⟨a', b'⟩ l)) b
      ⊢ Membership.mem (List.cons ⟨a', b'⟩ l) ⟨a, b⟩
    -/
    by_cases h : a = a'
      /-
        case pos
        α : Type u
        β : α → Type v
        inst✝ : DecidableEq α
        a : α
        b : β a
        a' : α
        b' : β a'
        l : List (Sigma β)
        H : Membership.mem (List.dlookup a (List.cons ⟨a', b'⟩ l)) b
        h : Eq a a'
        ⊢ Membership.mem (List.cons ⟨a', b'⟩ l) ⟨a, b⟩
      -/
    · subst a'
      /-
        case pos
        α : Type u
        β : α → Type v
        inst✝ : DecidableEq α
        a : α
        b : β a
        l : List (Sigma β)
        b' : β a
        H : Membership.mem (List.dlookup a (List.cons ⟨a, b'⟩ l)) b
        ⊢ Membership.mem (List.cons ⟨a, b'⟩ l) ⟨a, b⟩
      -/
      simp? at H says simp only [dlookup_cons_eq, Option.mem_def, Option.some.injEq] at H
      /-
        case pos
        α : Type u
        β : α → Type v
        inst✝ : DecidableEq α
        a : α
        b : β a
        l : List (Sigma β)
        b' : β a
        H : Eq b' b
        ⊢ Membership.mem (List.cons ⟨a, b'⟩ l) ⟨a, b⟩
      -/
      simp [H]
      /-
        🎉 no goals
      -/
      /-
        case neg
        α : Type u
        β : α → Type v
        inst✝ : DecidableEq α
        a : α
        b : β a
        a' : α
        b' : β a'
        l : List (Sigma β)
        H : Membership.mem (List.dlookup a (List.cons ⟨a', b'⟩ l)) b
        h : Not (Eq a a')
        ⊢ Membership.mem (List.cons ⟨a', b'⟩ l) ⟨a, b⟩
      -/
    · simp only [ne_eq, h, not_false_iff, dlookup_cons_ne] at H
      /-
        case neg
        α : Type u
        β : α → Type v
        inst✝ : DecidableEq α
        a : α
        b : β a
        a' : α
        b' : β a'
        l : List (Sigma β)
        h : Not (Eq a a')
        H : Membership.mem (List.dlookup a l) b
        ⊢ Membership.mem (List.cons ⟨a', b'⟩ l) ⟨a, b⟩
      -/
      simp [of_mem_dlookup H]
      /-
        🎉 no goals
      -/


theorem mem_dlookup {a} {b : β a} {l : List (Sigma β)} (nd : l.NodupKeys) (h : Sigma.mk a b ∈ l) :
    b ∈ dlookup a l := by
  /-
    α : Type u
    β : α → Type v
    inst✝ : DecidableEq α
    a : α
    b : β a
    l : List (Sigma β)
    nd : l.NodupKeys
    h : Membership.mem l ⟨a, b⟩
    ⊢ Membership.mem (List.dlookup a l) b
  -/
  cases' Option.isSome_iff_exists.mp (dlookup_isSome.mpr (mem_keys_of_mem h)) with b' h'
  /-
    case intro
    α : Type u
    β : α → Type v
    inst✝ : DecidableEq α
    a : α
    b : β a
    l : List (Sigma β)
    nd : l.NodupKeys
    h : Membership.mem l ⟨a, b⟩
    b' : β ⟨a, b⟩.fst
    h' : Eq (List.dlookup ⟨a, b⟩.fst l) (Option.some b')
    ⊢ Membership.mem (List.dlookup a l) b
  -/
  cases nd.eq_of_mk_mem h (of_mem_dlookup h')
  /-
    case intro.refl
    α : Type u
    β : α → Type v
    inst✝ : DecidableEq α
    a : α
    b : β a
    l : List (Sigma β)
    nd : l.NodupKeys
    h : Membership.mem l ⟨a, b⟩
    h' : Eq (List.dlookup ⟨a, b⟩.fst l) (Option.some b)
    ⊢ Membership.mem (List.dlookup a l) b
  -/
  exact h'
  /-
    🎉 no goals
  -/


theorem map_dlookup_eq_find (a : α) :
    ∀ l : List (Sigma β), (dlookup a l).map (Sigma.mk a) = find? (fun s => a = s.1) l
  | [] => rfl
  | ⟨a', b'⟩ :: l => by
    /-
      α : Type u
      β : α → Type v
      inst✝ : DecidableEq α
      a a' : α
      b' : β a'
      l : List (Sigma β)
      ⊢ Eq (Option.map (Sigma.mk a) (List.dlookup a (List.cons ⟨a', b'⟩ l))) (List.f …
    -/
    by_cases h : a = a'
      /-
        case pos
        α : Type u
        β : α → Type v
        inst✝ : DecidableEq α
        a a' : α
        b' : β a'
        l : List (Sigma β)
        h : Eq a a'
        ⊢ Eq (Option.map (Sigma.mk a) (List.dlookup a (List.cons ⟨a', b'⟩ l))) (List.f …
      -/
    · subst a'
      /-
        case pos
        α : Type u
        β : α → Type v
        inst✝ : DecidableEq α
        a : α
        l : List (Sigma β)
        b' : β a
        ⊢ Eq (Option.map (Sigma.mk a) (List.dlookup a (List.cons ⟨a, b'⟩ l))) (List.fi …
      -/
      simp
      /-
        🎉 no goals
      -/
      /-
        case neg
        α : Type u
        β : α → Type v
        inst✝ : DecidableEq α
        a a' : α
        b' : β a'
        l : List (Sigma β)
        h : Not (Eq a a')
        ⊢ Eq (Option.map (Sigma.mk a) (List.dlookup a (List.cons ⟨a', b'⟩ l))) (List.f …
      -/
    · simpa [h] using map_dlookup_eq_find a l
      /-
        🎉 no goals
      -/


theorem mem_dlookup_iff {a : α} {b : β a} {l : List (Sigma β)} (nd : l.NodupKeys) :
    b ∈ dlookup a l ↔ Sigma.mk a b ∈ l :=
  ⟨of_mem_dlookup, mem_dlookup nd⟩


theorem perm_dlookup (a : α) {l₁ l₂ : List (Sigma β)} (nd₁ : l₁.NodupKeys) (nd₂ : l₂.NodupKeys)
    (p : l₁ ~ l₂) : dlookup a l₁ = dlookup a l₂ := by
  /-
    α : Type u
    β : α → Type v
    inst✝ : DecidableEq α
    a : α
    l₁ l₂ : List (Sigma β)
    nd₁ : l₁.NodupKeys
    nd₂ : l₂.NodupKeys
    p : l₁.Perm l₂
    ⊢ Eq (List.dlookup a l₁) (List.dlookup a l₂)
  -/
  ext b; simp only [mem_dlookup_iff nd₁, mem_dlookup_iff nd₂]; exact p.mem_iff
                                                               /-
                                                                 🎉 no goals
                                                               -/


theorem lookup_ext {l₀ l₁ : List (Sigma β)} (nd₀ : l₀.NodupKeys) (nd₁ : l₁.NodupKeys)
    (h : ∀ x y, y ∈ l₀.dlookup x ↔ y ∈ l₁.dlookup x) : l₀ ~ l₁ :=
  mem_ext nd₀.nodup nd₁.nodup fun ⟨a, b⟩ => by
    /-
      α : Type u
      β : α → Type v
      inst✝ : DecidableEq α
      l₀ l₁ : List (Sigma β)
      nd₀ : l₀.NodupKeys
      nd₁ : l₁.NodupKeys
      h : ∀ (x : α) (y : β x), Iff (Membership.mem (List.dlookup x l₀) y) (Membershi …
      x✝ : Sigma β
      a : α
      b : β a
      ⊢ Iff (Membership.mem l₀ ⟨a, b⟩) (Membership.mem l₁ ⟨a, b⟩)
    -/
                                                     /-
                                                       🎉 no goals
                                                     -/
    rw [← mem_dlookup_iff, ← mem_dlookup_iff, h] <;> assumption
                                                     /-
                                                       🎉 no goals
                                                     -/


theorem dlookup_map (l : List (Sigma β))
    {f : α → α'} (hf : Function.Injective f) (g : ∀ a, β a → β' (f a)) (a : α) :
    (l.map fun x => ⟨f x.1, g _ x.2⟩).dlookup (f a) = (l.dlookup a).map (g a) := by
  /-
    α : Type u
    α' : Type u'
    β : α → Type v
    β' : α' → Type v'
    inst✝¹ : DecidableEq α
    inst✝ : DecidableEq α'
    l : List (Sigma β)
    f : α → α'
    hf : Function.Injective f
    g : (a : α) → β a → β' (f a)
    a : α
    ⊢ Eq (List.dlookup (f a) (List.map (fun x => ⟨f x.fst, g x.fst x.snd⟩) l)) (Op …
  -/
  induction' l with b l IH
    /-
      case nil
      α : Type u
      α' : Type u'
      β : α → Type v
      β' : α' → Type v'
      inst✝¹ : DecidableEq α
      inst✝ : DecidableEq α'
      f : α → α'
      hf : Function.Injective f
      g : (a : α) → β a → β' (f a)
      a : α
      ⊢ Eq (List.dlookup (f a) (List.map (fun x => ⟨f x.fst, g x.fst x.snd⟩) List.ni …
    -/
  · rw [map_nil, dlookup_nil, dlookup_nil, Option.map_none']
    /-
      🎉 no goals
    -/
    /-
      case cons
      α : Type u
      α' : Type u'
      β : α → Type v
      β' : α' → Type v'
      inst✝¹ : DecidableEq α
      inst✝ : DecidableEq α'
      f : α → α'
      hf : Function.Injective f
      g : (a : α) → β a → β' (f a)
      a : α
      b : Sigma β
      l : List (Sigma β)
      IH : Eq (List.dlookup (f a) (List.map (fun x => ⟨f x.fst, g x.fst x.snd⟩) l))  …
      ⊢ Eq (List.dlookup (f a) (List.map (fun x => ⟨f x.fst, g x.fst x.snd⟩) (List.c …
    -/
  · rw [map_cons]
    /-
      case cons
      α : Type u
      α' : Type u'
      β : α → Type v
      β' : α' → Type v'
      inst✝¹ : DecidableEq α
      inst✝ : DecidableEq α'
      f : α → α'
      hf : Function.Injective f
      g : (a : α) → β a → β' (f a)
      a : α
      b : Sigma β
      l : List (Sigma β)
      IH : Eq (List.dlookup (f a) (List.map (fun x => ⟨f x.fst, g x.fst x.snd⟩) l))  …
      ⊢ Eq (List.dlookup (f a) (List.cons ⟨f b.fst, g b.fst b.snd⟩ (List.map (fun x  …
    -/
    obtain rfl | h := eq_or_ne a b.1
      /-
        case cons.inl
        α : Type u
        α' : Type u'
        β : α → Type v
        β' : α' → Type v'
        inst✝¹ : DecidableEq α
        inst✝ : DecidableEq α'
        f : α → α'
        hf : Function.Injective f
        g : (a : α) → β a → β' (f a)
        b : Sigma β
        l : List (Sigma β)
        IH : Eq (List.dlookup (f b.fst) (List.map (fun x => ⟨f x.fst, g x.fst x.snd⟩)  …
        ⊢ Eq (List.dlookup (f b.fst) (List.cons ⟨f b.fst, g b.fst b.snd⟩ (List.map (fu …
      -/
    · rw [dlookup_cons_eq, dlookup_cons_eq, Option.map_some']
      /-
        🎉 no goals
      -/
      /-
        case cons.inr
        α : Type u
        α' : Type u'
        β : α → Type v
        β' : α' → Type v'
        inst✝¹ : DecidableEq α
        inst✝ : DecidableEq α'
        f : α → α'
        hf : Function.Injective f
        g : (a : α) → β a → β' (f a)
        a : α
        b : Sigma β
        l : List (Sigma β)
        IH : Eq (List.dlookup (f a) (List.map (fun x => ⟨f x.fst, g x.fst x.snd⟩) l))  …
        h : Ne a b.fst
        ⊢ Eq (List.dlookup (f a) (List.cons ⟨f b.fst, g b.fst b.snd⟩ (List.map (fun x  …
      -/
    · rw [dlookup_cons_ne _ _ h, dlookup_cons_ne _ _ (fun he => h <| hf he), IH]
      /-
        🎉 no goals
      -/


theorem dlookup_map₁ {β : Type v} (l : List (Σ _ : α, β))
    {f : α → α'} (hf : Function.Injective f) (a : α) :
    (l.map fun x => ⟨f x.1, x.2⟩ : List (Σ _ : α', β)).dlookup (f a) = l.dlookup a := by
  /-
    α : Type u
    α' : Type u'
    inst✝¹ : DecidableEq α
    inst✝ : DecidableEq α'
    β : Type v
    l : List (Sigma fun x => β)
    f : α → α'
    hf : Function.Injective f
    a : α
    ⊢ Eq (List.dlookup (f a) (List.map (fun x => ⟨f x.fst, x.snd⟩) l)) (List.dlook …
  -/
  rw [dlookup_map (β' := fun _ => β) l hf (fun _ x => x) a, Option.map_id']
  /-
    🎉 no goals
  -/


theorem dlookup_map₂ {γ δ : α → Type*} {l : List (Σ a, γ a)} {f : ∀ a, γ a → δ a} (a : α) :
    (l.map fun x => ⟨x.1, f _ x.2⟩ : List (Σ a, δ a)).dlookup a = (l.dlookup a).map (f a) :=
  dlookup_map l Function.injective_id _ _


/-- `lookup_all a l` is the list of all values in `l` corresponding to the key `a`. -/
def lookupAll (a : α) : List (Sigma β) → List (β a)
  | [] => []
  | ⟨a', b⟩ :: l => if h : a' = a then Eq.recOn h b :: lookupAll a l else lookupAll a l


@[simp]
theorem lookupAll_nil (a : α) : lookupAll a [] = @nil (β a) :=
  rfl


@[simp]
theorem lookupAll_cons_eq (l) (a : α) (b : β a) : lookupAll a (⟨a, b⟩ :: l) = b :: lookupAll a l :=
  dif_pos rfl


@[simp]
theorem lookupAll_cons_ne (l) {a} : ∀ s : Sigma β, a ≠ s.1 → lookupAll a (s :: l) = lookupAll a l
  | ⟨_, _⟩, h => dif_neg h.symm


theorem lookupAll_eq_nil {a : α} :
    ∀ {l : List (Sigma β)}, lookupAll a l = [] ↔ ∀ b : β a, Sigma.mk a b ∉ l
             /-
               α : Type u
               β : α → Type v
               inst✝ : DecidableEq α
               a : α
               ⊢ Iff (Eq (List.lookupAll a List.nil) List.nil) (∀ (b : β a), Not (Membership. …
             -/
  | [] => by simp
             /-
               🎉 no goals
             -/
  | ⟨a', b⟩ :: l => by
    /-
      α : Type u
      β : α → Type v
      inst✝ : DecidableEq α
      a a' : α
      b : β a'
      l : List (Sigma β)
      ⊢ Iff (Eq (List.lookupAll a (List.cons ⟨a', b⟩ l)) List.nil) (∀ (b_1 : β a), N …
    -/
    by_cases h : a = a'
      /-
        case pos
        α : Type u
        β : α → Type v
        inst✝ : DecidableEq α
        a a' : α
        b : β a'
        l : List (Sigma β)
        h : Eq a a'
        ⊢ Iff (Eq (List.lookupAll a (List.cons ⟨a', b⟩ l)) List.nil) (∀ (b_1 : β a), N …
      -/
    · subst a'
      simp only [lookupAll_cons_eq, mem_cons, Sigma.mk.inj_iff, heq_eq_eq, true_and, not_or,
        false_iff, not_forall, not_and, not_not, reduceCtorEq]
      /-
        case pos
        α : Type u
        β : α → Type v
        inst✝ : DecidableEq α
        a : α
        l : List (Sigma β)
        b : β a
        ⊢ Exists fun x => Not (Eq x b) → Membership.mem l ⟨a, x⟩
      -/
      use b
      /-
        case h
        α : Type u
        β : α → Type v
        inst✝ : DecidableEq α
        a : α
        l : List (Sigma β)
        b : β a
        ⊢ Not (Eq b b) → Membership.mem l ⟨a, b⟩
      -/
      simp
      /-
        🎉 no goals
      -/
      /-
        case neg
        α : Type u
        β : α → Type v
        inst✝ : DecidableEq α
        a a' : α
        b : β a'
        l : List (Sigma β)
        h : Not (Eq a a')
        ⊢ Iff (Eq (List.lookupAll a (List.cons ⟨a', b⟩ l)) List.nil) (∀ (b_1 : β a), N …
      -/
    · simp [h, lookupAll_eq_nil]
      /-
        🎉 no goals
      -/


theorem head?_lookupAll (a : α) : ∀ l : List (Sigma β), head? (lookupAll a l) = dlookup a l
             /-
               α : Type u
               β : α → Type v
               inst✝ : DecidableEq α
               a : α
               ⊢ Eq (List.lookupAll a List.nil).head? (List.dlookup a List.nil)
             -/
  | [] => by simp
             /-
               🎉 no goals
             -/
  | ⟨a', b⟩ :: l => by
    /-
      α : Type u
      β : α → Type v
      inst✝ : DecidableEq α
      a a' : α
      b : β a'
      l : List (Sigma β)
      ⊢ Eq (List.lookupAll a (List.cons ⟨a', b⟩ l)).head? (List.dlookup a (List.cons …
    -/
    by_cases h : a = a'
      /-
        case pos
        α : Type u
        β : α → Type v
        inst✝ : DecidableEq α
        a a' : α
        b : β a'
        l : List (Sigma β)
        h : Eq a a'
        ⊢ Eq (List.lookupAll a (List.cons ⟨a', b⟩ l)).head? (List.dlookup a (List.cons …
      -/
    · subst h; simp
               /-
                 🎉 no goals
               -/
      /-
        case neg
        α : Type u
        β : α → Type v
        inst✝ : DecidableEq α
        a a' : α
        b : β a'
        l : List (Sigma β)
        h : Not (Eq a a')
        ⊢ Eq (List.lookupAll a (List.cons ⟨a', b⟩ l)).head? (List.dlookup a (List.cons …
      -/
                                                                       /-
                                                                         🎉 no goals
                                                                       -/
    · rw [lookupAll_cons_ne, dlookup_cons_ne, head?_lookupAll a l] <;> assumption
                                                                       /-
                                                                         🎉 no goals
                                                                       -/


theorem mem_lookupAll {a : α} {b : β a} :
    ∀ {l : List (Sigma β)}, b ∈ lookupAll a l ↔ Sigma.mk a b ∈ l
             /-
               α : Type u
               β : α → Type v
               inst✝ : DecidableEq α
               a : α
               b : β a
               ⊢ Iff (Membership.mem (List.lookupAll a List.nil) b) (Membership.mem List.nil  …
             -/
  | [] => by simp
             /-
               🎉 no goals
             -/
  | ⟨a', b'⟩ :: l => by
    /-
      α : Type u
      β : α → Type v
      inst✝ : DecidableEq α
      a : α
      b : β a
      a' : α
      b' : β a'
      l : List (Sigma β)
      ⊢ Iff (Membership.mem (List.lookupAll a (List.cons ⟨a', b'⟩ l)) b) (Membership …
    -/
    by_cases h : a = a'
      /-
        case pos
        α : Type u
        β : α → Type v
        inst✝ : DecidableEq α
        a : α
        b : β a
        a' : α
        b' : β a'
        l : List (Sigma β)
        h : Eq a a'
        ⊢ Iff (Membership.mem (List.lookupAll a (List.cons ⟨a', b'⟩ l)) b) (Membership …
      -/
    · subst h
      /-
        case pos
        α : Type u
        β : α → Type v
        inst✝ : DecidableEq α
        a : α
        b : β a
        l : List (Sigma β)
        b' : β a
        ⊢ Iff (Membership.mem (List.lookupAll a (List.cons ⟨a, b'⟩ l)) b) (Membership. …
      -/
      simp [*, mem_lookupAll]
      /-
        🎉 no goals
      -/
      /-
        case neg
        α : Type u
        β : α → Type v
        inst✝ : DecidableEq α
        a : α
        b : β a
        a' : α
        b' : β a'
        l : List (Sigma β)
        h : Not (Eq a a')
        ⊢ Iff (Membership.mem (List.lookupAll a (List.cons ⟨a', b'⟩ l)) b) (Membership …
      -/
    · simp [*, mem_lookupAll]
      /-
        🎉 no goals
      -/


theorem lookupAll_sublist (a : α) : ∀ l : List (Sigma β), (lookupAll a l).map (Sigma.mk a) <+ l
             /-
               α : Type u
               β : α → Type v
               inst✝ : DecidableEq α
               a : α
               ⊢ (List.map (Sigma.mk a) (List.lookupAll a List.nil)).Sublist List.nil
             -/
  | [] => by simp
             /-
               🎉 no goals
             -/
  | ⟨a', b'⟩ :: l => by
    /-
      α : Type u
      β : α → Type v
      inst✝ : DecidableEq α
      a a' : α
      b' : β a'
      l : List (Sigma β)
      ⊢ (List.map (Sigma.mk a) (List.lookupAll a (List.cons ⟨a', b'⟩ l))).Sublist (L …
    -/
    by_cases h : a = a'
      /-
        case pos
        α : Type u
        β : α → Type v
        inst✝ : DecidableEq α
        a a' : α
        b' : β a'
        l : List (Sigma β)
        h : Eq a a'
        ⊢ (List.map (Sigma.mk a) (List.lookupAll a (List.cons ⟨a', b'⟩ l))).Sublist (L …
      -/
    · subst h
      /-
        case pos
        α : Type u
        β : α → Type v
        inst✝ : DecidableEq α
        a : α
        l : List (Sigma β)
        b' : β a
        ⊢ (List.map (Sigma.mk a) (List.lookupAll a (List.cons ⟨a, b'⟩ l))).Sublist (Li …
      -/
      simp only [ne_eq, not_true, lookupAll_cons_eq, List.map]
      /-
        case pos
        α : Type u
        β : α → Type v
        inst✝ : DecidableEq α
        a : α
        l : List (Sigma β)
        b' : β a
        ⊢ (List.cons ⟨a, b'⟩ (List.map (Sigma.mk a) (List.lookupAll a l))).Sublist (Li …
      -/
      exact (lookupAll_sublist a l).cons₂ _
      /-
        🎉 no goals
      -/
      /-
        case neg
        α : Type u
        β : α → Type v
        inst✝ : DecidableEq α
        a a' : α
        b' : β a'
        l : List (Sigma β)
        h : Not (Eq a a')
        ⊢ (List.map (Sigma.mk a) (List.lookupAll a (List.cons ⟨a', b'⟩ l))).Sublist (L …
      -/
    · simp only [ne_eq, h, not_false_iff, lookupAll_cons_ne]
      /-
        case neg
        α : Type u
        β : α → Type v
        inst✝ : DecidableEq α
        a a' : α
        b' : β a'
        l : List (Sigma β)
        h : Not (Eq a a')
        ⊢ (List.map (Sigma.mk a) (List.lookupAll a l)).Sublist (List.cons ⟨a', b'⟩ l)
      -/
      exact (lookupAll_sublist a l).cons _
      /-
        🎉 no goals
      -/


theorem lookupAll_length_le_one (a : α) {l : List (Sigma β)} (h : l.NodupKeys) :
    length (lookupAll a l) ≤ 1 := by
  /-
    α : Type u
    β : α → Type v
    inst✝ : DecidableEq α
    a : α
    l : List (Sigma β)
    h : l.NodupKeys
    ⊢ LE.le (List.lookupAll a l).length 1
  -/
  have := Nodup.sublist ((lookupAll_sublist a l).map _) h
  /-
    α : Type u
    β : α → Type v
    inst✝ : DecidableEq α
    a : α
    l : List (Sigma β)
    h : l.NodupKeys
    this : (List.map Sigma.fst (List.map (Sigma.mk a) (List.lookupAll a l))).Nodup
    ⊢ LE.le (List.lookupAll a l).length 1
  -/
  rw [map_map] at this
  /-
    α : Type u
    β : α → Type v
    inst✝ : DecidableEq α
    a : α
    l : List (Sigma β)
    h : l.NodupKeys
    this : (List.map (Function.comp Sigma.fst (Sigma.mk a)) (List.lookupAll a l)). …
    ⊢ LE.le (List.lookupAll a l).length 1
  -/
  rwa [← nodup_replicate, ← map_const]
  /-
    🎉 no goals
  -/


theorem lookupAll_eq_dlookup (a : α) {l : List (Sigma β)} (h : l.NodupKeys) :
    lookupAll a l = (dlookup a l).toList := by
  /-
    α : Type u
    β : α → Type v
    inst✝ : DecidableEq α
    a : α
    l : List (Sigma β)
    h : l.NodupKeys
    ⊢ Eq (List.lookupAll a l) (List.dlookup a l).toList
  -/
  rw [← head?_lookupAll]
  /-
    α : Type u
    β : α → Type v
    inst✝ : DecidableEq α
    a : α
    l : List (Sigma β)
    h : l.NodupKeys
    ⊢ Eq (List.lookupAll a l) (List.lookupAll a l).head?.toList
  -/
  have h1 := lookupAll_length_le_one a h; revert h1
  /-
    α : Type u
    β : α → Type v
    inst✝ : DecidableEq α
    a : α
    l : List (Sigma β)
    h : l.NodupKeys
    ⊢ LE.le (List.lookupAll a l).length 1 → Eq (List.lookupAll a l) (List.lookupAl …
  -/
                                                                   /-
                                                                     🎉 no goals
                                                                   -/
                                                                   /-
                                                                     🎉 no goals
                                                                   -/
  rcases lookupAll a l with (_ | ⟨b, _ | ⟨c, l⟩⟩) <;> intro h1 <;> try rfl
  /-
    case cons.cons
    α : Type u
    β : α → Type v
    inst✝ : DecidableEq α
    a : α
    l✝ : List (Sigma β)
    h : l✝.NodupKeys
    b c : β a
    l : List (β a)
    h1 : LE.le (List.cons b (List.cons c l)).length 1
    ⊢ Eq (List.cons b (List.cons c l)) (List.cons b (List.cons c l)).head?.toList
  -/
  exact absurd h1 (by simp)
  /-
    🎉 no goals
  -/


theorem lookupAll_nodup (a : α) {l : List (Sigma β)} (h : l.NodupKeys) : (lookupAll a l).Nodup := by
   /-
     α : Type u
     β : α → Type v
     inst✝ : DecidableEq α
     a : α
     l : List (Sigma β)
     h : l.NodupKeys
     ⊢ (List.lookupAll a l).Nodup
   -/
  (rw [lookupAll_eq_dlookup a h]; apply Option.toList_nodup)
                                  /-
                                    🎉 no goals
                                  -/


theorem perm_lookupAll (a : α) {l₁ l₂ : List (Sigma β)} (nd₁ : l₁.NodupKeys) (nd₂ : l₂.NodupKeys)
    (p : l₁ ~ l₂) : lookupAll a l₁ = lookupAll a l₂ := by
  /-
    α : Type u
    β : α → Type v
    inst✝ : DecidableEq α
    a : α
    l₁ l₂ : List (Sigma β)
    nd₁ : l₁.NodupKeys
    nd₂ : l₂.NodupKeys
    p : l₁.Perm l₂
    ⊢ Eq (List.lookupAll a l₁) (List.lookupAll a l₂)
  -/
  simp [lookupAll_eq_dlookup, nd₁, nd₂, perm_dlookup a nd₁ nd₂ p]
  /-
    🎉 no goals
  -/


theorem dlookup_append (l₁ l₂ : List (Sigma β)) (a : α) :
    (l₁ ++ l₂).dlookup a = (l₁.dlookup a).or (l₂.dlookup a) := by
  induction l₁ with
  | nil => rfl
  | cons x l₁ IH =>
    rw [cons_append]
    obtain rfl | hb := Decidable.eq_or_ne a x.1
    · rw [dlookup_cons_eq, dlookup_cons_eq, Option.or]
    · rw [dlookup_cons_ne _ _ hb, dlookup_cons_ne _ _ hb, IH]


/-- Replaces the first value with key `a` by `b`. -/
def kreplace (a : α) (b : β a) : List (Sigma β) → List (Sigma β) :=
  lookmap fun s => if a = s.1 then some ⟨a, b⟩ else none


theorem kreplace_of_forall_not (a : α) (b : β a) {l : List (Sigma β)}
    (H : ∀ b : β a, Sigma.mk a b ∉ l) : kreplace a b l = l :=
  lookmap_of_forall_not _ <| by
    /-
      α : Type u
      β : α → Type v
      inst✝ : DecidableEq α
      a : α
      b : β a
      l : List (Sigma β)
      H : ∀ (b : β a), Not (Membership.mem l ⟨a, b⟩)
      ⊢ ∀ (a_1 : Sigma β), Membership.mem l a_1 → Eq (ite (Eq a a_1.fst) (Option.som …
    -/
    rintro ⟨a', b'⟩ h; dsimp; split_ifs
      /-
        case pos
        α : Type u
        β : α → Type v
        inst✝ : DecidableEq α
        a : α
        b : β a
        l : List (Sigma β)
        H : ∀ (b : β a), Not (Membership.mem l ⟨a, b⟩)
        a' : α
        b' : β a'
        h : Membership.mem l ⟨a', b'⟩
        h✝ : Eq a a'
        ⊢ False
      -/
    · subst a'
      /-
        case pos
        α : Type u
        β : α → Type v
        inst✝ : DecidableEq α
        a : α
        b : β a
        l : List (Sigma β)
        H : ∀ (b : β a), Not (Membership.mem l ⟨a, b⟩)
        b' : β a
        h : Membership.mem l ⟨a, b'⟩
        ⊢ False
      -/
      exact H _ h
      /-
        🎉 no goals
      -/
      /-
        case neg
        α : Type u
        β : α → Type v
        inst✝ : DecidableEq α
        a : α
        b : β a
        l : List (Sigma β)
        H : ∀ (b : β a), Not (Membership.mem l ⟨a, b⟩)
        a' : α
        b' : β a'
        h : Membership.mem l ⟨a', b'⟩
        h✝ : Not (Eq a a')
        ⊢ Eq Option.none Option.none
      -/
    · rfl
      /-
        🎉 no goals
      -/


theorem kreplace_self {a : α} {b : β a} {l : List (Sigma β)} (nd : NodupKeys l)
    (h : Sigma.mk a b ∈ l) : kreplace a b l = l := by
  /-
    α : Type u
    β : α → Type v
    inst✝ : DecidableEq α
    a : α
    b : β a
    l : List (Sigma β)
    nd : l.NodupKeys
    h : Membership.mem l ⟨a, b⟩
    ⊢ Eq (List.kreplace a b l) l
  -/
  refine (lookmap_congr ?_).trans (lookmap_id' (Option.guard fun (s : Sigma β) => a = s.1) ?_ _)
    /-
      case refine_1
      α : Type u
      β : α → Type v
      inst✝ : DecidableEq α
      a : α
      b : β a
      l : List (Sigma β)
      nd : l.NodupKeys
      h : Membership.mem l ⟨a, b⟩
      ⊢ ∀ (a_1 : Sigma β), Membership.mem l a_1 → Eq (ite (Eq a a_1.fst) (Option.som …
    -/
  · rintro ⟨a', b'⟩ h'
    /-
      case refine_1.mk
      α : Type u
      β : α → Type v
      inst✝ : DecidableEq α
      a : α
      b : β a
      l : List (Sigma β)
      nd : l.NodupKeys
      h : Membership.mem l ⟨a, b⟩
      a' : α
      b' : β a'
      h' : Membership.mem l ⟨a', b'⟩
      ⊢ Eq (ite (Eq a ⟨a', b'⟩.fst) (Option.some ⟨a, b⟩) Option.none) (Option.guard  …
    -/
    dsimp [Option.guard]
    /-
      case refine_1.mk
      α : Type u
      β : α → Type v
      inst✝ : DecidableEq α
      a : α
      b : β a
      l : List (Sigma β)
      nd : l.NodupKeys
      h : Membership.mem l ⟨a, b⟩
      a' : α
      b' : β a'
      h' : Membership.mem l ⟨a', b'⟩
      ⊢ Eq (ite (Eq a a') (Option.some ⟨a, b⟩) Option.none) (ite (Eq a a') (Option.s …
    -/
    split_ifs
      /-
        case pos
        α : Type u
        β : α → Type v
        inst✝ : DecidableEq α
        a : α
        b : β a
        l : List (Sigma β)
        nd : l.NodupKeys
        h : Membership.mem l ⟨a, b⟩
        a' : α
        b' : β a'
        h' : Membership.mem l ⟨a', b'⟩
        h✝ : Eq a a'
        ⊢ Eq (Option.some ⟨a, b⟩) (Option.some ⟨a', b'⟩)
      -/
    · subst a'
      /-
        case pos
        α : Type u
        β : α → Type v
        inst✝ : DecidableEq α
        a : α
        b : β a
        l : List (Sigma β)
        nd : l.NodupKeys
        h : Membership.mem l ⟨a, b⟩
        b' : β a
        h' : Membership.mem l ⟨a, b'⟩
        ⊢ Eq (Option.some ⟨a, b⟩) (Option.some ⟨a, b'⟩)
      -/
      simp [nd.eq_of_mk_mem h h']
      /-
        🎉 no goals
      -/
      /-
        case neg
        α : Type u
        β : α → Type v
        inst✝ : DecidableEq α
        a : α
        b : β a
        l : List (Sigma β)
        nd : l.NodupKeys
        h : Membership.mem l ⟨a, b⟩
        a' : α
        b' : β a'
        h' : Membership.mem l ⟨a', b'⟩
        h✝ : Not (Eq a a')
        ⊢ Eq Option.none Option.none
      -/
    · rfl
      /-
        🎉 no goals
      -/
    /-
      case refine_2
      α : Type u
      β : α → Type v
      inst✝ : DecidableEq α
      a : α
      b : β a
      l : List (Sigma β)
      nd : l.NodupKeys
      h : Membership.mem l ⟨a, b⟩
      ⊢ ∀ (a_1 b : Sigma β), Membership.mem (Option.guard (fun s => Eq a s.fst) a_1) …
    -/
  · rintro ⟨a₁, b₁⟩ ⟨a₂, b₂⟩
    /-
      case refine_2.mk.mk
      α : Type u
      β : α → Type v
      inst✝ : DecidableEq α
      a : α
      b : β a
      l : List (Sigma β)
      nd : l.NodupKeys
      h : Membership.mem l ⟨a, b⟩
      a₁ : α
      b₁ : β a₁
      a₂ : α
      b₂ : β a₂
      ⊢ Membership.mem (Option.guard (fun s => Eq a s.fst) ⟨a₁, b₁⟩) ⟨a₂, b₂⟩ → Eq ⟨ …
    -/
    dsimp [Option.guard]
    /-
      case refine_2.mk.mk
      α : Type u
      β : α → Type v
      inst✝ : DecidableEq α
      a : α
      b : β a
      l : List (Sigma β)
      nd : l.NodupKeys
      h : Membership.mem l ⟨a, b⟩
      a₁ : α
      b₁ : β a₁
      a₂ : α
      b₂ : β a₂
      ⊢ Membership.mem (ite (Eq a a₁) (Option.some ⟨a₁, b₁⟩) Option.none) ⟨a₂, b₂⟩ → …
    -/
    split_ifs
      /-
        case pos
        α : Type u
        β : α → Type v
        inst✝ : DecidableEq α
        a : α
        b : β a
        l : List (Sigma β)
        nd : l.NodupKeys
        h : Membership.mem l ⟨a, b⟩
        a₁ : α
        b₁ : β a₁
        a₂ : α
        b₂ : β a₂
        h✝ : Eq a a₁
        ⊢ Membership.mem (Option.some ⟨a₁, b₁⟩) ⟨a₂, b₂⟩ → Eq ⟨a₁, b₁⟩ ⟨a₂, b₂⟩
      -/
    · simp
      /-
        🎉 no goals
      -/
      /-
        case neg
        α : Type u
        β : α → Type v
        inst✝ : DecidableEq α
        a : α
        b : β a
        l : List (Sigma β)
        nd : l.NodupKeys
        h : Membership.mem l ⟨a, b⟩
        a₁ : α
        b₁ : β a₁
        a₂ : α
        b₂ : β a₂
        h✝ : Not (Eq a a₁)
        ⊢ Membership.mem Option.none ⟨a₂, b₂⟩ → Eq ⟨a₁, b₁⟩ ⟨a₂, b₂⟩
      -/
    · rintro ⟨⟩
      /-
        🎉 no goals
      -/


theorem keys_kreplace (a : α) (b : β a) : ∀ l : List (Sigma β), (kreplace a b l).keys = l.keys :=
  lookmap_map_eq _ _ <| by
    /-
      α : Type u
      β : α → Type v
      inst✝ : DecidableEq α
      a : α
      b : β a
      ⊢ ∀ (a_1 b_1 : Sigma β), Membership.mem (ite (Eq a a_1.fst) (Option.some ⟨a, b …
    -/
    rintro ⟨a₁, b₂⟩ ⟨a₂, b₂⟩
    /-
      case mk.mk
      α : Type u
      β : α → Type v
      inst✝ : DecidableEq α
      a : α
      b : β a
      a₁ : α
      b₂✝ : β a₁
      a₂ : α
      b₂ : β a₂
      ⊢ Membership.mem (ite (Eq a ⟨a₁, b₂✝⟩.fst) (Option.some ⟨a, b⟩) Option.none) ⟨ …
    -/
    dsimp
    /-
      case mk.mk
      α : Type u
      β : α → Type v
      inst✝ : DecidableEq α
      a : α
      b : β a
      a₁ : α
      b₂✝ : β a₁
      a₂ : α
      b₂ : β a₂
      ⊢ Membership.mem (ite (Eq a a₁) (Option.some ⟨a, b⟩) Option.none) ⟨a₂, b₂⟩ → E …
    -/
                         /-
                           🎉 no goals
                         -/
    split_ifs with h <;> simp +contextual [h]
                         /-
                           🎉 no goals
                         -/


theorem kreplace_nodupKeys (a : α) (b : β a) {l : List (Sigma β)} :
                                                   /-
                                                     α : Type u
                                                     β : α → Type v
                                                     inst✝ : DecidableEq α
                                                     a : α
                                                     b : β a
                                                     l : List (Sigma β)
                                                     ⊢ Iff (List.kreplace a b l).NodupKeys l.NodupKeys
                                                   -/
    (kreplace a b l).NodupKeys ↔ l.NodupKeys := by simp [NodupKeys, keys_kreplace]
                                                   /-
                                                     🎉 no goals
                                                   -/


theorem Perm.kreplace {a : α} {b : β a} {l₁ l₂ : List (Sigma β)} (nd : l₁.NodupKeys) :
    l₁ ~ l₂ → kreplace a b l₁ ~ kreplace a b l₂ :=
  perm_lookmap _ <| by
    /-
      α : Type u
      β : α → Type v
      inst✝ : DecidableEq α
      a : α
      b : β a
      l₁ l₂ : List (Sigma β)
      nd : l₁.NodupKeys
      ⊢ List.Pairwise (fun a_1 b_1 => ∀ (c : Sigma β), Membership.mem (ite (Eq a a_1 …
    -/
    refine nd.pairwise_ne.imp ?_
    /-
      α : Type u
      β : α → Type v
      inst✝ : DecidableEq α
      a : α
      b : β a
      l₁ l₂ : List (Sigma β)
      nd : l₁.NodupKeys
      ⊢ ∀ {a_1 b_1 : Sigma β}, Ne a_1.fst b_1.fst → ∀ (c : Sigma β), Membership.mem  …
    -/
    intro x y h z h₁ w h₂
    /-
      α : Type u
      β : α → Type v
      inst✝ : DecidableEq α
      a : α
      b : β a
      l₁ l₂ : List (Sigma β)
      nd : l₁.NodupKeys
      x y : Sigma β
      h : Ne x.fst y.fst
      z : Sigma β
      h₁ : Membership.mem (ite (Eq a x.fst) (Option.some ⟨a, b⟩) Option.none) z
      w : Sigma β
      h₂ : Membership.mem (ite (Eq a y.fst) (Option.some ⟨a, b⟩) Option.none) w
      ⊢ And (Eq x y) (Eq z w)
    -/
                                        /-
                                          🎉 no goals
                                        -/
                                        /-
                                          🎉 no goals
                                        -/
    split_ifs at h₁ h₂ with h_2 h_1 <;> cases h₁ <;> cases h₂
                                                     /-
                                                       🎉 no goals
                                                     -/
    /-
      case pos.refl.refl
      α : Type u
      β : α → Type v
      inst✝ : DecidableEq α
      a : α
      b : β a
      l₁ l₂ : List (Sigma β)
      nd : l₁.NodupKeys
      x y : Sigma β
      h : Ne x.fst y.fst
      h_2 : Eq a x.fst
      h_1 : Eq a y.fst
      ⊢ And (Eq x y) (Eq ⟨a, b⟩ ⟨a, b⟩)
    -/
    exact (h (h_2.symm.trans h_1)).elim
    /-
      🎉 no goals
    -/


/-- Remove the first pair with the key `a`. -/
def kerase (a : α) : List (Sigma β) → List (Sigma β) :=
  eraseP fun s => a = s.1


@[simp]
theorem kerase_nil {a} : @kerase _ β _ a [] = [] :=
  rfl


@[simp]
theorem kerase_cons_eq {a} {s : Sigma β} {l : List (Sigma β)} (h : a = s.1) :
                                /-
                                  α : Type u
                                  β : α → Type v
                                  inst✝ : DecidableEq α
                                  a : α
                                  s : Sigma β
                                  l : List (Sigma β)
                                  h : Eq a s.fst
                                  ⊢ Eq (List.kerase a (List.cons s l)) l
                                -/
    kerase a (s :: l) = l := by simp [kerase, h]
                                /-
                                  🎉 no goals
                                -/


@[simp]
theorem kerase_cons_ne {a} {s : Sigma β} {l : List (Sigma β)} (h : a ≠ s.1) :
                                              /-
                                                α : Type u
                                                β : α → Type v
                                                inst✝ : DecidableEq α
                                                a : α
                                                s : Sigma β
                                                l : List (Sigma β)
                                                h : Ne a s.fst
                                                ⊢ Eq (List.kerase a (List.cons s l)) (List.cons s (List.kerase a l))
                                              -/
    kerase a (s :: l) = s :: kerase a l := by simp [kerase, h]
                                              /-
                                                🎉 no goals
                                              -/


@[simp]
theorem kerase_of_not_mem_keys {a} {l : List (Sigma β)} (h : a ∉ l.keys) : kerase a l = l := by
  induction l with
  | nil => rfl
  | cons _ _ ih => simp [not_or] at h; simp [h.1, ih h.2]


theorem kerase_sublist (a : α) (l : List (Sigma β)) : kerase a l <+ l :=
  eraseP_sublist _


theorem kerase_keys_subset (a) (l : List (Sigma β)) : (kerase a l).keys ⊆ l.keys :=
  ((kerase_sublist a l).map _).subset


theorem mem_keys_of_mem_keys_kerase {a₁ a₂} {l : List (Sigma β)} :
    a₁ ∈ (kerase a₂ l).keys → a₁ ∈ l.keys :=
  @kerase_keys_subset _ _ _ _ _ _


theorem exists_of_kerase {a : α} {l : List (Sigma β)} (h : a ∈ l.keys) :
    ∃ (b : β a) (l₁ l₂ : List (Sigma β)),
      a ∉ l₁.keys ∧ l = l₁ ++ ⟨a, b⟩ :: l₂ ∧ kerase a l = l₁ ++ l₂ := by
  induction l with
  | nil => cases h
  | cons hd tl ih =>
    by_cases e : a = hd.1
    · subst e
      exact ⟨hd.2, [], tl, by simp, by cases hd; rfl, by simp⟩
    · simp only [keys_cons, mem_cons] at h
      cases' h with h h
      · exact absurd h e
      rcases ih h with ⟨b, tl₁, tl₂, h₁, h₂, h₃⟩
      exact ⟨b, hd :: tl₁, tl₂, not_mem_cons_of_ne_of_not_mem e h₁, by (rw [h₂]; rfl), by
            simp [e, h₃]⟩


@[simp]
theorem mem_keys_kerase_of_ne {a₁ a₂} {l : List (Sigma β)} (h : a₁ ≠ a₂) :
    a₁ ∈ (kerase a₂ l).keys ↔ a₁ ∈ l.keys :=
  (Iff.intro mem_keys_of_mem_keys_kerase) fun p =>
    if q : a₂ ∈ l.keys then
      match l, kerase a₂ l, exists_of_kerase q, p with
                                              /-
                                                α : Type u
                                                β : α → Type v
                                                inst✝ : DecidableEq α
                                                a₁ a₂ : α
                                                l : List (Sigma β)
                                                h : Ne a₁ a₂
                                                p✝ : Membership.mem l.keys a₁
                                                w✝² : β a₂
                                                w✝¹ w✝ : List (Sigma β)
                                                left✝ : Not (Membership.mem w✝¹.keys a₂)
                                                p : Membership.mem (HAppend.hAppend w✝¹ (List.cons ⟨a₂, w✝²⟩ w✝)).keys a₁
                                                q : Membership.mem (HAppend.hAppend w✝¹ (List.cons ⟨a₂, w✝²⟩ w✝)).keys a₂
                                                ⊢ Membership.mem (HAppend.hAppend w✝¹ w✝).keys a₁
                                              -/
      | _, _, ⟨_, _, _, _, rfl, rfl⟩, p => by simpa [keys, h] using p
                                              /-
                                                🎉 no goals
                                              -/
            /-
              α : Type u
              β : α → Type v
              inst✝ : DecidableEq α
              a₁ a₂ : α
              l : List (Sigma β)
              h : Ne a₁ a₂
              p : Membership.mem l.keys a₁
              q : Not (Membership.mem l.keys a₂)
              ⊢ Membership.mem (List.kerase a₂ l).keys a₁
            -/
    else by simp [q, p]
            /-
              🎉 no goals
            -/


theorem keys_kerase {a} {l : List (Sigma β)} : (kerase a l).keys = l.keys.erase a := by
  /-
    α : Type u
    β : α → Type v
    inst✝ : DecidableEq α
    a : α
    l : List (Sigma β)
    ⊢ Eq (List.kerase a l).keys (l.keys.erase a)
  -/
  rw [keys, kerase, erase_eq_eraseP, eraseP_map, Function.comp_def]
  /-
    α : Type u
    β : α → Type v
    inst✝ : DecidableEq α
    a : α
    l : List (Sigma β)
    ⊢ Eq (List.map Sigma.fst (List.eraseP (fun s => Decidable.decide (Eq a s.fst)) …
  -/
  congr
  /-
    🎉 no goals
  -/


theorem kerase_kerase {a a'} {l : List (Sigma β)} :
    (kerase a' l).kerase a = (kerase a l).kerase a' := by
  /-
    α : Type u
    β : α → Type v
    inst✝ : DecidableEq α
    a a' : α
    l : List (Sigma β)
    ⊢ Eq (List.kerase a (List.kerase a' l)) (List.kerase a' (List.kerase a l))
  -/
  by_cases h : a = a'
    /-
      case pos
      α : Type u
      β : α → Type v
      inst✝ : DecidableEq α
      a a' : α
      l : List (Sigma β)
      h : Eq a a'
      ⊢ Eq (List.kerase a (List.kerase a' l)) (List.kerase a' (List.kerase a l))
    -/
  · subst a'; rfl
              /-
                🎉 no goals
              -/
  /-
    case neg
    α : Type u
    β : α → Type v
    inst✝ : DecidableEq α
    a a' : α
    l : List (Sigma β)
    h : Not (Eq a a')
    ⊢ Eq (List.kerase a (List.kerase a' l)) (List.kerase a' (List.kerase a l))
  -/
  induction' l with x xs
    /-
      case neg.nil
      α : Type u
      β : α → Type v
      inst✝ : DecidableEq α
      a a' : α
      h : Not (Eq a a')
      ⊢ Eq (List.kerase a (List.kerase a' List.nil)) (List.kerase a' (List.kerase a  …
    -/
  · rfl
    /-
      🎉 no goals
    -/
    /-
      case neg.cons
      α : Type u
      β : α → Type v
      inst✝ : DecidableEq α
      a a' : α
      h : Not (Eq a a')
      x : Sigma β
      xs : List (Sigma β)
      tail_ih✝ : Eq (List.kerase a (List.kerase a' xs)) (List.kerase a' (List.kerase …
      ⊢ Eq (List.kerase a (List.kerase a' (List.cons x xs))) (List.kerase a' (List.k …
    -/
  · by_cases a' = x.1
      /-
        case pos
        α : Type u
        β : α → Type v
        inst✝ : DecidableEq α
        a a' : α
        h : Not (Eq a a')
        x : Sigma β
        xs : List (Sigma β)
        tail_ih✝ : Eq (List.kerase a (List.kerase a' xs)) (List.kerase a' (List.kerase …
        h✝ : Eq a' x.fst
        ⊢ Eq (List.kerase a (List.kerase a' (List.cons x xs))) (List.kerase a' (List.k …
      -/
    · subst a'
      /-
        case pos
        α : Type u
        β : α → Type v
        inst✝ : DecidableEq α
        a : α
        x : Sigma β
        xs : List (Sigma β)
        h : Not (Eq a x.fst)
        tail_ih✝ : Eq (List.kerase a (List.kerase x.fst xs)) (List.kerase x.fst (List. …
        ⊢ Eq (List.kerase a (List.kerase x.fst (List.cons x xs))) (List.kerase x.fst ( …
      -/
      simp [kerase_cons_ne h, kerase_cons_eq rfl]
      /-
        🎉 no goals
      -/
    /-
      case neg
      α : Type u
      β : α → Type v
      inst✝ : DecidableEq α
      a a' : α
      h : Not (Eq a a')
      x : Sigma β
      xs : List (Sigma β)
      tail_ih✝ : Eq (List.kerase a (List.kerase a' xs)) (List.kerase a' (List.kerase …
      h✝ : Not (Eq a' x.fst)
      ⊢ Eq (List.kerase a (List.kerase a' (List.cons x xs))) (List.kerase a' (List.k …
    -/
    by_cases h' : a = x.1
      /-
        case pos
        α : Type u
        β : α → Type v
        inst✝ : DecidableEq α
        a a' : α
        h : Not (Eq a a')
        x : Sigma β
        xs : List (Sigma β)
        tail_ih✝ : Eq (List.kerase a (List.kerase a' xs)) (List.kerase a' (List.kerase …
        h✝ : Not (Eq a' x.fst)
        h' : Eq a x.fst
        ⊢ Eq (List.kerase a (List.kerase a' (List.cons x xs))) (List.kerase a' (List.k …
      -/
    · subst a
      /-
        case pos
        α : Type u
        β : α → Type v
        inst✝ : DecidableEq α
        a' : α
        x : Sigma β
        xs : List (Sigma β)
        h✝ : Not (Eq a' x.fst)
        h : Not (Eq x.fst a')
        tail_ih✝ : Eq (List.kerase x.fst (List.kerase a' xs)) (List.kerase a' (List.ke …
        ⊢ Eq (List.kerase x.fst (List.kerase a' (List.cons x xs))) (List.kerase a' (Li …
      -/
      simp [kerase_cons_eq rfl, kerase_cons_ne (Ne.symm h)]
      /-
        🎉 no goals
      -/
      /-
        case neg
        α : Type u
        β : α → Type v
        inst✝ : DecidableEq α
        a a' : α
        h : Not (Eq a a')
        x : Sigma β
        xs : List (Sigma β)
        tail_ih✝ : Eq (List.kerase a (List.kerase a' xs)) (List.kerase a' (List.kerase …
        h✝ : Not (Eq a' x.fst)
        h' : Not (Eq a x.fst)
        ⊢ Eq (List.kerase a (List.kerase a' (List.cons x xs))) (List.kerase a' (List.k …
      -/
    · simp [kerase_cons_ne, *]
      /-
        🎉 no goals
      -/


theorem NodupKeys.kerase (a : α) : NodupKeys l → (kerase a l).NodupKeys :=
  NodupKeys.sublist <| kerase_sublist _ _


theorem Perm.kerase {a : α} {l₁ l₂ : List (Sigma β)} (nd : l₁.NodupKeys) :
    l₁ ~ l₂ → kerase a l₁ ~ kerase a l₂ := by
  /-
    α : Type u
    β : α → Type v
    inst✝ : DecidableEq α
    a : α
    l₁ l₂ : List (Sigma β)
    nd : l₁.NodupKeys
    ⊢ l₁.Perm l₂ → (List.kerase a l₁).Perm (List.kerase a l₂)
  -/
  apply Perm.eraseP
  /-
    case H
    α : Type u
    β : α → Type v
    inst✝ : DecidableEq α
    a : α
    l₁ l₂ : List (Sigma β)
    nd : l₁.NodupKeys
    ⊢ List.Pairwise (fun a_1 b => Eq (Decidable.decide (Eq a a_1.fst)) Bool.true → …
  -/
  apply (nodupKeys_iff_pairwise.1 nd).imp
  /-
    case H
    α : Type u
    β : α → Type v
    inst✝ : DecidableEq α
    a : α
    l₁ l₂ : List (Sigma β)
    nd : l₁.NodupKeys
    ⊢ ∀ {a_1 b : Sigma β}, Ne a_1.fst b.fst → Eq (Decidable.decide (Eq a a_1.fst)) …
  -/
  intros; simp_all
          /-
            🎉 no goals
          -/


@[simp]
theorem not_mem_keys_kerase (a) {l : List (Sigma β)} (nd : l.NodupKeys) :
    a ∉ (kerase a l).keys := by
  induction l with
  | nil => simp
  | cons hd tl ih =>
    simp? at nd says simp only [nodupKeys_cons] at nd
    by_cases h : a = hd.1
    · subst h
      simp [nd.1]
    · simp [h, ih nd.2]


@[simp]
theorem dlookup_kerase (a) {l : List (Sigma β)} (nd : l.NodupKeys) :
    dlookup a (kerase a l) = none :=
  dlookup_eq_none.mpr (not_mem_keys_kerase a nd)


@[simp]
theorem dlookup_kerase_ne {a a'} {l : List (Sigma β)} (h : a ≠ a') :
    dlookup a (kerase a' l) = dlookup a l := by
  induction l with
  | nil => rfl
  | cons hd tl ih =>
    cases' hd with ah bh
    by_cases h₁ : a = ah <;> by_cases h₂ : a' = ah
    · substs h₁ h₂
      cases Ne.irrefl h
    · subst h₁
      simp [h₂]
    · subst h₂
      simp [h]
    · simp [h₁, h₂, ih]


theorem kerase_append_left {a} :
    ∀ {l₁ l₂ : List (Sigma β)}, a ∈ l₁.keys → kerase a (l₁ ++ l₂) = kerase a l₁ ++ l₂
                   /-
                     α : Type u
                     β : α → Type v
                     inst✝ : DecidableEq α
                     a : α
                     x✝ : List (Sigma β)
                     h : Membership.mem List.nil.keys a
                     ⊢ Eq (List.kerase a (HAppend.hAppend List.nil x✝)) (HAppend.hAppend (List.kera …
                   -/
  | [], _, h => by cases h
                   /-
                     🎉 no goals
                   -/
  | s :: l₁, l₂, h₁ => by
    if h₂ : a = s.1 then simp [h₂]
    else simp at h₁; cases' h₁ with h₁ h₁ <;> [exact absurd h₁ h₂; simp [h₂, kerase_append_left h₁]]


theorem kerase_append_right {a} :
    ∀ {l₁ l₂ : List (Sigma β)}, a ∉ l₁.keys → kerase a (l₁ ++ l₂) = l₁ ++ kerase a l₂
  | [], _, _ => rfl
  | _ :: l₁, l₂, h => by
    /-
      α : Type u
      β : α → Type v
      inst✝ : DecidableEq α
      a : α
      head✝ : Sigma β
      l₁ : List (Sigma β)
      l₂ : List (Sigma β)
      h : Not (Membership.mem (List.cons head✝ l₁).keys a)
      ⊢ Eq (List.kerase a (HAppend.hAppend (List.cons head✝ l₁) l₂)) (HAppend.hAppen …
    -/
    simp only [keys_cons, mem_cons, not_or] at h
    /-
      α : Type u
      β : α → Type v
      inst✝ : DecidableEq α
      a : α
      head✝ : Sigma β
      l₁ : List (Sigma β)
      l₂ : List (Sigma β)
      h : And (Not (Eq a head✝.fst)) (Not (Membership.mem l₁.keys a))
      ⊢ Eq (List.kerase a (HAppend.hAppend (List.cons head✝ l₁) l₂)) (HAppend.hAppen …
    -/
    simp [h.1, kerase_append_right h.2]
    /-
      🎉 no goals
    -/


theorem kerase_comm (a₁ a₂) (l : List (Sigma β)) :
    kerase a₂ (kerase a₁ l) = kerase a₁ (kerase a₂ l) :=
                         /-
                           α : Type u
                           β : α → Type v
                           inst✝ : DecidableEq α
                           a₁ a₂ : α
                           l : List (Sigma β)
                           h : Eq a₁ a₂
                           ⊢ Eq (List.kerase a₂ (List.kerase a₁ l)) (List.kerase a₁ (List.kerase a₂ l))
                         -/
  if h : a₁ = a₂ then by simp [h]
                         /-
                           🎉 no goals
                         -/
  else
    if ha₁ : a₁ ∈ l.keys then
      if ha₂ : a₂ ∈ l.keys then
        match l, kerase a₁ l, exists_of_kerase ha₁, ha₂ with
        | _, _, ⟨b₁, l₁, l₂, a₁_nin_l₁, rfl, rfl⟩, _ =>
          if h' : a₂ ∈ l₁.keys then by
            simp [kerase_append_left h',
              kerase_append_right (mt (mem_keys_kerase_of_ne h).mp a₁_nin_l₁)]
          else by
            simp [kerase_append_right h', kerase_append_right a₁_nin_l₁,
              @kerase_cons_ne _ _ _ a₂ ⟨a₁, b₁⟩ _ (Ne.symm h)]
              /-
                α : Type u
                β : α → Type v
                inst✝ : DecidableEq α
                a₁ a₂ : α
                l : List (Sigma β)
                h : Not (Eq a₁ a₂)
                ha₁ : Membership.mem l.keys a₁
                ha₂ : Not (Membership.mem l.keys a₂)
                ⊢ Eq (List.kerase a₂ (List.kerase a₁ l)) (List.kerase a₁ (List.kerase a₂ l))
              -/
      else by simp [ha₂, mt mem_keys_of_mem_keys_kerase ha₂]
              /-
                🎉 no goals
              -/
            /-
              α : Type u
              β : α → Type v
              inst✝ : DecidableEq α
              a₁ a₂ : α
              l : List (Sigma β)
              h : Not (Eq a₁ a₂)
              ha₁ : Not (Membership.mem l.keys a₁)
              ⊢ Eq (List.kerase a₂ (List.kerase a₁ l)) (List.kerase a₁ (List.kerase a₂ l))
            -/
    else by simp [ha₁, mt mem_keys_of_mem_keys_kerase ha₁]
            /-
              🎉 no goals
            -/


theorem sizeOf_kerase [SizeOf (Sigma β)] (x : α)
    (xs : List (Sigma β)) : SizeOf.sizeOf (List.kerase x xs) ≤ SizeOf.sizeOf xs := by
  /-
    α : Type u
    β : α → Type v
    inst✝¹ : DecidableEq α
    inst✝ : SizeOf (Sigma β)
    x : α
    xs : List (Sigma β)
    ⊢ LE.le (SizeOf.sizeOf (List.kerase x xs)) (SizeOf.sizeOf xs)
  -/
  simp only [SizeOf.sizeOf, _sizeOf_1]
  /-
    α : Type u
    β : α → Type v
    inst✝¹ : DecidableEq α
    inst✝ : SizeOf (Sigma β)
    x : α
    xs : List (Sigma β)
    ⊢ LE.le (List.rec 1 (fun head tail tail_ih => HAdd.hAdd (HAdd.hAdd 1 (SizeOf.s …
  -/
  induction' xs with y ys
    /-
      case nil
      α : Type u
      β : α → Type v
      inst✝¹ : DecidableEq α
      inst✝ : SizeOf (Sigma β)
      x : α
      ⊢ LE.le (List.rec 1 (fun head tail tail_ih => HAdd.hAdd (HAdd.hAdd 1 (SizeOf.s …
    -/
  · simp
    /-
      🎉 no goals
    -/
    /-
      case cons
      α : Type u
      β : α → Type v
      inst✝¹ : DecidableEq α
      inst✝ : SizeOf (Sigma β)
      x : α
      y : Sigma β
      ys : List (Sigma β)
      tail_ih✝ : LE.le (List.rec 1 (fun head tail tail_ih => HAdd.hAdd (HAdd.hAdd 1  …
      ⊢ LE.le (List.rec 1 (fun head tail tail_ih => HAdd.hAdd (HAdd.hAdd 1 (SizeOf.s …
    -/
                         /-
                           🎉 no goals
                         -/
  · by_cases x = y.1 <;> simp [*]
                         /-
                           🎉 no goals
                         -/


/-- Insert the pair `⟨a, b⟩` and erase the first pair with the key `a`. -/
def kinsert (a : α) (b : β a) (l : List (Sigma β)) : List (Sigma β) :=
  ⟨a, b⟩ :: kerase a l


@[simp]
theorem kinsert_def {a} {b : β a} {l : List (Sigma β)} : kinsert a b l = ⟨a, b⟩ :: kerase a l :=
  rfl


theorem mem_keys_kinsert {a a'} {b' : β a'} {l : List (Sigma β)} :
                                                           /-
                                                             α : Type u
                                                             β : α → Type v
                                                             inst✝ : DecidableEq α
                                                             a a' : α
                                                             b' : β a'
                                                             l : List (Sigma β)
                                                             ⊢ Iff (Membership.mem (List.kinsert a' b' l).keys a) (Or (Eq a a') (Membership …
                                                           -/
                                                                                   /-
                                                                                     🎉 no goals
                                                                                   -/
    a ∈ (kinsert a' b' l).keys ↔ a = a' ∨ a ∈ l.keys := by by_cases h : a = a' <;> simp [h]
                                                                                   /-
                                                                                     🎉 no goals
                                                                                   -/


theorem kinsert_nodupKeys (a) (b : β a) {l : List (Sigma β)} (nd : l.NodupKeys) :
    (kinsert a b l).NodupKeys :=
  nodupKeys_cons.mpr ⟨not_mem_keys_kerase a nd, nd.kerase a⟩


theorem Perm.kinsert {a} {b : β a} {l₁ l₂ : List (Sigma β)} (nd₁ : l₁.NodupKeys) (p : l₁ ~ l₂) :
    kinsert a b l₁ ~ kinsert a b l₂ :=
  (p.kerase nd₁).cons _


theorem dlookup_kinsert {a} {b : β a} (l : List (Sigma β)) :
    dlookup a (kinsert a b l) = some b := by
  /-
    α : Type u
    β : α → Type v
    inst✝ : DecidableEq α
    a : α
    b : β a
    l : List (Sigma β)
    ⊢ Eq (List.dlookup a (List.kinsert a b l)) (Option.some b)
  -/
  simp only [kinsert, dlookup_cons_eq]
  /-
    🎉 no goals
  -/


theorem dlookup_kinsert_ne {a a'} {b' : β a'} {l : List (Sigma β)} (h : a ≠ a') :
                                                    /-
                                                      α : Type u
                                                      β : α → Type v
                                                      inst✝ : DecidableEq α
                                                      a a' : α
                                                      b' : β a'
                                                      l : List (Sigma β)
                                                      h : Ne a a'
                                                      ⊢ Eq (List.dlookup a (List.kinsert a' b' l)) (List.dlookup a l)
                                                    -/
    dlookup a (kinsert a' b' l) = dlookup a l := by simp [h]
                                                    /-
                                                      🎉 no goals
                                                    -/


/-- Finds the first entry with a given key `a` and returns its value (as an `Option` because there
might be no entry with key `a`) alongside with the rest of the entries. -/
def kextract (a : α) : List (Sigma β) → Option (β a) × List (Sigma β)
  | [] => (none, [])
  | s :: l =>
    if h : s.1 = a then (some (Eq.recOn h s.2), l)
    else
      let (b', l') := kextract a l
      (b', s :: l')


@[simp]
theorem kextract_eq_dlookup_kerase (a : α) :
    ∀ l : List (Sigma β), kextract a l = (dlookup a l, kerase a l)
  | [] => rfl
  | ⟨a', b⟩ :: l => by
    /-
      α : Type u
      β : α → Type v
      inst✝ : DecidableEq α
      a a' : α
      b : β a'
      l : List (Sigma β)
      ⊢ Eq (List.kextract a (List.cons ⟨a', b⟩ l)) { fst := List.dlookup a (List.con …
    -/
    simp only [kextract]; dsimp; split_ifs with h
      /-
        case pos
        α : Type u
        β : α → Type v
        inst✝ : DecidableEq α
        a a' : α
        b : β a'
        l : List (Sigma β)
        h : Eq a' a
        ⊢ Eq { fst := Option.some (Eq.rec b ⋯), snd := l } { fst := List.dlookup a (Li …
      -/
    · subst a'
      /-
        case pos
        α : Type u
        β : α → Type v
        inst✝ : DecidableEq α
        a : α
        l : List (Sigma β)
        b : β a
        ⊢ Eq { fst := Option.some (Eq.rec b ⋯), snd := l } { fst := List.dlookup a (Li …
      -/
      simp [kerase]
      /-
        🎉 no goals
      -/
      /-
        case neg
        α : Type u
        β : α → Type v
        inst✝ : DecidableEq α
        a a' : α
        b : β a'
        l : List (Sigma β)
        h : Not (Eq a' a)
        ⊢ Eq { fst := (List.kextract a l).1, snd := List.cons ⟨a', b⟩ (List.kextract a …
      -/
    · simp [kextract, Ne.symm h, kextract_eq_dlookup_kerase a l, kerase]
      /-
        🎉 no goals
      -/


/-- Remove entries with duplicate keys from `l : List (Sigma β)`. -/
def dedupKeys : List (Sigma β) → List (Sigma β) :=
  List.foldr (fun x => kinsert x.1 x.2) []


theorem dedupKeys_cons {x : Sigma β} (l : List (Sigma β)) :
    dedupKeys (x :: l) = kinsert x.1 x.2 (dedupKeys l) :=
  rfl



theorem nodupKeys_dedupKeys (l : List (Sigma β)) : NodupKeys (dedupKeys l) := by
  /-
    α : Type u
    β : α → Type v
    inst✝ : DecidableEq α
    l : List (Sigma β)
    ⊢ l.dedupKeys.NodupKeys
  -/
  dsimp [dedupKeys]
  /-
    α : Type u
    β : α → Type v
    inst✝ : DecidableEq α
    l : List (Sigma β)
    ⊢ (List.foldr (fun x => List.kinsert x.fst x.snd) List.nil l).NodupKeys
  -/
  generalize hl : nil = l'
  have : NodupKeys l' := by
    rw [← hl]
    apply nodup_nil
  /-
    α : Type u
    β : α → Type v
    inst✝ : DecidableEq α
    l : List (Sigma β)
    l' : List (Sigma β)
    hl : Eq List.nil l'
    this : l'.NodupKeys
    ⊢ (List.foldr (fun x => List.kinsert x.fst x.snd) l' l).NodupKeys
  -/
  clear hl
  /-
    α : Type u
    β : α → Type v
    inst✝ : DecidableEq α
    l : List (Sigma β)
    l' : List (Sigma β)
    this : l'.NodupKeys
    ⊢ (List.foldr (fun x => List.kinsert x.fst x.snd) l' l).NodupKeys
  -/
  induction' l with x xs l_ih
    /-
      case nil
      α : Type u
      β : α → Type v
      inst✝ : DecidableEq α
      l' : List (Sigma β)
      this : l'.NodupKeys
      ⊢ (List.foldr (fun x => List.kinsert x.fst x.snd) l' List.nil).NodupKeys
    -/
  · apply this
    /-
      🎉 no goals
    -/
    /-
      case cons
      α : Type u
      β : α → Type v
      inst✝ : DecidableEq α
      l' : List (Sigma β)
      this : l'.NodupKeys
      x : Sigma β
      xs : List (Sigma β)
      l_ih : (List.foldr (fun x => List.kinsert x.fst x.snd) l' xs).NodupKeys
      ⊢ (List.foldr (fun x => List.kinsert x.fst x.snd) l' (List.cons x xs)).NodupKeys
    -/
  · cases x
    /-
      case cons.mk
      α : Type u
      β : α → Type v
      inst✝ : DecidableEq α
      l' : List (Sigma β)
      this : l'.NodupKeys
      xs : List (Sigma β)
      l_ih : (List.foldr (fun x => List.kinsert x.fst x.snd) l' xs).NodupKeys
      fst✝ : α
      snd✝ : β fst✝
      ⊢ (List.foldr (fun x => List.kinsert x.fst x.snd) l' (List.cons ⟨fst✝, snd✝⟩ x …
    -/
    simp only [foldr_cons, kinsert_def, nodupKeys_cons, ne_eq, not_true]
    /-
      case cons.mk
      α : Type u
      β : α → Type v
      inst✝ : DecidableEq α
      l' : List (Sigma β)
      this : l'.NodupKeys
      xs : List (Sigma β)
      l_ih : (List.foldr (fun x => List.kinsert x.fst x.snd) l' xs).NodupKeys
      fst✝ : α
      snd✝ : β fst✝
      ⊢ And (Not (Membership.mem (List.kerase fst✝ (List.foldr (fun x => List.kinser …
    -/
    constructor
      /-
        case cons.mk.left
        α : Type u
        β : α → Type v
        inst✝ : DecidableEq α
        l' : List (Sigma β)
        this : l'.NodupKeys
        xs : List (Sigma β)
        l_ih : (List.foldr (fun x => List.kinsert x.fst x.snd) l' xs).NodupKeys
        fst✝ : α
        snd✝ : β fst✝
        ⊢ Not (Membership.mem (List.kerase fst✝ (List.foldr (fun x => List.kinsert x.f …
      -/
    · simp only [keys_kerase]
      /-
        case cons.mk.left
        α : Type u
        β : α → Type v
        inst✝ : DecidableEq α
        l' : List (Sigma β)
        this : l'.NodupKeys
        xs : List (Sigma β)
        l_ih : (List.foldr (fun x => List.kinsert x.fst x.snd) l' xs).NodupKeys
        fst✝ : α
        snd✝ : β fst✝
        ⊢ Not (Membership.mem ((List.foldr (fun x => List.kinsert x.fst x.snd) l' xs). …
      -/
      apply l_ih.not_mem_erase
      /-
        🎉 no goals
      -/
      /-
        case cons.mk.right
        α : Type u
        β : α → Type v
        inst✝ : DecidableEq α
        l' : List (Sigma β)
        this : l'.NodupKeys
        xs : List (Sigma β)
        l_ih : (List.foldr (fun x => List.kinsert x.fst x.snd) l' xs).NodupKeys
        fst✝ : α
        snd✝ : β fst✝
        ⊢ (List.kerase fst✝ (List.foldr (fun x => List.kinsert x.fst x.snd) l' xs)).No …
      -/
    · exact l_ih.kerase _
      /-
        🎉 no goals
      -/


theorem dlookup_dedupKeys (a : α) (l : List (Sigma β)) : dlookup a (dedupKeys l) = dlookup a l := by
  /-
    α : Type u
    β : α → Type v
    inst✝ : DecidableEq α
    a : α
    l : List (Sigma β)
    ⊢ Eq (List.dlookup a l.dedupKeys) (List.dlookup a l)
  -/
  induction' l with l_hd _ l_ih
    /-
      case nil
      α : Type u
      β : α → Type v
      inst✝ : DecidableEq α
      a : α
      ⊢ Eq (List.dlookup a List.nil.dedupKeys) (List.dlookup a List.nil)
    -/
  · rfl
    /-
      🎉 no goals
    -/
  /-
    case cons
    α : Type u
    β : α → Type v
    inst✝ : DecidableEq α
    a : α
    l_hd : Sigma β
    tail✝ : List (Sigma β)
    l_ih : Eq (List.dlookup a tail✝.dedupKeys) (List.dlookup a tail✝)
    ⊢ Eq (List.dlookup a (List.cons l_hd tail✝).dedupKeys) (List.dlookup a (List.c …
  -/
  cases' l_hd with a' b
  /-
    case cons.mk
    α : Type u
    β : α → Type v
    inst✝ : DecidableEq α
    a : α
    tail✝ : List (Sigma β)
    l_ih : Eq (List.dlookup a tail✝.dedupKeys) (List.dlookup a tail✝)
    a' : α
    b : β a'
    ⊢ Eq (List.dlookup a (List.cons ⟨a', b⟩ tail✝).dedupKeys) (List.dlookup a (Lis …
  -/
  by_cases h : a = a'
    /-
      case pos
      α : Type u
      β : α → Type v
      inst✝ : DecidableEq α
      a : α
      tail✝ : List (Sigma β)
      l_ih : Eq (List.dlookup a tail✝.dedupKeys) (List.dlookup a tail✝)
      a' : α
      b : β a'
      h : Eq a a'
      ⊢ Eq (List.dlookup a (List.cons ⟨a', b⟩ tail✝).dedupKeys) (List.dlookup a (Lis …
    -/
  · subst a'
    /-
      case pos
      α : Type u
      β : α → Type v
      inst✝ : DecidableEq α
      a : α
      tail✝ : List (Sigma β)
      l_ih : Eq (List.dlookup a tail✝.dedupKeys) (List.dlookup a tail✝)
      b : β a
      ⊢ Eq (List.dlookup a (List.cons ⟨a, b⟩ tail✝).dedupKeys) (List.dlookup a (List …
    -/
    rw [dedupKeys_cons, dlookup_kinsert, dlookup_cons_eq]
    /-
      🎉 no goals
    -/
    /-
      case neg
      α : Type u
      β : α → Type v
      inst✝ : DecidableEq α
      a : α
      tail✝ : List (Sigma β)
      l_ih : Eq (List.dlookup a tail✝.dedupKeys) (List.dlookup a tail✝)
      a' : α
      b : β a'
      h : Not (Eq a a')
      ⊢ Eq (List.dlookup a (List.cons ⟨a', b⟩ tail✝).dedupKeys) (List.dlookup a (Lis …
    -/
  · rw [dedupKeys_cons, dlookup_kinsert_ne h, l_ih, dlookup_cons_ne]
    /-
      case neg.a
      α : Type u
      β : α → Type v
      inst✝ : DecidableEq α
      a : α
      tail✝ : List (Sigma β)
      l_ih : Eq (List.dlookup a tail✝.dedupKeys) (List.dlookup a tail✝)
      a' : α
      b : β a'
      h : Not (Eq a a')
      ⊢ Ne a ⟨a', b⟩.fst
    -/
    exact h
    /-
      🎉 no goals
    -/


theorem sizeOf_dedupKeys [SizeOf (Sigma β)]
    (xs : List (Sigma β)) : SizeOf.sizeOf (dedupKeys xs) ≤ SizeOf.sizeOf xs := by
  /-
    α : Type u
    β : α → Type v
    inst✝¹ : DecidableEq α
    inst✝ : SizeOf (Sigma β)
    xs : List (Sigma β)
    ⊢ LE.le (SizeOf.sizeOf xs.dedupKeys) (SizeOf.sizeOf xs)
  -/
  simp only [SizeOf.sizeOf, _sizeOf_1]
  /-
    α : Type u
    β : α → Type v
    inst✝¹ : DecidableEq α
    inst✝ : SizeOf (Sigma β)
    xs : List (Sigma β)
    ⊢ LE.le (List.rec 1 (fun head tail tail_ih => HAdd.hAdd (HAdd.hAdd 1 (SizeOf.s …
  -/
  induction' xs with x xs
    /-
      case nil
      α : Type u
      β : α → Type v
      inst✝¹ : DecidableEq α
      inst✝ : SizeOf (Sigma β)
      ⊢ LE.le (List.rec 1 (fun head tail tail_ih => HAdd.hAdd (HAdd.hAdd 1 (SizeOf.s …
    -/
  · simp [dedupKeys]
    /-
      🎉 no goals
    -/
    /-
      case cons
      α : Type u
      β : α → Type v
      inst✝¹ : DecidableEq α
      inst✝ : SizeOf (Sigma β)
      x : Sigma β
      xs : List (Sigma β)
      tail_ih✝ : LE.le (List.rec 1 (fun head tail tail_ih => HAdd.hAdd (HAdd.hAdd 1  …
      ⊢ LE.le (List.rec 1 (fun head tail tail_ih => HAdd.hAdd (HAdd.hAdd 1 (SizeOf.s …
    -/
  · simp only [dedupKeys_cons, kinsert_def, Nat.add_le_add_iff_left, Sigma.eta]
    /-
      case cons
      α : Type u
      β : α → Type v
      inst✝¹ : DecidableEq α
      inst✝ : SizeOf (Sigma β)
      x : Sigma β
      xs : List (Sigma β)
      tail_ih✝ : LE.le (List.rec 1 (fun head tail tail_ih => HAdd.hAdd (HAdd.hAdd 1  …
      ⊢ LE.le (List.rec 1 (fun head tail tail_ih => HAdd.hAdd (HAdd.hAdd 1 (SizeOf.s …
    -/
    trans
      /-
        α : Type u
        β : α → Type v
        inst✝¹ : DecidableEq α
        inst✝ : SizeOf (Sigma β)
        x : Sigma β
        xs : List (Sigma β)
        tail_ih✝ : LE.le (List.rec 1 (fun head tail tail_ih => HAdd.hAdd (HAdd.hAdd 1  …
        ⊢ LE.le (List.rec 1 (fun head tail tail_ih => HAdd.hAdd (HAdd.hAdd 1 (SizeOf.s …
      -/
    · apply sizeOf_kerase
      /-
        🎉 no goals
      -/
      /-
        α : Type u
        β : α → Type v
        inst✝¹ : DecidableEq α
        inst✝ : SizeOf (Sigma β)
        x : Sigma β
        xs : List (Sigma β)
        tail_ih✝ : LE.le (List.rec 1 (fun head tail tail_ih => HAdd.hAdd (HAdd.hAdd 1  …
        ⊢ LE.le (SizeOf.sizeOf xs.dedupKeys) (List.rec 1 (fun head tail tail_ih => HAd …
      -/
    · assumption
      /-
        🎉 no goals
      -/


/-- `kunion l₁ l₂` is the append to l₁ of l₂ after, for each key in l₁, the
first matching pair in l₂ is erased. -/
def kunion : List (Sigma β) → List (Sigma β) → List (Sigma β)
  | [], l₂ => l₂
  | s :: l₁, l₂ => s :: kunion l₁ (kerase s.1 l₂)


@[simp]
theorem nil_kunion {l : List (Sigma β)} : kunion [] l = l :=
  rfl


@[simp]
theorem kunion_nil : ∀ {l : List (Sigma β)}, kunion l [] = l
  | [] => rfl
                 /-
                   α : Type u
                   β : α → Type v
                   inst✝ : DecidableEq α
                   head✝ : Sigma β
                   l : List (Sigma β)
                   ⊢ Eq ((List.cons head✝ l).kunion List.nil) (List.cons head✝ l)
                 -/
  | _ :: l => by rw [kunion, kerase_nil, kunion_nil]
                 /-
                   🎉 no goals
                 -/


@[simp]
theorem kunion_cons {s} {l₁ l₂ : List (Sigma β)} :
    kunion (s :: l₁) l₂ = s :: kunion l₁ (kerase s.1 l₂) :=
  rfl


@[simp]
theorem mem_keys_kunion {a} {l₁ l₂ : List (Sigma β)} :
    a ∈ (kunion l₁ l₂).keys ↔ a ∈ l₁.keys ∨ a ∈ l₂.keys := by
  induction l₁ generalizing l₂ with
  | nil => simp
  | cons s l₁ ih => by_cases h : a = s.1 <;> [simp [h]; simp [h, ih]]


@[simp]
theorem kunion_kerase {a} :
    ∀ {l₁ l₂ : List (Sigma β)}, kunion (kerase a l₁) (kerase a l₂) = kerase a (kunion l₁ l₂)
  | [], _ => rfl
                    /-
                      α : Type u
                      β : α → Type v
                      inst✝ : DecidableEq α
                      a : α
                      s : Sigma β
                      tail✝ : List (Sigma β)
                      l : List (Sigma β)
                      ⊢ Eq ((List.kerase a (List.cons s tail✝)).kunion (List.kerase a l)) (List.kera …
                    -/
                                             /-
                                               🎉 no goals
                                             -/
  | s :: _, l => by by_cases h : a = s.1 <;> simp [h, kerase_comm a s.1 l, kunion_kerase]
                                             /-
                                               🎉 no goals
                                             -/


theorem NodupKeys.kunion (nd₁ : l₁.NodupKeys) (nd₂ : l₂.NodupKeys) : (kunion l₁ l₂).NodupKeys := by
  induction l₁ generalizing l₂ with
  | nil => simp only [nil_kunion, nd₂]
  | cons s l₁ ih =>
    simp? at nd₁ says simp only [nodupKeys_cons] at nd₁
    simp [not_or, nd₁.1, nd₂, ih nd₁.2 (nd₂.kerase s.1)]


theorem Perm.kunion_right {l₁ l₂ : List (Sigma β)} (p : l₁ ~ l₂) (l) :
    kunion l₁ l ~ kunion l₂ l := by
  induction p generalizing l with
  | nil => rfl
  | cons hd _ ih =>
    simp [ih (List.kerase _ _), Perm.cons]
  | swap s₁ s₂ l => simp [kerase_comm, Perm.swap]
  | trans _ _ ih₁₂ ih₂₃ => exact Perm.trans (ih₁₂ l) (ih₂₃ l)


theorem Perm.kunion_left :
    ∀ (l) {l₁ l₂ : List (Sigma β)}, l₁.NodupKeys → l₁ ~ l₂ → kunion l l₁ ~ kunion l l₂
  | [], _, _, _, p => p
  | s :: l, _, _, nd₁, p => ((p.kerase nd₁).kunion_left l <| nd₁.kerase s.1).cons s


theorem Perm.kunion {l₁ l₂ l₃ l₄ : List (Sigma β)} (nd₃ : l₃.NodupKeys) (p₁₂ : l₁ ~ l₂)
    (p₃₄ : l₃ ~ l₄) : kunion l₁ l₃ ~ kunion l₂ l₄ :=
  (p₁₂.kunion_right l₃).trans (p₃₄.kunion_left l₂ nd₃)


@[simp]
theorem dlookup_kunion_left {a} {l₁ l₂ : List (Sigma β)} (h : a ∈ l₁.keys) :
    dlookup a (kunion l₁ l₂) = dlookup a l₁ := by
  /-
    α : Type u
    β : α → Type v
    inst✝ : DecidableEq α
    a : α
    l₁ l₂ : List (Sigma β)
    h : Membership.mem l₁.keys a
    ⊢ Eq (List.dlookup a (l₁.kunion l₂)) (List.dlookup a l₁)
  -/
                                                /-
                                                  🎉 no goals
                                                -/
  induction' l₁ with s _ ih generalizing l₂ <;> simp at h; cases' h with h h <;> cases' s with a'
    /-
      case cons.inl.mk
      α : Type u
      β : α → Type v
      inst✝ : DecidableEq α
      a : α
      tail✝ : List (Sigma β)
      ih : ∀ {l₂ : List (Sigma β)}, Membership.mem tail✝.keys a → Eq (List.dlookup a …
      l₂ : List (Sigma β)
      a' : α
      snd✝ : β a'
      h : Eq a ⟨a', snd✝⟩.fst
      ⊢ Eq (List.dlookup a ((List.cons ⟨a', snd✝⟩ tail✝).kunion l₂)) (List.dlookup a …
    -/
  · subst h
    /-
      case cons.inl.mk
      α : Type u
      β : α → Type v
      inst✝ : DecidableEq α
      a : α
      tail✝ : List (Sigma β)
      ih : ∀ {l₂ : List (Sigma β)}, Membership.mem tail✝.keys a → Eq (List.dlookup a …
      l₂ : List (Sigma β)
      snd✝ : β a
      ⊢ Eq (List.dlookup a ((List.cons ⟨a, snd✝⟩ tail✝).kunion l₂)) (List.dlookup a  …
    -/
    simp
    /-
      🎉 no goals
    -/
    /-
      case cons.inr.mk
      α : Type u
      β : α → Type v
      inst✝ : DecidableEq α
      a : α
      tail✝ : List (Sigma β)
      ih : ∀ {l₂ : List (Sigma β)}, Membership.mem tail✝.keys a → Eq (List.dlookup a …
      l₂ : List (Sigma β)
      h : Membership.mem tail✝.keys a
      a' : α
      snd✝ : β a'
      ⊢ Eq (List.dlookup a ((List.cons ⟨a', snd✝⟩ tail✝).kunion l₂)) (List.dlookup a …
    -/
  · rw [kunion_cons]
    /-
      case cons.inr.mk
      α : Type u
      β : α → Type v
      inst✝ : DecidableEq α
      a : α
      tail✝ : List (Sigma β)
      ih : ∀ {l₂ : List (Sigma β)}, Membership.mem tail✝.keys a → Eq (List.dlookup a …
      l₂ : List (Sigma β)
      h : Membership.mem tail✝.keys a
      a' : α
      snd✝ : β a'
      ⊢ Eq (List.dlookup a (List.cons ⟨a', snd✝⟩ (tail✝.kunion (List.kerase ⟨a', snd …
    -/
    by_cases h' : a = a'
      /-
        case pos
        α : Type u
        β : α → Type v
        inst✝ : DecidableEq α
        a : α
        tail✝ : List (Sigma β)
        ih : ∀ {l₂ : List (Sigma β)}, Membership.mem tail✝.keys a → Eq (List.dlookup a …
        l₂ : List (Sigma β)
        h : Membership.mem tail✝.keys a
        a' : α
        snd✝ : β a'
        h' : Eq a a'
        ⊢ Eq (List.dlookup a (List.cons ⟨a', snd✝⟩ (tail✝.kunion (List.kerase ⟨a', snd …
      -/
    · subst h'
      /-
        case pos
        α : Type u
        β : α → Type v
        inst✝ : DecidableEq α
        a : α
        tail✝ : List (Sigma β)
        ih : ∀ {l₂ : List (Sigma β)}, Membership.mem tail✝.keys a → Eq (List.dlookup a …
        l₂ : List (Sigma β)
        h : Membership.mem tail✝.keys a
        snd✝ : β a
        ⊢ Eq (List.dlookup a (List.cons ⟨a, snd✝⟩ (tail✝.kunion (List.kerase ⟨a, snd✝⟩ …
      -/
      simp
      /-
        🎉 no goals
      -/
      /-
        case neg
        α : Type u
        β : α → Type v
        inst✝ : DecidableEq α
        a : α
        tail✝ : List (Sigma β)
        ih : ∀ {l₂ : List (Sigma β)}, Membership.mem tail✝.keys a → Eq (List.dlookup a …
        l₂ : List (Sigma β)
        h : Membership.mem tail✝.keys a
        a' : α
        snd✝ : β a'
        h' : Not (Eq a a')
        ⊢ Eq (List.dlookup a (List.cons ⟨a', snd✝⟩ (tail✝.kunion (List.kerase ⟨a', snd …
      -/
    · simp [h', ih h]
      /-
        🎉 no goals
      -/


@[simp]
theorem dlookup_kunion_right {a} {l₁ l₂ : List (Sigma β)} (h : a ∉ l₁.keys) :
    dlookup a (kunion l₁ l₂) = dlookup a l₂ := by
  induction l₁ generalizing l₂ with
  | nil => simp
  | cons _ _ ih => simp_all [not_or]


theorem mem_dlookup_kunion {a} {b : β a} {l₁ l₂ : List (Sigma β)} :
    b ∈ dlookup a (kunion l₁ l₂) ↔ b ∈ dlookup a l₁ ∨ a ∉ l₁.keys ∧ b ∈ dlookup a l₂ := by
  induction l₁ generalizing l₂ with
  | nil => simp
  | cons s _ ih =>
    cases' s with a'
    by_cases h₁ : a = a'
    · subst h₁
      simp
    · let h₂ := @ih (kerase a' l₂)
      simp? [h₁] at h₂ says
        simp only [Option.mem_def, ne_eq, h₁, not_false_eq_true, dlookup_kerase_ne] at h₂
      simp [h₁, h₂]


@[simp]
theorem dlookup_kunion_eq_some {a} {b : β a} {l₁ l₂ : List (Sigma β)} :
    dlookup a (kunion l₁ l₂) = some b ↔
      dlookup a l₁ = some b ∨ a ∉ l₁.keys ∧ dlookup a l₂ = some b :=
  mem_dlookup_kunion


theorem mem_dlookup_kunion_middle {a} {b : β a} {l₁ l₂ l₃ : List (Sigma β)}
    (h₁ : b ∈ dlookup a (kunion l₁ l₃)) (h₂ : a ∉ keys l₂) :
    b ∈ dlookup a (kunion (kunion l₁ l₂) l₃) :=
  match mem_dlookup_kunion.mp h₁ with
  | Or.inl h => mem_dlookup_kunion.mpr (Or.inl (mem_dlookup_kunion.mpr (Or.inl h)))
  | Or.inr h => mem_dlookup_kunion.mpr <| Or.inr ⟨mt mem_keys_kunion.mp (not_or.mpr ⟨h.1, h₂⟩), h.2⟩


