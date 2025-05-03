/-- `AList β` is a key-value map stored as a `List` (i.e. a linked list).
  It is a wrapper around certain `List` functions with the added constraint
  that the list have unique keys. -/
structure AList (β : α → Type v) : Type max u v where
  /-- The underlying `List` of an `AList` -/
  entries : List (Sigma β)
  /-- There are no duplicate keys in `entries` -/
  nodupKeys : entries.NodupKeys


/-- Given `l : List (Sigma β)`, create a term of type `AList β` by removing
entries with duplicate keys. -/
def List.toAList [DecidableEq α] {β : α → Type v} (l : List (Sigma β)) : AList β where
  entries := _
  nodupKeys := nodupKeys_dedupKeys l


@[ext]
theorem ext : ∀ {s t : AList β}, s.entries = t.entries → s = t
                               /-
                                 α : Type u
                                 β : α → Type v
                                 l₁ : List (Sigma β)
                                 h₁ : l₁.NodupKeys
                                 l₂ : List (Sigma β)
                                 nodupKeys✝ : l₂.NodupKeys
                                 H : Eq { entries := l₁, nodupKeys := h₁ }.entries { entries := l₂, nodupKeys : …
                                 ⊢ Eq { entries := l₁, nodupKeys := h₁ } { entries := l₂, nodupKeys := nodupKey …
                               -/
  | ⟨l₁, h₁⟩, ⟨l₂, _⟩, H => by congr
                               /-
                                 🎉 no goals
                               -/


instance [DecidableEq α] [∀ a, DecidableEq (β a)] : DecidableEq (AList β) := fun xs ys => by
  /-
    α : Type u
    β : α → Type v
    inst✝¹ : DecidableEq α
    inst✝ : (a : α) → DecidableEq (β a)
    xs ys : AList β
    ⊢ Decidable (Eq xs ys)
  -/
  rw [AList.ext_iff]; infer_instance
                      /-
                        🎉 no goals
                      -/


/-- The list of keys of an association list. -/
def keys (s : AList β) : List α :=
  s.entries.keys


theorem keys_nodup (s : AList β) : s.keys.Nodup :=
  s.nodupKeys


/-- The predicate `a ∈ s` means that `s` has a value associated to the key `a`. -/
instance : Membership α (AList β) :=
  ⟨fun s a => a ∈ s.keys⟩


theorem mem_keys {a : α} {s : AList β} : a ∈ s ↔ a ∈ s.keys :=
  Iff.rfl


theorem mem_of_perm {a : α} {s₁ s₂ : AList β} (p : s₁.entries ~ s₂.entries) : a ∈ s₁ ↔ a ∈ s₂ :=
  (p.map Sigma.fst).mem_iff


/-- The empty association list. -/
instance : EmptyCollection (AList β) :=
  ⟨⟨[], nodupKeys_nil⟩⟩


instance : Inhabited (AList β) :=
  ⟨∅⟩


@[simp]
theorem not_mem_empty (a : α) : a ∉ (∅ : AList β) :=
  not_mem_nil a


@[simp]
theorem empty_entries : (∅ : AList β).entries = [] :=
  rfl


@[simp]
theorem keys_empty : (∅ : AList β).keys = [] :=
  rfl


/-- The singleton association list. -/
def singleton (a : α) (b : β a) : AList β :=
  ⟨[⟨a, b⟩], nodupKeys_singleton _⟩


@[simp]
theorem singleton_entries (a : α) (b : β a) : (singleton a b).entries = [Sigma.mk a b] :=
  rfl


@[simp]
theorem keys_singleton (a : α) (b : β a) : (singleton a b).keys = [a] :=
  rfl


/-- Look up the value associated to a key in an association list. -/
def lookup (a : α) (s : AList β) : Option (β a) :=
  s.entries.dlookup a


@[simp]
theorem lookup_empty (a) : lookup a (∅ : AList β) = none :=
  rfl


theorem lookup_isSome {a : α} {s : AList β} : (s.lookup a).isSome ↔ a ∈ s :=
  dlookup_isSome


theorem lookup_eq_none {a : α} {s : AList β} : lookup a s = none ↔ a ∉ s :=
  dlookup_eq_none


theorem mem_lookup_iff {a : α} {b : β a} {s : AList β} :
    b ∈ lookup a s ↔ Sigma.mk a b ∈ s.entries :=
  mem_dlookup_iff s.nodupKeys


theorem perm_lookup {a : α} {s₁ s₂ : AList β} (p : s₁.entries ~ s₂.entries) :
    s₁.lookup a = s₂.lookup a :=
  perm_dlookup _ s₁.nodupKeys s₂.nodupKeys p


instance (a : α) (s : AList β) : Decidable (a ∈ s) :=
  decidable_of_iff _ lookup_isSome


theorem keys_subset_keys_of_entries_subset_entries
    {s₁ s₂ : AList β} (h : s₁.entries ⊆ s₂.entries) : s₁.keys ⊆ s₂.keys := by
  /-
    α : Type u
    β : α → Type v
    s₁ s₂ : AList β
    h : HasSubset.Subset s₁.entries s₂.entries
    ⊢ HasSubset.Subset s₁.keys s₂.keys
  -/
  intro k hk
  /-
    α : Type u
    β : α → Type v
    s₁ s₂ : AList β
    h : HasSubset.Subset s₁.entries s₂.entries
    k : α
    hk : Membership.mem s₁.keys k
    ⊢ Membership.mem s₂.keys k
  -/
  letI : DecidableEq α := Classical.decEq α
  /-
    α : Type u
    β : α → Type v
    s₁ s₂ : AList β
    h : HasSubset.Subset s₁.entries s₂.entries
    k : α
    hk : Membership.mem s₁.keys k
    this : DecidableEq α := Classical.decEq α
    ⊢ Membership.mem s₂.keys k
  -/
  have := h (mem_lookup_iff.1 (Option.get_mem (lookup_isSome.2 hk)))
  /-
    α : Type u
    β : α → Type v
    s₁ s₂ : AList β
    h : HasSubset.Subset s₁.entries s₂.entries
    k : α
    hk : Membership.mem s₁.keys k
    this✝ : DecidableEq α := Classical.decEq α
    this : Membership.mem s₂.entries ⟨k, (AList.lookup k s₁).get ⋯⟩
    ⊢ Membership.mem s₂.keys k
  -/
  rw [← mem_lookup_iff, Option.mem_def] at this
  /-
    α : Type u
    β : α → Type v
    s₁ s₂ : AList β
    h : HasSubset.Subset s₁.entries s₂.entries
    k : α
    hk : Membership.mem s₁.keys k
    this✝ : DecidableEq α := Classical.decEq α
    this : Eq (AList.lookup k s₂) (Option.some ((AList.lookup k s₁).get ⋯))
    ⊢ Membership.mem s₂.keys k
  -/
  rw [← mem_keys, ← lookup_isSome, this]
  /-
    α : Type u
    β : α → Type v
    s₁ s₂ : AList β
    h : HasSubset.Subset s₁.entries s₂.entries
    k : α
    hk : Membership.mem s₁.keys k
    this✝ : DecidableEq α := Classical.decEq α
    this : Eq (AList.lookup k s₂) (Option.some ((AList.lookup k s₁).get ⋯))
    ⊢ Eq (Option.some ((AList.lookup k s₁).get ⋯)).isSome Bool.true
  -/
  exact Option.isSome_some
  /-
    🎉 no goals
  -/


/-- Replace a key with a given value in an association list.
  If the key is not present it does nothing. -/
def replace (a : α) (b : β a) (s : AList β) : AList β :=
  ⟨kreplace a b s.entries, (kreplace_nodupKeys a b).2 s.nodupKeys⟩


@[simp]
theorem keys_replace (a : α) (b : β a) (s : AList β) : (replace a b s).keys = s.keys :=
  keys_kreplace _ _ _


@[simp]
theorem mem_replace {a a' : α} {b : β a} {s : AList β} : a' ∈ replace a b s ↔ a' ∈ s := by
  /-
    α : Type u
    β : α → Type v
    inst✝ : DecidableEq α
    a a' : α
    b : β a
    s : AList β
    ⊢ Iff (Membership.mem (AList.replace a b s) a') (Membership.mem s a')
  -/
  rw [mem_keys, keys_replace, ← mem_keys]
  /-
    🎉 no goals
  -/


theorem perm_replace {a : α} {b : β a} {s₁ s₂ : AList β} :
    s₁.entries ~ s₂.entries → (replace a b s₁).entries ~ (replace a b s₂).entries :=
  Perm.kreplace s₁.nodupKeys


/-- Fold a function over the key-value pairs in the map. -/
def foldl {δ : Type w} (f : δ → ∀ a, β a → δ) (d : δ) (m : AList β) : δ :=
  m.entries.foldl (fun r a => f r a.1 a.2) d


/-- Erase a key from the map. If the key is not present, do nothing. -/
def erase (a : α) (s : AList β) : AList β :=
  ⟨s.entries.kerase a, s.nodupKeys.kerase a⟩


@[simp]
theorem keys_erase (a : α) (s : AList β) : (erase a s).keys = s.keys.erase a :=
  keys_kerase


@[simp]
theorem mem_erase {a a' : α} {s : AList β} : a' ∈ erase a s ↔ a' ≠ a ∧ a' ∈ s := by
  /-
    α : Type u
    β : α → Type v
    inst✝ : DecidableEq α
    a a' : α
    s : AList β
    ⊢ Iff (Membership.mem (AList.erase a s) a') (And (Ne a' a) (Membership.mem s a …
  -/
  rw [mem_keys, keys_erase, s.keys_nodup.mem_erase_iff, ← mem_keys]
  /-
    🎉 no goals
  -/


theorem perm_erase {a : α} {s₁ s₂ : AList β} :
    s₁.entries ~ s₂.entries → (erase a s₁).entries ~ (erase a s₂).entries :=
  Perm.kerase s₁.nodupKeys


@[simp]
theorem lookup_erase (a) (s : AList β) : lookup a (erase a s) = none :=
  dlookup_kerase a s.nodupKeys


@[simp]
theorem lookup_erase_ne {a a'} {s : AList β} (h : a ≠ a') : lookup a (erase a' s) = lookup a s :=
  dlookup_kerase_ne h


theorem erase_erase (a a' : α) (s : AList β) : (s.erase a).erase a' = (s.erase a').erase a :=
  ext <| kerase_kerase


/-- Insert a key-value pair into an association list and erase any existing pair
  with the same key. -/
def insert (a : α) (b : β a) (s : AList β) : AList β :=
  ⟨kinsert a b s.entries, kinsert_nodupKeys a b s.nodupKeys⟩


@[simp]
theorem entries_insert {a} {b : β a} {s : AList β} :
    (insert a b s).entries = Sigma.mk a b :: kerase a s.entries :=
  rfl


@[deprecated (since := "2024-12-17")] alias insert_entries := entries_insert


theorem entries_insert_of_not_mem {a} {b : β a} {s : AList β} (h : a ∉ s) :
                                                       /-
                                                         α : Type u
                                                         β : α → Type v
                                                         inst✝ : DecidableEq α
                                                         a : α
                                                         b : β a
                                                         s : AList β
                                                         h : Not (Membership.mem s a)
                                                         ⊢ Eq (AList.insert a b s).entries (List.cons ⟨a, b⟩ s.entries)
                                                       -/
    (insert a b s).entries = ⟨a, b⟩ :: s.entries := by rw [entries_insert, kerase_of_not_mem_keys h]
                                                       /-
                                                         🎉 no goals
                                                       -/


theorem insert_of_not_mem {a} {b : β a} {s : AList β} (h : a ∉ s) :
    insert a b s = ⟨⟨a, b⟩ :: s.entries, nodupKeys_cons.2 ⟨h, s.2⟩⟩ :=
  ext <| entries_insert_of_not_mem h


@[deprecated (since := "2024-12-14")] alias insert_entries_of_neg := entries_insert_of_not_mem

@[deprecated (since := "2024-12-14")] alias insert_of_neg := insert_of_not_mem


@[simp]
theorem insert_empty (a) (b : β a) : insert a b ∅ = singleton a b :=
  rfl


@[simp]
theorem mem_insert {a a'} {b' : β a'} (s : AList β) : a ∈ insert a' b' s ↔ a = a' ∨ a ∈ s :=
  mem_keys_kinsert


@[simp]
theorem keys_insert {a} {b : β a} (s : AList β) : (insert a b s).keys = a :: s.keys.erase a := by
  /-
    α : Type u
    β : α → Type v
    inst✝ : DecidableEq α
    a : α
    b : β a
    s : AList β
    ⊢ Eq (AList.insert a b s).keys (List.cons a (s.keys.erase a))
  -/
  simp [insert, keys, keys_kerase]
  /-
    🎉 no goals
  -/


theorem perm_insert {a} {b : β a} {s₁ s₂ : AList β} (p : s₁.entries ~ s₂.entries) :
    (insert a b s₁).entries ~ (insert a b s₂).entries := by
  /-
    α : Type u
    β : α → Type v
    inst✝ : DecidableEq α
    a : α
    b : β a
    s₁ s₂ : AList β
    p : s₁.entries.Perm s₂.entries
    ⊢ (AList.insert a b s₁).entries.Perm (AList.insert a b s₂).entries
  -/
  simp only [entries_insert]; exact p.kinsert s₁.nodupKeys
                              /-
                                🎉 no goals
                              -/


@[simp]
theorem lookup_insert {a} {b : β a} (s : AList β) : lookup a (insert a b s) = some b := by
  /-
    α : Type u
    β : α → Type v
    inst✝ : DecidableEq α
    a : α
    b : β a
    s : AList β
    ⊢ Eq (AList.lookup a (AList.insert a b s)) (Option.some b)
  -/
  simp only [lookup, insert, dlookup_kinsert]
  /-
    🎉 no goals
  -/


@[simp]
theorem lookup_insert_ne {a a'} {b' : β a'} {s : AList β} (h : a ≠ a') :
    lookup a (insert a' b' s) = lookup a s :=
  dlookup_kinsert_ne h


@[simp] theorem lookup_insert_eq_none {l : AList β} {k k' : α} {v : β k} :
    (l.insert k v).lookup k' = none ↔ (k' ≠ k) ∧ l.lookup k' = none := by
  /-
    α : Type u
    β : α → Type v
    inst✝ : DecidableEq α
    l : AList β
    k k' : α
    v : β k
    ⊢ Iff (Eq (AList.lookup k' (AList.insert k v l)) Option.none) (And (Ne k' k) ( …
  -/
  by_cases h : k' = k
    /-
      case pos
      α : Type u
      β : α → Type v
      inst✝ : DecidableEq α
      l : AList β
      k k' : α
      v : β k
      h : Eq k' k
      ⊢ Iff (Eq (AList.lookup k' (AList.insert k v l)) Option.none) (And (Ne k' k) ( …
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
      l : AList β
      k k' : α
      v : β k
      h : Not (Eq k' k)
      ⊢ Iff (Eq (AList.lookup k' (AList.insert k v l)) Option.none) (And (Ne k' k) ( …
    -/
  · simp_all [lookup_insert_ne h]
    /-
      🎉 no goals
    -/


@[simp]
theorem lookup_to_alist {a} (s : List (Sigma β)) : lookup a s.toAList = s.dlookup a := by
  /-
    α : Type u
    β : α → Type v
    inst✝ : DecidableEq α
    a : α
    s : List (Sigma β)
    ⊢ Eq (AList.lookup a s.toAList) (List.dlookup a s)
  -/
  rw [List.toAList, lookup, dlookup_dedupKeys]
  /-
    🎉 no goals
  -/


@[simp]
theorem insert_insert {a} {b b' : β a} (s : AList β) :
    (s.insert a b).insert a b' = s.insert a b' := by
  /-
    α : Type u
    β : α → Type v
    inst✝ : DecidableEq α
    a : α
    b b' : β a
    s : AList β
    ⊢ Eq (AList.insert a b' (AList.insert a b s)) (AList.insert a b' s)
  -/
  ext : 1; simp only [AList.entries_insert, List.kerase_cons_eq]
           /-
             🎉 no goals
           -/


theorem insert_insert_of_ne {a a'} {b : β a} {b' : β a'} (s : AList β) (h : a ≠ a') :
    ((s.insert a b).insert a' b').entries ~ ((s.insert a' b').insert a b).entries := by
  /-
    α : Type u
    β : α → Type v
    inst✝ : DecidableEq α
    a a' : α
    b : β a
    b' : β a'
    s : AList β
    h : Ne a a'
    ⊢ (AList.insert a' b' (AList.insert a b s)).entries.Perm (AList.insert a b (AL …
  -/
  simp only [entries_insert]; rw [kerase_cons_ne, kerase_cons_ne, kerase_comm] <;>
    [apply Perm.swap; exact h; exact h.symm]


@[simp]
theorem insert_singleton_eq {a : α} {b b' : β a} : insert a b (singleton a b') = singleton a b :=
  ext <| by
    simp only [AList.entries_insert, List.kerase_cons_eq, and_self_iff, AList.singleton_entries,
      heq_iff_eq, eq_self_iff_true]


@[simp]
theorem entries_toAList (xs : List (Sigma β)) : (List.toAList xs).entries = dedupKeys xs :=
  rfl


theorem toAList_cons (a : α) (b : β a) (xs : List (Sigma β)) :
    List.toAList (⟨a, b⟩ :: xs) = insert a b xs.toAList :=
  rfl


theorem mk_cons_eq_insert (c : Sigma β) (l : List (Sigma β)) (h : (c :: l).NodupKeys) :
    (⟨c :: l, h⟩ : AList β) = insert c.1 c.2 ⟨l, nodupKeys_of_nodupKeys_cons h⟩ := by
  /-
    α : Type u
    β : α → Type v
    inst✝ : DecidableEq α
    c : Sigma β
    l : List (Sigma β)
    h : (List.cons c l).NodupKeys
    ⊢ Eq { entries := List.cons c l, nodupKeys := h } (AList.insert c.fst c.snd {  …
  -/
  simpa [insert] using (kerase_of_not_mem_keys <| not_mem_keys_of_nodupKeys_cons h).symm
  /-
    🎉 no goals
  -/


/-- Recursion on an `AList`, using `insert`. Use as `induction l`. -/
@[elab_as_elim, induction_eliminator]
def insertRec {C : AList β → Sort*} (H0 : C ∅)
    (IH : ∀ (a : α) (b : β a) (l : AList β), a ∉ l → C l → C (l.insert a b)) :
    ∀ l : AList β, C l
  | ⟨[], _⟩ => H0
  | ⟨c :: l, h⟩ => by
    /-
      α : Type u
      β : α → Type v
      inst✝ : DecidableEq α
      C : AList β → Sort u_1
      H0 : C EmptyCollection.emptyCollection
      IH : (a : α) → (b : β a) → (l : AList β) → Not (Membership.mem l a) → C l → C  …
      c : Sigma β
      l : List (Sigma β)
      h : (List.cons c l).NodupKeys
      ⊢ C { entries := List.cons c l, nodupKeys := h }
    -/
    rw [mk_cons_eq_insert]
    /-
      α : Type u
      β : α → Type v
      inst✝ : DecidableEq α
      C : AList β → Sort u_1
      H0 : C EmptyCollection.emptyCollection
      IH : (a : α) → (b : β a) → (l : AList β) → Not (Membership.mem l a) → C l → C  …
      c : Sigma β
      l : List (Sigma β)
      h : (List.cons c l).NodupKeys
      ⊢ C (AList.insert c.fst c.snd { entries := l, nodupKeys := ⋯ })
    -/
    refine IH _ _ _ ?_ (insertRec H0 IH _)
    /-
      α : Type u
      β : α → Type v
      inst✝ : DecidableEq α
      C : AList β → Sort u_1
      H0 : C EmptyCollection.emptyCollection
      IH : (a : α) → (b : β a) → (l : AList β) → Not (Membership.mem l a) → C l → C  …
      c : Sigma β
      l : List (Sigma β)
      h : (List.cons c l).NodupKeys
      ⊢ Not (Membership.mem { entries := l, nodupKeys := ⋯ } c.fst)
    -/
    exact not_mem_keys_of_nodupKeys_cons h
    /-
      🎉 no goals
    -/

-- Test that the `induction` tactic works on `insertRec`.

@[simp]
theorem insertRec_empty {C : AList β → Sort*} (H0 : C ∅)
    (IH : ∀ (a : α) (b : β a) (l : AList β), a ∉ l → C l → C (l.insert a b)) :
    @insertRec α β _ C H0 IH ∅ = H0 := by
  /-
    α : Type u
    β : α → Type v
    inst✝ : DecidableEq α
    C : AList β → Sort u_1
    H0 : C EmptyCollection.emptyCollection
    IH : (a : α) → (b : β a) → (l : AList β) → Not (Membership.mem l a) → C l → C  …
    ⊢ Eq (AList.insertRec H0 IH EmptyCollection.emptyCollection) H0
  -/
  change @insertRec α β _ C H0 IH ⟨[], _⟩ = H0
  /-
    α : Type u
    β : α → Type v
    inst✝ : DecidableEq α
    C : AList β → Sort u_1
    H0 : C EmptyCollection.emptyCollection
    IH : (a : α) → (b : β a) → (l : AList β) → Not (Membership.mem l a) → C l → C  …
    ⊢ Eq (AList.insertRec H0 IH { entries := List.nil, nodupKeys := ⋯ }) H0
  -/
  rw [insertRec]
  /-
    🎉 no goals
  -/


theorem insertRec_insert {C : AList β → Sort*} (H0 : C ∅)
    (IH : ∀ (a : α) (b : β a) (l : AList β), a ∉ l → C l → C (l.insert a b)) {c : Sigma β}
    {l : AList β} (h : c.1 ∉ l) :
    @insertRec α β _ C H0 IH (l.insert c.1 c.2) = IH c.1 c.2 l h (@insertRec α β _ C H0 IH l) := by
  /-
    α : Type u
    β : α → Type v
    inst✝ : DecidableEq α
    C : AList β → Sort u_1
    H0 : C EmptyCollection.emptyCollection
    IH : (a : α) → (b : β a) → (l : AList β) → Not (Membership.mem l a) → C l → C  …
    c : Sigma β
    l : AList β
    h : Not (Membership.mem l c.fst)
    ⊢ Eq (AList.insertRec H0 IH (AList.insert c.fst c.snd l)) (IH c.fst c.snd l h  …
  -/
  cases' l with l hl
  suffices HEq (@insertRec α β _ C H0 IH ⟨c :: l, nodupKeys_cons.2 ⟨h, hl⟩⟩)
      (IH c.1 c.2 ⟨l, hl⟩ h (@insertRec α β _ C H0 IH ⟨l, hl⟩)) by
    cases c
    apply eq_of_heq
    convert this <;> rw [insert_of_not_mem h]
  /-
    case mk
    α : Type u
    β : α → Type v
    inst✝ : DecidableEq α
    C : AList β → Sort u_1
    H0 : C EmptyCollection.emptyCollection
    IH : (a : α) → (b : β a) → (l : AList β) → Not (Membership.mem l a) → C l → C  …
    c : Sigma β
    l : List (Sigma β)
    hl : l.NodupKeys
    h : Not (Membership.mem { entries := l, nodupKeys := hl } c.fst)
    ⊢ HEq (AList.insertRec H0 IH { entries := List.cons c l, nodupKeys := ⋯ }) (IH …
  -/
  rw [insertRec]
  /-
    case mk
    α : Type u
    β : α → Type v
    inst✝ : DecidableEq α
    C : AList β → Sort u_1
    H0 : C EmptyCollection.emptyCollection
    IH : (a : α) → (b : β a) → (l : AList β) → Not (Membership.mem l a) → C l → C  …
    c : Sigma β
    l : List (Sigma β)
    hl : l.NodupKeys
    h : Not (Membership.mem { entries := l, nodupKeys := hl } c.fst)
    ⊢ HEq (⋯.mpr (IH c.fst c.snd { entries := l, nodupKeys := ⋯ } ⋯ (AList.insertR …
  -/
  apply cast_heq
  /-
    🎉 no goals
  -/


theorem insertRec_insert_mk {C : AList β → Sort*} (H0 : C ∅)
    (IH : ∀ (a : α) (b : β a) (l : AList β), a ∉ l → C l → C (l.insert a b)) {a : α} (b : β a)
    {l : AList β} (h : a ∉ l) :
    @insertRec α β _ C H0 IH (l.insert a b) = IH a b l h (@insertRec α β _ C H0 IH l) :=
  @insertRec_insert α β _ C H0 IH ⟨a, b⟩ l h


/-- Erase a key from the map, and return the corresponding value, if found. -/
def extract (a : α) (s : AList β) : Option (β a) × AList β :=
  have : (kextract a s.entries).2.NodupKeys := by
    /-
      α : Type u
      β : α → Type v
      inst✝ : DecidableEq α
      a : α
      s : AList β
      ⊢ (List.kextract a s.entries).2.NodupKeys
    -/
    rw [kextract_eq_dlookup_kerase]; exact s.nodupKeys.kerase _
                                     /-
                                       🎉 no goals
                                     -/
  match kextract a s.entries, this with
  | (b, l), h => (b, ⟨l, h⟩)


@[simp]
theorem extract_eq_lookup_erase (a : α) (s : AList β) : extract a s = (lookup a s, erase a s) := by
  /-
    α : Type u
    β : α → Type v
    inst✝ : DecidableEq α
    a : α
    s : AList β
    ⊢ Eq (AList.extract a s) { fst := AList.lookup a s, snd := AList.erase a s }
  -/
                                  /-
                                    🎉 no goals
                                  -/
  simp [extract]; constructor <;> rfl
                                  /-
                                    🎉 no goals
                                  -/


/-- `s₁ ∪ s₂` is the key-based union of two association lists. It is
left-biased: if there exists an `a ∈ s₁`, `lookup a (s₁ ∪ s₂) = lookup a s₁`.
-/
def union (s₁ s₂ : AList β) : AList β :=
  ⟨s₁.entries.kunion s₂.entries, s₁.nodupKeys.kunion s₂.nodupKeys⟩


instance : Union (AList β) :=
  ⟨union⟩


@[simp]
theorem union_entries {s₁ s₂ : AList β} : (s₁ ∪ s₂).entries = kunion s₁.entries s₂.entries :=
  rfl


@[simp]
theorem empty_union {s : AList β} : (∅ : AList β) ∪ s = s :=
  ext rfl


@[simp]
theorem union_empty {s : AList β} : s ∪ (∅ : AList β) = s :=
            /-
              α : Type u
              β : α → Type v
              inst✝ : DecidableEq α
              s : AList β
              ⊢ Eq (Union.union s EmptyCollection.emptyCollection).entries s.entries
            -/
  ext <| by simp
            /-
              🎉 no goals
            -/


@[simp]
theorem mem_union {a} {s₁ s₂ : AList β} : a ∈ s₁ ∪ s₂ ↔ a ∈ s₁ ∨ a ∈ s₂ :=
  mem_keys_kunion


theorem perm_union {s₁ s₂ s₃ s₄ : AList β} (p₁₂ : s₁.entries ~ s₂.entries)
    (p₃₄ : s₃.entries ~ s₄.entries) : (s₁ ∪ s₃).entries ~ (s₂ ∪ s₄).entries := by
  /-
    α : Type u
    β : α → Type v
    inst✝ : DecidableEq α
    s₁ s₂ s₃ s₄ : AList β
    p₁₂ : s₁.entries.Perm s₂.entries
    p₃₄ : s₃.entries.Perm s₄.entries
    ⊢ (Union.union s₁ s₃).entries.Perm (Union.union s₂ s₄).entries
  -/
  simp [p₁₂.kunion s₃.nodupKeys p₃₄]
  /-
    🎉 no goals
  -/


theorem union_erase (a : α) (s₁ s₂ : AList β) : erase a (s₁ ∪ s₂) = erase a s₁ ∪ erase a s₂ :=
  ext kunion_kerase.symm


@[simp]
theorem lookup_union_left {a} {s₁ s₂ : AList β} : a ∈ s₁ → lookup a (s₁ ∪ s₂) = lookup a s₁ :=
  dlookup_kunion_left


@[simp]
theorem lookup_union_right {a} {s₁ s₂ : AList β} : a ∉ s₁ → lookup a (s₁ ∪ s₂) = lookup a s₂ :=
  dlookup_kunion_right

-- Porting note: removing simp, LHS not in SNF, new theorem added instead.

theorem mem_lookup_union {a} {b : β a} {s₁ s₂ : AList β} :
    b ∈ lookup a (s₁ ∪ s₂) ↔ b ∈ lookup a s₁ ∨ a ∉ s₁ ∧ b ∈ lookup a s₂ :=
  mem_dlookup_kunion


@[simp]
theorem lookup_union_eq_some {a} {b : β a} {s₁ s₂ : AList β} :
    lookup a (s₁ ∪ s₂) = some b ↔ lookup a s₁ = some b ∨ a ∉ s₁ ∧ lookup a s₂ = some b :=
  mem_dlookup_kunion


theorem mem_lookup_union_middle {a} {b : β a} {s₁ s₂ s₃ : AList β} :
    b ∈ lookup a (s₁ ∪ s₃) → a ∉ s₂ → b ∈ lookup a (s₁ ∪ s₂ ∪ s₃) :=
  mem_dlookup_kunion_middle


theorem insert_union {a} {b : β a} {s₁ s₂ : AList β} :
                                                    /-
                                                      α : Type u
                                                      β : α → Type v
                                                      inst✝ : DecidableEq α
                                                      a : α
                                                      b : β a
                                                      s₁ s₂ : AList β
                                                      ⊢ Eq (AList.insert a b (Union.union s₁ s₂)) (Union.union (AList.insert a b s₁) …
                                                    -/
    insert a b (s₁ ∪ s₂) = insert a b s₁ ∪ s₂ := by ext; simp
                                                         /-
                                                           🎉 no goals
                                                         -/


theorem union_assoc {s₁ s₂ s₃ : AList β} : (s₁ ∪ s₂ ∪ s₃).entries ~ (s₁ ∪ (s₂ ∪ s₃)).entries :=
  lookup_ext (AList.nodupKeys _) (AList.nodupKeys _)
        /-
          α : Type u
          β : α → Type v
          inst✝ : DecidableEq α
          s₁ s₂ s₃ : AList β
          ⊢ ∀ (x : α) (y : β x), Iff (Membership.mem (List.dlookup x (Union.union (Union …
        -/
    (by simp [not_or, or_assoc, and_or_left, and_assoc])
        /-
          🎉 no goals
        -/


/-- Two associative lists are disjoint if they have no common keys. -/
def Disjoint (s₁ s₂ : AList β) : Prop :=
  ∀ k ∈ s₁.keys, ¬k ∈ s₂.keys


theorem union_comm_of_disjoint {s₁ s₂ : AList β} (h : Disjoint s₁ s₂) :
    (s₁ ∪ s₂).entries ~ (s₂ ∪ s₁).entries :=
  lookup_ext (AList.nodupKeys _) (AList.nodupKeys _)
    (by
      /-
        α : Type u
        β : α → Type v
        inst✝ : DecidableEq α
        s₁ s₂ : AList β
        h : s₁.Disjoint s₂
        ⊢ ∀ (x : α) (y : β x), Iff (Membership.mem (List.dlookup x (Union.union s₁ s₂) …
      -/
      intros; simp only [union_entries, Option.mem_def, dlookup_kunion_eq_some]
      /-
        α : Type u
        β : α → Type v
        inst✝ : DecidableEq α
        s₁ s₂ : AList β
        h : s₁.Disjoint s₂
        x✝ : α
        y✝ : β x✝
        ⊢ Iff (Or (Eq (List.dlookup x✝ s₁.entries) (Option.some y✝)) (And (Not (Member …
      -/
      constructor <;> intro h'
        /-
          case mp
          α : Type u
          β : α → Type v
          inst✝ : DecidableEq α
          s₁ s₂ : AList β
          h : s₁.Disjoint s₂
          x✝ : α
          y✝ : β x✝
          h' : Or (Eq (List.dlookup x✝ s₁.entries) (Option.some y✝)) (And (Not (Membersh …
          ⊢ Or (Eq (List.dlookup x✝ s₂.entries) (Option.some y✝)) (And (Not (Membership. …
        -/
      · cases' h' with h' h'
          /-
            case mp.inl
            α : Type u
            β : α → Type v
            inst✝ : DecidableEq α
            s₁ s₂ : AList β
            h : s₁.Disjoint s₂
            x✝ : α
            y✝ : β x✝
            h' : Eq (List.dlookup x✝ s₁.entries) (Option.some y✝)
            ⊢ Or (Eq (List.dlookup x✝ s₂.entries) (Option.some y✝)) (And (Not (Membership. …
          -/
        · right
          /-
            case mp.inl.h
            α : Type u
            β : α → Type v
            inst✝ : DecidableEq α
            s₁ s₂ : AList β
            h : s₁.Disjoint s₂
            x✝ : α
            y✝ : β x✝
            h' : Eq (List.dlookup x✝ s₁.entries) (Option.some y✝)
            ⊢ And (Not (Membership.mem s₂.entries.keys x✝)) (Eq (List.dlookup x✝ s₁.entrie …
          -/
          refine ⟨?_, h'⟩
          /-
            case mp.inl.h
            α : Type u
            β : α → Type v
            inst✝ : DecidableEq α
            s₁ s₂ : AList β
            h : s₁.Disjoint s₂
            x✝ : α
            y✝ : β x✝
            h' : Eq (List.dlookup x✝ s₁.entries) (Option.some y✝)
            ⊢ Not (Membership.mem s₂.entries.keys x✝)
          -/
          apply h
          /-
            case mp.inl.h.a
            α : Type u
            β : α → Type v
            inst✝ : DecidableEq α
            s₁ s₂ : AList β
            h : s₁.Disjoint s₂
            x✝ : α
            y✝ : β x✝
            h' : Eq (List.dlookup x✝ s₁.entries) (Option.some y✝)
            ⊢ Membership.mem s₁.keys x✝
          -/
          rw [keys, ← List.dlookup_isSome, h']
          /-
            case mp.inl.h.a
            α : Type u
            β : α → Type v
            inst✝ : DecidableEq α
            s₁ s₂ : AList β
            h : s₁.Disjoint s₂
            x✝ : α
            y✝ : β x✝
            h' : Eq (List.dlookup x✝ s₁.entries) (Option.some y✝)
            ⊢ Eq (Option.some y✝).isSome Bool.true
          -/
          exact rfl
          /-
            🎉 no goals
          -/
          /-
            case mp.inr
            α : Type u
            β : α → Type v
            inst✝ : DecidableEq α
            s₁ s₂ : AList β
            h : s₁.Disjoint s₂
            x✝ : α
            y✝ : β x✝
            h' : And (Not (Membership.mem s₁.entries.keys x✝)) (Eq (List.dlookup x✝ s₂.ent …
            ⊢ Or (Eq (List.dlookup x✝ s₂.entries) (Option.some y✝)) (And (Not (Membership. …
          -/
        · left
          /-
            case mp.inr.h
            α : Type u
            β : α → Type v
            inst✝ : DecidableEq α
            s₁ s₂ : AList β
            h : s₁.Disjoint s₂
            x✝ : α
            y✝ : β x✝
            h' : And (Not (Membership.mem s₁.entries.keys x✝)) (Eq (List.dlookup x✝ s₂.ent …
            ⊢ Eq (List.dlookup x✝ s₂.entries) (Option.some y✝)
          -/
          rw [h'.2]
          /-
            🎉 no goals
          -/
        /-
          case mpr
          α : Type u
          β : α → Type v
          inst✝ : DecidableEq α
          s₁ s₂ : AList β
          h : s₁.Disjoint s₂
          x✝ : α
          y✝ : β x✝
          h' : Or (Eq (List.dlookup x✝ s₂.entries) (Option.some y✝)) (And (Not (Membersh …
          ⊢ Or (Eq (List.dlookup x✝ s₁.entries) (Option.some y✝)) (And (Not (Membership. …
        -/
      · cases' h' with h' h'
          /-
            case mpr.inl
            α : Type u
            β : α → Type v
            inst✝ : DecidableEq α
            s₁ s₂ : AList β
            h : s₁.Disjoint s₂
            x✝ : α
            y✝ : β x✝
            h' : Eq (List.dlookup x✝ s₂.entries) (Option.some y✝)
            ⊢ Or (Eq (List.dlookup x✝ s₁.entries) (Option.some y✝)) (And (Not (Membership. …
          -/
        · right
          /-
            case mpr.inl.h
            α : Type u
            β : α → Type v
            inst✝ : DecidableEq α
            s₁ s₂ : AList β
            h : s₁.Disjoint s₂
            x✝ : α
            y✝ : β x✝
            h' : Eq (List.dlookup x✝ s₂.entries) (Option.some y✝)
            ⊢ And (Not (Membership.mem s₁.entries.keys x✝)) (Eq (List.dlookup x✝ s₂.entrie …
          -/
          refine ⟨?_, h'⟩
          /-
            case mpr.inl.h
            α : Type u
            β : α → Type v
            inst✝ : DecidableEq α
            s₁ s₂ : AList β
            h : s₁.Disjoint s₂
            x✝ : α
            y✝ : β x✝
            h' : Eq (List.dlookup x✝ s₂.entries) (Option.some y✝)
            ⊢ Not (Membership.mem s₁.entries.keys x✝)
          -/
          intro h''
          /-
            case mpr.inl.h
            α : Type u
            β : α → Type v
            inst✝ : DecidableEq α
            s₁ s₂ : AList β
            h : s₁.Disjoint s₂
            x✝ : α
            y✝ : β x✝
            h' : Eq (List.dlookup x✝ s₂.entries) (Option.some y✝)
            h'' : Membership.mem s₁.entries.keys x✝
            ⊢ False
          -/
          apply h _ h''
          /-
            case mpr.inl.h
            α : Type u
            β : α → Type v
            inst✝ : DecidableEq α
            s₁ s₂ : AList β
            h : s₁.Disjoint s₂
            x✝ : α
            y✝ : β x✝
            h' : Eq (List.dlookup x✝ s₂.entries) (Option.some y✝)
            h'' : Membership.mem s₁.entries.keys x✝
            ⊢ Membership.mem s₂.keys x✝
          -/
          rw [keys, ← List.dlookup_isSome, h']
          /-
            case mpr.inl.h
            α : Type u
            β : α → Type v
            inst✝ : DecidableEq α
            s₁ s₂ : AList β
            h : s₁.Disjoint s₂
            x✝ : α
            y✝ : β x✝
            h' : Eq (List.dlookup x✝ s₂.entries) (Option.some y✝)
            h'' : Membership.mem s₁.entries.keys x✝
            ⊢ Eq (Option.some y✝).isSome Bool.true
          -/
          exact rfl
          /-
            🎉 no goals
          -/
          /-
            case mpr.inr
            α : Type u
            β : α → Type v
            inst✝ : DecidableEq α
            s₁ s₂ : AList β
            h : s₁.Disjoint s₂
            x✝ : α
            y✝ : β x✝
            h' : And (Not (Membership.mem s₂.entries.keys x✝)) (Eq (List.dlookup x✝ s₁.ent …
            ⊢ Or (Eq (List.dlookup x✝ s₁.entries) (Option.some y✝)) (And (Not (Membership. …
          -/
        · left
          /-
            case mpr.inr.h
            α : Type u
            β : α → Type v
            inst✝ : DecidableEq α
            s₁ s₂ : AList β
            h : s₁.Disjoint s₂
            x✝ : α
            y✝ : β x✝
            h' : And (Not (Membership.mem s₂.entries.keys x✝)) (Eq (List.dlookup x✝ s₁.ent …
            ⊢ Eq (List.dlookup x✝ s₁.entries) (Option.some y✝)
          -/
          rw [h'.2])
          /-
            🎉 no goals
          -/


