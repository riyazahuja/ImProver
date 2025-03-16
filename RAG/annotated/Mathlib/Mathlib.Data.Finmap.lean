/-- Multiset of keys of an association multiset. -/
def keys (s : Multiset (Sigma β)) : Multiset α :=
  s.map Sigma.fst


@[simp]
theorem coe_keys {l : List (Sigma β)} : keys (l : Multiset (Sigma β)) = (l.keys : Multiset α) :=
  rfl

-- Porting note: Fixed Nodupkeys -> NodupKeys

/-- `NodupKeys s` means that `s` has no duplicate keys. -/
def NodupKeys (s : Multiset (Sigma β)) : Prop :=
  Quot.liftOn s List.NodupKeys fun _ _ p => propext <| perm_nodupKeys p


@[simp]
theorem coe_nodupKeys {l : List (Sigma β)} : @NodupKeys α β l ↔ l.NodupKeys :=
  Iff.rfl


lemma nodup_keys {m : Multiset (Σ a, β a)} : m.keys.Nodup ↔ m.NodupKeys := by
  /-
    α : Type u
    β : α → Type v
    m : Multiset (Sigma fun a => β a)
    ⊢ Iff m.keys.Nodup m.NodupKeys
  -/
  rcases m with ⟨l⟩; rfl
                     /-
                       🎉 no goals
                     -/


alias ⟨_, NodupKeys.nodup_keys⟩ := nodup_keys


protected lemma NodupKeys.nodup {m : Multiset (Σ a, β a)} (h : m.NodupKeys) : m.Nodup :=
  h.nodup_keys.of_map _


/-- `Finmap β` is the type of finite maps over a multiset. It is effectively
  a quotient of `AList β` by permutation of the underlying list. -/
structure Finmap (β : α → Type v) : Type max u v where
  /-- The underlying `Multiset` of a `Finmap` -/
  entries : Multiset (Sigma β)
  /-- There are no duplicate keys in `entries` -/
  nodupKeys : entries.NodupKeys


/-- The quotient map from `AList` to `Finmap`. -/
def AList.toFinmap (s : AList β) : Finmap β :=
  ⟨s.entries, s.nodupKeys⟩


local notation:arg "⟦" a "⟧" => AList.toFinmap a


theorem AList.toFinmap_eq {s₁ s₂ : AList β} :
    toFinmap s₁ = toFinmap s₂ ↔ s₁.entries ~ s₂.entries := by
  /-
    α : Type u
    β : α → Type v
    s₁ s₂ : AList β
    ⊢ Iff (Eq s₁.toFinmap s₂.toFinmap) (s₁.entries.Perm s₂.entries)
  -/
  cases s₁
  /-
    case mk
    α : Type u
    β : α → Type v
    s₂ : AList β
    entries✝ : List (Sigma β)
    nodupKeys✝ : entries✝.NodupKeys
    ⊢ Iff (Eq { entries := entries✝, nodupKeys := nodupKeys✝ }.toFinmap s₂.toFinma …
  -/
  cases s₂
  /-
    case mk.mk
    α : Type u
    β : α → Type v
    entries✝¹ : List (Sigma β)
    nodupKeys✝¹ : entries✝¹.NodupKeys
    entries✝ : List (Sigma β)
    nodupKeys✝ : entries✝.NodupKeys
    ⊢ Iff (Eq { entries := entries✝¹, nodupKeys := nodupKeys✝¹ }.toFinmap { entrie …
  -/
  simp [AList.toFinmap]
  /-
    🎉 no goals
  -/


@[simp]
theorem AList.toFinmap_entries (s : AList β) : ⟦s⟧.entries = s.entries :=
  rfl


/-- Given `l : List (Sigma β)`, create a term of type `Finmap β` by removing
entries with duplicate keys. -/
def List.toFinmap [DecidableEq α] (s : List (Sigma β)) : Finmap β :=
  s.toAList.toFinmap


lemma nodup_entries (f : Finmap β) : f.entries.Nodup := f.nodupKeys.nodup


/-- Lift a permutation-respecting function on `AList` to `Finmap`. -/
def liftOn {γ} (s : Finmap β) (f : AList β → γ)
    (H : ∀ a b : AList β, a.entries ~ b.entries → f a = f b) : γ := by
  refine
    (Quotient.liftOn s.entries
      (fun (l : List (Sigma β)) => (⟨_, fun nd => f ⟨l, nd⟩⟩ : Part γ))
      (fun l₁ l₂ p => Part.ext' (perm_nodupKeys p) ?_) : Part γ).get ?_
    /-
      case refine_1
      α : Type u
      β : α → Type v
      γ : Type ?u.3957
      s : Finmap β
      f : AList β → γ
      H : ∀ (a b : AList β), a.entries.Perm b.entries → Eq (f a) (f b)
      l₁ l₂ : List (Sigma β)
      p : HasEquiv.Equiv l₁ l₂
      ⊢ ∀ (h₁ : ((fun l => { Dom := l.NodupKeys, get := fun nd => f { entries := l,  …
    -/
  · exact fun h1 h2 => H _ _ p
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      α : Type u
      β : α → Type v
      γ : Type ?u.3957
      s : Finmap β
      f : AList β → γ
      H : ∀ (a b : AList β), a.entries.Perm b.entries → Eq (f a) (f b)
      ⊢ (Quotient.liftOn s.entries (fun l => { Dom := l.NodupKeys, get := fun nd =>  …
    -/
  · have := s.nodupKeys
    -- Porting note: `revert` required because `rcases` behaves differently
    /-
      case refine_2
      α : Type u
      β : α → Type v
      γ : Type ?u.3957
      s : Finmap β
      f : AList β → γ
      H : ∀ (a b : AList β), a.entries.Perm b.entries → Eq (f a) (f b)
      this : s.entries.NodupKeys
      ⊢ (Quotient.liftOn s.entries (fun l => { Dom := l.NodupKeys, get := fun nd =>  …
    -/
    revert this
    /-
      case refine_2
      α : Type u
      β : α → Type v
      γ : Type ?u.3957
      s : Finmap β
      f : AList β → γ
      H : ∀ (a b : AList β), a.entries.Perm b.entries → Eq (f a) (f b)
      ⊢ s.entries.NodupKeys → (Quotient.liftOn s.entries (fun l => { Dom := l.NodupK …
    -/
    rcases s.entries with ⟨l⟩
    /-
      case refine_2.mk
      α : Type u
      β : α → Type v
      γ : Type ?u.3957
      s : Finmap β
      f : AList β → γ
      H : ∀ (a b : AList β), a.entries.Perm b.entries → Eq (f a) (f b)
      x✝ : Multiset (Sigma β)
      l : List (Sigma β)
      ⊢ Multiset.NodupKeys (Quot.mk (⇑(List.isSetoid (Sigma β))) l) → (Quotient.lift …
    -/
    exact id
    /-
      🎉 no goals
    -/


@[simp]
theorem liftOn_toFinmap {γ} (s : AList β) (f : AList β → γ) (H) : liftOn ⟦s⟧ f H = f s := by
  /-
    α : Type u
    β : α → Type v
    γ : Type u_1
    s : AList β
    f : AList β → γ
    H : ∀ (a b : AList β), a.entries.Perm b.entries → Eq (f a) (f b)
    ⊢ Eq (s.toFinmap.liftOn f H) (f s)
  -/
  cases s
  /-
    case mk
    α : Type u
    β : α → Type v
    γ : Type u_1
    f : AList β → γ
    H : ∀ (a b : AList β), a.entries.Perm b.entries → Eq (f a) (f b)
    entries✝ : List (Sigma β)
    nodupKeys✝ : entries✝.NodupKeys
    ⊢ Eq ({ entries := entries✝, nodupKeys := nodupKeys✝ }.toFinmap.liftOn f H) (f …
  -/
  rfl
  /-
    🎉 no goals
  -/


/-- Lift a permutation-respecting function on 2 `AList`s to 2 `Finmap`s. -/
def liftOn₂ {γ} (s₁ s₂ : Finmap β) (f : AList β → AList β → γ)
    (H : ∀ a₁ b₁ a₂ b₂ : AList β,
      a₁.entries ~ a₂.entries → b₁.entries ~ b₂.entries → f a₁ b₁ = f a₂ b₂) : γ :=
  liftOn s₁ (fun l₁ => liftOn s₂ (f l₁) fun _ _ p => H _ _ _ _ (Perm.refl _) p) fun a₁ a₂ p => by
    /-
      α : Type u
      β : α → Type v
      γ : Type ?u.4728
      s₁ s₂ : Finmap β
      f : AList β → AList β → γ
      H : ∀ (a₁ b₁ a₂ b₂ : AList β), a₁.entries.Perm a₂.entries → b₁.entries.Perm b₂ …
      a₁ a₂ : AList β
      p : a₁.entries.Perm a₂.entries
      ⊢ Eq ((fun l₁ => s₂.liftOn (f l₁) ⋯) a₁) ((fun l₁ => s₂.liftOn (f l₁) ⋯) a₂)
    -/
    have H' : f a₁ = f a₂ := funext fun _ => H _ _ _ _ p (Perm.refl _)
    /-
      α : Type u
      β : α → Type v
      γ : Type ?u.4728
      s₁ s₂ : Finmap β
      f : AList β → AList β → γ
      H : ∀ (a₁ b₁ a₂ b₂ : AList β), a₁.entries.Perm a₂.entries → b₁.entries.Perm b₂ …
      a₁ a₂ : AList β
      p : a₁.entries.Perm a₂.entries
      H' : Eq (f a₁) (f a₂)
      ⊢ Eq ((fun l₁ => s₂.liftOn (f l₁) ⋯) a₁) ((fun l₁ => s₂.liftOn (f l₁) ⋯) a₂)
    -/
    simp only [H']
    /-
      🎉 no goals
    -/


@[simp]
theorem liftOn₂_toFinmap {γ} (s₁ s₂ : AList β) (f : AList β → AList β → γ) (H) :
    liftOn₂ ⟦s₁⟧ ⟦s₂⟧ f H = f s₁ s₂ := by
      /-
        α : Type u
        β : α → Type v
        γ : Type u_1
        s₁ s₂ : AList β
        f : AList β → AList β → γ
        H : ∀ (a₁ b₁ a₂ b₂ : AList β), a₁.entries.Perm a₂.entries → b₁.entries.Perm b₂ …
        ⊢ Eq (s₁.toFinmap.liftOn₂ s₂.toFinmap f H) (f s₁ s₂)
      -/
      cases s₁; cases s₂; rfl
                          /-
                            🎉 no goals
                          -/


@[elab_as_elim]
theorem induction_on {C : Finmap β → Prop} (s : Finmap β) (H : ∀ a : AList β, C ⟦a⟧) : C s := by
  /-
    α : Type u
    β : α → Type v
    C : Finmap β → Prop
    s : Finmap β
    H : ∀ (a : AList β), C a.toFinmap
    ⊢ C s
  -/
  rcases s with ⟨⟨a⟩, h⟩; exact H ⟨a, h⟩
                          /-
                            🎉 no goals
                          -/


@[elab_as_elim]
theorem induction_on₂ {C : Finmap β → Finmap β → Prop} (s₁ s₂ : Finmap β)
    (H : ∀ a₁ a₂ : AList β, C ⟦a₁⟧ ⟦a₂⟧) : C s₁ s₂ :=
  induction_on s₁ fun l₁ => induction_on s₂ fun l₂ => H l₁ l₂


@[elab_as_elim]
theorem induction_on₃ {C : Finmap β → Finmap β → Finmap β → Prop} (s₁ s₂ s₃ : Finmap β)
    (H : ∀ a₁ a₂ a₃ : AList β, C ⟦a₁⟧ ⟦a₂⟧ ⟦a₃⟧) : C s₁ s₂ s₃ :=
  induction_on₂ s₁ s₂ fun l₁ l₂ => induction_on s₃ fun l₃ => H l₁ l₂ l₃


@[ext]
theorem ext : ∀ {s t : Finmap β}, s.entries = t.entries → s = t
                               /-
                                 α : Type u
                                 β : α → Type v
                                 l₁ : Multiset (Sigma β)
                                 h₁ : l₁.NodupKeys
                                 l₂ : Multiset (Sigma β)
                                 nodupKeys✝ : l₂.NodupKeys
                                 H : Eq { entries := l₁, nodupKeys := h₁ }.entries { entries := l₂, nodupKeys : …
                                 ⊢ Eq { entries := l₁, nodupKeys := h₁ } { entries := l₂, nodupKeys := nodupKey …
                               -/
  | ⟨l₁, h₁⟩, ⟨l₂, _⟩, H => by congr
                               /-
                                 🎉 no goals
                               -/


@[simp]
theorem ext_iff' {s t : Finmap β} : s.entries = t.entries ↔ s = t :=
  Finmap.ext_iff.symm


/-- The predicate `a ∈ s` means that `s` has a value associated to the key `a`. -/
instance : Membership α (Finmap β) :=
  ⟨fun s a => a ∈ s.entries.keys⟩


theorem mem_def {a : α} {s : Finmap β} : a ∈ s ↔ a ∈ s.entries.keys :=
  Iff.rfl


@[simp]
theorem mem_toFinmap {a : α} {s : AList β} : a ∈ toFinmap s ↔ a ∈ s :=
  Iff.rfl


/-- The set of keys of a finite map. -/
def keys (s : Finmap β) : Finset α :=
  ⟨s.entries.keys, s.nodupKeys.nodup_keys⟩


@[simp]
theorem keys_val (s : AList β) : (keys ⟦s⟧).val = s.keys :=
  rfl


@[simp]
theorem keys_ext {s₁ s₂ : AList β} : keys ⟦s₁⟧ = keys ⟦s₂⟧ ↔ s₁.keys ~ s₂.keys := by
  /-
    α : Type u
    β : α → Type v
    s₁ s₂ : AList β
    ⊢ Iff (Eq s₁.toFinmap.keys s₂.toFinmap.keys) (s₁.keys.Perm s₂.keys)
  -/
  simp [keys, AList.keys]
  /-
    🎉 no goals
  -/


theorem mem_keys {a : α} {s : Finmap β} : a ∈ s.keys ↔ a ∈ s :=
  induction_on s fun _ => AList.mem_keys


/-- The empty map. -/
instance : EmptyCollection (Finmap β) :=
  ⟨⟨0, nodupKeys_nil⟩⟩


instance : Inhabited (Finmap β) :=
  ⟨∅⟩


@[simp]
theorem empty_toFinmap : (⟦∅⟧ : Finmap β) = ∅ :=
  rfl


@[simp]
theorem toFinmap_nil [DecidableEq α] : ([].toFinmap : Finmap β) = ∅ :=
  rfl


theorem not_mem_empty {a : α} : a ∉ (∅ : Finmap β) :=
  Multiset.not_mem_zero a


@[simp]
theorem keys_empty : (∅ : Finmap β).keys = ∅ :=
  rfl


/-- The singleton map. -/
def singleton (a : α) (b : β a) : Finmap β :=
  ⟦AList.singleton a b⟧


@[simp]
theorem keys_singleton (a : α) (b : β a) : (singleton a b).keys = {a} :=
  rfl


@[simp]
theorem mem_singleton (x y : α) (b : β y) : x ∈ singleton y b ↔ x = y := by
  /-
    α : Type u
    β : α → Type v
    x y : α
    b : β y
    ⊢ Iff (Membership.mem (Finmap.singleton y b) x) (Eq x y)
  -/
  simp only [singleton]; erw [mem_cons, mem_nil_iff, or_false]
                         /-
                           🎉 no goals
                         -/


instance decidableEq [∀ a, DecidableEq (β a)] : DecidableEq (Finmap β)
  | _, _ => decidable_of_iff _ Finmap.ext_iff.symm


/-- Look up the value associated to a key in a map. -/
def lookup (a : α) (s : Finmap β) : Option (β a) :=
  liftOn s (AList.lookup a) fun _ _ => perm_lookup


@[simp]
theorem lookup_toFinmap (a : α) (s : AList β) : lookup a ⟦s⟧ = s.lookup a :=
  rfl

-- Porting note: renaming to `List.dlookup` since `List.lookup` already exists

@[simp]
theorem dlookup_list_toFinmap (a : α) (s : List (Sigma β)) : lookup a s.toFinmap = s.dlookup a := by
  /-
    α : Type u
    β : α → Type v
    inst✝ : DecidableEq α
    a : α
    s : List (Sigma β)
    ⊢ Eq (Finmap.lookup a s.toFinmap) (List.dlookup a s)
  -/
  rw [List.toFinmap, lookup_toFinmap, lookup_to_alist]
  /-
    🎉 no goals
  -/


@[simp]
theorem lookup_empty (a) : lookup a (∅ : Finmap β) = none :=
  rfl


theorem lookup_isSome {a : α} {s : Finmap β} : (s.lookup a).isSome ↔ a ∈ s :=
  induction_on s fun _ => AList.lookup_isSome


theorem lookup_eq_none {a} {s : Finmap β} : lookup a s = none ↔ a ∉ s :=
  induction_on s fun _ => AList.lookup_eq_none


lemma mem_lookup_iff {s : Finmap β} {a : α} {b : β a} :
    b ∈ s.lookup a ↔ Sigma.mk a b ∈ s.entries := by
  /-
    α : Type u
    β : α → Type v
    inst✝ : DecidableEq α
    s : Finmap β
    a : α
    b : β a
    ⊢ Iff (Membership.mem (Finmap.lookup a s) b) (Membership.mem s.entries ⟨a, b⟩)
  -/
  rcases s with ⟨⟨l⟩, hl⟩; exact List.mem_dlookup_iff hl
                           /-
                             🎉 no goals
                           -/


lemma lookup_eq_some_iff {s : Finmap β} {a : α} {b : β a} :
    s.lookup a = b ↔ Sigma.mk a b ∈ s.entries := mem_lookup_iff


@[simp] lemma sigma_keys_lookup (s : Finmap β) :
    s.keys.sigma (fun i => (s.lookup i).toFinset) = ⟨s.entries, s.nodup_entries⟩ := by
  /-
    α : Type u
    β : α → Type v
    inst✝ : DecidableEq α
    s : Finmap β
    ⊢ Eq (s.keys.sigma fun i => (Finmap.lookup i s).toFinset) { val := s.entries,  …
  -/
  ext x
  /-
    case h
    α : Type u
    β : α → Type v
    inst✝ : DecidableEq α
    s : Finmap β
    x : Sigma fun i => β i
    ⊢ Iff (Membership.mem (s.keys.sigma fun i => (Finmap.lookup i s).toFinset) x)  …
  -/
  have : x ∈ s.entries → x.1 ∈ s.keys := Multiset.mem_map_of_mem _
  /-
    case h
    α : Type u
    β : α → Type v
    inst✝ : DecidableEq α
    s : Finmap β
    x : Sigma fun i => β i
    this : Membership.mem s.entries x → Membership.mem s.keys x.fst
    ⊢ Iff (Membership.mem (s.keys.sigma fun i => (Finmap.lookup i s).toFinset) x)  …
  -/
  simpa [lookup_eq_some_iff]
  /-
    🎉 no goals
  -/


@[simp]
theorem lookup_singleton_eq {a : α} {b : β a} : (singleton a b).lookup a = some b := by
  /-
    α : Type u
    β : α → Type v
    inst✝ : DecidableEq α
    a : α
    b : β a
    ⊢ Eq (Finmap.lookup a (Finmap.singleton a b)) (Option.some b)
  -/
  rw [singleton, lookup_toFinmap, AList.singleton, AList.lookup, dlookup_cons_eq]
  /-
    🎉 no goals
  -/


instance (a : α) (s : Finmap β) : Decidable (a ∈ s) :=
  decidable_of_iff _ lookup_isSome


theorem mem_iff {a : α} {s : Finmap β} : a ∈ s ↔ ∃ b, s.lookup a = some b :=
  induction_on s fun s =>
    Iff.trans List.mem_keys <| exists_congr fun _ => (mem_dlookup_iff s.nodupKeys).symm


theorem mem_of_lookup_eq_some {a : α} {b : β a} {s : Finmap β} (h : s.lookup a = some b) : a ∈ s :=
  mem_iff.mpr ⟨_, h⟩


theorem ext_lookup {s₁ s₂ : Finmap β} : (∀ x, s₁.lookup x = s₂.lookup x) → s₁ = s₂ :=
  induction_on₂ s₁ s₂ fun s₁ s₂ h => by
    /-
      α : Type u
      β : α → Type v
      inst✝ : DecidableEq α
      s₁✝ s₂✝ : Finmap β
      s₁ s₂ : AList β
      h : ∀ (x : α), Eq (Finmap.lookup x s₁.toFinmap) (Finmap.lookup x s₂.toFinmap)
      ⊢ Eq s₁.toFinmap s₂.toFinmap
    -/
    simp only [AList.lookup, lookup_toFinmap] at h
    /-
      α : Type u
      β : α → Type v
      inst✝ : DecidableEq α
      s₁✝ s₂✝ : Finmap β
      s₁ s₂ : AList β
      h : ∀ (x : α), Eq (List.dlookup x s₁.entries) (List.dlookup x s₂.entries)
      ⊢ Eq s₁.toFinmap s₂.toFinmap
    -/
    rw [AList.toFinmap_eq]
    /-
      α : Type u
      β : α → Type v
      inst✝ : DecidableEq α
      s₁✝ s₂✝ : Finmap β
      s₁ s₂ : AList β
      h : ∀ (x : α), Eq (List.dlookup x s₁.entries) (List.dlookup x s₂.entries)
      ⊢ s₁.entries.Perm s₂.entries
    -/
    apply lookup_ext s₁.nodupKeys s₂.nodupKeys
    /-
      α : Type u
      β : α → Type v
      inst✝ : DecidableEq α
      s₁✝ s₂✝ : Finmap β
      s₁ s₂ : AList β
      h : ∀ (x : α), Eq (List.dlookup x s₁.entries) (List.dlookup x s₂.entries)
      ⊢ ∀ (x : α) (y : β x), Iff (Membership.mem (List.dlookup x s₁.entries) y) (Mem …
    -/
    intro x y
    /-
      α : Type u
      β : α → Type v
      inst✝ : DecidableEq α
      s₁✝ s₂✝ : Finmap β
      s₁ s₂ : AList β
      h : ∀ (x : α), Eq (List.dlookup x s₁.entries) (List.dlookup x s₂.entries)
      x : α
      y : β x
      ⊢ Iff (Membership.mem (List.dlookup x s₁.entries) y) (Membership.mem (List.dlo …
    -/
    rw [h]
    /-
      🎉 no goals
    -/


/-- An equivalence between `Finmap β` and pairs `(keys : Finset α, lookup : ∀ a, Option (β a))` such
that `(lookup a).isSome ↔ a ∈ keys`. -/
@[simps apply_coe_fst apply_coe_snd]
def keysLookupEquiv :
    Finmap β ≃ { f : Finset α × (∀ a, Option (β a)) // ∀ i, (f.2 i).isSome ↔ i ∈ f.1 } where
  toFun s := ⟨(s.keys, fun i => s.lookup i), fun _ => lookup_isSome⟩
  invFun f := mk (f.1.1.sigma fun i => (f.1.2 i).toFinset).val <| by
    /-
      α : Type u
      β : α → Type v
      inst✝ : DecidableEq α
      f : Subtype fun f => ∀ (i : α), Iff (Eq (f.2 i).isSome Bool.true) (Membership. …
      ⊢ ((↑f).1.sigma fun i => ((↑f).2 i).toFinset).val.NodupKeys
    -/
    refine Multiset.nodup_keys.1 ((Finset.nodup _).map_on ?_)
    /-
      α : Type u
      β : α → Type v
      inst✝ : DecidableEq α
      f : Subtype fun f => ∀ (i : α), Iff (Eq (f.2 i).isSome Bool.true) (Membership. …
      ⊢ ∀ (x : Sigma fun a => β a), Membership.mem ((↑f).1.sigma fun i => ((↑f).2 i) …
    -/
    simp only [Finset.mem_val, Finset.mem_sigma, Option.mem_toFinset, Option.mem_def]
    /-
      α : Type u
      β : α → Type v
      inst✝ : DecidableEq α
      f : Subtype fun f => ∀ (i : α), Iff (Eq (f.2 i).isSome Bool.true) (Membership. …
      ⊢ ∀ (x : Sigma fun a => β a), And (Membership.mem (↑f).1 x.fst) (Eq ((↑f).2 x. …
    -/
    rintro ⟨i, x⟩ ⟨_, hx⟩ ⟨j, y⟩ ⟨_, hy⟩ (rfl : i = j)
    /-
      case mk.intro.mk.intro
      α : Type u
      β : α → Type v
      inst✝ : DecidableEq α
      f : Subtype fun f => ∀ (i : α), Iff (Eq (f.2 i).isSome Bool.true) (Membership. …
      i : α
      x : β i
      left✝¹ : Membership.mem (↑f).1 ⟨i, x⟩.fst
      hx : Eq ((↑f).2 ⟨i, x⟩.fst) (Option.some ⟨i, x⟩.snd)
      y : β i
      left✝ : Membership.mem (↑f).1 ⟨i, y⟩.fst
      hy : Eq ((↑f).2 ⟨i, y⟩.fst) (Option.some ⟨i, y⟩.snd)
      ⊢ Eq ⟨i, x⟩ ⟨i, y⟩
    -/
    simpa using hx.symm.trans hy
    /-
      🎉 no goals
    -/
                          /-
                            α : Type u
                            β : α → Type v
                            inst✝ : DecidableEq α
                            f : Finmap β
                            ⊢ Eq ((fun f => { entries := ((↑f).1.sigma fun i => ((↑f).2 i).toFinset).val,  …
                          -/
  left_inv f := ext <| by simp
                          /-
                            🎉 no goals
                          -/
  right_inv := fun ⟨(s, f), hf⟩ => by
    /-
      α : Type u
      β : α → Type v
      inst✝ : DecidableEq α
      x✝ : Subtype fun f => ∀ (i : α), Iff (Eq (f.2 i).isSome Bool.true) (Membership …
      s : Finset α
      f : (a : α) → Option (β a)
      hf : ∀ (i : α), Iff (Eq ({ fst := s, snd := f }.2 i).isSome Bool.true) (Member …
      ⊢ Eq ((fun s => ⟨{ fst := s.keys, snd := fun i => Finmap.lookup i s }, ⋯⟩) ((f …
    -/
    dsimp only at hf
    /-
      α : Type u
      β : α → Type v
      inst✝ : DecidableEq α
      x✝ : Subtype fun f => ∀ (i : α), Iff (Eq (f.2 i).isSome Bool.true) (Membership …
      s : Finset α
      f : (a : α) → Option (β a)
      hf : ∀ (i : α), Iff (Eq (f i).isSome Bool.true) (Membership.mem s i)
      ⊢ Eq ((fun s => ⟨{ fst := s.keys, snd := fun i => Finmap.lookup i s }, ⋯⟩) ((f …
    -/
    ext
      /-
        case a.fst.h
        α : Type u
        β : α → Type v
        inst✝ : DecidableEq α
        x✝ : Subtype fun f => ∀ (i : α), Iff (Eq (f.2 i).isSome Bool.true) (Membership …
        s : Finset α
        f : (a : α) → Option (β a)
        hf : ∀ (i : α), Iff (Eq (f i).isSome Bool.true) (Membership.mem s i)
        a✝ : α
        ⊢ Iff (Membership.mem (↑((fun s => ⟨{ fst := s.keys, snd := fun i => Finmap.lo …
      -/
    · simp [keys, Multiset.keys, ← hf, Option.isSome_iff_exists]
      /-
        🎉 no goals
      -/
      /-
        case a.snd.h.a
        α : Type u
        β : α → Type v
        inst✝ : DecidableEq α
        x✝¹ : Subtype fun f => ∀ (i : α), Iff (Eq (f.2 i).isSome Bool.true) (Membershi …
        s : Finset α
        f : (a : α) → Option (β a)
        hf : ∀ (i : α), Iff (Eq (f i).isSome Bool.true) (Membership.mem s i)
        x✝ : α
        a✝ : β x✝
        ⊢ Iff (Membership.mem ((↑((fun s => ⟨{ fst := s.keys, snd := fun i => Finmap.l …
      -/
    · simp +contextual [lookup_eq_some_iff, ← hf]
      /-
        🎉 no goals
      -/


@[simp] lemma keysLookupEquiv_symm_apply_keys :
    ∀ f : {f : Finset α × (∀ a, Option (β a)) // ∀ i, (f.2 i).isSome ↔ i ∈ f.1},
      (keysLookupEquiv.symm f).keys = f.1.1 :=
  keysLookupEquiv.surjective.forall.2 fun _ => by
    /-
      α : Type u
      β : α → Type v
      inst✝ : DecidableEq α
      x✝ : Finmap β
      ⊢ Eq (Finmap.keysLookupEquiv.symm (Finmap.keysLookupEquiv x✝)).keys (↑(Finmap. …
    -/
    simp only [Equiv.symm_apply_apply, keysLookupEquiv_apply_coe_fst]
    /-
      🎉 no goals
    -/


@[simp] lemma keysLookupEquiv_symm_apply_lookup :
    ∀ (f : {f : Finset α × (∀ a, Option (β a)) // ∀ i, (f.2 i).isSome ↔ i ∈ f.1}) a,
      (keysLookupEquiv.symm f).lookup a = f.1.2 a :=
  keysLookupEquiv.surjective.forall.2 fun _ _ => by
    /-
      α : Type u
      β : α → Type v
      inst✝ : DecidableEq α
      x✝¹ : Finmap β
      x✝ : α
      ⊢ Eq (Finmap.lookup x✝ (Finmap.keysLookupEquiv.symm (Finmap.keysLookupEquiv x✝ …
    -/
    simp only [Equiv.symm_apply_apply, keysLookupEquiv_apply_coe_snd]
    /-
      🎉 no goals
    -/


/-- Replace a key with a given value in a finite map.
  If the key is not present it does nothing. -/
def replace (a : α) (b : β a) (s : Finmap β) : Finmap β :=
  (liftOn s fun t => AList.toFinmap (AList.replace a b t))
    fun _ _ p => toFinmap_eq.2 <| perm_replace p

-- Porting note: explicit type required because of the ambiguity

@[simp]
theorem replace_toFinmap (a : α) (b : β a) (s : AList β) :
    replace a b ⟦s⟧ = (⟦s.replace a b⟧ : Finmap β) := by
  /-
    α : Type u
    β : α → Type v
    inst✝ : DecidableEq α
    a : α
    b : β a
    s : AList β
    ⊢ Eq (Finmap.replace a b s.toFinmap) (AList.replace a b s).toFinmap
  -/
  simp [replace]
  /-
    🎉 no goals
  -/


@[simp]
theorem keys_replace (a : α) (b : β a) (s : Finmap β) : (replace a b s).keys = s.keys :=
                             /-
                               α : Type u
                               β : α → Type v
                               inst✝ : DecidableEq α
                               a : α
                               b : β a
                               s✝ : Finmap β
                               s : AList β
                               ⊢ Eq (Finmap.replace a b s.toFinmap).keys s.toFinmap.keys
                             -/
  induction_on s fun s => by simp
                             /-
                               🎉 no goals
                             -/


@[simp]
theorem mem_replace {a a' : α} {b : β a} {s : Finmap β} : a' ∈ replace a b s ↔ a' ∈ s :=
                             /-
                               α : Type u
                               β : α → Type v
                               inst✝ : DecidableEq α
                               a a' : α
                               b : β a
                               s✝ : Finmap β
                               s : AList β
                               ⊢ Iff (Membership.mem (Finmap.replace a b s.toFinmap) a') (Membership.mem s.to …
                             -/
  induction_on s fun s => by simp
                             /-
                               🎉 no goals
                             -/


/-- Fold a commutative function over the key-value pairs in the map -/
def foldl {δ : Type w} (f : δ → ∀ a, β a → δ)
    (H : ∀ d a₁ b₁ a₂ b₂, f (f d a₁ b₁) a₂ b₂ = f (f d a₂ b₂) a₁ b₁) (d : δ) (m : Finmap β) : δ :=
  letI : RightCommutative fun d (s : Sigma β) ↦ f d s.1 s.2 := ⟨fun _ _ _ ↦ H _ _ _ _ _⟩
  m.entries.foldl (fun d s => f d s.1 s.2) d


/-- `any f s` returns `true` iff there exists a value `v` in `s` such that `f v = true`. -/
def any (f : ∀ x, β x → Bool) (s : Finmap β) : Bool :=
  s.foldl (fun x y z => x || f y z)
                       /-
                         α : Type u
                         β : α → Type v
                         f : (x : α) → β x → Bool
                         s : Finmap β
                         x✝³ : Bool
                         x✝² : α
                         x✝¹ : β x✝²
                         x✝ : α
                         ⊢ ∀ (b₂ : β x✝), Eq ((fun x y z => x.or (f y z)) ((fun x y z => x.or (f y z))  …
                       -/
    (fun _ _ _ _ => by simp_rw [Bool.or_assoc, Bool.or_comm, imp_true_iff]) false
                       /-
                         🎉 no goals
                       -/


/-- `all f s` returns `true` iff `f v = true` for all values `v` in `s`. -/
def all (f : ∀ x, β x → Bool) (s : Finmap β) : Bool :=
  s.foldl (fun x y z => x && f y z)
                       /-
                         α : Type u
                         β : α → Type v
                         f : (x : α) → β x → Bool
                         s : Finmap β
                         x✝³ : Bool
                         x✝² : α
                         x✝¹ : β x✝²
                         x✝ : α
                         ⊢ ∀ (b₂ : β x✝), Eq ((fun x y z => x.and (f y z)) ((fun x y z => x.and (f y z) …
                       -/
    (fun _ _ _ _ => by simp_rw [Bool.and_assoc, Bool.and_comm, imp_true_iff]) true
                       /-
                         🎉 no goals
                       -/


/-- Erase a key from the map. If the key is not present it does nothing. -/
def erase (a : α) (s : Finmap β) : Finmap β :=
  (liftOn s fun t => AList.toFinmap (AList.erase a t)) fun _ _ p => toFinmap_eq.2 <| perm_erase p


@[simp]
theorem erase_toFinmap (a : α) (s : AList β) : erase a ⟦s⟧ = AList.toFinmap (s.erase a) := by
  /-
    α : Type u
    β : α → Type v
    inst✝ : DecidableEq α
    a : α
    s : AList β
    ⊢ Eq (Finmap.erase a s.toFinmap) (AList.erase a s).toFinmap
  -/
  simp [erase]
  /-
    🎉 no goals
  -/


@[simp]
theorem keys_erase_toFinset (a : α) (s : AList β) : keys ⟦s.erase a⟧ = (keys ⟦s⟧).erase a := by
  /-
    α : Type u
    β : α → Type v
    inst✝ : DecidableEq α
    a : α
    s : AList β
    ⊢ Eq (AList.erase a s).toFinmap.keys (s.toFinmap.keys.erase a)
  -/
  simp [Finset.erase, keys, AList.erase, keys_kerase]
  /-
    🎉 no goals
  -/


@[simp]
theorem keys_erase (a : α) (s : Finmap β) : (erase a s).keys = s.keys.erase a :=
                             /-
                               α : Type u
                               β : α → Type v
                               inst✝ : DecidableEq α
                               a : α
                               s✝ : Finmap β
                               s : AList β
                               ⊢ Eq (Finmap.erase a s.toFinmap).keys (s.toFinmap.keys.erase a)
                             -/
  induction_on s fun s => by simp
                             /-
                               🎉 no goals
                             -/


@[simp]
theorem mem_erase {a a' : α} {s : Finmap β} : a' ∈ erase a s ↔ a' ≠ a ∧ a' ∈ s :=
                             /-
                               α : Type u
                               β : α → Type v
                               inst✝ : DecidableEq α
                               a a' : α
                               s✝ : Finmap β
                               s : AList β
                               ⊢ Iff (Membership.mem (Finmap.erase a s.toFinmap) a') (And (Ne a' a) (Membersh …
                             -/
  induction_on s fun s => by simp
                             /-
                               🎉 no goals
                             -/


theorem not_mem_erase_self {a : α} {s : Finmap β} : ¬a ∈ erase a s := by
  /-
    α : Type u
    β : α → Type v
    inst✝ : DecidableEq α
    a : α
    s : Finmap β
    ⊢ Not (Membership.mem (Finmap.erase a s) a)
  -/
  rw [mem_erase, not_and_or, not_not]
  /-
    α : Type u
    β : α → Type v
    inst✝ : DecidableEq α
    a : α
    s : Finmap β
    ⊢ Or (Eq a a) (Not (Membership.mem s a))
  -/
  left
  /-
    case h
    α : Type u
    β : α → Type v
    inst✝ : DecidableEq α
    a : α
    s : Finmap β
    ⊢ Eq a a
  -/
  rfl
  /-
    🎉 no goals
  -/


@[simp]
theorem lookup_erase (a) (s : Finmap β) : lookup a (erase a s) = none :=
  induction_on s <| AList.lookup_erase a


@[simp]
theorem lookup_erase_ne {a a'} {s : Finmap β} (h : a ≠ a') : lookup a (erase a' s) = lookup a s :=
  induction_on s fun _ => AList.lookup_erase_ne h


theorem erase_erase {a a' : α} {s : Finmap β} : erase a (erase a' s) = erase a' (erase a s) :=
                                  /-
                                    α : Type u
                                    β : α → Type v
                                    inst✝ : DecidableEq α
                                    a a' : α
                                    s✝ : Finmap β
                                    s : AList β
                                    ⊢ Eq (Finmap.erase a (Finmap.erase a' s.toFinmap)).entries (Finmap.erase a' (F …
                                  -/
  induction_on s fun s => ext (by simp only [AList.erase_erase, erase_toFinmap])
                                  /-
                                    🎉 no goals
                                  -/


/-- `sdiff s s'` consists of all key-value pairs from `s` and `s'` where the keys are in `s` or
`s'` but not both. -/
def sdiff (s s' : Finmap β) : Finmap β :=
  s'.foldl (fun s x _ => s.erase x) (fun _ _ _ _ _ => erase_erase) s


instance : SDiff (Finmap β) :=
  ⟨sdiff⟩


/-- Insert a key-value pair into a finite map, replacing any existing pair with
  the same key. -/
def insert (a : α) (b : β a) (s : Finmap β) : Finmap β :=
  (liftOn s fun t => AList.toFinmap (AList.insert a b t)) fun _ _ p =>
    toFinmap_eq.2 <| perm_insert p


@[simp]
theorem insert_toFinmap (a : α) (b : β a) (s : AList β) :
    insert a b (AList.toFinmap s) = AList.toFinmap (s.insert a b) := by
  /-
    α : Type u
    β : α → Type v
    inst✝ : DecidableEq α
    a : α
    b : β a
    s : AList β
    ⊢ Eq (Finmap.insert a b s.toFinmap) (AList.insert a b s).toFinmap
  -/
  simp [insert]
  /-
    🎉 no goals
  -/


theorem entries_insert_of_not_mem {a : α} {b : β a} {s : Finmap β} :
    a ∉ s → (insert a b s).entries = ⟨a, b⟩ ::ₘ s.entries :=
  induction_on s fun s h => by
    -- Porting note: `-entries_insert` required
    /-
      α : Type u
      β : α → Type v
      inst✝ : DecidableEq α
      a : α
      b : β a
      s✝ : Finmap β
      s : AList β
      h : Not (Membership.mem s.toFinmap a)
      ⊢ Eq (Finmap.insert a b s.toFinmap).entries (Multiset.cons ⟨a, b⟩ s.toFinmap.e …
    -/
    simp [AList.entries_insert_of_not_mem (mt mem_toFinmap.1 h), -entries_insert]
    /-
      🎉 no goals
    -/


@[deprecated (since := "2024-12-14")] alias insert_entries_of_neg := entries_insert_of_not_mem


@[simp]
theorem mem_insert {a a' : α} {b' : β a'} {s : Finmap β} : a ∈ insert a' b' s ↔ a = a' ∨ a ∈ s :=
  induction_on s AList.mem_insert


@[simp]
theorem lookup_insert {a} {b : β a} (s : Finmap β) : lookup a (insert a b s) = some b :=
                             /-
                               α : Type u
                               β : α → Type v
                               inst✝ : DecidableEq α
                               a : α
                               b : β a
                               s✝ : Finmap β
                               s : AList β
                               ⊢ Eq (Finmap.lookup a (Finmap.insert a b s.toFinmap)) (Option.some b)
                             -/
  induction_on s fun s => by simp only [insert_toFinmap, lookup_toFinmap, AList.lookup_insert]
                             /-
                               🎉 no goals
                             -/


@[simp]
theorem lookup_insert_of_ne {a a'} {b : β a} (s : Finmap β) (h : a' ≠ a) :
    lookup a' (insert a b s) = lookup a' s :=
                             /-
                               α : Type u
                               β : α → Type v
                               inst✝ : DecidableEq α
                               a a' : α
                               b : β a
                               s✝ : Finmap β
                               h : Ne a' a
                               s : AList β
                               ⊢ Eq (Finmap.lookup a' (Finmap.insert a b s.toFinmap)) (Finmap.lookup a' s.toF …
                             -/
  induction_on s fun s => by simp only [insert_toFinmap, lookup_toFinmap, lookup_insert_ne h]
                             /-
                               🎉 no goals
                             -/


@[simp]
theorem insert_insert {a} {b b' : β a} (s : Finmap β) :
    (s.insert a b).insert a b' = s.insert a b' :=
                             /-
                               α : Type u
                               β : α → Type v
                               inst✝ : DecidableEq α
                               a : α
                               b b' : β a
                               s✝ : Finmap β
                               s : AList β
                               ⊢ Eq (Finmap.insert a b' (Finmap.insert a b s.toFinmap)) (Finmap.insert a b' s …
                             -/
  induction_on s fun s => by simp only [insert_toFinmap, AList.insert_insert]
                             /-
                               🎉 no goals
                             -/


theorem insert_insert_of_ne {a a'} {b : β a} {b' : β a'} (s : Finmap β) (h : a ≠ a') :
    (s.insert a b).insert a' b' = (s.insert a' b').insert a b :=
  induction_on s fun s => by
    /-
      α : Type u
      β : α → Type v
      inst✝ : DecidableEq α
      a a' : α
      b : β a
      b' : β a'
      s✝ : Finmap β
      h : Ne a a'
      s : AList β
      ⊢ Eq (Finmap.insert a' b' (Finmap.insert a b s.toFinmap)) (Finmap.insert a b ( …
    -/
    simp only [insert_toFinmap, AList.toFinmap_eq, AList.insert_insert_of_ne _ h]
    /-
      🎉 no goals
    -/


theorem toFinmap_cons (a : α) (b : β a) (xs : List (Sigma β)) :
    List.toFinmap (⟨a, b⟩ :: xs) = insert a b xs.toFinmap :=
  rfl


theorem mem_list_toFinmap (a : α) (xs : List (Sigma β)) :
    a ∈ xs.toFinmap ↔ ∃ b : β a, Sigma.mk a b ∈ xs := by
  -- Porting note: golfed
  /-
    α : Type u
    β : α → Type v
    inst✝ : DecidableEq α
    a : α
    xs : List (Sigma β)
    ⊢ Iff (Membership.mem xs.toFinmap a) (Exists fun b => Membership.mem xs ⟨a, b⟩)
  -/
  induction' xs with x xs
    /-
      case nil
      α : Type u
      β : α → Type v
      inst✝ : DecidableEq α
      a : α
      ⊢ Iff (Membership.mem List.nil.toFinmap a) (Exists fun b => Membership.mem Lis …
    -/
  · simp only [toFinmap_nil, not_mem_empty, find?, not_mem_nil, exists_false]
    /-
      🎉 no goals
    -/
  /-
    case cons
    α : Type u
    β : α → Type v
    inst✝ : DecidableEq α
    a : α
    x : Sigma β
    xs : List (Sigma β)
    tail_ih✝ : Iff (Membership.mem xs.toFinmap a) (Exists fun b => Membership.mem  …
    ⊢ Iff (Membership.mem (List.cons x xs).toFinmap a) (Exists fun b => Membership …
  -/
  cases' x with fst_i snd_i
  -- Porting note: `Sigma.mk.inj_iff` required because `simp` behaves differently
  /-
    case cons.mk
    α : Type u
    β : α → Type v
    inst✝ : DecidableEq α
    a : α
    xs : List (Sigma β)
    tail_ih✝ : Iff (Membership.mem xs.toFinmap a) (Exists fun b => Membership.mem  …
    fst_i : α
    snd_i : β fst_i
    ⊢ Iff (Membership.mem (List.cons ⟨fst_i, snd_i⟩ xs).toFinmap a) (Exists fun b  …
  -/
  simp only [toFinmap_cons, *, exists_or, mem_cons, mem_insert, exists_and_left, Sigma.mk.inj_iff]
  /-
    case cons.mk
    α : Type u
    β : α → Type v
    inst✝ : DecidableEq α
    a : α
    xs : List (Sigma β)
    tail_ih✝ : Iff (Membership.mem xs.toFinmap a) (Exists fun b => Membership.mem  …
    fst_i : α
    snd_i : β fst_i
    ⊢ Iff (Or (Eq a fst_i) (Exists fun b => Membership.mem xs ⟨a, b⟩)) (Or (And (E …
  -/
  refine (or_congr_left <| and_iff_left_of_imp ?_).symm
  /-
    case cons.mk
    α : Type u
    β : α → Type v
    inst✝ : DecidableEq α
    a : α
    xs : List (Sigma β)
    tail_ih✝ : Iff (Membership.mem xs.toFinmap a) (Exists fun b => Membership.mem  …
    fst_i : α
    snd_i : β fst_i
    ⊢ Eq a fst_i → Exists fun x => HEq x snd_i
  -/
  rintro rfl
  /-
    case cons.mk
    α : Type u
    β : α → Type v
    inst✝ : DecidableEq α
    a : α
    xs : List (Sigma β)
    tail_ih✝ : Iff (Membership.mem xs.toFinmap a) (Exists fun b => Membership.mem  …
    snd_i : β a
    ⊢ Exists fun x => HEq x snd_i
  -/
  simp only [exists_eq, heq_iff_eq]
  /-
    🎉 no goals
  -/


@[simp]
theorem insert_singleton_eq {a : α} {b b' : β a} : insert a b (singleton a b') = singleton a b := by
  /-
    α : Type u
    β : α → Type v
    inst✝ : DecidableEq α
    a : α
    b b' : β a
    ⊢ Eq (Finmap.insert a b (Finmap.singleton a b')) (Finmap.singleton a b)
  -/
  simp only [singleton, Finmap.insert_toFinmap, AList.insert_singleton_eq]
  /-
    🎉 no goals
  -/


/-- Erase a key from the map, and return the corresponding value, if found. -/
def extract (a : α) (s : Finmap β) : Option (β a) × Finmap β :=
  (liftOn s fun t => Prod.map id AList.toFinmap (AList.extract a t)) fun s₁ s₂ p => by
    /-
      α : Type u
      β : α → Type v
      inst✝ : DecidableEq α
      a : α
      s : Finmap β
      s₁ s₂ : AList β
      p : s₁.entries.Perm s₂.entries
      ⊢ Eq (Prod.map id AList.toFinmap (AList.extract a s₁)) (Prod.map id AList.toFi …
    -/
    simp [perm_lookup p, toFinmap_eq, perm_erase p]
    /-
      🎉 no goals
    -/


@[simp]
theorem extract_eq_lookup_erase (a : α) (s : Finmap β) : extract a s = (lookup a s, erase a s) :=
                             /-
                               α : Type u
                               β : α → Type v
                               inst✝ : DecidableEq α
                               a : α
                               s✝ : Finmap β
                               s : AList β
                               ⊢ Eq (Finmap.extract a s.toFinmap) { fst := Finmap.lookup a s.toFinmap, snd := …
                             -/
  induction_on s fun s => by simp [extract]
                             /-
                               🎉 no goals
                             -/


/-- `s₁ ∪ s₂` is the key-based union of two finite maps. It is left-biased: if
there exists an `a ∈ s₁`, `lookup a (s₁ ∪ s₂) = lookup a s₁`. -/
def union (s₁ s₂ : Finmap β) : Finmap β :=
  (liftOn₂ s₁ s₂ fun s₁ s₂ => (AList.toFinmap (s₁ ∪ s₂))) fun _ _ _ _ p₁₃ p₂₄ =>
    toFinmap_eq.mpr <| perm_union p₁₃ p₂₄


instance : Union (Finmap β) :=
  ⟨union⟩


@[simp]
theorem mem_union {a} {s₁ s₂ : Finmap β} : a ∈ s₁ ∪ s₂ ↔ a ∈ s₁ ∨ a ∈ s₂ :=
  induction_on₂ s₁ s₂ fun _ _ => AList.mem_union


@[simp]
theorem union_toFinmap (s₁ s₂ : AList β) : (toFinmap s₁) ∪ (toFinmap s₂) = toFinmap (s₁ ∪ s₂) := by
  /-
    α : Type u
    β : α → Type v
    inst✝ : DecidableEq α
    s₁ s₂ : AList β
    ⊢ Eq (Union.union s₁.toFinmap s₂.toFinmap) (Union.union s₁ s₂).toFinmap
  -/
  simp [(· ∪ ·), union]
  /-
    🎉 no goals
  -/


theorem keys_union {s₁ s₂ : Finmap β} : (s₁ ∪ s₂).keys = s₁.keys ∪ s₂.keys :=
                                                    /-
                                                      α : Type u
                                                      β : α → Type v
                                                      inst✝ : DecidableEq α
                                                      s₁✝ s₂✝ : Finmap β
                                                      s₁ s₂ : AList β
                                                      ⊢ ∀ (a : α), Iff (Membership.mem (Union.union s₁.toFinmap s₂.toFinmap).keys a) …
                                                    -/
  induction_on₂ s₁ s₂ fun s₁ s₂ => Finset.ext <| by simp [keys]
                                                    /-
                                                      🎉 no goals
                                                    -/


@[simp]
theorem lookup_union_left {a} {s₁ s₂ : Finmap β} : a ∈ s₁ → lookup a (s₁ ∪ s₂) = lookup a s₁ :=
  induction_on₂ s₁ s₂ fun _ _ => AList.lookup_union_left


@[simp]
theorem lookup_union_right {a} {s₁ s₂ : Finmap β} : a ∉ s₁ → lookup a (s₁ ∪ s₂) = lookup a s₂ :=
  induction_on₂ s₁ s₂ fun _ _ => AList.lookup_union_right


theorem lookup_union_left_of_not_in {a} {s₁ s₂ : Finmap β} (h : a ∉ s₂) :
    lookup a (s₁ ∪ s₂) = lookup a s₁ := by
  /-
    α : Type u
    β : α → Type v
    inst✝ : DecidableEq α
    a : α
    s₁ s₂ : Finmap β
    h : Not (Membership.mem s₂ a)
    ⊢ Eq (Finmap.lookup a (Union.union s₁ s₂)) (Finmap.lookup a s₁)
  -/
  by_cases h' : a ∈ s₁
    /-
      case pos
      α : Type u
      β : α → Type v
      inst✝ : DecidableEq α
      a : α
      s₁ s₂ : Finmap β
      h : Not (Membership.mem s₂ a)
      h' : Membership.mem s₁ a
      ⊢ Eq (Finmap.lookup a (Union.union s₁ s₂)) (Finmap.lookup a s₁)
    -/
  · rw [lookup_union_left h']
    /-
      🎉 no goals
    -/
    /-
      case neg
      α : Type u
      β : α → Type v
      inst✝ : DecidableEq α
      a : α
      s₁ s₂ : Finmap β
      h : Not (Membership.mem s₂ a)
      h' : Not (Membership.mem s₁ a)
      ⊢ Eq (Finmap.lookup a (Union.union s₁ s₂)) (Finmap.lookup a s₁)
    -/
  · rw [lookup_union_right h', lookup_eq_none.mpr h, lookup_eq_none.mpr h']
    /-
      🎉 no goals
    -/


/-- `simp`-normal form of `mem_lookup_union` -/
@[simp]
theorem mem_lookup_union' {a} {b : β a} {s₁ s₂ : Finmap β} :
    lookup a (s₁ ∪ s₂) = some b ↔ b ∈ lookup a s₁ ∨ a ∉ s₁ ∧ b ∈ lookup a s₂ :=
  induction_on₂ s₁ s₂ fun _ _ => AList.mem_lookup_union


theorem mem_lookup_union {a} {b : β a} {s₁ s₂ : Finmap β} :
    b ∈ lookup a (s₁ ∪ s₂) ↔ b ∈ lookup a s₁ ∨ a ∉ s₁ ∧ b ∈ lookup a s₂ :=
  induction_on₂ s₁ s₂ fun _ _ => AList.mem_lookup_union


theorem mem_lookup_union_middle {a} {b : β a} {s₁ s₂ s₃ : Finmap β} :
    b ∈ lookup a (s₁ ∪ s₃) → a ∉ s₂ → b ∈ lookup a (s₁ ∪ s₂ ∪ s₃) :=
  induction_on₃ s₁ s₂ s₃ fun _ _ _ => AList.mem_lookup_union_middle


theorem insert_union {a} {b : β a} {s₁ s₂ : Finmap β} : insert a b (s₁ ∪ s₂) = insert a b s₁ ∪ s₂ :=
                                      /-
                                        α : Type u
                                        β : α → Type v
                                        inst✝ : DecidableEq α
                                        a : α
                                        b : β a
                                        s₁ s₂ : Finmap β
                                        a₁ a₂ : AList β
                                        ⊢ Eq (Finmap.insert a b (Union.union a₁.toFinmap a₂.toFinmap)) (Union.union (F …
                                      -/
  induction_on₂ s₁ s₂ fun a₁ a₂ => by simp [AList.insert_union]
                                      /-
                                        🎉 no goals
                                      -/


theorem union_assoc {s₁ s₂ s₃ : Finmap β} : s₁ ∪ s₂ ∪ s₃ = s₁ ∪ (s₂ ∪ s₃) :=
  induction_on₃ s₁ s₂ s₃ fun s₁ s₂ s₃ => by
    /-
      α : Type u
      β : α → Type v
      inst✝ : DecidableEq α
      s₁✝ s₂✝ s₃✝ : Finmap β
      s₁ s₂ s₃ : AList β
      ⊢ Eq (Union.union (Union.union s₁.toFinmap s₂.toFinmap) s₃.toFinmap) (Union.un …
    -/
    simp only [AList.toFinmap_eq, union_toFinmap, AList.union_assoc]
    /-
      🎉 no goals
    -/


@[simp]
theorem empty_union {s₁ : Finmap β} : ∅ ∪ s₁ = s₁ :=
  induction_on s₁ fun s₁ => by
    /-
      α : Type u
      β : α → Type v
      inst✝ : DecidableEq α
      s₁✝ : Finmap β
      s₁ : AList β
      ⊢ Eq (Union.union EmptyCollection.emptyCollection s₁.toFinmap) s₁.toFinmap
    -/
    rw [← empty_toFinmap]
    /-
      α : Type u
      β : α → Type v
      inst✝ : DecidableEq α
      s₁✝ : Finmap β
      s₁ : AList β
      ⊢ Eq (Union.union EmptyCollection.emptyCollection.toFinmap s₁.toFinmap) s₁.toF …
    -/
    simp [-empty_toFinmap, AList.toFinmap_eq, union_toFinmap, AList.union_assoc]
    /-
      🎉 no goals
    -/


@[simp]
theorem union_empty {s₁ : Finmap β} : s₁ ∪ ∅ = s₁ :=
  induction_on s₁ fun s₁ => by
    /-
      α : Type u
      β : α → Type v
      inst✝ : DecidableEq α
      s₁✝ : Finmap β
      s₁ : AList β
      ⊢ Eq (Union.union s₁.toFinmap EmptyCollection.emptyCollection) s₁.toFinmap
    -/
    rw [← empty_toFinmap]
    /-
      α : Type u
      β : α → Type v
      inst✝ : DecidableEq α
      s₁✝ : Finmap β
      s₁ : AList β
      ⊢ Eq (Union.union s₁.toFinmap EmptyCollection.emptyCollection.toFinmap) s₁.toF …
    -/
    simp [-empty_toFinmap, AList.toFinmap_eq, union_toFinmap, AList.union_assoc]
    /-
      🎉 no goals
    -/


theorem erase_union_singleton (a : α) (b : β a) (s : Finmap β) (h : s.lookup a = some b) :
    s.erase a ∪ singleton a b = s :=
  ext_lookup fun x => by
    /-
      α : Type u
      β : α → Type v
      inst✝ : DecidableEq α
      a : α
      b : β a
      s : Finmap β
      h : Eq (Finmap.lookup a s) (Option.some b)
      x : α
      ⊢ Eq (Finmap.lookup x (Union.union (Finmap.erase a s) (Finmap.singleton a b))) …
    -/
    by_cases h' : x = a
      /-
        case pos
        α : Type u
        β : α → Type v
        inst✝ : DecidableEq α
        a : α
        b : β a
        s : Finmap β
        h : Eq (Finmap.lookup a s) (Option.some b)
        x : α
        h' : Eq x a
        ⊢ Eq (Finmap.lookup x (Union.union (Finmap.erase a s) (Finmap.singleton a b))) …
      -/
    · subst a
      /-
        case pos
        α : Type u
        β : α → Type v
        inst✝ : DecidableEq α
        s : Finmap β
        x : α
        b : β x
        h : Eq (Finmap.lookup x s) (Option.some b)
        ⊢ Eq (Finmap.lookup x (Union.union (Finmap.erase x s) (Finmap.singleton x b))) …
      -/
      rw [lookup_union_right not_mem_erase_self, lookup_singleton_eq, h]
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
        s : Finmap β
        h : Eq (Finmap.lookup a s) (Option.some b)
        x : α
        h' : Not (Eq x a)
        ⊢ Eq (Finmap.lookup x (Union.union (Finmap.erase a s) (Finmap.singleton a b))) …
      -/
    · have : x ∉ singleton a b := by rwa [mem_singleton]
      /-
        case neg
        α : Type u
        β : α → Type v
        inst✝ : DecidableEq α
        a : α
        b : β a
        s : Finmap β
        h : Eq (Finmap.lookup a s) (Option.some b)
        x : α
        h' : Not (Eq x a)
        this : Not (Membership.mem (Finmap.singleton a b) x)
        ⊢ Eq (Finmap.lookup x (Union.union (Finmap.erase a s) (Finmap.singleton a b))) …
      -/
      rw [lookup_union_left_of_not_in this, lookup_erase_ne h']
      /-
        🎉 no goals
      -/


/-- `Disjoint s₁ s₂` holds if `s₁` and `s₂` have no keys in common. -/
def Disjoint (s₁ s₂ : Finmap β) : Prop :=
  ∀ x ∈ s₁, ¬x ∈ s₂


theorem disjoint_empty (x : Finmap β) : Disjoint ∅ x :=
  nofun


@[symm]
theorem Disjoint.symm (x y : Finmap β) (h : Disjoint x y) : Disjoint y x := fun p hy hx => h p hx hy


theorem Disjoint.symm_iff (x y : Finmap β) : Disjoint x y ↔ Disjoint y x :=
  ⟨Disjoint.symm x y, Disjoint.symm y x⟩


                                                         /-
                                                           α : Type u
                                                           β : α → Type v
                                                           inst✝ : DecidableEq α
                                                           x y : Finmap β
                                                           ⊢ Decidable (x.Disjoint y)
                                                         -/
instance : DecidableRel (@Disjoint α β) := fun x y => by dsimp only [Disjoint]; infer_instance
                                                                                /-
                                                                                  🎉 no goals
                                                                                -/


theorem disjoint_union_left (x y z : Finmap β) :
    Disjoint (x ∪ y) z ↔ Disjoint x z ∧ Disjoint y z := by
  /-
    α : Type u
    β : α → Type v
    inst✝ : DecidableEq α
    x y z : Finmap β
    ⊢ Iff ((Union.union x y).Disjoint z) (And (x.Disjoint z) (y.Disjoint z))
  -/
  simp [Disjoint, Finmap.mem_union, or_imp, forall_and]
  /-
    🎉 no goals
  -/


theorem disjoint_union_right (x y z : Finmap β) :
    Disjoint x (y ∪ z) ↔ Disjoint x y ∧ Disjoint x z := by
  /-
    α : Type u
    β : α → Type v
    inst✝ : DecidableEq α
    x y z : Finmap β
    ⊢ Iff (x.Disjoint (Union.union y z)) (And (x.Disjoint y) (x.Disjoint z))
  -/
  rw [Disjoint.symm_iff, disjoint_union_left, Disjoint.symm_iff _ x, Disjoint.symm_iff _ x]
  /-
    🎉 no goals
  -/


theorem union_comm_of_disjoint {s₁ s₂ : Finmap β} : Disjoint s₁ s₂ → s₁ ∪ s₂ = s₂ ∪ s₁ :=
  induction_on₂ s₁ s₂ fun s₁ s₂ => by
    /-
      α : Type u
      β : α → Type v
      inst✝ : DecidableEq α
      s₁✝ s₂✝ : Finmap β
      s₁ s₂ : AList β
      ⊢ s₁.toFinmap.Disjoint s₂.toFinmap → Eq (Union.union s₁.toFinmap s₂.toFinmap)  …
    -/
    intro h
    /-
      α : Type u
      β : α → Type v
      inst✝ : DecidableEq α
      s₁✝ s₂✝ : Finmap β
      s₁ s₂ : AList β
      h : s₁.toFinmap.Disjoint s₂.toFinmap
      ⊢ Eq (Union.union s₁.toFinmap s₂.toFinmap) (Union.union s₂.toFinmap s₁.toFinmap)
    -/
    simp only [AList.toFinmap_eq, union_toFinmap, AList.union_comm_of_disjoint h]
    /-
      🎉 no goals
    -/


theorem union_cancel {s₁ s₂ s₃ : Finmap β} (h : Disjoint s₁ s₃) (h' : Disjoint s₂ s₃) :
    s₁ ∪ s₃ = s₂ ∪ s₃ ↔ s₁ = s₂ :=
  ⟨fun h'' => by
    /-
      α : Type u
      β : α → Type v
      inst✝ : DecidableEq α
      s₁ s₂ s₃ : Finmap β
      h : s₁.Disjoint s₃
      h' : s₂.Disjoint s₃
      h'' : Eq (Union.union s₁ s₃) (Union.union s₂ s₃)
      ⊢ Eq s₁ s₂
    -/
    apply ext_lookup
    /-
      case a
      α : Type u
      β : α → Type v
      inst✝ : DecidableEq α
      s₁ s₂ s₃ : Finmap β
      h : s₁.Disjoint s₃
      h' : s₂.Disjoint s₃
      h'' : Eq (Union.union s₁ s₃) (Union.union s₂ s₃)
      ⊢ ∀ (x : α), Eq (Finmap.lookup x s₁) (Finmap.lookup x s₂)
    -/
    intro x
    /-
      case a
      α : Type u
      β : α → Type v
      inst✝ : DecidableEq α
      s₁ s₂ s₃ : Finmap β
      h : s₁.Disjoint s₃
      h' : s₂.Disjoint s₃
      h'' : Eq (Union.union s₁ s₃) (Union.union s₂ s₃)
      x : α
      ⊢ Eq (Finmap.lookup x s₁) (Finmap.lookup x s₂)
    -/
    have : (s₁ ∪ s₃).lookup x = (s₂ ∪ s₃).lookup x := h'' ▸ rfl
    /-
      case a
      α : Type u
      β : α → Type v
      inst✝ : DecidableEq α
      s₁ s₂ s₃ : Finmap β
      h : s₁.Disjoint s₃
      h' : s₂.Disjoint s₃
      h'' : Eq (Union.union s₁ s₃) (Union.union s₂ s₃)
      x : α
      this : Eq (Finmap.lookup x (Union.union s₁ s₃)) (Finmap.lookup x (Union.union  …
      ⊢ Eq (Finmap.lookup x s₁) (Finmap.lookup x s₂)
    -/
    by_cases hs₁ : x ∈ s₁
      /-
        case pos
        α : Type u
        β : α → Type v
        inst✝ : DecidableEq α
        s₁ s₂ s₃ : Finmap β
        h : s₁.Disjoint s₃
        h' : s₂.Disjoint s₃
        h'' : Eq (Union.union s₁ s₃) (Union.union s₂ s₃)
        x : α
        this : Eq (Finmap.lookup x (Union.union s₁ s₃)) (Finmap.lookup x (Union.union  …
        hs₁ : Membership.mem s₁ x
        ⊢ Eq (Finmap.lookup x s₁) (Finmap.lookup x s₂)
      -/
    · rwa [lookup_union_left hs₁, lookup_union_left_of_not_in (h _ hs₁)] at this
      /-
        🎉 no goals
      -/
      /-
        case neg
        α : Type u
        β : α → Type v
        inst✝ : DecidableEq α
        s₁ s₂ s₃ : Finmap β
        h : s₁.Disjoint s₃
        h' : s₂.Disjoint s₃
        h'' : Eq (Union.union s₁ s₃) (Union.union s₂ s₃)
        x : α
        this : Eq (Finmap.lookup x (Union.union s₁ s₃)) (Finmap.lookup x (Union.union  …
        hs₁ : Not (Membership.mem s₁ x)
        ⊢ Eq (Finmap.lookup x s₁) (Finmap.lookup x s₂)
      -/
    · by_cases hs₂ : x ∈ s₂
        /-
          case pos
          α : Type u
          β : α → Type v
          inst✝ : DecidableEq α
          s₁ s₂ s₃ : Finmap β
          h : s₁.Disjoint s₃
          h' : s₂.Disjoint s₃
          h'' : Eq (Union.union s₁ s₃) (Union.union s₂ s₃)
          x : α
          this : Eq (Finmap.lookup x (Union.union s₁ s₃)) (Finmap.lookup x (Union.union  …
          hs₁ : Not (Membership.mem s₁ x)
          hs₂ : Membership.mem s₂ x
          ⊢ Eq (Finmap.lookup x s₁) (Finmap.lookup x s₂)
        -/
      · rwa [lookup_union_left_of_not_in (h' _ hs₂), lookup_union_left hs₂] at this
        /-
          🎉 no goals
        -/
        /-
          case neg
          α : Type u
          β : α → Type v
          inst✝ : DecidableEq α
          s₁ s₂ s₃ : Finmap β
          h : s₁.Disjoint s₃
          h' : s₂.Disjoint s₃
          h'' : Eq (Union.union s₁ s₃) (Union.union s₂ s₃)
          x : α
          this : Eq (Finmap.lookup x (Union.union s₁ s₃)) (Finmap.lookup x (Union.union  …
          hs₁ : Not (Membership.mem s₁ x)
          hs₂ : Not (Membership.mem s₂ x)
          ⊢ Eq (Finmap.lookup x s₁) (Finmap.lookup x s₂)
        -/
      · rw [lookup_eq_none.mpr hs₁, lookup_eq_none.mpr hs₂], fun h => h ▸ rfl⟩
        /-
          🎉 no goals
        -/


