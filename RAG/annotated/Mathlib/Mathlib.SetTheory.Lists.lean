/-- Prelists, helper type to define `Lists`. `Lists' α false` are the "atoms", a copy of `α`.
`Lists' α true` are the "proper" ZFA prelists, inductively defined from the empty ZFA prelist and
from appending a ZFA prelist to a proper ZFA prelist. It is made so that you can't append anything
to an atom while having only one appending function for appending both atoms and proper ZFC prelists
to a proper ZFA prelist. -/
inductive Lists'.{u} (α : Type u) : Bool → Type u
  | atom : α → Lists' α false
  | nil : Lists' α true
  | cons' {b} : Lists' α b → Lists' α true → Lists' α true
  deriving DecidableEq

compile_inductive% Lists'


/-- Hereditarily finite list, aka ZFA list. A ZFA list is either an "atom" (`b = false`),
corresponding to an element of `α`, or a "proper" ZFA list, inductively defined from the empty ZFA
list and from appending a ZFA list to a proper ZFA list. -/
def Lists (α : Type*) :=
  Σb, Lists' α b


instance [Inhabited α] : ∀ b, Inhabited (Lists' α b)
  | true => ⟨nil⟩
  | false => ⟨atom default⟩


/-- Appending a ZFA list to a proper ZFA prelist. -/
def cons : Lists α → Lists' α true → Lists' α true
  | ⟨_, a⟩, l => cons' a l


/-- Converts a ZFA prelist to a `List` of ZFA lists. Atoms are sent to `[]`. -/
@[simp]
def toList : ∀ {b}, Lists' α b → List (Lists α)
  | _, atom _ => []
  | _, nil => []
  | _, cons' a l => ⟨_, a⟩ :: l.toList


@[simp]
theorem toList_cons (a : Lists α) (l) : toList (cons a l) = a :: l.toList := rfl


/-- Converts a `List` of ZFA lists to a proper ZFA prelist. -/
@[simp]
def ofList : List (Lists α) → Lists' α true
  | [] => nil
  | a :: l => cons a (ofList l)


@[simp]
                                                                     /-
                                                                       α : Type u_1
                                                                       l : List (Lists α)
                                                                       ⊢ Eq (Lists'.ofList l).toList l
                                                                     -/
                                                                                     /-
                                                                                       🎉 no goals
                                                                                     -/
theorem to_ofList (l : List (Lists α)) : toList (ofList l) = l := by induction l <;> simp [*]
                                                                                     /-
                                                                                       🎉 no goals
                                                                                     -/


@[simp]
theorem of_toList : ∀ l : Lists' α true, ofList (toList l) = l :=
  suffices
    ∀ (b) (h : true = b) (l : Lists' α b),
                                   /-
                                     α : Type u_1
                                     b : Bool
                                     h : Eq Bool.true b
                                     l : Lists' α b
                                     ⊢ Lists' α Bool.true
                                   -/
      let l' : Lists' α true := by rw [h]; exact l
                                           /-
                                             🎉 no goals
                                           -/
      ofList (toList l') = l'
    from this _ rfl
  fun b h l => by
    induction l with
    | atom => cases h
    | nil => simp
    | cons' b a _ IH => simpa [cons] using IH rfl


mutual
  /-- Equivalence of ZFA lists. Defined inductively. -/
  inductive Lists.Equiv : Lists α → Lists α → Prop
    | refl (l) : Lists.Equiv l l
    | antisymm {l₁ l₂ : Lists' α true} :
      Lists'.Subset l₁ l₂ → Lists'.Subset l₂ l₁ → Lists.Equiv ⟨_, l₁⟩ ⟨_, l₂⟩

  /-- Subset relation for ZFA lists. Defined inductively. -/
  inductive Lists'.Subset : Lists' α true → Lists' α true → Prop
    | nil {l} : Lists'.Subset Lists'.nil l
    | cons {a a' l l'} :
      Lists.Equiv a a' →
        a' ∈ Lists'.toList l' → Lists'.Subset l l' → Lists'.Subset (Lists'.cons a l) l'
end


local infixl:50 " ~ " => Lists.Equiv


instance : HasSubset (Lists' α true) :=
  ⟨Lists'.Subset⟩


/-- ZFA prelist membership. A ZFA list is in a ZFA prelist if some element of this ZFA prelist is
equivalent as a ZFA list to this ZFA list. -/
instance {b} : Membership (Lists α) (Lists' α b) :=
  ⟨fun l a => ∃ a' ∈ l.toList, a ~ a'⟩


theorem mem_def {b a} {l : Lists' α b} : a ∈ l ↔ ∃ a' ∈ l.toList, a ~ a' :=
  Iff.rfl


@[simp]
theorem mem_cons {a y l} : a ∈ @cons α y l ↔ a ~ y ∨ a ∈ l := by
  /-
    α : Type u_1
    a y : Lists α
    l : Lists' α Bool.true
    ⊢ Iff (Membership.mem (Lists'.cons y l) a) (Or (a.Equiv y) (Membership.mem l a))
  -/
  simp [mem_def, or_and_right, exists_or]
  /-
    🎉 no goals
  -/


theorem cons_subset {a} {l₁ l₂ : Lists' α true} : Lists'.cons a l₁ ⊆ l₂ ↔ a ∈ l₂ ∧ l₁ ⊆ l₂ := by
  /-
    α : Type u_1
    a : Lists α
    l₁ l₂ : Lists' α Bool.true
    ⊢ Iff (HasSubset.Subset (Lists'.cons a l₁) l₂) (And (Membership.mem l₂ a) (Has …
  -/
  refine ⟨fun h => ?_, fun ⟨⟨a', m, e⟩, s⟩ => Subset.cons e m s⟩
  /-
    α : Type u_1
    a : Lists α
    l₁ l₂ : Lists' α Bool.true
    h : HasSubset.Subset (Lists'.cons a l₁) l₂
    ⊢ And (Membership.mem l₂ a) (HasSubset.Subset l₁ l₂)
  -/
  generalize h' : Lists'.cons a l₁ = l₁' at h
  /-
    α : Type u_1
    a : Lists α
    l₁ l₂ l₁' : Lists' α Bool.true
    h' : Eq (Lists'.cons a l₁) l₁'
    h : HasSubset.Subset l₁' l₂
    ⊢ And (Membership.mem l₂ a) (HasSubset.Subset l₁ l₂)
  -/
  cases' h with l a' a'' l l' e m s
    /-
      case nil
      α : Type u_1
      a : Lists α
      l₁ l₂ : Lists' α Bool.true
      h' : Eq (Lists'.cons a l₁) Lists'.nil
      ⊢ And (Membership.mem l₂ a) (HasSubset.Subset l₁ l₂)
    -/
  · cases a
    /-
      case nil.mk
      α : Type u_1
      l₁ l₂ : Lists' α Bool.true
      fst✝ : Bool
      snd✝ : Lists' α fst✝
      h' : Eq (Lists'.cons ⟨fst✝, snd✝⟩ l₁) Lists'.nil
      ⊢ And (Membership.mem l₂ ⟨fst✝, snd✝⟩) (HasSubset.Subset l₁ l₂)
    -/
    cases h'
    /-
      🎉 no goals
    -/
  /-
    case cons
    α : Type u_1
    a : Lists α
    l₁ l₂ : Lists' α Bool.true
    a' a'' : Lists α
    l : Lists' α Bool.true
    e : a'.Equiv a''
    h' : Eq (Lists'.cons a l₁) (Lists'.cons a' l)
    m : Membership.mem l₂.toList a''
    s : l.Subset l₂
    ⊢ And (Membership.mem l₂ a) (HasSubset.Subset l₁ l₂)
  -/
  cases a; cases a'; cases h'; exact ⟨⟨_, m, e⟩, s⟩
                               /-
                                 🎉 no goals
                               -/


theorem ofList_subset {l₁ l₂ : List (Lists α)} (h : l₁ ⊆ l₂) :
    Lists'.ofList l₁ ⊆ Lists'.ofList l₂ := by
  induction l₁ with
  | nil => exact Subset.nil
  | cons _ _ l₁_ih =>
    refine Subset.cons (Lists.Equiv.refl _) ?_ (l₁_ih (List.subset_of_cons_subset h))
    simp only [List.cons_subset] at h; simp [h]


@[refl]
theorem Subset.refl {l : Lists' α true} : l ⊆ l := by
  /-
    α : Type u_1
    l : Lists' α Bool.true
    ⊢ HasSubset.Subset l l
  -/
  rw [← Lists'.of_toList l]; exact ofList_subset (List.Subset.refl _)
                             /-
                               🎉 no goals
                             -/


theorem subset_nil {l : Lists' α true} : l ⊆ Lists'.nil → l = Lists'.nil := by
  /-
    α : Type u_1
    l : Lists' α Bool.true
    ⊢ HasSubset.Subset l Lists'.nil → Eq l Lists'.nil
  -/
  rw [← of_toList l]
  /-
    α : Type u_1
    l : Lists' α Bool.true
    ⊢ HasSubset.Subset (Lists'.ofList l.toList) Lists'.nil → Eq (Lists'.ofList l.t …
  -/
  induction toList l <;> intro h
    /-
      case nil
      α : Type u_1
      l : Lists' α Bool.true
      h : HasSubset.Subset (Lists'.ofList List.nil) Lists'.nil
      ⊢ Eq (Lists'.ofList List.nil) Lists'.nil
    -/
  · rfl
    /-
      🎉 no goals
    -/
    /-
      case cons
      α : Type u_1
      l : Lists' α Bool.true
      head✝ : Lists α
      tail✝ : List (Lists α)
      tail_ih✝ : HasSubset.Subset (Lists'.ofList tail✝) Lists'.nil → Eq (Lists'.ofLi …
      h : HasSubset.Subset (Lists'.ofList (List.cons head✝ tail✝)) Lists'.nil
      ⊢ Eq (Lists'.ofList (List.cons head✝ tail✝)) Lists'.nil
    -/
  · rcases cons_subset.1 h with ⟨⟨_, ⟨⟩, _⟩, _⟩
    /-
      🎉 no goals
    -/


theorem mem_of_subset' {a} : ∀ {l₁ l₂ : Lists' α true} (_ : l₁ ⊆ l₂) (_ : a ∈ l₁.toList), a ∈ l₂
                                       /-
                                         α : Type u_1
                                         a : Lists α
                                         x✝ : Lists' α Bool.true
                                         h : Membership.mem Lists'.nil.toList a
                                         ⊢ Membership.mem x✝ a
                                       -/
  | nil, _, Lists'.Subset.nil, h => by cases h
                                       /-
                                         🎉 no goals
                                       -/
  | cons' a0 l0, l₂, s, h => by
    /-
      α : Type u_1
      a : Lists α
      b✝ : Bool
      a0 : Lists' α b✝
      l0 l₂ : Lists' α Bool.true
      s : HasSubset.Subset (a0.cons' l0) l₂
      h : Membership.mem (a0.cons' l0).toList a
      ⊢ Membership.mem l₂ a
    -/
    cases' s with _ _ _ _ _ e m s
    /-
      case cons
      α : Type u_1
      a : Lists α
      l0 l₂ : Lists' α Bool.true
      a✝ a'✝ : Lists α
      e : a✝.Equiv a'✝
      h : Membership.mem (a✝.snd.cons' l0).toList a
      m : Membership.mem l₂.toList a'✝
      s : l0.Subset l₂
      ⊢ Membership.mem l₂ a
    -/
    simp only [toList, Sigma.eta, List.find?, List.mem_cons] at h
    /-
      case cons
      α : Type u_1
      a : Lists α
      l0 l₂ : Lists' α Bool.true
      a✝ a'✝ : Lists α
      e : a✝.Equiv a'✝
      m : Membership.mem l₂.toList a'✝
      s : l0.Subset l₂
      h : Or (Eq a a✝) (Membership.mem l0.toList a)
      ⊢ Membership.mem l₂ a
    -/
    rcases h with (rfl | h)
      /-
        case cons.inl
        α : Type u_1
        a : Lists α
        l0 l₂ : Lists' α Bool.true
        a'✝ : Lists α
        m : Membership.mem l₂.toList a'✝
        s : l0.Subset l₂
        e : a.Equiv a'✝
        ⊢ Membership.mem l₂ a
      -/
    · exact ⟨_, m, e⟩
      /-
        🎉 no goals
      -/
      /-
        case cons.inr
        α : Type u_1
        a : Lists α
        l0 l₂ : Lists' α Bool.true
        a✝ a'✝ : Lists α
        e : a✝.Equiv a'✝
        m : Membership.mem l₂.toList a'✝
        s : l0.Subset l₂
        h : Membership.mem l0.toList a
        ⊢ Membership.mem l₂ a
      -/
    · exact mem_of_subset' s h
      /-
        🎉 no goals
      -/


theorem subset_def {l₁ l₂ : Lists' α true} : l₁ ⊆ l₂ ↔ ∀ a ∈ l₁.toList, a ∈ l₂ :=
  ⟨fun H _ => mem_of_subset' H, fun H => by
    /-
      α : Type u_1
      l₁ l₂ : Lists' α Bool.true
      H : ∀ (a : Lists α), Membership.mem l₁.toList a → Membership.mem l₂ a
      ⊢ HasSubset.Subset l₁ l₂
    -/
    rw [← of_toList l₁]
    /-
      α : Type u_1
      l₁ l₂ : Lists' α Bool.true
      H : ∀ (a : Lists α), Membership.mem l₁.toList a → Membership.mem l₂ a
      ⊢ HasSubset.Subset (Lists'.ofList l₁.toList) l₂
    -/
    revert H; induction' toList l₁ with h t t_ih <;> intro H
      /-
        case nil
        α : Type u_1
        l₁ l₂ : Lists' α Bool.true
        H : ∀ (a : Lists α), Membership.mem List.nil a → Membership.mem l₂ a
        ⊢ HasSubset.Subset (Lists'.ofList List.nil) l₂
      -/
    · exact Subset.nil
      /-
        🎉 no goals
      -/
      /-
        case cons
        α : Type u_1
        l₁ l₂ : Lists' α Bool.true
        h : Lists α
        t : List (Lists α)
        t_ih : (∀ (a : Lists α), Membership.mem t a → Membership.mem l₂ a) → HasSubset …
        H : ∀ (a : Lists α), Membership.mem (List.cons h t) a → Membership.mem l₂ a
        ⊢ HasSubset.Subset (Lists'.ofList (List.cons h t)) l₂
      -/
    · simp only [ofList, List.find?, List.mem_cons, forall_eq_or_imp] at *
      /-
        case cons
        α : Type u_1
        l₁ l₂ : Lists' α Bool.true
        h : Lists α
        t : List (Lists α)
        t_ih : (∀ (a : Lists α), Membership.mem t a → Membership.mem l₂ a) → HasSubset …
        H : And (Membership.mem l₂ h) (∀ (a : Lists α), Membership.mem t a → Membershi …
        ⊢ HasSubset.Subset (Lists'.cons h (Lists'.ofList t)) l₂
      -/
      exact cons_subset.2 ⟨H.1, t_ih H.2⟩⟩
      /-
        🎉 no goals
      -/


/-- Sends `a : α` to the corresponding atom in `Lists α`. -/
@[match_pattern]
def atom (a : α) : Lists α :=
  ⟨_, Lists'.atom a⟩


/-- Converts a proper ZFA prelist to a ZFA list. -/
@[match_pattern]
def of' (l : Lists' α true) : Lists α :=
  ⟨_, l⟩


/-- Converts a ZFA list to a `List` of ZFA lists. Atoms are sent to `[]`. -/
@[simp]
def toList : Lists α → List (Lists α)
  | ⟨_, l⟩ => l.toList


/-- Predicate stating that a ZFA list is proper. -/
def IsList (l : Lists α) : Prop :=
  l.1


/-- Converts a `List` of ZFA lists to a ZFA list. -/
def ofList (l : List (Lists α)) : Lists α :=
  of' (Lists'.ofList l)


theorem isList_toList (l : List (Lists α)) : IsList (ofList l) :=
  Eq.refl _


                                                                     /-
                                                                       α : Type u_1
                                                                       l : List (Lists α)
                                                                       ⊢ Eq (Lists.ofList l).toList l
                                                                     -/
theorem to_ofList (l : List (Lists α)) : toList (ofList l) = l := by simp [ofList, of']
                                                                     /-
                                                                       🎉 no goals
                                                                     -/


theorem of_toList : ∀ {l : Lists α}, IsList l → ofList (toList l) = l
                       /-
                         α : Type u_1
                         l : Lists' α Bool.true
                         x✝ : Lists.IsList ⟨Bool.true, l⟩
                         ⊢ Eq (Lists.ofList (Lists.toList ⟨Bool.true, l⟩)) ⟨Bool.true, l⟩
                       -/
  | ⟨true, l⟩, _ => by simp_all [ofList, of']
                       /-
                         🎉 no goals
                       -/


instance : Inhabited (Lists α) :=
  ⟨of' Lists'.nil⟩


                                                       /-
                                                         α : Type u_1
                                                         inst✝ : DecidableEq α
                                                         ⊢ DecidableEq (Lists α)
                                                       -/
instance [DecidableEq α] : DecidableEq (Lists α) := by unfold Lists; infer_instance
                                                                     /-
                                                                       🎉 no goals
                                                                     -/


                                             /-
                                               α : Type u_1
                                               inst✝ : SizeOf α
                                               ⊢ SizeOf (Lists α)
                                             -/
instance [SizeOf α] : SizeOf (Lists α) := by unfold Lists; infer_instance
                                                           /-
                                                             🎉 no goals
                                                           -/


/-- A recursion principle for pairs of ZFA lists and proper ZFA prelists. -/
def inductionMut (C : Lists α → Sort*) (D : Lists' α true → Sort*)
    (C0 : ∀ a, C (atom a)) (C1 : ∀ l, D l → C (of' l))
    (D0 : D Lists'.nil) (D1 : ∀ a l, C a → D l → D (Lists'.cons a l)) :
    PProd (∀ l, C l) (∀ l, D l) := by
  suffices
    ∀ {b} (l : Lists' α b),
      PProd (C ⟨_, l⟩)
        (match b, l with
        | true, l => D l
        | false, _ => PUnit)
    by exact ⟨fun ⟨b, l⟩ => (this _).1, fun l => (this l).2⟩
  /-
    α : Type u_1
    C : Lists α → Sort u_2
    D : Lists' α Bool.true → Sort u_3
    C0 : (a : α) → C (Lists.atom a)
    C1 : (l : Lists' α Bool.true) → D l → C (Lists.of' l)
    D0 : D Lists'.nil
    D1 : (a : Lists α) → (l : Lists' α Bool.true) → C a → D l → D (Lists'.cons a l)
    ⊢ {b : Bool} → (l : Lists' α b) → PProd (C ⟨b, l⟩) (Lists.inductionMut.match_1 …
  -/
  intros b l
  /-
    α : Type u_1
    C : Lists α → Sort u_2
    D : Lists' α Bool.true → Sort u_3
    C0 : (a : α) → C (Lists.atom a)
    C1 : (l : Lists' α Bool.true) → D l → C (Lists.of' l)
    D0 : D Lists'.nil
    D1 : (a : Lists α) → (l : Lists' α Bool.true) → C a → D l → D (Lists'.cons a l)
    b : Bool
    l : Lists' α b
    ⊢ PProd (C ⟨b, l⟩) (Lists.inductionMut.match_1 (fun b l => Sort u_3) b l (fun  …
  -/
  induction' l with a b a l IH₁ IH
    /-
      case atom
      α : Type u_1
      C : Lists α → Sort u_2
      D : Lists' α Bool.true → Sort u_3
      C0 : (a : α) → C (Lists.atom a)
      C1 : (l : Lists' α Bool.true) → D l → C (Lists.of' l)
      D0 : D Lists'.nil
      D1 : (a : Lists α) → (l : Lists' α Bool.true) → C a → D l → D (Lists'.cons a l)
      b : Bool
      a : α
      ⊢ PProd (C ⟨Bool.false, Lists'.atom a⟩) (Lists.inductionMut.match_1 (fun b l = …
    -/
  · exact ⟨C0 _, ⟨⟩⟩
    /-
      🎉 no goals
    -/
    /-
      case nil
      α : Type u_1
      C : Lists α → Sort u_2
      D : Lists' α Bool.true → Sort u_3
      C0 : (a : α) → C (Lists.atom a)
      C1 : (l : Lists' α Bool.true) → D l → C (Lists.of' l)
      D0 : D Lists'.nil
      D1 : (a : Lists α) → (l : Lists' α Bool.true) → C a → D l → D (Lists'.cons a l)
      b : Bool
      ⊢ PProd (C ⟨Bool.true, Lists'.nil⟩) (Lists.inductionMut.match_1 (fun b l => So …
    -/
  · exact ⟨C1 _ D0, D0⟩
    /-
      🎉 no goals
    -/
    /-
      case cons'
      α : Type u_1
      C : Lists α → Sort u_2
      D : Lists' α Bool.true → Sort u_3
      C0 : (a : α) → C (Lists.atom a)
      C1 : (l : Lists' α Bool.true) → D l → C (Lists.of' l)
      D0 : D Lists'.nil
      D1 : (a : Lists α) → (l : Lists' α Bool.true) → C a → D l → D (Lists'.cons a l)
      b✝ b : Bool
      a : Lists' α b
      l : Lists' α Bool.true
      IH₁ : PProd (C ⟨b, a⟩) (Lists.inductionMut.match_1 (fun b l => Sort u_3) b a ( …
      IH : PProd (C ⟨Bool.true, l⟩) (Lists.inductionMut.match_1 (fun b l => Sort u_3 …
      ⊢ PProd (C ⟨Bool.true, a.cons' l⟩) (Lists.inductionMut.match_1 (fun b l => Sor …
    -/
  · have : D (Lists'.cons' a l) := D1 ⟨_, _⟩ _ IH₁.1 IH.2
    /-
      case cons'
      α : Type u_1
      C : Lists α → Sort u_2
      D : Lists' α Bool.true → Sort u_3
      C0 : (a : α) → C (Lists.atom a)
      C1 : (l : Lists' α Bool.true) → D l → C (Lists.of' l)
      D0 : D Lists'.nil
      D1 : (a : Lists α) → (l : Lists' α Bool.true) → C a → D l → D (Lists'.cons a l)
      b✝ b : Bool
      a : Lists' α b
      l : Lists' α Bool.true
      IH₁ : PProd (C ⟨b, a⟩) (Lists.inductionMut.match_1 (fun b l => Sort u_3) b a ( …
      IH : PProd (C ⟨Bool.true, l⟩) (Lists.inductionMut.match_1 (fun b l => Sort u_3 …
      this : D (a.cons' l)
      ⊢ PProd (C ⟨Bool.true, a.cons' l⟩) (Lists.inductionMut.match_1 (fun b l => Sor …
    -/
    exact ⟨C1 _ this, this⟩
    /-
      🎉 no goals
    -/


/-- Membership of ZFA list. A ZFA list belongs to a proper ZFA list if it belongs to the latter as a
proper ZFA prelist. An atom has no members. -/
def mem (a : Lists α) : Lists α → Prop
  | ⟨false, _⟩ => False
  | ⟨_, l⟩ => a ∈ l


instance : Membership (Lists α) (Lists α) where
  mem ls l := mem l ls


theorem isList_of_mem {a : Lists α} : ∀ {l : Lists α}, a ∈ l → IsList l
  | ⟨_, Lists'.nil⟩, _ => rfl
  | ⟨_, Lists'.cons' _ _⟩, _ => rfl


theorem Equiv.antisymm_iff {l₁ l₂ : Lists' α true} : of' l₁ ~ of' l₂ ↔ l₁ ⊆ l₂ ∧ l₂ ⊆ l₁ := by
  /-
    α : Type u_1
    l₁ l₂ : Lists' α Bool.true
    ⊢ Iff ((Lists.of' l₁).Equiv (Lists.of' l₂)) (And (HasSubset.Subset l₁ l₂) (Has …
  -/
  refine ⟨fun h => ?_, fun ⟨h₁, h₂⟩ => Equiv.antisymm h₁ h₂⟩
  /-
    α : Type u_1
    l₁ l₂ : Lists' α Bool.true
    h : (Lists.of' l₁).Equiv (Lists.of' l₂)
    ⊢ And (HasSubset.Subset l₁ l₂) (HasSubset.Subset l₂ l₁)
  -/
  cases' h with _ _ _ h₁ h₂
    /-
      case refl
      α : Type u_1
      l₁ : Lists' α Bool.true
      ⊢ And (HasSubset.Subset l₁ l₁) (HasSubset.Subset l₁ l₁)
    -/
  · simp [Lists'.Subset.refl]
    /-
      🎉 no goals
    -/
    /-
      case antisymm
      α : Type u_1
      l₁ l₂ : Lists' α Bool.true
      h₁ : l₁.Subset l₂
      h₂ : l₂.Subset l₁
      ⊢ And (HasSubset.Subset l₁ l₂) (HasSubset.Subset l₂ l₁)
    -/
  · exact ⟨h₁, h₂⟩
    /-
      🎉 no goals
    -/


theorem equiv_atom {a} {l : Lists α} : atom a ~ l ↔ atom a = l :=
               /-
                 α : Type u_1
                 a : α
                 l : Lists α
                 h : (Lists.atom a).Equiv l
                 ⊢ Eq (Lists.atom a) l
               -/
  ⟨fun h => by cases h; rfl, fun h => h ▸ Equiv.refl _⟩
                        /-
                          🎉 no goals
                        -/


@[symm]
theorem Equiv.symm {l₁ l₂ : Lists α} (h : l₁ ~ l₂) : l₂ ~ l₁ := by
  /-
    α : Type u_1
    l₁ l₂ : Lists α
    h : l₁.Equiv l₂
    ⊢ l₂.Equiv l₁
  -/
  cases' h with _ _ _ h₁ h₂ <;> [rfl; exact Equiv.antisymm h₂ h₁]
  /-
    🎉 no goals
  -/


theorem Equiv.trans : ∀ {l₁ l₂ l₃ : Lists α}, l₁ ~ l₂ → l₂ ~ l₃ → l₁ ~ l₃ := by
  /-
    α : Type u_1
    ⊢ ∀ {l₁ l₂ l₃ : Lists α}, l₁.Equiv l₂ → l₂.Equiv l₃ → l₁.Equiv l₃
  -/
  let trans := fun l₁ : Lists α => ∀ ⦃l₂ l₃⦄, l₁ ~ l₂ → l₂ ~ l₃ → l₁ ~ l₃
  /-
    α : Type u_1
    trans : Lists α → Prop := fun l₁ => ∀ ⦃l₂ l₃ : Lists α⦄, l₁.Equiv l₂ → l₂.Equi …
    ⊢ ∀ {l₁ l₂ l₃ : Lists α}, l₁.Equiv l₂ → l₂.Equiv l₃ → l₁.Equiv l₃
  -/
  suffices PProd (∀ l₁, trans l₁) (∀ (l : Lists' α true), ∀ l' ∈ l.toList, trans l') by exact this.1
  /-
    α : Type u_1
    trans : Lists α → Prop := fun l₁ => ∀ ⦃l₂ l₃ : Lists α⦄, l₁.Equiv l₂ → l₂.Equi …
    ⊢ PProd (∀ (l₁ : Lists α), trans l₁) (∀ (l : Lists' α Bool.true) (l' : Lists α …
  -/
  apply inductionMut
    /-
      case C0
      α : Type u_1
      trans : Lists α → Prop := fun l₁ => ∀ ⦃l₂ l₃ : Lists α⦄, l₁.Equiv l₂ → l₂.Equi …
      ⊢ ∀ (a : α), trans (Lists.atom a)
    -/
  · intro a l₂ l₃ h₁ h₂
    /-
      case C0
      α : Type u_1
      trans : Lists α → Prop := fun l₁ => ∀ ⦃l₂ l₃ : Lists α⦄, l₁.Equiv l₂ → l₂.Equi …
      a : α
      l₂ l₃ : Lists α
      h₁ : (Lists.atom a).Equiv l₂
      h₂ : l₂.Equiv l₃
      ⊢ (Lists.atom a).Equiv l₃
    -/
    rwa [← equiv_atom.1 h₁] at h₂
    /-
      🎉 no goals
    -/
    /-
      case C1
      α : Type u_1
      trans : Lists α → Prop := fun l₁ => ∀ ⦃l₂ l₃ : Lists α⦄, l₁.Equiv l₂ → l₂.Equi …
      ⊢ ∀ (l : Lists' α Bool.true), (∀ (l' : Lists α), Membership.mem l.toList l' →  …
    -/
  · intro l₁ IH l₂ l₃ h₁ h₂
    /-
      case C1
      α : Type u_1
      trans : Lists α → Prop := fun l₁ => ∀ ⦃l₂ l₃ : Lists α⦄, l₁.Equiv l₂ → l₂.Equi …
      l₁ : Lists' α Bool.true
      IH : ∀ (l' : Lists α), Membership.mem l₁.toList l' → trans l'
      l₂ l₃ : Lists α
      h₁ : (Lists.of' l₁).Equiv l₂
      h₂ : l₂.Equiv l₃
      ⊢ (Lists.of' l₁).Equiv l₃
    -/
    cases' id h₁ with _ _ l₂
      /-
        case C1.refl
        α : Type u_1
        trans : Lists α → Prop := fun l₁ => ∀ ⦃l₂ l₃ : Lists α⦄, l₁.Equiv l₂ → l₂.Equi …
        l₁ : Lists' α Bool.true
        IH : ∀ (l' : Lists α), Membership.mem l₁.toList l' → trans l'
        l₃ : Lists α
        h₁ : (Lists.of' l₁).Equiv (Lists.of' l₁)
        h₂ : (Lists.of' l₁).Equiv l₃
        ⊢ (Lists.of' l₁).Equiv l₃
      -/
    · exact h₂
      /-
        🎉 no goals
      -/
    /-
      case C1.antisymm
      α : Type u_1
      trans : Lists α → Prop := fun l₁ => ∀ ⦃l₂ l₃ : Lists α⦄, l₁.Equiv l₂ → l₂.Equi …
      l₁ : Lists' α Bool.true
      IH : ∀ (l' : Lists α), Membership.mem l₁.toList l' → trans l'
      l₃ : Lists α
      l₂ : Lists' α Bool.true
      a✝¹ : l₁.Subset l₂
      a✝ : l₂.Subset l₁
      h₁ : (Lists.of' l₁).Equiv ⟨Bool.true, l₂⟩
      h₂ : Lists.Equiv ⟨Bool.true, l₂⟩ l₃
      ⊢ (Lists.of' l₁).Equiv l₃
    -/
    cases' id h₂ with _ _ l₃
      /-
        case C1.antisymm.refl
        α : Type u_1
        trans : Lists α → Prop := fun l₁ => ∀ ⦃l₂ l₃ : Lists α⦄, l₁.Equiv l₂ → l₂.Equi …
        l₁ : Lists' α Bool.true
        IH : ∀ (l' : Lists α), Membership.mem l₁.toList l' → trans l'
        l₂ : Lists' α Bool.true
        a✝¹ : l₁.Subset l₂
        a✝ : l₂.Subset l₁
        h₁ : (Lists.of' l₁).Equiv ⟨Bool.true, l₂⟩
        h₂ : Lists.Equiv ⟨Bool.true, l₂⟩ ⟨Bool.true, l₂⟩
        ⊢ (Lists.of' l₁).Equiv ⟨Bool.true, l₂⟩
      -/
    · exact h₁
      /-
        🎉 no goals
      -/
    /-
      case C1.antisymm.antisymm
      α : Type u_1
      trans : Lists α → Prop := fun l₁ => ∀ ⦃l₂ l₃ : Lists α⦄, l₁.Equiv l₂ → l₂.Equi …
      l₁ : Lists' α Bool.true
      IH : ∀ (l' : Lists α), Membership.mem l₁.toList l' → trans l'
      l₂ : Lists' α Bool.true
      a✝³ : l₁.Subset l₂
      a✝² : l₂.Subset l₁
      h₁ : (Lists.of' l₁).Equiv ⟨Bool.true, l₂⟩
      l₃ : Lists' α Bool.true
      a✝¹ : l₂.Subset l₃
      a✝ : l₃.Subset l₂
      h₂ : Lists.Equiv ⟨Bool.true, l₂⟩ ⟨Bool.true, l₃⟩
      ⊢ (Lists.of' l₁).Equiv ⟨Bool.true, l₃⟩
    -/
    cases' Equiv.antisymm_iff.1 h₁ with hl₁ hr₁
    /-
      case C1.antisymm.antisymm.intro
      α : Type u_1
      trans : Lists α → Prop := fun l₁ => ∀ ⦃l₂ l₃ : Lists α⦄, l₁.Equiv l₂ → l₂.Equi …
      l₁ : Lists' α Bool.true
      IH : ∀ (l' : Lists α), Membership.mem l₁.toList l' → trans l'
      l₂ : Lists' α Bool.true
      a✝³ : l₁.Subset l₂
      a✝² : l₂.Subset l₁
      h₁ : (Lists.of' l₁).Equiv ⟨Bool.true, l₂⟩
      l₃ : Lists' α Bool.true
      a✝¹ : l₂.Subset l₃
      a✝ : l₃.Subset l₂
      h₂ : Lists.Equiv ⟨Bool.true, l₂⟩ ⟨Bool.true, l₃⟩
      hl₁ : HasSubset.Subset l₁ l₂
      hr₁ : HasSubset.Subset l₂ l₁
      ⊢ (Lists.of' l₁).Equiv ⟨Bool.true, l₃⟩
    -/
    cases' Equiv.antisymm_iff.1 h₂ with hl₂ hr₂
    /-
      case C1.antisymm.antisymm.intro.intro
      α : Type u_1
      trans : Lists α → Prop := fun l₁ => ∀ ⦃l₂ l₃ : Lists α⦄, l₁.Equiv l₂ → l₂.Equi …
      l₁ : Lists' α Bool.true
      IH : ∀ (l' : Lists α), Membership.mem l₁.toList l' → trans l'
      l₂ : Lists' α Bool.true
      a✝³ : l₁.Subset l₂
      a✝² : l₂.Subset l₁
      h₁ : (Lists.of' l₁).Equiv ⟨Bool.true, l₂⟩
      l₃ : Lists' α Bool.true
      a✝¹ : l₂.Subset l₃
      a✝ : l₃.Subset l₂
      h₂ : Lists.Equiv ⟨Bool.true, l₂⟩ ⟨Bool.true, l₃⟩
      hl₁ : HasSubset.Subset l₁ l₂
      hr₁ : HasSubset.Subset l₂ l₁
      hl₂ : HasSubset.Subset l₂ l₃
      hr₂ : HasSubset.Subset l₃ l₂
      ⊢ (Lists.of' l₁).Equiv ⟨Bool.true, l₃⟩
    -/
    apply Equiv.antisymm_iff.2; constructor <;> apply Lists'.subset_def.2
      /-
        case C1.antisymm.antisymm.intro.intro.left
        α : Type u_1
        trans : Lists α → Prop := fun l₁ => ∀ ⦃l₂ l₃ : Lists α⦄, l₁.Equiv l₂ → l₂.Equi …
        l₁ : Lists' α Bool.true
        IH : ∀ (l' : Lists α), Membership.mem l₁.toList l' → trans l'
        l₂ : Lists' α Bool.true
        a✝³ : l₁.Subset l₂
        a✝² : l₂.Subset l₁
        h₁ : (Lists.of' l₁).Equiv ⟨Bool.true, l₂⟩
        l₃ : Lists' α Bool.true
        a✝¹ : l₂.Subset l₃
        a✝ : l₃.Subset l₂
        h₂ : Lists.Equiv ⟨Bool.true, l₂⟩ ⟨Bool.true, l₃⟩
        hl₁ : HasSubset.Subset l₁ l₂
        hr₁ : HasSubset.Subset l₂ l₁
        hl₂ : HasSubset.Subset l₂ l₃
        hr₂ : HasSubset.Subset l₃ l₂
        ⊢ ∀ (a : Lists α), Membership.mem l₁.toList a → Membership.mem l₃ a
      -/
    · intro a₁ m₁
      /-
        case C1.antisymm.antisymm.intro.intro.left
        α : Type u_1
        trans : Lists α → Prop := fun l₁ => ∀ ⦃l₂ l₃ : Lists α⦄, l₁.Equiv l₂ → l₂.Equi …
        l₁ : Lists' α Bool.true
        IH : ∀ (l' : Lists α), Membership.mem l₁.toList l' → trans l'
        l₂ : Lists' α Bool.true
        a✝³ : l₁.Subset l₂
        a✝² : l₂.Subset l₁
        h₁ : (Lists.of' l₁).Equiv ⟨Bool.true, l₂⟩
        l₃ : Lists' α Bool.true
        a✝¹ : l₂.Subset l₃
        a✝ : l₃.Subset l₂
        h₂ : Lists.Equiv ⟨Bool.true, l₂⟩ ⟨Bool.true, l₃⟩
        hl₁ : HasSubset.Subset l₁ l₂
        hr₁ : HasSubset.Subset l₂ l₁
        hl₂ : HasSubset.Subset l₂ l₃
        hr₂ : HasSubset.Subset l₃ l₂
        a₁ : Lists α
        m₁ : Membership.mem l₁.toList a₁
        ⊢ Membership.mem l₃ a₁
      -/
      rcases Lists'.mem_of_subset' hl₁ m₁ with ⟨a₂, m₂, e₁₂⟩
      /-
        case C1.antisymm.antisymm.intro.intro.left.intro.intro
        α : Type u_1
        trans : Lists α → Prop := fun l₁ => ∀ ⦃l₂ l₃ : Lists α⦄, l₁.Equiv l₂ → l₂.Equi …
        l₁ : Lists' α Bool.true
        IH : ∀ (l' : Lists α), Membership.mem l₁.toList l' → trans l'
        l₂ : Lists' α Bool.true
        a✝³ : l₁.Subset l₂
        a✝² : l₂.Subset l₁
        h₁ : (Lists.of' l₁).Equiv ⟨Bool.true, l₂⟩
        l₃ : Lists' α Bool.true
        a✝¹ : l₂.Subset l₃
        a✝ : l₃.Subset l₂
        h₂ : Lists.Equiv ⟨Bool.true, l₂⟩ ⟨Bool.true, l₃⟩
        hl₁ : HasSubset.Subset l₁ l₂
        hr₁ : HasSubset.Subset l₂ l₁
        hl₂ : HasSubset.Subset l₂ l₃
        hr₂ : HasSubset.Subset l₃ l₂
        a₁ : Lists α
        m₁ : Membership.mem l₁.toList a₁
        a₂ : Lists α
        m₂ : Membership.mem l₂.toList a₂
        e₁₂ : a₁.Equiv a₂
        ⊢ Membership.mem l₃ a₁
      -/
      rcases Lists'.mem_of_subset' hl₂ m₂ with ⟨a₃, m₃, e₂₃⟩
      /-
        case C1.antisymm.antisymm.intro.intro.left.intro.intro.intro.intro
        α : Type u_1
        trans : Lists α → Prop := fun l₁ => ∀ ⦃l₂ l₃ : Lists α⦄, l₁.Equiv l₂ → l₂.Equi …
        l₁ : Lists' α Bool.true
        IH : ∀ (l' : Lists α), Membership.mem l₁.toList l' → trans l'
        l₂ : Lists' α Bool.true
        a✝³ : l₁.Subset l₂
        a✝² : l₂.Subset l₁
        h₁ : (Lists.of' l₁).Equiv ⟨Bool.true, l₂⟩
        l₃ : Lists' α Bool.true
        a✝¹ : l₂.Subset l₃
        a✝ : l₃.Subset l₂
        h₂ : Lists.Equiv ⟨Bool.true, l₂⟩ ⟨Bool.true, l₃⟩
        hl₁ : HasSubset.Subset l₁ l₂
        hr₁ : HasSubset.Subset l₂ l₁
        hl₂ : HasSubset.Subset l₂ l₃
        hr₂ : HasSubset.Subset l₃ l₂
        a₁ : Lists α
        m₁ : Membership.mem l₁.toList a₁
        a₂ : Lists α
        m₂ : Membership.mem l₂.toList a₂
        e₁₂ : a₁.Equiv a₂
        a₃ : Lists α
        m₃ : Membership.mem l₃.toList a₃
        e₂₃ : a₂.Equiv a₃
        ⊢ Membership.mem l₃ a₁
      -/
      exact ⟨a₃, m₃, IH _ m₁ e₁₂ e₂₃⟩
      /-
        🎉 no goals
      -/
      /-
        case C1.antisymm.antisymm.intro.intro.right
        α : Type u_1
        trans : Lists α → Prop := fun l₁ => ∀ ⦃l₂ l₃ : Lists α⦄, l₁.Equiv l₂ → l₂.Equi …
        l₁ : Lists' α Bool.true
        IH : ∀ (l' : Lists α), Membership.mem l₁.toList l' → trans l'
        l₂ : Lists' α Bool.true
        a✝³ : l₁.Subset l₂
        a✝² : l₂.Subset l₁
        h₁ : (Lists.of' l₁).Equiv ⟨Bool.true, l₂⟩
        l₃ : Lists' α Bool.true
        a✝¹ : l₂.Subset l₃
        a✝ : l₃.Subset l₂
        h₂ : Lists.Equiv ⟨Bool.true, l₂⟩ ⟨Bool.true, l₃⟩
        hl₁ : HasSubset.Subset l₁ l₂
        hr₁ : HasSubset.Subset l₂ l₁
        hl₂ : HasSubset.Subset l₂ l₃
        hr₂ : HasSubset.Subset l₃ l₂
        ⊢ ∀ (a : Lists α), Membership.mem l₃.toList a → Membership.mem l₁ a
      -/
    · intro a₃ m₃
      /-
        case C1.antisymm.antisymm.intro.intro.right
        α : Type u_1
        trans : Lists α → Prop := fun l₁ => ∀ ⦃l₂ l₃ : Lists α⦄, l₁.Equiv l₂ → l₂.Equi …
        l₁ : Lists' α Bool.true
        IH : ∀ (l' : Lists α), Membership.mem l₁.toList l' → trans l'
        l₂ : Lists' α Bool.true
        a✝³ : l₁.Subset l₂
        a✝² : l₂.Subset l₁
        h₁ : (Lists.of' l₁).Equiv ⟨Bool.true, l₂⟩
        l₃ : Lists' α Bool.true
        a✝¹ : l₂.Subset l₃
        a✝ : l₃.Subset l₂
        h₂ : Lists.Equiv ⟨Bool.true, l₂⟩ ⟨Bool.true, l₃⟩
        hl₁ : HasSubset.Subset l₁ l₂
        hr₁ : HasSubset.Subset l₂ l₁
        hl₂ : HasSubset.Subset l₂ l₃
        hr₂ : HasSubset.Subset l₃ l₂
        a₃ : Lists α
        m₃ : Membership.mem l₃.toList a₃
        ⊢ Membership.mem l₁ a₃
      -/
      rcases Lists'.mem_of_subset' hr₂ m₃ with ⟨a₂, m₂, e₃₂⟩
      /-
        case C1.antisymm.antisymm.intro.intro.right.intro.intro
        α : Type u_1
        trans : Lists α → Prop := fun l₁ => ∀ ⦃l₂ l₃ : Lists α⦄, l₁.Equiv l₂ → l₂.Equi …
        l₁ : Lists' α Bool.true
        IH : ∀ (l' : Lists α), Membership.mem l₁.toList l' → trans l'
        l₂ : Lists' α Bool.true
        a✝³ : l₁.Subset l₂
        a✝² : l₂.Subset l₁
        h₁ : (Lists.of' l₁).Equiv ⟨Bool.true, l₂⟩
        l₃ : Lists' α Bool.true
        a✝¹ : l₂.Subset l₃
        a✝ : l₃.Subset l₂
        h₂ : Lists.Equiv ⟨Bool.true, l₂⟩ ⟨Bool.true, l₃⟩
        hl₁ : HasSubset.Subset l₁ l₂
        hr₁ : HasSubset.Subset l₂ l₁
        hl₂ : HasSubset.Subset l₂ l₃
        hr₂ : HasSubset.Subset l₃ l₂
        a₃ : Lists α
        m₃ : Membership.mem l₃.toList a₃
        a₂ : Lists α
        m₂ : Membership.mem l₂.toList a₂
        e₃₂ : a₃.Equiv a₂
        ⊢ Membership.mem l₁ a₃
      -/
      rcases Lists'.mem_of_subset' hr₁ m₂ with ⟨a₁, m₁, e₂₁⟩
      /-
        case C1.antisymm.antisymm.intro.intro.right.intro.intro.intro.intro
        α : Type u_1
        trans : Lists α → Prop := fun l₁ => ∀ ⦃l₂ l₃ : Lists α⦄, l₁.Equiv l₂ → l₂.Equi …
        l₁ : Lists' α Bool.true
        IH : ∀ (l' : Lists α), Membership.mem l₁.toList l' → trans l'
        l₂ : Lists' α Bool.true
        a✝³ : l₁.Subset l₂
        a✝² : l₂.Subset l₁
        h₁ : (Lists.of' l₁).Equiv ⟨Bool.true, l₂⟩
        l₃ : Lists' α Bool.true
        a✝¹ : l₂.Subset l₃
        a✝ : l₃.Subset l₂
        h₂ : Lists.Equiv ⟨Bool.true, l₂⟩ ⟨Bool.true, l₃⟩
        hl₁ : HasSubset.Subset l₁ l₂
        hr₁ : HasSubset.Subset l₂ l₁
        hl₂ : HasSubset.Subset l₂ l₃
        hr₂ : HasSubset.Subset l₃ l₂
        a₃ : Lists α
        m₃ : Membership.mem l₃.toList a₃
        a₂ : Lists α
        m₂ : Membership.mem l₂.toList a₂
        e₃₂ : a₃.Equiv a₂
        a₁ : Lists α
        m₁ : Membership.mem l₁.toList a₁
        e₂₁ : a₂.Equiv a₁
        ⊢ Membership.mem l₁ a₃
      -/
      exact ⟨a₁, m₁, (IH _ m₁ e₂₁.symm e₃₂.symm).symm⟩
      /-
        🎉 no goals
      -/
    /-
      case D0
      α : Type u_1
      trans : Lists α → Prop := fun l₁ => ∀ ⦃l₂ l₃ : Lists α⦄, l₁.Equiv l₂ → l₂.Equi …
      ⊢ ∀ (l' : Lists α), Membership.mem Lists'.nil.toList l' → trans l'
    -/
  · rintro _ ⟨⟩
    /-
      🎉 no goals
    -/
    /-
      case D1
      α : Type u_1
      trans : Lists α → Prop := fun l₁ => ∀ ⦃l₂ l₃ : Lists α⦄, l₁.Equiv l₂ → l₂.Equi …
      ⊢ ∀ (a : Lists α) (l : Lists' α Bool.true), trans a → (∀ (l' : Lists α), Membe …
    -/
  · intro a l IH₁ IH₂
    /-
      case D1
      α : Type u_1
      trans : Lists α → Prop := fun l₁ => ∀ ⦃l₂ l₃ : Lists α⦄, l₁.Equiv l₂ → l₂.Equi …
      a : Lists α
      l : Lists' α Bool.true
      IH₁ : trans a
      IH₂ : ∀ (l' : Lists α), Membership.mem l.toList l' → trans l'
      ⊢ ∀ (l' : Lists α), Membership.mem (Lists'.cons a l).toList l' → trans l'
    -/
    simpa using ⟨IH₁, IH₂⟩
    /-
      🎉 no goals
    -/


instance instSetoidLists : Setoid (Lists α) :=
  ⟨(· ~ ·), Equiv.refl, @Equiv.symm _, @Equiv.trans _⟩


theorem sizeof_pos {b} (l : Lists' α b) : 0 < SizeOf.sizeOf l := by
  /-
    α : Type u_1
    b : Bool
    l : Lists' α b
    ⊢ LT.lt 0 (SizeOf.sizeOf l)
  -/
  cases l <;> simp only [Lists'.atom.sizeOf_spec, Lists'.nil.sizeOf_spec, Lists'.cons'.sizeOf_spec,
    true_or, add_pos_iff, zero_lt_one]


theorem lt_sizeof_cons' {b} (a : Lists' α b) (l) :
    SizeOf.sizeOf (⟨b, a⟩ : Lists α) < SizeOf.sizeOf (Lists'.cons' a l) := by
  /-
    α : Type u_1
    b : Bool
    a : Lists' α b
    l : Lists' α Bool.true
    ⊢ LT.lt (SizeOf.sizeOf ⟨b, a⟩) (SizeOf.sizeOf (a.cons' l))
  -/
  simp only [Sigma.mk.sizeOf_spec, Lists'.cons'.sizeOf_spec, lt_add_iff_pos_right]
  /-
    α : Type u_1
    b : Bool
    a : Lists' α b
    l : Lists' α Bool.true
    ⊢ LT.lt 0 (SizeOf.sizeOf l)
  -/
  apply sizeof_pos
  /-
    🎉 no goals
  -/


mutual
  instance Equiv.decidable : ∀ l₁ l₂ : Lists α, Decidable (l₁ ~ l₂)
    | ⟨false, l₁⟩, ⟨false, l₂⟩ =>
      decidable_of_iff' (l₁ = l₂) <| by
        /-
          α : Type u_1
          inst✝ : DecidableEq α
          l₁ l₂ : Lists' α Bool.false
          ⊢ Iff (Lists.Equiv ⟨Bool.false, l₁⟩ ⟨Bool.false, l₂⟩) (Eq l₁ l₂)
        -/
        cases l₁
        /-
          case atom
          α : Type u_1
          inst✝ : DecidableEq α
          l₂ : Lists' α Bool.false
          a✝ : α
          ⊢ Iff (Lists.Equiv ⟨Bool.false, Lists'.atom a✝⟩ ⟨Bool.false, l₂⟩) (Eq (Lists'. …
        -/
        apply equiv_atom.trans
        /-
          case atom
          α : Type u_1
          inst✝ : DecidableEq α
          l₂ : Lists' α Bool.false
          a✝ : α
          ⊢ Iff (Eq (Lists.atom a✝) ⟨Bool.false, l₂⟩) (Eq (Lists'.atom a✝) l₂)
        -/
        simp only [atom]
        /-
          case atom
          α : Type u_1
          inst✝ : DecidableEq α
          l₂ : Lists' α Bool.false
          a✝ : α
          ⊢ Iff (Eq ⟨Bool.false, Lists'.atom a✝⟩ ⟨Bool.false, l₂⟩) (Eq (Lists'.atom a✝)  …
        -/
                                       /-
                                         🎉 no goals
                                       -/
        constructor <;> (rintro ⟨rfl⟩; rfl)
                                       /-
                                         🎉 no goals
                                       -/
                                               /-
                                                 α : Type u_1
                                                 inst✝ : DecidableEq α
                                                 l₁ : Lists' α Bool.false
                                                 l₂ : Lists' α Bool.true
                                                 ⊢ Not (Lists.Equiv ⟨Bool.false, l₁⟩ ⟨Bool.true, l₂⟩)
                                               -/
    | ⟨false, l₁⟩, ⟨true, l₂⟩ => isFalse <| by rintro ⟨⟩
                                               /-
                                                 🎉 no goals
                                               -/
                                               /-
                                                 α : Type u_1
                                                 inst✝ : DecidableEq α
                                                 l₁ : Lists' α Bool.true
                                                 l₂ : Lists' α Bool.false
                                                 ⊢ Not (Lists.Equiv ⟨Bool.true, l₁⟩ ⟨Bool.false, l₂⟩)
                                               -/
    | ⟨true, l₁⟩, ⟨false, l₂⟩ => isFalse <| by rintro ⟨⟩
                                               /-
                                                 🎉 no goals
                                               -/
    | ⟨true, l₁⟩, ⟨true, l₂⟩ => by
      haveI : Decidable (l₁ ⊆ l₂) :=
        have : SizeOf.sizeOf l₁ + SizeOf.sizeOf l₂ <
            SizeOf.sizeOf (⟨true, l₁⟩ : Lists α) + SizeOf.sizeOf (⟨true, l₂⟩ : Lists α) := by
          decreasing_tactic
        Subset.decidable l₁ l₂
      haveI : Decidable (l₂ ⊆ l₁) :=
        have : SizeOf.sizeOf l₂ + SizeOf.sizeOf l₁ <
            SizeOf.sizeOf (⟨true, l₁⟩ : Lists α) + SizeOf.sizeOf (⟨true, l₂⟩ : Lists α) := by
          decreasing_tactic
        Subset.decidable l₂ l₁
      /-
        α : Type u_1
        inst✝ : DecidableEq α
        l₁ l₂ : Lists' α Bool.true
        this✝ : Decidable (HasSubset.Subset l₁ l₂)
        this : Decidable (HasSubset.Subset l₂ l₁)
        ⊢ Decidable (Lists.Equiv ⟨Bool.true, l₁⟩ ⟨Bool.true, l₂⟩)
      -/
      exact decidable_of_iff' _ Equiv.antisymm_iff
      /-
        🎉 no goals
      -/
  termination_by x y => sizeOf x + sizeOf y
  instance Subset.decidable : ∀ l₁ l₂ : Lists' α true, Decidable (l₁ ⊆ l₂)
    | Lists'.nil, _ => isTrue Lists'.Subset.nil
    | @Lists'.cons' _ b a l₁, l₂ => by
      haveI :=
        have : sizeOf (⟨b, a⟩ : Lists α) < 1 + 1 + sizeOf a + sizeOf l₁ := by simp [sizeof_pos]
        mem.decidable ⟨b, a⟩ l₂
      haveI :=
        have : SizeOf.sizeOf l₁ + SizeOf.sizeOf l₂ <
            SizeOf.sizeOf (Lists'.cons' a l₁) + SizeOf.sizeOf l₂ := by
          decreasing_tactic
        Subset.decidable l₁ l₂
      /-
        α : Type u_1
        inst✝ : DecidableEq α
        b : Bool
        a : Lists' α b
        l₁ l₂ : Lists' α Bool.true
        this✝ : Decidable (Membership.mem l₂ ⟨b, a⟩)
        this : Decidable (HasSubset.Subset l₁ l₂)
        ⊢ Decidable (HasSubset.Subset (a.cons' l₁) l₂)
      -/
      exact decidable_of_iff' _ (@Lists'.cons_subset _ ⟨_, _⟩ _ _)
      /-
        🎉 no goals
      -/
  termination_by x y => sizeOf x + sizeOf y
  instance mem.decidable : ∀ (a : Lists α) (l : Lists' α true), Decidable (a ∈ l)
                                     /-
                                       α : Type u_1
                                       inst✝ : DecidableEq α
                                       a : Lists α
                                       ⊢ Not (Membership.mem Lists'.nil a)
                                     -/
    | a, Lists'.nil => isFalse <| by rintro ⟨_, ⟨⟩, _⟩
                                     /-
                                       🎉 no goals
                                     -/
    | a, Lists'.cons' b l₂ => by
      haveI :=
        have : sizeOf (⟨_, b⟩ : Lists α) < 1 + 1 + sizeOf b + sizeOf l₂ := by simp [sizeof_pos]
        Equiv.decidable a ⟨_, b⟩
      haveI :=
        have :
          SizeOf.sizeOf a + SizeOf.sizeOf l₂ <
            SizeOf.sizeOf a + SizeOf.sizeOf (Lists'.cons' b l₂) := by
          decreasing_tactic
        mem.decidable a l₂
      /-
        α : Type u_1
        inst✝ : DecidableEq α
        a : Lists α
        b✝ : Bool
        b : Lists' α b✝
        l₂ : Lists' α Bool.true
        this✝ : Decidable (a.Equiv ⟨b✝, b⟩)
        this : Decidable (Membership.mem l₂ a)
        ⊢ Decidable (Membership.mem (b.cons' l₂) a)
      -/
      refine decidable_of_iff' (a ~ ⟨_, b⟩ ∨ a ∈ l₂) ?_
      /-
        α : Type u_1
        inst✝ : DecidableEq α
        a : Lists α
        b✝ : Bool
        b : Lists' α b✝
        l₂ : Lists' α Bool.true
        this✝ : Decidable (a.Equiv ⟨b✝, b⟩)
        this : Decidable (Membership.mem l₂ a)
        ⊢ Iff (Membership.mem (b.cons' l₂) a) (Or (a.Equiv ⟨b✝, b⟩) (Membership.mem l₂ …
      -/
      rw [← Lists'.mem_cons]; rfl
                              /-
                                🎉 no goals
                              -/
  termination_by x y => sizeOf x + sizeOf y
end

-- This is an autogenerated declaration, so there's nothing we can do about it.

theorem mem_equiv_left {l : Lists' α true} : ∀ {a a'}, a ~ a' → (a ∈ l ↔ a' ∈ l) :=
  suffices ∀ {a a'}, a ~ a' → a ∈ l → a' ∈ l from fun e => ⟨this e, this e.symm⟩
  fun e₁ ⟨_, m₃, e₂⟩ => ⟨_, m₃, e₁.symm.trans e₂⟩


theorem mem_of_subset {a} {l₁ l₂ : Lists' α true} (s : l₁ ⊆ l₂) : a ∈ l₁ → a ∈ l₂
  | ⟨_, m, e⟩ => (mem_equiv_left e).2 (mem_of_subset' s m)


theorem Subset.trans {l₁ l₂ l₃ : Lists' α true} (h₁ : l₁ ⊆ l₂) (h₂ : l₂ ⊆ l₃) : l₁ ⊆ l₃ :=
  subset_def.2 fun _ m₁ => mem_of_subset h₂ <| mem_of_subset' h₁ m₁


/-- `Finsets` are defined via equivalence classes of `Lists` -/
def Finsets (α : Type*) :=
  Quotient (@Lists.instSetoidLists α)


instance : EmptyCollection (Finsets α) :=
  ⟨⟦Lists.of' Lists'.nil⟧⟩


instance : Inhabited (Finsets α) :=
  ⟨∅⟩


instance [DecidableEq α] : DecidableEq (Finsets α) := by
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    ⊢ DecidableEq (Finsets α)
  -/
  unfold Finsets
  -- Porting note: infer_instance does not work for some reason
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    ⊢ DecidableEq (Quotient Lists.instSetoidLists)
  -/
  exact (Quotient.decidableEq (d := fun _ _ => Lists.Equiv.decidable _ _))
  /-
    🎉 no goals
  -/


