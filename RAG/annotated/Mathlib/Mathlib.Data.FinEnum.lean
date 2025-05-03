/-- `FinEnum α` means that `α` is finite and can be enumerated in some order,
  i.e. `α` has an explicit bijection with `Fin n` for some n. -/
class FinEnum (α : Sort*) where
  /-- `FinEnum.card` is the cardinality of the `FinEnum` -/
  card : ℕ
  /-- `FinEnum.Equiv` states that type `α` is in bijection with `Fin card`,
    the size of the `FinEnum` -/
  equiv : α ≃ Fin card
  [decEq : DecidableEq α]


/-- transport a `FinEnum` instance across an equivalence -/
def ofEquiv (α) {β} [FinEnum α] (h : β ≃ α) : FinEnum β where
  card := card α
  equiv := h.trans (equiv)
  decEq := (h.trans (equiv)).decidableEq


/-- create a `FinEnum` instance from an exhaustive list without duplicates -/
def ofNodupList [DecidableEq α] (xs : List α) (h : ∀ x : α, x ∈ xs) (h' : List.Nodup xs) :
    FinEnum α where
  card := xs.length
  equiv :=
                                /-
                                  α : Type u
                                  β : α → Type v
                                  inst✝ : DecidableEq α
                                  xs : List α
                                  h : ∀ (x : α), Membership.mem xs x
                                  h' : xs.Nodup
                                  x : α
                                  ⊢ LT.lt (List.indexOf x xs) xs.length
                                -/
                                                             /-
                                                               🎉 no goals
                                                             -/
    ⟨fun x => ⟨xs.indexOf x, by rw [List.indexOf_lt_length]; apply h⟩, xs.get, fun x => by simp,
                                                                                           /-
                                                                                             🎉 no goals
                                                                                           -/
                  /-
                    α : Type u
                    β : α → Type v
                    inst✝ : DecidableEq α
                    xs : List α
                    h : ∀ (x : α), Membership.mem xs x
                    h' : xs.Nodup
                    i : Fin xs.length
                    ⊢ Eq ((fun x => ⟨List.indexOf x xs, ⋯⟩) (xs.get i)) i
                  -/
      fun i => by ext; simp [List.indexOf_getElem h']⟩
                       /-
                         🎉 no goals
                       -/


/-- create a `FinEnum` instance from an exhaustive list; duplicates are removed -/
def ofList [DecidableEq α] (xs : List α) (h : ∀ x : α, x ∈ xs) : FinEnum α :=
                           /-
                             α : Type u
                             β : α → Type v
                             inst✝ : DecidableEq α
                             xs : List α
                             h : ∀ (x : α), Membership.mem xs x
                             ⊢ ∀ (x : α), Membership.mem xs.dedup x
                           -/
  ofNodupList xs.dedup (by simp [*]) (List.nodup_dedup _)
                           /-
                             🎉 no goals
                           -/


/-- create an exhaustive list of the values of a given type -/
def toList (α) [FinEnum α] : List α :=
  (List.finRange (card α)).map (equiv).symm


@[simp]
theorem mem_toList [FinEnum α] (x : α) : x ∈ toList α := by
  /-
    α : Type u
    inst✝ : FinEnum α
    x : α
    ⊢ Membership.mem (FinEnum.toList α) x
  -/
  simp [toList]; exists equiv x; simp
                                 /-
                                   🎉 no goals
                                 -/


@[simp]
theorem nodup_toList [FinEnum α] : List.Nodup (toList α) := by
  /-
    α : Type u
    inst✝ : FinEnum α
    ⊢ (FinEnum.toList α).Nodup
  -/
  simp [toList]; apply List.Nodup.map <;> [apply Equiv.injective; apply List.nodup_finRange]
                 /-
                   🎉 no goals
                 -/


/-- create a `FinEnum` instance using a surjection -/
def ofSurjective {β} (f : β → α) [DecidableEq α] [FinEnum β] (h : Surjective f) : FinEnum α :=
                                /-
                                  α : Type u
                                  β✝ : α → Type v
                                  β : Type ?u.3350
                                  f : β → α
                                  inst✝¹ : DecidableEq α
                                  inst✝ : FinEnum β
                                  h : Function.Surjective f
                                  ⊢ ∀ (x : α), Membership.mem (List.map f (FinEnum.toList β)) x
                                -/
  ofList ((toList β).map f) (by intro; simpa using h _)
                                       /-
                                         🎉 no goals
                                       -/


/-- create a `FinEnum` instance using an injection -/
noncomputable def ofInjective {α β} (f : α → β) [DecidableEq α] [FinEnum β] (h : Injective f) :
    FinEnum α :=
  ofList ((toList β).filterMap (partialInv f))
    (by
      /-
        α✝ : Type u
        β✝ : α✝ → Type v
        α : Type ?u.5075
        β : Type ?u.5150
        f : α → β
        inst✝¹ : DecidableEq α
        inst✝ : FinEnum β
        h : Function.Injective f
        ⊢ ∀ (x : α), Membership.mem (List.filterMap (Function.partialInv f) (FinEnum.t …
      -/
      intro x
      /-
        α✝ : Type u
        β✝ : α✝ → Type v
        α : Type ?u.5075
        β : Type ?u.5150
        f : α → β
        inst✝¹ : DecidableEq α
        inst✝ : FinEnum β
        h : Function.Injective f
        x : α
        ⊢ Membership.mem (List.filterMap (Function.partialInv f) (FinEnum.toList β)) x
      -/
      simp only [mem_toList, true_and, List.mem_filterMap]
      /-
        α✝ : Type u
        β✝ : α✝ → Type v
        α : Type ?u.5075
        β : Type ?u.5150
        f : α → β
        inst✝¹ : DecidableEq α
        inst✝ : FinEnum β
        h : Function.Injective f
        x : α
        ⊢ Exists fun a => Eq (Function.partialInv f a) (Option.some x)
      -/
      use f x
      /-
        case h
        α✝ : Type u
        β✝ : α✝ → Type v
        α : Type ?u.5075
        β : Type ?u.5150
        f : α → β
        inst✝¹ : DecidableEq α
        inst✝ : FinEnum β
        h : Function.Injective f
        x : α
        ⊢ Eq (Function.partialInv f (f x)) (Option.some x)
      -/
      simp only [h, Function.partialInv_left])
      /-
        🎉 no goals
      -/


instance _root_.ULift.instFinEnum [FinEnum α] : FinEnum (ULift α) :=
  ⟨card α, Equiv.ulift.trans equiv⟩


@[simp]
theorem card_ulift [FinEnum (ULift α)] [FinEnum α] : card (ULift α) = card α :=
  Fin.equiv_iff_eq.mp ⟨equiv.symm.trans Equiv.ulift |>.trans equiv⟩


@[simp] lemma equiv_up : equiv (ULift.up a) = equiv a := rfl

@[simp] lemma equiv_down : equiv a'.down = equiv a' := rfl

@[simp] lemma up_equiv_symm : ULift.up (equiv.symm i) = (equiv (α := ULift α)).symm i := rfl

@[simp] lemma down_equiv_symm : ((equiv (α := ULift α)).symm i).down = equiv.symm i := rfl


instance pempty : FinEnum PEmpty :=
  ofList [] fun x => PEmpty.elim x


instance empty : FinEnum Empty :=
  ofList [] fun x => Empty.elim x


instance punit : FinEnum PUnit :=
                                  /-
                                    α : Type u
                                    β : α → Type v
                                    x : PUnit.{?u.7079 + 1}
                                    ⊢ Membership.mem (List.cons PUnit.unit List.nil) x
                                  -/
  ofList [PUnit.unit] fun x => by cases x; simp
                                           /-
                                             🎉 no goals
                                           -/


instance prod {β} [FinEnum α] [FinEnum β] : FinEnum (α × β) :=
                                            /-
                                              α : Type u
                                              β✝ : α → Type v
                                              β : Type ?u.7392
                                              inst✝¹ : FinEnum α
                                              inst✝ : FinEnum β
                                              x : Prod α β
                                              ⊢ Membership.mem (SProd.sprod (FinEnum.toList α) (FinEnum.toList β)) x
                                            -/
  ofList (toList α ×ˢ toList β) fun x => by cases x; simp
                                                     /-
                                                       🎉 no goals
                                                     -/


instance sum {β} [FinEnum α] [FinEnum β] : FinEnum (α ⊕ β) :=
                                                                        /-
                                                                          α : Type u
                                                                          β✝ : α → Type v
                                                                          β : Type ?u.7908
                                                                          inst✝¹ : FinEnum α
                                                                          inst✝ : FinEnum β
                                                                          x : Sum α β
                                                                          ⊢ Membership.mem (HAppend.hAppend (List.map Sum.inl (FinEnum.toList α)) (List. …
                                                                        -/
                                                                                    /-
                                                                                      🎉 no goals
                                                                                    -/
  ofList ((toList α).map Sum.inl ++ (toList β).map Sum.inr) fun x => by cases x <;> simp
                                                                                    /-
                                                                                      🎉 no goals
                                                                                    -/


instance fin {n} : FinEnum (Fin n) :=
                               /-
                                 α : Type u
                                 β : α → Type v
                                 n : Nat
                                 ⊢ ∀ (x : Fin n), Membership.mem (List.finRange n) x
                               -/
  ofList (List.finRange _) (by simp)
                               /-
                                 🎉 no goals
                               -/


@[simp]
theorem card_fin {n} [FinEnum (Fin n)] : card (Fin n) = n := Fin.equiv_iff_eq.mp ⟨equiv.symm⟩


instance Quotient.enum [FinEnum α] (s : Setoid α) [DecidableRel ((· ≈ ·) : α → α → Prop)] :
    FinEnum (Quotient s) :=
  FinEnum.ofSurjective Quotient.mk'' fun x => Quotient.inductionOn x fun x => ⟨x, rfl⟩


/-- enumerate all finite sets of a given type -/
def Finset.enum [DecidableEq α] : List α → List (Finset α)
  | [] => [∅]
  | x :: xs => do
    let r ← Finset.enum xs
    [r, insert x r]


@[simp]
theorem Finset.mem_enum [DecidableEq α] (s : Finset α) (xs : List α) :
    s ∈ Finset.enum xs ↔ ∀ x ∈ s, x ∈ xs := by
  induction xs generalizing s with
  | nil => simp [enum, eq_empty_iff_forall_not_mem]
  | cons x xs ih =>
      simp only [enum, List.bind_eq_flatMap, List.mem_flatMap, List.mem_cons, List.mem_singleton,
        List.not_mem_nil, or_false, ih]
      refine ⟨by aesop, fun hs => ⟨s.erase x, ?_⟩⟩
      simp only [or_iff_not_imp_left] at hs
      simp +contextual [eq_comm (a := s), or_iff_not_imp_left, hs]


instance Finset.finEnum [FinEnum α] : FinEnum (Finset α) :=
                                      /-
                                        α : Type u
                                        β : α → Type v
                                        inst✝ : FinEnum α
                                        ⊢ ∀ (x : Finset α), Membership.mem (FinEnum.Finset.enum (FinEnum.toList α)) x
                                      -/
  ofList (Finset.enum (toList α)) (by intro; simp)
                                             /-
                                               🎉 no goals
                                             -/


instance Subtype.finEnum [FinEnum α] (p : α → Prop) [DecidablePred p] : FinEnum { x // p x } :=
  ofList ((toList α).filterMap fun x => if h : p x then some ⟨_, h⟩ else none)
        /-
          α : Type u
          β : α → Type v
          inst✝¹ : FinEnum α
          p : α → Prop
          inst✝ : DecidablePred p
          ⊢ ∀ (x : Subtype fun x => p x), Membership.mem (List.filterMap (fun x => dite  …
        -/
    (by rintro ⟨x, h⟩; simpa)
                       /-
                         🎉 no goals
                       -/


instance (β : α → Type v) [FinEnum α] [∀ a, FinEnum (β a)] : FinEnum (Sigma β) :=
  ofList ((toList α).flatMap fun a => (toList (β a)).map <| Sigma.mk a)
        /-
          α : Type u
          β✝ β : α → Type v
          inst✝¹ : FinEnum α
          inst✝ : (a : α) → FinEnum (β a)
          ⊢ ∀ (x : Sigma β), Membership.mem ((FinEnum.toList α).flatMap fun a => List.ma …
        -/
    (by intro x; cases x; simp)
                          /-
                            🎉 no goals
                          -/


instance PSigma.finEnum [FinEnum α] [∀ a, FinEnum (β a)] : FinEnum (Σ'a, β a) :=
  FinEnum.ofEquiv _ (Equiv.psigmaEquivSigma _)


instance PSigma.finEnumPropLeft {α : Prop} {β : α → Type v} [∀ a, FinEnum (β a)] [Decidable α] :
    FinEnum (Σ'a, β a) :=
                                                                             /-
                                                                               α✝ : Type u
                                                                               β✝ : α✝ → Type v
                                                                               α : Prop
                                                                               β : α → Type v
                                                                               inst✝¹ : (a : α) → FinEnum (β a)
                                                                               inst✝ : Decidable α
                                                                               h : α
                                                                               x✝ : PSigma fun a => β a
                                                                               a : α
                                                                               Ba : β a
                                                                               ⊢ Membership.mem (List.map (PSigma.mk h) (FinEnum.toList (β h))) ⟨a, Ba⟩
                                                                             -/
  if h : α then ofList ((toList (β h)).map <| PSigma.mk h) fun ⟨a, Ba⟩ => by simp
                                                                             /-
                                                                               🎉 no goals
                                                                             -/
  else ofList [] fun ⟨a, _⟩ => (h a).elim


instance PSigma.finEnumPropRight {β : α → Prop} [FinEnum α] [∀ a, Decidable (β a)] :
    FinEnum (Σ'a, β a) :=
  FinEnum.ofEquiv { a // β a }
    ⟨fun ⟨x, y⟩ => ⟨x, y⟩, fun ⟨x, y⟩ => ⟨x, y⟩, fun ⟨_, _⟩ => rfl, fun ⟨_, _⟩ => rfl⟩


instance PSigma.finEnumPropProp {α : Prop} {β : α → Prop} [Decidable α] [∀ a, Decidable (β a)] :
    FinEnum (Σ'a, β a) :=
                                                   /-
                                                     α✝ : Type u
                                                     β✝ : α✝ → Type v
                                                     α : Prop
                                                     β : α → Prop
                                                     inst✝¹ : Decidable α
                                                     inst✝ : (a : α) → Decidable (β a)
                                                     h : Exists fun a => β a
                                                     ⊢ ∀ (x : PSigma fun a => β a), Membership.mem (List.cons ⟨⋯, ⋯⟩ List.nil) x
                                                   -/
  if h : ∃ a, β a then ofList [⟨h.fst, h.snd⟩] (by rintro ⟨⟩; simp)
                                                              /-
                                                                🎉 no goals
                                                              -/
  else ofList [] fun a => (h ⟨a.fst, a.snd⟩).elim


                                                                                             /-
                                                                                               α : Type u
                                                                                               β : α → Type v
                                                                                               inst✝ : DecidableEq α
                                                                                               xs : List α
                                                                                               ⊢ ∀ (x : Subtype fun x => Membership.mem xs x), Membership.mem xs.attach x
                                                                                             -/
instance [DecidableEq α] (xs : List α) : FinEnum { x : α // x ∈ xs } := ofList xs.attach (by simp)
                                                                                             /-
                                                                                               🎉 no goals
                                                                                             -/


instance (priority := 100) [FinEnum α] : Fintype α where
  elems := univ.map (equiv).symm.toEmbedding
                 /-
                   α : Type u
                   β : α → Type v
                   inst✝ : FinEnum α
                   ⊢ ∀ (x : α), Membership.mem (Finset.map FinEnum.equiv.symm.toEmbedding Finset. …
                 -/
  complete := by intros; simp
                         /-
                           🎉 no goals
                         -/


/-- The enumeration merely adds an ordering, leaving the cardinality as is. -/
theorem card_eq_fintypeCard {α : Type u} [FinEnum α] [Fintype α] : card α = Fintype.card α :=
  Fintype.truncEquivFin α |>.inductionOn (fun h ↦ Fin.equiv_iff_eq.mp ⟨equiv.symm.trans h⟩)


/-- Any two enumerations of the same type have the same length. -/
theorem card_unique {α : Type u} (e₁ e₂ : FinEnum α) : e₁.card = e₂.card :=
  calc _
  _ = _ := @card_eq_fintypeCard _ e₁ inferInstance
  _ = _ := Fintype.card_congr' rfl
  _ = _ := @card_eq_fintypeCard _ e₂ inferInstance |>.symm


/-- A type indexable by `Fin 0` is empty and vice versa. -/
theorem card_eq_zero_iff {α : Type u} [FinEnum α] : card α = 0 ↔ IsEmpty α :=
  Eq.congr_left card_eq_fintypeCard |>.trans Fintype.card_eq_zero_iff


/-- Any enumeration of an empty type has length 0. -/
theorem card_eq_zero {α : Type u} [FinEnum α] [IsEmpty α] : card α = 0 :=
  card_eq_zero_iff.mpr ‹_›


/-- A type indexable by `Fin n` with positive `n` is inhabited and vice versa. -/
theorem card_pos_iff {α : Type u} [FinEnum α] : 0 < card α ↔ Nonempty α :=
  card_eq_fintypeCard (α := α) ▸ Fintype.card_pos_iff


/-- Any non-empty enumeration has more than one element. -/
lemma card_pos {α : Type*} [FinEnum α] [Nonempty α] : 0 < card α :=
  card_pos_iff.mpr ‹_›


/-- No non-empty enumeration has 0 elements. -/
lemma card_ne_zero {α : Type*} [FinEnum α] [Nonempty α] : card α ≠ 0 := card_pos.ne'


/-- Any enumeration of a type with unique inhabitant has length 1. -/
theorem card_eq_one (α : Type u) [FinEnum α] [Unique α] : card α = 1 :=
  card_eq_fintypeCard.trans <| Fintype.card_eq_one_iff_nonempty_unique.mpr ⟨‹_›⟩


instance [IsEmpty α] : Unique (FinEnum α) where
  default := ⟨0, Equiv.equivOfIsEmpty α (Fin 0)⟩
  uniq e := by
    /-
      α : Type u
      β : α → Type v
      inst✝ : IsEmpty α
      e : FinEnum α
      ⊢ Eq e Inhabited.default
    -/
    show FinEnum.mk e.1 e.2 = _
    /-
      α : Type u
      β : α → Type v
      inst✝ : IsEmpty α
      e : FinEnum α
      ⊢ Eq (FinEnum.mk (FinEnum.card α) FinEnum.equiv) Inhabited.default
    -/
    congr 1
      /-
        case h.e_2
        α : Type u
        β : α → Type v
        inst✝ : IsEmpty α
        e : FinEnum α
        ⊢ Eq (FinEnum.card α) 0
      -/
    · exact card_eq_zero
      /-
        🎉 no goals
      -/
      /-
        case h.e_3
        α : Type u
        β : α → Type v
        inst✝ : IsEmpty α
        e : FinEnum α
        ⊢ HEq FinEnum.equiv (Equiv.equivOfIsEmpty α (Fin 0))
      -/
    · refine heq_of_cast_eq ?_ (Subsingleton.allEq _ _)
      /-
        case h.e_3
        α : Type u
        β : α → Type v
        inst✝ : IsEmpty α
        e : FinEnum α
        ⊢ Eq (Equiv α (Fin (FinEnum.card α))) (Equiv α (Fin 0))
      -/
      exact congrArg (α ≃ Fin ·) <| card_eq_zero
      /-
        🎉 no goals
      -/
      /-
        case h.e_4.h
        α : Type u
        β : α → Type v
        inst✝ : IsEmpty α
        e : FinEnum α
        ⊢ Eq (fun a b => FinEnum.decEq a b) fun a b => decidableEq_of_subsingleton a b
      -/
    · funext x
      /-
        case h.e_4.h.h
        α : Type u
        β : α → Type v
        inst✝ : IsEmpty α
        e : FinEnum α
        x : α
        ⊢ Eq (fun b => FinEnum.decEq x b) fun b => decidableEq_of_subsingleton x b
      -/
      exact ‹IsEmpty α›.elim x
      /-
        🎉 no goals
      -/


/-- An empty type has a trivial enumeration. Not registered as an instance, to make sure that there
aren't two definitionally differing instances around. -/
def ofIsEmpty [IsEmpty α] : FinEnum α := default


instance [Unique α] : Unique (FinEnum α) where
  default := ⟨1, Equiv.ofUnique α (Fin 1)⟩
  uniq e := by
    /-
      α : Type u
      β : α → Type v
      inst✝ : Unique α
      e : FinEnum α
      ⊢ Eq e Inhabited.default
    -/
    show FinEnum.mk e.1 e.2 = _
    /-
      α : Type u
      β : α → Type v
      inst✝ : Unique α
      e : FinEnum α
      ⊢ Eq (FinEnum.mk (FinEnum.card α) FinEnum.equiv) Inhabited.default
    -/
    congr 1
      /-
        case h.e_2
        α : Type u
        β : α → Type v
        inst✝ : Unique α
        e : FinEnum α
        ⊢ Eq (FinEnum.card α) 1
      -/
    · exact card_eq_one α
      /-
        🎉 no goals
      -/
      /-
        case h.e_3
        α : Type u
        β : α → Type v
        inst✝ : Unique α
        e : FinEnum α
        ⊢ HEq FinEnum.equiv (Equiv.ofUnique α (Fin 1))
      -/
    · refine heq_of_cast_eq ?_ (Subsingleton.allEq _ _)
      /-
        case h.e_3
        α : Type u
        β : α → Type v
        inst✝ : Unique α
        e : FinEnum α
        ⊢ Eq (Equiv α (Fin (FinEnum.card α))) (Equiv α (Fin 1))
      -/
      exact congrArg (α ≃ Fin ·) <| card_eq_one α
      /-
        🎉 no goals
      -/
      /-
        case h.e_4.h
        α : Type u
        β : α → Type v
        inst✝ : Unique α
        e : FinEnum α
        ⊢ Eq (fun a b => FinEnum.decEq a b) fun a b => decidableEq_of_subsingleton a b
      -/
    · funext x y
      /-
        case h.e_4.h.h.h
        α : Type u
        β : α → Type v
        inst✝ : Unique α
        e : FinEnum α
        x y : α
        ⊢ Eq (FinEnum.decEq x y) (decidableEq_of_subsingleton x y)
      -/
      cases decEq x y <;> cases decidableEq_of_subsingleton x y <;>
      /-
        case h.e_4.h.h.h.isFalse.isFalse
        α : Type u
        β : α → Type v
        inst✝ : Unique α
        e : FinEnum α
        x y : α
        h✝¹ h✝ : Not (Eq x y)
        ⊢ Eq (Decidable.isFalse h✝¹) (Decidable.isFalse h✝)
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
      first | rfl | contradiction
      /-
        🎉 no goals
      -/


/-- A type with unique inhabitant has a trivial enumeration. Not registered as an instance, to make
sure that there aren't two definitionally differing instances around. -/
def ofUnique [Unique α] : FinEnum α := default


theorem mem_pi_toList (xs : List α)
    (f : ∀ a, a ∈ xs → β a) : f ∈ pi xs fun x => toList (β x) :=
  (mem_pi _ _).mpr fun _ _ ↦ mem_toList _


/-- enumerate all functions whose domain and range are finitely enumerable -/
def Pi.enum (β : α → Type*) [∀ a, FinEnum (β a)] : List (∀ a, β a) :=
  (pi (toList α) fun x => toList (β x)).map (fun f x => f x (mem_toList _))


theorem Pi.mem_enum (f : ∀ a, β a) :
                        /-
                          α : Type u_1
                          inst✝¹ : FinEnum α
                          β : α → Type u_2
                          inst✝ : (a : α) → FinEnum (β a)
                          f : (a : α) → β a
                          ⊢ Membership.mem (List.Pi.enum β) f
                        -/
    f ∈ Pi.enum β := by simpa [Pi.enum] using ⟨fun a _ => f a, mem_pi_toList _ _, rfl⟩
                        /-
                          🎉 no goals
                        -/


instance Pi.finEnum : FinEnum (∀ a, β a) :=
  ofList (Pi.enum _) fun _ => Pi.mem_enum _


instance pfunFinEnum (p : Prop) [Decidable p] (α : p → Type) [∀ hp, FinEnum (α hp)] :
    FinEnum (∀ hp : p, α hp) :=
  if hp : p then
                                                  /-
                                                    α✝ : Type u_1
                                                    inst✝³ : FinEnum α✝
                                                    β : α✝ → Type u_2
                                                    inst✝² : (a : α✝) → FinEnum (β a)
                                                    p : Prop
                                                    inst✝¹ : Decidable p
                                                    α : p → Type
                                                    inst✝ : (hp : p) → FinEnum (α hp)
                                                    hp : p
                                                    ⊢ ∀ (x : (hp : p) → α hp), Membership.mem (List.map (fun x x_1 => x) (FinEnum. …
                                                  -/
    ofList ((toList (α hp)).map fun x _ => x) (by intro x; simpa using ⟨x hp, rfl⟩)
                                                           /-
                                                             🎉 no goals
                                                           -/
                                             /-
                                               α✝ : Type u_1
                                               inst✝³ : FinEnum α✝
                                               β : α✝ → Type u_2
                                               inst✝² : (a : α✝) → FinEnum (β a)
                                               p : Prop
                                               inst✝¹ : Decidable p
                                               α : p → Type
                                               inst✝ : (hp : p) → FinEnum (α hp)
                                               hp : Not p
                                               ⊢ ∀ (x : (hp : p) → α hp), Membership.mem (List.cons (fun hp' => ⋯.elim) List. …
                                             -/
  else ofList [fun hp' => (hp hp').elim] (by intro; simp; ext hp'; cases hp hp')
                                                                   /-
                                                                     🎉 no goals
                                                                   -/


